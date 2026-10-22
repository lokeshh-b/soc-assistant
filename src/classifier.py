"""src/classifier.py

Machine Learning classification and SHAP explainability module.
Responsible for loading trained ML models, performing threat/anomaly inference
on log feature windows, and computing local/global SHAP feature attributions.

Architectural Note:
Kept strictly independent from GenAI agents and dashboard UI presentation logic.
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier

from src.schemas import ClassifierPrediction, FeatureContribution

logger = logging.getLogger(__name__)

# =============================================================================
# Behavioral Feature Specification & Physical Constraints Contract
# =============================================================================
# The project specification lists 13 behavioral features extracted over sliding log windows.
# NOTE FOR TEAM COORDINATION: The project plan mentions "14 features" in passing
# but enumerates only 13 features below. We do NOT invent a 14th feature.
# This discrepancy is flagged for Member 1 (Data Prep Lead) to confirm before
# final dataset freeze.
EXPECTED_FEATURES: List[str] = [
    "total_event_count",
    "events_per_second_max",
    "byte_length_mean",
    "failed_auth_count",
    "accepted_auth_count",
    "failed_to_success_ratio",
    "distinct_usernames_count",
    "root_or_admin_attempt_flag",
    "distinct_dest_ports_count",
    "ephemeral_port_ratio",
    "privileged_port_ratio",
    "unique_drain_templates_count",
    "rare_template_ratio",
]

# Sensitive or leakage-prone columns that must never be used as predictive features
LEAKAGE_COLUMNS: List[str] = [
    "timestamp",
    "time",
    "start_time",
    "end_time",
    "source_ip",
    "src_ip",
    "destination_ip",
    "dest_ip",
    "hostname",
    "host",
    "user",
    "username",
    "label",
    "target",
    "is_attack",
    "attack_type",
]

# Feature physical constraints specification
# Each feature has individual constraints based on its physical meaning:
# - Counts and rates: [0.0, inf)
# - failed_to_success_ratio: [0.0, inf). May legitimately exceed 1.0 during brute-force attacks!
# - Unit-interval ratios: [0.0, 1.0]
# - Binary flags: in {0.0, 1.0}
FEATURE_CONSTRAINTS: Dict[str, Dict[str, Any]] = {
    "total_event_count": {"min": 0.0, "max": None, "type": "count"},
    "events_per_second_max": {"min": 0.0, "max": None, "type": "rate"},
    "byte_length_mean": {"min": 0.0, "max": None, "type": "continuous"},
    "failed_auth_count": {"min": 0.0, "max": None, "type": "count"},
    "accepted_auth_count": {"min": 0.0, "max": None, "type": "count"},
    "failed_to_success_ratio": {
        "min": 0.0,
        "max": None,
        "type": "unbounded_ratio",
        "description": "Ratio of failed to successful authentications. Can legitimately be >> 1.0 during attacks.",
    },
    "distinct_usernames_count": {"min": 0.0, "max": None, "type": "count"},
    "root_or_admin_attempt_flag": {"min": 0.0, "max": 1.0, "type": "binary"},
    "distinct_dest_ports_count": {"min": 0.0, "max": None, "type": "count"},
    "ephemeral_port_ratio": {"min": 0.0, "max": 1.0, "type": "unit_ratio"},
    "privileged_port_ratio": {"min": 0.0, "max": 1.0, "type": "unit_ratio"},
    "unique_drain_templates_count": {"min": 0.0, "max": None, "type": "count"},
    "rare_template_ratio": {"min": 0.0, "max": 1.0, "type": "unit_ratio"},
}


class SOCClassifier:
    """Random Forest Classifier and SHAP Explainer for Behavioral SOC Log Windows.

    This class encapsulates:
    - Feature validation, type checking, and physical constraint sanitization.
    - Model training with reproducible seeds and class-imbalance weighting.
    - Threat risk score calculation (suspicious-class probability).
      NOTE: The risk score represents uncalibrated Random Forest tree voting probability
      for class 1 (Suspicious); it is not an empirical real-world attack probability.
    - Configurable decision thresholding.
    - Local SHAP feature attribution generation for model transparency.
    - Serialization and deserialization via joblib.
    """

    def __init__(
        self,
        model_name: str = "RandomForest-SOC-v1",
        threshold: float = 0.5,
        random_state: int = 42,
        feature_names: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize the SOCClassifier.

        Args:
            model_name: Identifier name for the model.
            threshold: Decision threshold for mapping risk score to binary class [0.0, 1.0].
                       Default is 0.5. Values >= threshold are flagged Suspicious (1).
            random_state: Seed for reproducibility across trees and splits.
            feature_names: Ordered list of feature column names. Defaults to EXPECTED_FEATURES.

        Raises:
            ValueError: If threshold is not in [0.0, 1.0].
        """
        threshold_float = float(threshold)
        if not (0.0 <= threshold_float <= 1.0):
            raise ValueError(f"Decision threshold must be within [0.0, 1.0]. Got: {threshold}")

        self.model_name = model_name
        self.threshold = threshold_float
        self.random_state = int(random_state)

        # Enforce canonical 13-feature contract: reject any conflicting feature specification
        if feature_names is not None and list(feature_names) != EXPECTED_FEATURES:
            raise ValueError(
                f"Conflicting feature contract provided to SOCClassifier: {list(feature_names)}. "
                f"The canonical 13-feature contract must strictly match: {EXPECTED_FEATURES}."
            )
        self.feature_names = list(EXPECTED_FEATURES)

        self.model: Optional[RandomForestClassifier] = None
        self.explainer: Optional[shap.TreeExplainer] = None
        self.impute_values_: Dict[str, float] = {feat: 0.0 for feat in self.feature_names}
        self.last_explanation_statuses_: List[str] = []

    # -------------------------------------------------------------------------
    # Input Validation & Physical Constraint Sanitization
    # -------------------------------------------------------------------------
    def validate_features(self, df: Union[pd.DataFrame, pd.Series, Dict[str, Any]]) -> pd.DataFrame:
        """Validate that input data contains required behavioral features and applies physical constraints.

        Handles:
        - Type conversion from dict, Series, or DataFrame.
        - Checks for missing required features (raises ValueError).
        - Deliberately filters out extra/metadata columns with informative debug logging.
        - Validates numeric types and raises ValueError on unparseable non-numeric strings.
        - Resolves division-by-zero in failed_to_success_ratio (when accepted_auth_count == 0).
        - Preserves legitimate failed_to_success_ratio values greater than 1.0 without upper clamping.
        - Sanitizes infinite values (np.inf, -np.inf) by replacing with NaN and imputing.
        - Imputes missing/NaN values using training statistics (or 0.0 default).
        - Enforces lower bounds (>= 0.0) on count/rate features.
        - Enforces [0.0, 1.0] bounds on unit-interval ratio features only.
        - Enforces binary flag binarization on root_or_admin_attempt_flag.

        Args:
            df: Input data as a pandas DataFrame, Series, or dictionary.

        Returns:
            pd.DataFrame: Cleaned DataFrame containing strictly the expected feature columns
                          in canonical order, with np.float64 dtypes.

        Raises:
            ValueError: If input is empty, missing required columns, or contains invalid non-numeric data.
        """
        if isinstance(df, dict):
            df = pd.DataFrame([df])
        elif isinstance(df, pd.Series):
            df = df.to_frame().T
        elif not isinstance(df, pd.DataFrame):
            raise ValueError(
                f"Input must be a pandas DataFrame, Series, or dict. Got type: {type(df).__name__}"
            )

        if df.empty:
            raise ValueError("Input DataFrame is empty. At least one sample is required for inference.")

        # Check for duplicate column names to prevent ambiguous extraction
        if df.columns.duplicated().any():
            dup_cols = df.columns[df.columns.duplicated()].unique().tolist()
            raise ValueError(
                f"Input data contains duplicate column names: {dup_cols}. "
                f"All column names must be unique to avoid ambiguous feature selection."
            )

        # Check for missing required behavioral features
        missing_features = [feat for feat in self.feature_names if feat not in df.columns]
        if missing_features:
            raise ValueError(
                f"Missing required behavioral feature(s) ({len(missing_features)}/{len(self.feature_names)}): "
                f"{missing_features}. Expected features are: {self.feature_names}"
            )

        # Handle extra columns deliberately: log their exclusion
        extra_columns = [col for col in df.columns if col not in self.feature_names]
        if extra_columns:
            logger.debug(
                "Input contains %d extra columns not part of the behavioral feature contract: %s. "
                "Excluding them from model inference.",
                len(extra_columns),
                extra_columns,
            )

        # Extract only the expected features in canonical order
        feature_df = df[self.feature_names].copy()

        # Check for unparseable non-numeric types before coercing
        for col in self.feature_names:
            series = feature_df[col]
            if not pd.api.types.is_numeric_dtype(series):
                coerced = pd.to_numeric(series, errors="coerce")
                invalid_mask = series.notna() & coerced.isna()
                if invalid_mask.any():
                    invalid_samples = series[invalid_mask].unique()[:3].tolist()
                    raise ValueError(
                        f"Column '{col}' contains non-numeric data that cannot be converted to float: "
                        f"{invalid_samples}. All behavioral features must be numeric."
                    )
                feature_df[col] = coerced
            else:
                feature_df[col] = pd.to_numeric(series, errors="coerce")

        # Specific domain contract: division-by-zero and indeterminate handling in failed_to_success_ratio.
        # Upstream feature calculation produces two distinct boundary edge cases:
        # 1. Indeterminate 0/0 (failed_auth_count == 0 and accepted_auth_count == 0):
        #    Zero authentication events took place in the window. The mathematical ratio 0/0 is indeterminate (NaN).
        #    EXPLICIT POLICY: Resolved to 0.0 because there is zero evidence of authentication brute-force or failure.
        #    Must NOT fall through to arbitrary median-imputation from training distributions.
        # 2. Division-by-zero x/0 (failed_auth_count > 0 and accepted_auth_count == 0):
        #    Authentication failures occurred with zero successes. Upstream division yields inf (or NaN).
        #    EXPLICIT POLICY: Resolved to failed_auth_count / 1.0 (preserving full brute-force telemetry).
        if "failed_to_success_ratio" in feature_df.columns:
            ratio_series = feature_df["failed_to_success_ratio"]
            failed = feature_df["failed_auth_count"].fillna(0.0) if "failed_auth_count" in feature_df.columns else None
            accepted = feature_df["accepted_auth_count"].fillna(0.0) if "accepted_auth_count" in feature_df.columns else None

            if failed is not None and accepted is not None:
                # Case 1: 0/0 indeterminate ratio -> explicitly 0.0
                zero_zero_mask = (failed == 0.0) & (accepted == 0.0) & (ratio_series.isna() | np.isinf(ratio_series))
                if zero_zero_mask.any():
                    feature_df.loc[zero_zero_mask, "failed_to_success_ratio"] = 0.0
                    logger.info(
                        "Resolved indeterminate 0/0 'failed_to_success_ratio' to 0.0 for %d sample(s) "
                        "(zero authentication events in window; explicit policy avoids arbitrary median imputation).",
                        int(zero_zero_mask.sum()),
                    )

                # Case 2: x/0 division-by-zero -> failed_auth_count / 1.0
                div_zero_mask = (failed > 0.0) & (accepted == 0.0) & (ratio_series.isna() | np.isinf(ratio_series))
                if div_zero_mask.any():
                    feature_df.loc[div_zero_mask, "failed_to_success_ratio"] = failed.loc[div_zero_mask]
                    logger.info(
                        "Resolved division-by-zero 'failed_to_success_ratio' for %d sample(s) "
                        "using failed_auth_count / 1.0 to preserve brute-force signal.",
                        int(div_zero_mask.sum()),
                    )

        # Detect and sanitize any remaining infinite values (np.inf, -np.inf)
        inf_mask = np.isinf(feature_df.values)
        if inf_mask.any():
            inf_cols = [
                self.feature_names[j]
                for j in range(len(self.feature_names))
                if np.isinf(feature_df.iloc[:, j]).any()
            ]
            logger.warning(
                "Input contains infinite values (inf/-inf) in columns: %s. Replacing with NaN and imputing.",
                inf_cols,
            )
            feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

        # Impute missing/NaN values using fitted training medians (or 0.0 default)
        if feature_df.isna().any().any():
            nan_cols = feature_df.columns[feature_df.isna().any()].tolist()
            logger.warning(
                "Input contains NaN/null values in columns: %s. Imputing with learned feature baseline.",
                nan_cols,
            )
            feature_df = feature_df.fillna(value=self.impute_values_)

        # Apply documented individual physical constraints:
        # 1. Non-negative features (counts, rates, and failed_to_success_ratio)
        # failed_to_success_ratio is bounded ONLY at lower=0.0; values > 1.0 are strictly preserved!
        for col, spec in FEATURE_CONSTRAINTS.items():
            if spec["min"] == 0.0 and col in feature_df.columns:
                neg_mask = feature_df[col] < 0.0
                if neg_mask.any():
                    logger.warning(
                        "Negative values detected in non-negative feature '%s' (%d samples). Clipping to 0.0.",
                        col,
                        neg_mask.sum(),
                    )
                    feature_df[col] = feature_df[col].clip(lower=0.0)

        # 2. Unit-interval ratio features: bounded in [0.0, 1.0]
        # (ephemeral_port_ratio, privileged_port_ratio, rare_template_ratio)
        unit_ratio_cols = [col for col, spec in FEATURE_CONSTRAINTS.items() if spec["type"] == "unit_ratio"]
        for col in unit_ratio_cols:
            if col in feature_df.columns:
                out_of_bounds = (feature_df[col] < 0.0) | (feature_df[col] > 1.0)
                if out_of_bounds.any():
                    logger.warning(
                        "Out-of-bounds ratio values detected in '%s' (%d samples). Clipping to [0.0, 1.0].",
                        col,
                        out_of_bounds.sum(),
                    )
                    feature_df[col] = feature_df[col].clip(lower=0.0, upper=1.0)

        # 3. Binary flag: root_or_admin_attempt_flag should be 0.0 or 1.0
        if "root_or_admin_attempt_flag" in feature_df.columns:
            flag_col = feature_df["root_or_admin_attempt_flag"]
            non_binary = ~flag_col.isin([0.0, 1.0])
            if non_binary.any():
                logger.warning(
                    "Non-binary values detected in 'root_or_admin_attempt_flag' (%d samples). "
                    "Binarizing: values > 0 -> 1.0, otherwise 0.0.",
                    non_binary.sum(),
                )
                feature_df["root_or_admin_attempt_flag"] = (flag_col > 0.0).astype(np.float64)

        return feature_df.astype(np.float64)

    # -------------------------------------------------------------------------
    # Model Training
    # -------------------------------------------------------------------------
    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray, Sequence[int]],
        n_estimators: int = 100,
        max_depth: Optional[int] = 12,
        min_samples_split: int = 4,
        class_weight: Optional[str] = "balanced",
        **kwargs: Any,
    ) -> "SOCClassifier":
        """Train the Random Forest classifier and fit the SHAP TreeExplainer.

        Preconditions:
        - X must contain the expected 13 behavioral features.
        - y must be strictly binary integers: 0 (Benign) and 1 (Suspicious).
        - Both classes (0 and 1) must be present in the training data.

        Args:
            X: Training feature matrix.
            y: Target binary labels (0 = Benign, 1 = Suspicious).
            n_estimators: Number of trees in the forest.
            max_depth: Maximum tree depth.
            min_samples_split: Minimum samples required to split an internal node.
            class_weight: Strategy for handling class imbalance ('balanced' recommended).
            **kwargs: Additional parameters passed to RandomForestClassifier.

        Returns:
            self: The fitted SOCClassifier instance.

        Raises:
            ValueError: If target labels are invalid, mismatched in length, or contain < 2 classes.
        """
        X_clean = self.validate_features(X)

        # Validate target labels y: do not silently coerce fractional floats or NaNs
        if pd.isna(y).any():
            raise ValueError("Target labels y contain NaN/null values. All labels must be non-null binary integers.")

        y_raw = np.asarray(y)

        # Reject fractional/floating-point values (e.g. 0.5, 0.8) without silent truncation
        if np.issubdtype(y_raw.dtype, np.floating):
            fractional_mask = ~np.isclose(y_raw, np.round(y_raw), atol=1e-7)
            if np.any(fractional_mask):
                invalid_samples = y_raw[fractional_mask][:3].tolist()
                raise ValueError(
                    f"Target labels y contain fractional/non-integer values: {invalid_samples}. "
                    f"Labels must be strictly binary integers 0 (Benign) or 1 (Suspicious)."
                )

        try:
            y_arr = y_raw.astype(int)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Target labels y cannot be converted to integers: {exc}") from exc

        if len(X_clean) != len(y_arr):
            raise ValueError(
                f"Length mismatch: X has {len(X_clean)} samples, but y has {len(y_arr)} labels."
            )

        unique_classes = np.unique(y_arr)
        # Verify binary classes strictly in {0, 1}
        if not set(unique_classes).issubset({0, 1}):
            raise ValueError(
                f"Invalid target classes: {unique_classes.tolist()}. Target labels must be binary "
                f"integers where 0 = Benign and 1 = Suspicious."
            )

        if len(unique_classes) < 2:
            raise ValueError(
                f"Training data must contain both classes (0 = Benign and 1 = Suspicious) "
                f"to train a binary classifier. Found only class: {unique_classes.tolist()}."
            )

        # Learn feature imputation values strictly from training data
        self.impute_values_ = X_clean.median().to_dict()

        # Initialize and fit scikit-learn Random Forest
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight=class_weight,
            random_state=self.random_state,
            n_jobs=-1,
            **kwargs,
        )
        self.model.fit(X_clean.values, y_arr)

        # Initialize SHAP TreeExplainer
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception as exc:
            logger.error("Failed to initialize SHAP TreeExplainer: %s", exc)
            self.explainer = None

        return self

    # -------------------------------------------------------------------------
    # Inference & Scoring
    # -------------------------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Compute the suspicious-class probability (risk score) for input samples.

        Identifies the exact column index corresponding to class 1 (Suspicious)
        from model.classes_. Raises RuntimeError if class 1 is absent.

        Args:
            X: Input feature DataFrame.

        Returns:
            np.ndarray: 1D array of floats in [0.0, 1.0] representing risk scores.

        Raises:
            RuntimeError: If model is not trained/loaded or cannot produce class 1 probabilities.
        """
        if self.model is None:
            raise RuntimeError("Model is not trained or loaded. Call .fit() or .load() before inference.")

        if not hasattr(self.model, "predict_proba"):
            raise RuntimeError(
                f"Underlying estimator '{type(self.model).__name__}' does not support 'predict_proba'. "
                f"Probability-based risk scoring and thresholding require an estimator with predict_proba."
            )

        classes = getattr(self.model, "classes_", np.array([]))
        if 1 not in classes:
            raise RuntimeError(
                f"Model '{self.model_name}' cannot produce a valid suspicious-class probability: "
                f"class 1 ('Suspicious') is absent from model.classes_ ({classes.tolist()}). "
                f"Cannot triage alerts with a model that lacks the suspicious class."
            )

        X_clean = self.validate_features(X)
        proba = self.model.predict_proba(X_clean.values)

        class_1_idx = int(np.where(classes == 1)[0][0])
        return proba[:, class_1_idx].astype(np.float64)

    def predict(
        self,
        X: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Generate binary class predictions using the configured decision threshold.

        Boundary behavior:
        Samples with risk_score >= threshold are flagged as 1 (Suspicious).
        Samples with risk_score < threshold are classified as 0 (Benign).

        Args:
            X: Input feature DataFrame.
            threshold: Optional custom threshold in [0.0, 1.0] to override instance threshold.

        Returns:
            np.ndarray: 1D array of binary ints (0 = Benign, 1 = Suspicious).

        Raises:
            ValueError: If custom threshold is outside [0.0, 1.0].
            RuntimeError: If model cannot produce valid suspicious-class probabilities.
        """
        if threshold is not None:
            thresh = float(threshold)
            if not (0.0 <= thresh <= 1.0):
                raise ValueError(f"Decision threshold must be within [0.0, 1.0]. Got: {threshold}")
        else:
            thresh = self.threshold

        risk_scores = self.predict_proba(X)
        return (risk_scores >= thresh).astype(int)

    # -------------------------------------------------------------------------
    # SHAP Explanations
    # -------------------------------------------------------------------------
    def explain(
        self,
        X: pd.DataFrame,
        top_k: int = 5,
    ) -> List[List[FeatureContribution]]:
        """Compute SHAP feature attributions for samples relative to the suspicious class.

        Args:
            X: Validated feature DataFrame.
            top_k: Maximum number of top contributing features to return per sample.

        Returns:
            List of lists: For each sample, an ordered list of FeatureContribution objects
                           sorted by absolute SHAP attribution magnitude descending.

        Raises:
            RuntimeError: If model cannot produce suspicious-class explanations.
        """
        top_k_clean = max(0, int(top_k))
        if top_k_clean == 0:
            self.last_explanation_statuses_ = ["bypassed"] * len(X)
            return [[] for _ in range(len(X))]

        if self.model is None:
            logger.warning("Classifier model is not fitted. Explanations cannot be computed.")
            self.last_explanation_statuses_ = ["unavailable"] * len(X)
            return [[] for _ in range(len(X))]

        classes = getattr(self.model, "classes_", np.array([]))
        if len(classes) > 0 and 1 not in classes:
            raise RuntimeError(
                f"Model '{self.model_name}' cannot compute SHAP explanations for suspicious class: "
                f"class 1 ('Suspicious') is absent from model.classes_ ({classes.tolist()})."
            )

        if self.explainer is None:
            logger.warning("SHAP explainer is not available. Explanations cannot be computed.")
            self.last_explanation_statuses_ = ["unavailable"] * len(X)
            return [[] for _ in range(len(X))]

        X_clean = self.validate_features(X)

        try:
            raw_shap_values = self.explainer.shap_values(X_clean.values)
        except Exception as exc:
            logger.error(
                "Error computing SHAP values: %s. Surfacing empty explanations rather than fabricating values.",
                exc,
            )
            self.last_explanation_statuses_ = ["error"] * len(X_clean)
            return [[] for _ in range(len(X_clean))]

        class_1_idx = int(np.where(classes == 1)[0][0])

        # Handle different SHAP output formats across shap/scikit-learn versions:
        # Case A: 3D ndarray of shape (n_samples, n_features, n_classes)
        if isinstance(raw_shap_values, np.ndarray) and raw_shap_values.ndim == 3:
            target_shap = raw_shap_values[:, :, class_1_idx]
        # Case B: List of arrays [class_0_array, class_1_array]
        elif isinstance(raw_shap_values, list) and len(raw_shap_values) > class_1_idx:
            target_shap = raw_shap_values[class_1_idx]
        # Case C: 2D ndarray of shape (n_samples, n_features)
        elif isinstance(raw_shap_values, np.ndarray) and raw_shap_values.ndim == 2:
            target_shap = raw_shap_values
        else:
            logger.warning("Unexpected SHAP output structure (%s). Returning empty explanations.", type(raw_shap_values))
            self.last_explanation_statuses_ = ["error"] * len(X_clean)
            return [[] for _ in range(len(X_clean))]

        all_explanations: List[List[FeatureContribution]] = []
        sample_statuses: List[str] = []
        feature_names = self.feature_names

        # Validate output dimensions match expected feature count
        if target_shap.shape[1] != len(feature_names):
            logger.error(
                "SHAP output feature dimension mismatch: explainer produced %d features, but expected %d. "
                "Rejecting corrupted explanations.",
                target_shap.shape[1],
                len(feature_names),
            )
            self.last_explanation_statuses_ = ["error"] * len(X_clean)
            return [[] for _ in range(len(X_clean))]

        for row_idx in range(len(X_clean)):
            row_features = X_clean.iloc[row_idx]
            row_shap = target_shap[row_idx]

            # Detect non-finite values in SHAP attributions (NaN, +inf, -inf)
            if np.isnan(row_shap).any() or np.isinf(row_shap).any():
                logger.error(
                    "Non-finite SHAP values (NaN or Inf) detected in sample %d. "
                    "Mathematical explanation failed; rejecting explanation for this sample.",
                    row_idx,
                )
                all_explanations.append([])
                sample_statuses.append("error")
                continue

            contributions: List[FeatureContribution] = []
            for feat_idx, feat_name in enumerate(feature_names):
                observed_val = float(row_features[feat_name])
                shap_val = float(row_shap[feat_idx])
                contributions.append(
                    FeatureContribution(
                        feature_name=feat_name,
                        feature_value=observed_val,
                        shap_value=round(shap_val, 6),
                    )
                )

            # Rank features strictly by absolute SHAP attribution magnitude descending
            contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)
            all_explanations.append(contributions[:top_k_clean])
            sample_statuses.append("success")

        self.last_explanation_statuses_ = sample_statuses
        return all_explanations

    # -------------------------------------------------------------------------
    # Structured Prediction Contract
    # -------------------------------------------------------------------------
    def predict_record(
        self,
        record: Union[pd.Series, Dict[str, Any], pd.DataFrame],
        top_k: int = 5,
        threshold: Optional[float] = None,
        explain: bool = True,
    ) -> ClassifierPrediction:
        """Run single-instance inference and return a standardized ClassifierPrediction.

        Args:
            record: Single behavioral log window representation (dict, Series, or single-row DataFrame).
            top_k: Number of top contributing features to include in SHAP explanation.
            threshold: Optional custom threshold in [0.0, 1.0] to override default.
            explain: Whether to compute SHAP explanations.

        Returns:
            ClassifierPrediction: Pydantic schema validated prediction object.
        """
        start_time = time.perf_counter()

        if threshold is not None:
            thresh = float(threshold)
            if not (0.0 <= thresh <= 1.0):
                raise ValueError(f"Decision threshold must be within [0.0, 1.0]. Got: {threshold}")
        else:
            thresh = self.threshold

        X_clean = self.validate_features(record)

        risk_scores = self.predict_proba(X_clean)
        risk_score = float(risk_scores[0])
        predicted_class = int(risk_score >= thresh)

        if explain:
            explanations = self.explain(X_clean, top_k=top_k)
            top_features = explanations[0] if explanations else []
            statuses = getattr(self, "last_explanation_statuses_", ["success"])
            expl_status = statuses[0] if statuses else "unavailable"
        else:
            top_features = []
            expl_status = "bypassed"

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return ClassifierPrediction(
            model_name=self.model_name,
            risk_score=round(risk_score, 4),
            predicted_class=predicted_class,
            threshold_applied=round(thresh, 4),
            top_contributing_features=top_features,
            explanation_status=expl_status,
            inference_latency_ms=round(latency_ms, 2),
        )

    def predict_batch(
        self,
        df: pd.DataFrame,
        top_k: int = 5,
        threshold: Optional[float] = None,
        explain: bool = True,
    ) -> List[ClassifierPrediction]:
        """Run batch inference on multiple behavioral log windows.

        Args:
            df: DataFrame containing behavioral feature columns.
            top_k: Number of top features per explanation.
            threshold: Optional threshold override in [0.0, 1.0].
            explain: Whether to compute SHAP explanations. Set False for faster throughput.

        Returns:
            List[ClassifierPrediction]: Formatted predictions for each row in df.
        """
        start_time = time.perf_counter()

        if threshold is not None:
            thresh = float(threshold)
            if not (0.0 <= thresh <= 1.0):
                raise ValueError(f"Decision threshold must be within [0.0, 1.0]. Got: {threshold}")
        else:
            thresh = self.threshold

        X_clean = self.validate_features(df)
        n_samples = len(X_clean)

        risk_scores = self.predict_proba(X_clean)
        predicted_classes = (risk_scores >= thresh).astype(int)

        if explain:
            explanations = self.explain(X_clean, top_k=top_k)
            statuses = getattr(self, "last_explanation_statuses_", ["success"] * n_samples)
        else:
            explanations = [[] for _ in range(n_samples)]
            statuses = ["bypassed"] * n_samples

        total_elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        avg_latency_ms = round(total_elapsed_ms / max(n_samples, 1), 2)

        predictions: List[ClassifierPrediction] = []
        for i in range(n_samples):
            predictions.append(
                ClassifierPrediction(
                    model_name=self.model_name,
                    risk_score=round(float(risk_scores[i]), 4),
                    predicted_class=int(predicted_classes[i]),
                    threshold_applied=round(thresh, 4),
                    top_contributing_features=explanations[i],
                    explanation_status=statuses[i] if i < len(statuses) else "success",
                    inference_latency_ms=avg_latency_ms,
                )
            )

        return predictions

    # -------------------------------------------------------------------------
    # Model Persistence
    # -------------------------------------------------------------------------
    def save(self, filepath: Union[str, Path]) -> Path:
        """Serialize and save the trained model, explainer, and metadata to disk.

        Args:
            filepath: Destination file path (e.g. 'models/classifier.joblib').

        Returns:
            Path: Absolute path of saved file.

        Raises:
            RuntimeError: If model has not been fitted prior to saving.
        """
        if self.model is None:
            raise RuntimeError("Cannot save an unfitted model. Train the model using .fit() first.")

        save_path = Path(filepath).resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)

        feature_hash = hashlib.sha256(",".join(self.feature_names).encode("utf-8")).hexdigest()[:16]
        payload = {
            "model_name": self.model_name,
            "threshold": self.threshold,
            "random_state": self.random_state,
            "feature_names": self.feature_names,
            "feature_names_hash": feature_hash,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": self.model,
            "explainer": self.explainer,
            "impute_values_": self.impute_values_,
        }

        joblib.dump(payload, save_path)
        logger.info("Saved SOCClassifier artifact to %s", save_path)
        return save_path

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "SOCClassifier":
        """Load a serialized SOCClassifier artifact from disk.

        Validates artifact integrity, checks for required dictionary keys, ensures
        the 13-feature contract matches EXPECTED_FEATURES, and ensures class 1
        (Suspicious) is present in model classes.

        Args:
            filepath: Path to the serialized .joblib artifact.

        Returns:
            SOCClassifier: Loaded and ready-to-infer classifier instance.

        Raises:
            FileNotFoundError: If artifact does not exist at filepath.
            ValueError: If the file is not a valid serialized SOCClassifier artifact,
                        violates the feature contract, or lacks class 1.
        """
        load_path = Path(filepath).resolve()
        if not load_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at '{load_path}'. Ensure the model is trained first."
            )

        try:
            payload = joblib.load(load_path)
        except Exception as exc:
            raise ValueError(f"Failed to deserialize model artifact from '{load_path}': {exc}") from exc

        if not isinstance(payload, dict) or "model" not in payload:
            raise ValueError(f"Invalid artifact format in '{load_path}'. Expected dictionary with 'model' key.")

        # Validate feature contract against expected features
        if "feature_names" not in payload or payload["feature_names"] is None:
            raise ValueError(f"Model artifact in '{load_path}' is missing required 'feature_names' metadata.")

        artifact_features = list(payload["feature_names"])
        if artifact_features != EXPECTED_FEATURES:
            raise ValueError(
                f"Model artifact feature contract mismatch in '{load_path}': "
                f"Artifact was serialized with features {artifact_features}, "
                f"but runtime expects {EXPECTED_FEATURES}."
            )

        # Integrity check: verify feature_names_hash if present
        saved_hash = payload.get("feature_names_hash")
        expected_hash = hashlib.sha256(",".join(EXPECTED_FEATURES).encode("utf-8")).hexdigest()[:16]
        if saved_hash is not None and saved_hash != expected_hash:
            raise ValueError(
                f"Model artifact integrity check failed in '{load_path}': "
                f"feature_names_hash '{saved_hash}' does not match expected schema fingerprint '{expected_hash}'."
            )

        model_obj = payload["model"]
        if not hasattr(model_obj, "predict_proba"):
            raise ValueError(
                f"Artifact 'model' in '{load_path}' is invalid: missing predict_proba method."
            )

        # Check estimator feature dimension matches expected feature count
        n_features_in = getattr(model_obj, "n_features_in_", None)
        if n_features_in is not None and n_features_in != len(EXPECTED_FEATURES):
            raise ValueError(
                f"Underlying estimator in '{load_path}' was trained on {n_features_in} features, "
                f"which does not match runtime expectation of {len(EXPECTED_FEATURES)} features."
            )

        model_classes = getattr(model_obj, "classes_", None)
        if model_classes is not None and 1 not in model_classes:
            raise ValueError(
                f"Loaded model artifact from '{load_path}' is defective: class 1 ('Suspicious') "
                f"is absent from model.classes_ ({model_classes.tolist()})."
            )

        classifier = cls(
            model_name=payload.get("model_name", "RandomForest-SOC"),
            threshold=payload.get("threshold", 0.5),
            random_state=payload.get("random_state", 42),
            feature_names=payload.get("feature_names", EXPECTED_FEATURES),
        )
        classifier.model = model_obj
        classifier.explainer = payload.get("explainer")
        classifier.impute_values_ = payload.get(
            "impute_values_", {feat: 0.0 for feat in classifier.feature_names}
        )

        # Attempt to reconstruct explainer if missing
        if classifier.explainer is None and classifier.model is not None:
            try:
                classifier.explainer = shap.TreeExplainer(classifier.model)
            except Exception as exc:
                logger.warning("Could not reconstruct SHAP TreeExplainer from loaded model: %s", exc)

        return classifier


# =============================================================================
# Synthetic Data Generator (For Testing Prior to Real Data Delivery)
# =============================================================================
def generate_synthetic_behavioral_data(
    n_samples: int = 300,
    attack_ratio: float = 0.25,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic behavioral log window features strictly for testing and validation.

    DISCLAIMER:
    This dataset is synthetic and intended exclusively for interface testing,
    pipeline integration, and threshold selection exercises while Member 1 prepares
    the genuine CAM-LDS/security log dataset. It must NOT be cited as real attack data.

    Args:
        n_samples: Total number of synthetic windows to generate.
        attack_ratio: Proportion of samples labeled as suspicious/attack (1).
        random_state: Seed for reproducible synthetic distributions.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Feature matrix X and binary target labels y.
    """
    rng = np.random.default_rng(random_state)
    n_attack = int(n_samples * attack_ratio)
    n_benign = n_samples - n_attack

    # Benign distribution: low failed auth, normal event volume, low rare templates
    benign_data = {
        "total_event_count": rng.integers(10, 200, size=n_benign),
        "events_per_second_max": rng.uniform(0.5, 10.0, size=n_benign),
        "byte_length_mean": rng.uniform(100.0, 500.0, size=n_benign),
        "failed_auth_count": rng.integers(0, 3, size=n_benign),
        "accepted_auth_count": rng.integers(1, 20, size=n_benign),
        "failed_to_success_ratio": rng.uniform(0.0, 0.2, size=n_benign),
        "distinct_usernames_count": rng.integers(1, 3, size=n_benign),
        "root_or_admin_attempt_flag": rng.choice([0, 1], p=[0.97, 0.03], size=n_benign),
        "distinct_dest_ports_count": rng.integers(1, 5, size=n_benign),
        "ephemeral_port_ratio": rng.uniform(0.1, 0.4, size=n_benign),
        "privileged_port_ratio": rng.uniform(0.6, 0.9, size=n_benign),
        "unique_drain_templates_count": rng.integers(2, 10, size=n_benign),
        "rare_template_ratio": rng.uniform(0.0, 0.05, size=n_benign),
    }

    # Attack distribution: bursty events, high failed logins, port scanning, rare log templates
    # failed_to_success_ratio legitimately exceeds 1.0 during attacks (e.g. 1.5 to 25.0)
    attack_data = {
        "total_event_count": rng.integers(150, 1500, size=n_attack),
        "events_per_second_max": rng.uniform(15.0, 150.0, size=n_attack),
        "byte_length_mean": rng.uniform(50.0, 1200.0, size=n_attack),
        "failed_auth_count": rng.integers(5, 100, size=n_attack),
        "accepted_auth_count": rng.integers(0, 5, size=n_attack),
        "failed_to_success_ratio": rng.uniform(1.5, 25.0, size=n_attack),
        "distinct_usernames_count": rng.integers(3, 25, size=n_attack),
        "root_or_admin_attempt_flag": rng.choice([0, 1], p=[0.3, 0.7], size=n_attack),
        "distinct_dest_ports_count": rng.integers(5, 50, size=n_attack),
        "ephemeral_port_ratio": rng.uniform(0.5, 0.95, size=n_attack),
        "privileged_port_ratio": rng.uniform(0.05, 0.5, size=n_attack),
        "unique_drain_templates_count": rng.integers(8, 30, size=n_attack),
        "rare_template_ratio": rng.uniform(0.15, 0.85, size=n_attack),
    }

    df_benign = pd.DataFrame(benign_data)
    df_benign["target"] = 0

    df_attack = pd.DataFrame(attack_data)
    df_attack["target"] = 1

    combined_df = pd.concat([df_benign, df_attack], ignore_index=True)
    # Shuffle dataset
    shuffled_df = combined_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    X = shuffled_df[EXPECTED_FEATURES]
    y = shuffled_df["target"].astype(int)

    return X, y
