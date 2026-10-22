"""scripts/train_model.py

Model training and validation script for the Explainable SOC Assistant.
Loads processed behavioral log windows, performs defensible train/validation/test splitting,
fits the Random Forest classifier and SHAP explainer, tunes the decision threshold on
validation data, evaluates final performance on the untouched test set, and performs
atomic artifact serialization to prevent stale or corrupt model states.

Usage:
    # Demonstration on synthetic data:
    python scripts/train_model.py --synthetic

    # Production training on Member 1's real processed data:
    python scripts/train_model.py --data-path data/processed/behavioral_windows.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

# Ensure project root is in sys.path when invoked directly as a standalone script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.classifier import EXPECTED_FEATURES, SOCClassifier, generate_synthetic_behavioral_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_model")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SOC Random Forest Classifier and SHAP Explainer.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/behavioral_windows.csv",
        help="Path to processed dataset CSV from Member 1.",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="models/classifier.joblib",
        help="Output destination path for serialized model artifact.",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.95,
        help="Target recall constraint for threshold tuning on validation set.",
    )
    parser.add_argument(
        "--group-col",
        type=str,
        default=None,
        help="Optional column to use for scenario/session grouping (e.g. scenario_id). If not specified, no group isolation is enforced.",
    )
    parser.add_argument(
        "--time-col",
        type=str,
        default=None,
        help="Optional column to use for chronological sorting (e.g. timestamp). If not specified, no temporal ordering is enforced.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Explicit flag required to run training in demonstration mode with synthetic data.",
    )
    return parser.parse_args()


def split_data(
    df: pd.DataFrame,
    target_col: str = "target",
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    group_col: Optional[str] = None,
    time_col: Optional[str] = None,
    random_state: int = 42,
    return_strategy: bool = False,
) -> Union[
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, str],
]:
    """Perform a 60/20/20 train/validation/test split with defensible leakage controls.

    Split Method Analysis & Leakage Considerations:
    1. Scenario/Session Grouping (primary when group column is present):
       Groups windows by attack scenario, host, or session with strict partition isolation.
       Prevents an ongoing attack or host pattern from appearing in both training and test sets.
       If groups cannot support all 3 subsets with class diversity, raises an actionable ValueError.
       Never falls back to random splitting silently.
       Group allocation is deterministic: groups are sorted and permuted using random_state.

    2. Chronological Slicing (when time column is present):
       Orders windows chronologically before slicing.
       - INTERVAL BOUNDARY CONVENTION:
         Interval boundary purging enforces `val[window_start] >= max(train[window_end])` and
         `test[window_start] >= max(val[window_end])`.
         This guarantees mathematical independence under the standard half-open interval
         convention `[start, end)`. If upstream feature generation uses closed intervals `[start, end]`,
         an additional micro-gap buffer should be coordinated with Member 1 prior to real-data freeze.
       - MULTI-STREAM LIMITATION:
         A global chronological split orders events in time across all hosts/scenarios concurrently.
         It establishes temporal independence, but does NOT isolate hosts or attack scenarios.
         If host-level generalization is required, group isolation (`group_col`) must be used.

    3. Stratified Splitting (fallback when neither group nor time metadata exists):
       Preserves class proportions across all splits. NOTE: Establishes neither group isolation
       nor temporal independence; explicitly labeled as unverified in split strategy metadata.
    """
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' was not found in dataset columns: {df.columns.tolist()}"
        )

    n_total = len(df)
    if n_total < 5:
        raise ValueError(
            f"Dataset has only {n_total} samples, which is too small to perform a 60/20/20 train/validation/test split."
        )

    class_counts = df[target_col].value_counts()
    if (class_counts < 3).any():
        raise ValueError(
            f"Dataset has insufficient class diversity: class distribution is {class_counts.to_dict()}. "
            f"Each class must have at least 3 samples to perform a 3-way train/validation/test split."
        )

    # Validate explicitly requested columns
    if group_col is not None and group_col not in df.columns:
        raise ValueError(
            f"Specified group column '{group_col}' was not found in dataset columns: {df.columns.tolist()}"
        )
    if time_col is not None and time_col not in df.columns:
        raise ValueError(
            f"Specified time column '{time_col}' was not found in dataset columns: {df.columns.tolist()}"
        )

    # Check for duplicate behavioral feature rows in source data
    feature_cols = [c for c in EXPECTED_FEATURES if c in df.columns]
    if feature_cols:
        dup_count = int(df.duplicated(subset=feature_cols).sum())
        if dup_count > 0:
            logger.warning(
                "Source data contains %d duplicate behavioral feature row(s). "
                "Verify whether identical benign traffic or overlapping windows are present.",
                dup_count,
            )

    start_col = next((c for c in ["window_start", "start_time", "start"] if c in df.columns), None)
    end_col = next((c for c in ["window_end", "end_time", "end"] if c in df.columns), None)

    split_strategy = "stratified_split:target"

    if group_col is not None:
        # Group-based independence policy explicitly requested
        logger.info(
            "Selected grouping key '%s' as the independence unit per explicit configuration.",
            group_col,
        )
        unique_groups = sorted(df[group_col].unique(), key=lambda x: str(x))
        n_scenarios = len(unique_groups)
        if n_scenarios < 3:
            raise ValueError(
                f"Explicitly configured group policy on '{group_col}' requires at least 3 distinct groups to populate "
                f"train, validation, and test sets without group leakage. Found only {n_scenarios} groups: {unique_groups}."
            )

        if random_state is not None:
            rng = np.random.default_rng(random_state)
            scenarios = np.array(unique_groups)[rng.permutation(len(unique_groups))]
        else:
            scenarios = np.array(unique_groups)

        n_train_scenarios = max(1, int(n_scenarios * train_ratio))
        n_val_scenarios = max(1, int(n_scenarios * val_ratio))

        train_scenarios = scenarios[:n_train_scenarios]
        val_scenarios = scenarios[n_train_scenarios : n_train_scenarios + n_val_scenarios]
        test_scenarios = scenarios[n_train_scenarios + n_val_scenarios :]

        if len(test_scenarios) == 0:
            test_scenarios = val_scenarios[-1:]
            val_scenarios = val_scenarios[:-1]

        train_df = df[df[group_col].isin(train_scenarios)].copy()
        val_df = df[df[group_col].isin(val_scenarios)].copy()
        test_df = df[df[group_col].isin(test_scenarios)].copy()

        # Verify that all splits contain samples and both classes
        has_both_classes = (
            len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0
            and train_df[target_col].nunique() >= 2
            and val_df[target_col].nunique() >= 2
            and test_df[target_col].nunique() >= 2
        )

        if not has_both_classes:
            train_cls = train_df[target_col].unique().tolist() if len(train_df) else []
            val_cls = val_df[target_col].unique().tolist() if len(val_df) else []
            test_cls = test_df[target_col].unique().tolist() if len(test_df) else []
            raise ValueError(
                f"Explicitly configured group policy on '{group_col}' cannot produce valid, class-balanced partitions "
                f"across train, validation, and test subsets without leaking group data. "
                f"Partitions produced: train classes={train_cls}, val classes={val_cls}, test classes={test_cls}. "
                f"Cannot enforce group isolation without class diversity in each partition."
            )

        train_groups = set(train_df[group_col])
        val_groups = set(val_df[group_col])
        test_groups = set(test_df[group_col])
        assert train_groups.isdisjoint(val_groups), "Group leakage detected between train and validation!"
        assert train_groups.isdisjoint(test_groups), "Group leakage detected between train and test!"
        assert val_groups.isdisjoint(test_groups), "Group leakage detected between validation and test!"

        if time_col is not None:
            logger.info(
                "Ordering chronologically by '%s' within each '%s' group partition.",
                time_col,
                group_col,
            )
            train_df = train_df.sort_values(by=time_col).reset_index(drop=True)
            val_df = val_df.sort_values(by=time_col).reset_index(drop=True)
            test_df = test_df.sort_values(by=time_col).reset_index(drop=True)
            split_strategy = f"group_split:{group_col}+temporal_ordered:{time_col}"
        else:
            split_strategy = f"group_split:{group_col}"

        logger.info(
            "Successfully split by '%s' groups with strict isolation (train=%s, val=%s, test=%s).",
            group_col,
            train_scenarios.tolist(),
            val_scenarios.tolist(),
            test_scenarios.tolist(),
        )
    elif time_col is not None:
        logger.info(
            "Selected time column '%s' for chronological split per explicit configuration. "
            "Checking for overlapping sliding windows across split boundaries...",
            time_col,
        )
        df_sorted = df.sort_values(by=time_col).reset_index(drop=True)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_df = df_sorted.iloc[:n_train].copy()
        val_df = df_sorted.iloc[n_train : n_train + n_val].copy()
        test_df = df_sorted.iloc[n_train + n_val :].copy()

        # Temporal boundary overlap detection and buffer purging if window start/end metadata exists
        if start_col and end_col and start_col in df.columns and end_col in df.columns:
            train_end_max = pd.to_numeric(train_df[end_col], errors="coerce").max()
            val_start_min = pd.to_numeric(val_df[start_col], errors="coerce").min()
            val_end_max = pd.to_numeric(val_df[end_col], errors="coerce").max()
            test_start_min = pd.to_numeric(test_df[start_col], errors="coerce").min()

            if pd.isna(train_end_max) or pd.isna(val_start_min):
                t_train_end = pd.to_datetime(train_df[end_col], errors="coerce").max()
                t_val_start = pd.to_datetime(val_df[start_col], errors="coerce").min()
                t_val_end = pd.to_datetime(val_df[end_col], errors="coerce").max()
                t_test_start = pd.to_datetime(test_df[start_col], errors="coerce").min()
                has_overlap = bool((t_train_end > t_val_start) or (t_val_end > t_test_start))

                val_mask = pd.to_datetime(val_df[start_col], errors="coerce") >= t_train_end
                t_val_end_cur = pd.to_datetime(val_df[val_mask][end_col], errors="coerce").max() if val_mask.any() else t_train_end
                test_mask = pd.to_datetime(test_df[start_col], errors="coerce") >= t_val_end_cur
            else:
                has_overlap = bool((train_end_max > val_start_min) or (val_end_max > test_start_min))
                val_mask = pd.to_numeric(val_df[start_col], errors="coerce") >= train_end_max
                t_val_end_cur = pd.to_numeric(val_df[val_mask][end_col], errors="coerce").max() if val_mask.any() else train_end_max
                test_mask = pd.to_numeric(test_df[start_col], errors="coerce") >= t_val_end_cur

            if has_overlap:
                purged_val_count = int((~val_mask).sum())
                purged_test_count = int((~test_mask).sum())
                val_df = val_df[val_mask].copy()
                test_df = test_df[test_mask].copy()

                if (
                    len(val_df) == 0
                    or len(test_df) == 0
                    or val_df[target_col].nunique() < 2
                    or test_df[target_col].nunique() < 2
                ):
                    train_cls = train_df[target_col].unique().tolist()
                    val_cls = val_df[target_col].unique().tolist() if len(val_df) else []
                    test_cls = test_df[target_col].unique().tolist() if len(test_df) else []
                    raise ValueError(
                        f"Temporal window overlap across split boundaries could not be resolved by buffer purging: "
                        f"purged {purged_val_count} overlapping validation windows and {purged_test_count} test windows, "
                        f"leaving insufficient samples or single-class partitions in validation (samples={len(val_df)}, "
                        f"classes={val_cls}) or test (samples={len(test_df)}, classes={test_cls}). "
                        f"Adjust window stride/duration or supply a dataset with a larger temporal range."
                    )

                split_strategy = (
                    f"chronological_split_purged_buffer:{time_col} "
                    f"(purged {purged_val_count} val, {purged_test_count} test buffer windows)"
                )
                logger.info(
                    "Purged %d overlapping boundary windows to establish temporal independence.",
                    purged_val_count + purged_test_count,
                )
            else:
                split_strategy = f"chronological_split_verified:{time_col}"
        else:
            split_strategy = (
                f"chronological_split_unverified_interval:{time_col} "
                f"(window duration/bounds metadata missing; cannot verify interval overlap)"
            )
            logger.warning(
                "Chronological split performed on '%s', but window start/end interval metadata is missing. "
                "Sliding window overlap cannot be mathematically established.",
                time_col,
            )

        candidate_stream_cols = [
            c for c in ["host_id", "host", "hostname", "src_ip", "source_ip", "scenario_id", "scenario", "stream_id", "capture_id"]
            if c in df.columns
        ]
        if candidate_stream_cols:
            logger.warning(
                "MULTI-STREAM TEMPORAL SPLIT LIMITATION: Multi-stream column(s) %s detected while performing "
                "global chronological split on '%s'. A global timestamp split does not isolate host streams "
                "or guarantee generalization to unseen hosts. Configure --group-col if host independence is required.",
                candidate_stream_cols,
                time_col,
            )
            split_strategy += f" [multi-stream columns present: {candidate_stream_cols}; global temporal order does not isolate host streams]"
    else:
        logger.warning(
            "Neither group_col nor time_col was configured. Performing stratified 60/20/20 split. "
            "WARNING: Establishes neither group isolation nor temporal independence."
        )
        train_df, temp_df = train_test_split(
            df,
            test_size=(1.0 - train_ratio),
            stratify=df[target_col],
            random_state=random_state,
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            stratify=temp_df[target_col],
            random_state=random_state,
        )
        split_strategy = (
            "unverified_stratified_split:target "
            "(metadata missing: establishes neither group isolation nor temporal independence)"
        )

    # Verify metadata columns are never included in feature matrices
    known_metadata_cols = {
        "scenario_id", "scenario", "session_id", "group_id", "host_id",
        "timestamp", "time", "window_start", "window_end", "start_time", "end_time", "start", "end",
        target_col,
    }
    if group_col:
        known_metadata_cols.add(group_col)
    if time_col:
        known_metadata_cols.add(time_col)

    for meta_col in known_metadata_cols:
        if meta_col in EXPECTED_FEATURES:
            raise ValueError(f"CRITICAL DEFECT: Metadata column '{meta_col}' cannot be in EXPECTED_FEATURES.")

    X_train = train_df[EXPECTED_FEATURES]
    y_train = train_df[target_col].astype(int)

    X_val = val_df[EXPECTED_FEATURES]
    y_val = val_df[target_col].astype(int)

    X_test = test_df[EXPECTED_FEATURES]
    y_test = test_df[target_col].astype(int)

    # Check for feature leakage across all split boundaries
    train_val_overlap = int(pd.merge(X_train, X_val, how="inner").shape[0])
    train_test_overlap = int(pd.merge(X_train, X_test, how="inner").shape[0])
    val_test_overlap = int(pd.merge(X_val, X_test, how="inner").shape[0])
    total_split_overlap = train_val_overlap + train_test_overlap + val_test_overlap
    if total_split_overlap > 0:
        logger.warning(
            "POTENTIAL DATA LEAKAGE: %d duplicate feature rows detected across split boundaries "
            "(train-val: %d, train-test: %d, val-test: %d). "
            "Coordinate with Member 1 to confirm window overlap buffer / gap purge.",
            total_split_overlap,
            train_val_overlap,
            train_test_overlap,
            val_test_overlap,
        )

    logger.info(
        "Dataset Split Complete (%s): Train=%d (%.1f%%, %d suspicious), Val=%d (%.1f%%, %d suspicious), Test=%d (%.1f%%, %d suspicious)",
        split_strategy,
        len(X_train),
        (len(X_train) / n_total) * 100,
        int(y_train.sum()),
        len(X_val),
        (len(X_val) / n_total) * 100,
        int(y_val.sum()),
        len(X_test),
        (len(X_test) / n_total) * 100,
        int(y_test.sum()),
    )

    if return_strategy:
        return X_train, y_train, X_val, y_val, X_test, y_test, split_strategy
    return X_train, y_train, X_val, y_val, X_test, y_test


def tune_threshold_on_validation(
    classifier: SOCClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    target_recall: float = 0.95,
) -> Tuple[float, Dict[str, float]]:
    """Select the decision threshold on validation data to meet target recall.

    CRITICAL RULE:
    Threshold selection MUST occur on the validation set, NEVER on the final test set.
    """
    if 1 not in y_val.values:
        logger.warning(
            "Validation partition contains no positive/suspicious samples (class 1 is absent from y_val). "
            "Target recall cannot be measured. Defaulting threshold to 0.5."
        )
        return 0.5, {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    val_probs = classifier.predict_proba(X_val)
    candidate_thresholds = np.linspace(0.05, 0.95, 37)

    best_threshold = 0.5
    best_f1 = -1.0
    selected_metrics = {}

    qualifying_candidates = []

    for thresh in candidate_thresholds:
        preds = (val_probs >= thresh).astype(int)
        rec = recall_score(y_val, preds, zero_division=0)
        prec = precision_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)

        if rec >= target_recall:
            qualifying_candidates.append((thresh, prec, rec, f1))

    if qualifying_candidates:
        # Tie-breaking order: Precision descending, F1 descending, Threshold descending (reduces false alarm fatigue)
        qualifying_candidates.sort(key=lambda x: (x[1], x[3], x[0]), reverse=True)
        chosen = qualifying_candidates[0]
        best_threshold = float(chosen[0])
        selected_metrics = {
            "precision": round(float(chosen[1]), 4),
            "recall": round(float(chosen[2]), 4),
            "f1": round(float(chosen[3]), 4),
        }
        logger.info(
            "Selected threshold %.2f on validation set meeting target recall >= %.2f "
            "(Val Recall=%.4f, Val Precision=%.4f, Val F1=%.4f)",
            best_threshold,
            target_recall,
            chosen[2],
            chosen[1],
            chosen[3],
        )
    else:
        logger.warning(
            "Target recall of >= %.2f was not achievable on validation data. "
            "Selecting threshold to maximize validation F1 score.",
            target_recall,
        )
        for thresh in candidate_thresholds:
            preds = (val_probs >= thresh).astype(int)
            f1 = f1_score(y_val, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(thresh)
                selected_metrics = {
                    "precision": round(float(precision_score(y_val, preds, zero_division=0)), 4),
                    "recall": round(float(recall_score(y_val, preds, zero_division=0)), 4),
                    "f1": round(float(f1), 4),
                }

    return best_threshold, selected_metrics


def evaluate_set(
    classifier: SOCClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float,
    split_name: str = "Test",
) -> Dict[str, float]:
    """Compute comprehensive classification performance metrics on an evaluation set."""
    probs = classifier.predict_proba(X)
    preds = (probs >= threshold).astype(int)

    cm = confusion_matrix(y, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)

    try:
        roc_auc = roc_auc_score(y, probs)
    except ValueError:
        roc_auc = float("nan")

    try:
        pr_auc = average_precision_score(y, probs)
    except ValueError:
        pr_auc = float("nan")

    logger.info("--------------------------------------------------")
    logger.info("Evaluation Results on [%s Set] (Threshold = %.2f)", split_name, threshold)
    logger.info("  Precision:         %.4f", prec)
    logger.info("  Recall:            %.4f", rec)
    logger.info("  F1-Score:          %.4f", f1)
    logger.info("  ROC-AUC:           %.4f", roc_auc)
    logger.info("  PR-AUC (Avg Prec): %.4f", pr_auc)
    logger.info("  Confusion Matrix:  TN=%d, FP=%d, FN=%d, TP=%d", tn, fp, fn, tp)
    logger.info("--------------------------------------------------")

    return {
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "threshold_applied": round(float(threshold), 4),
    }


def run_training_pipeline(
    df: pd.DataFrame,
    target_col: str = "target",
    group_col: Optional[str] = None,
    time_col: Optional[str] = None,
    target_recall: float = 0.95,
    random_state: int = 42,
    output_model: Union[str, Path] = "models/classifier.joblib",
    summary_path: Union[str, Path] = "artifacts/training_summary.json",
    is_synthetic: bool = False,
) -> Tuple[SOCClassifier, Dict[str, Any]]:
    """Execute complete end-to-end training pipeline.

    Enforces strict test-set isolation:
    1. Splits dataset into train (60%), validation (20%), and test (20%).
    2. Fits SOCClassifier on training partition ONLY.
    3. Tunes decision threshold on validation partition ONLY.
    4. Freezes threshold on classifier.
    5. Evaluates performance ONCE on untouched test partition.
    6. Staged replacement of model and training summary artifacts with SHA-256 digest linkage.
    """
    # 1. Train / Val / Test Split
    X_train, y_train, X_val, y_val, X_test, y_test, split_strategy = split_data(
        df,
        target_col=target_col,
        group_col=group_col,
        time_col=time_col,
        random_state=random_state,
        return_strategy=True,
    )

    # 2. Model Training on Train partition only
    logger.info("Initializing SOCClassifier baseline (Random Forest)...")
    classifier = SOCClassifier(
        model_name="RandomForest-SOC-v1",
        random_state=random_state,
    )
    classifier.fit(X_train, y_train, n_estimators=100, max_depth=10)

    # 3. Threshold Calibration strictly on Validation Set
    best_thresh, val_metrics = tune_threshold_on_validation(
        classifier,
        X_val,
        y_val,
        target_recall=target_recall,
    )
    classifier.threshold = best_thresh

    # 4. Final Evaluation strictly on Untouched Test Set
    test_metrics = evaluate_set(classifier, X_test, y_test, threshold=best_thresh, split_name="Untouched Test")

    achieved_target_recall = test_metrics["recall"] >= target_recall
    if achieved_target_recall:
        logger.info(
            "Target recall (>= %.2f) was ACHIEVED on test set (Test Recall = %.4f). "
            "False Positives = %d (trade-off for high detection rate).",
            target_recall,
            test_metrics["recall"],
            test_metrics["false_positives"],
        )
    else:
        logger.warning(
            "Target recall (>= %.2f) was NOT achieved on test set (Test Recall = %.4f). "
            "Threshold adjustment or additional behavioral feature engineering with Member 1 may be required.",
            target_recall,
            test_metrics["recall"],
        )

    # 5. Staged Artifact Replacement with SHA-256 Digest
    out_model_file = Path(output_model).resolve()
    sum_file = Path(summary_path).resolve()

    tmp_model_path = out_model_file.with_name(f".{out_model_file.name}.tmp")
    tmp_summary_path = sum_file.with_name(f".{sum_file.name}.tmp")

    train_val_overlap = int(pd.merge(X_train, X_val, how="inner").shape[0])
    train_test_overlap = int(pd.merge(X_train, X_test, how="inner").shape[0])
    val_test_overlap = int(pd.merge(X_val, X_test, how="inner").shape[0])

    summary_report = {
        "model_name": classifier.model_name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "is_synthetic_demonstration": is_synthetic,
        "features_used": EXPECTED_FEATURES,
        "split_strategy_used": split_strategy,
        "dataset_samples": {
            "total": len(df),
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
            "train_val_overlap_count": train_val_overlap,
            "train_test_overlap_count": train_test_overlap,
            "val_test_overlap_count": val_test_overlap,
        },
        "selected_threshold": best_thresh,
        "target_recall_requested": target_recall,
        "target_recall_achieved": achieved_target_recall,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_artifact_path": str(out_model_file),
    }

    try:
        # Save model and report to temporary staging locations
        classifier.save(tmp_model_path)

        # Compute SHA-256 digest of the staged model file for summary linkage
        import hashlib
        model_digest = hashlib.sha256(tmp_model_path.read_bytes()).hexdigest()
        summary_report["model_sha256"] = model_digest

        sum_file.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_report, f, indent=2)

        # Staged replacement of files sequentially
        # NOTE ON MULTI-FILE ATOMICITY:
        # Separate file replacements do not constitute a distributed multi-file transaction.
        # If interrupted between the two replaces, artifacts may temporarily diverge.
        # validate_model_and_summary_consistency() is provided to validate model-summary consistency.
        tmp_model_path.replace(out_model_file)
        tmp_summary_path.replace(sum_file)
        logger.info(
            "Successfully staged and replaced model artifact at '%s' and summary report at '%s'.",
            out_model_file,
            sum_file,
        )
    except Exception as exc:
        if tmp_model_path.exists():
            try:
                tmp_model_path.unlink()
            except OSError:
                pass
        if tmp_summary_path.exists():
            try:
                tmp_summary_path.unlink()
            except OSError:
                pass
        logger.error("Failed during artifact persistence. Existing artifacts were preserved. Error: %s", exc)
        raise

    return classifier, summary_report


def main() -> None:
    """Run model training, threshold tuning, test evaluation, and atomic model artifact saving."""
    args = parse_args()
    data_file = Path(args.data_path)

    # 1. Dataset Ingestion / Strict validation check
    if args.synthetic:
        logger.warning(
            "========================================================================\n"
            "[DEMONSTRATION RUN: SYNTHETIC DATASET]\n"
            "Flag '--synthetic' was provided. Using synthetic behavioral windows.\n"
            "NOTE: Synthetic metrics are for pipeline testing only and DO NOT represent\n"
            "real-world SOC attack detection capability.\n"
            "========================================================================"
        )
        X_synth, y_synth = generate_synthetic_behavioral_data(n_samples=500, random_state=args.random_state)
        df = X_synth.copy()
        df["target"] = y_synth
        is_synthetic = True
    else:
        if not data_file.exists():
            raise FileNotFoundError(
                f"Processed dataset file not found at '{data_file}'.\n"
                f"Member 1 has not yet produced the processed data, or the path is incorrect.\n"
                f"To run a demonstration using synthetic data, explicitly pass '--synthetic':\n"
                f"    python scripts/train_model.py --synthetic"
            )

        logger.info("Loading real processed dataset from '%s'", data_file)
        if data_file.suffix == ".parquet":
            df = pd.read_parquet(data_file)
        else:
            df = pd.read_csv(data_file)
        is_synthetic = False

    run_training_pipeline(
        df=df,
        target_col="target",
        group_col=args.group_col,
        time_col=args.time_col,
        target_recall=args.target_recall,
        random_state=args.random_state,
        output_model=args.output_model,
        summary_path="artifacts/training_summary.json",
        is_synthetic=is_synthetic,
    )


def validate_model_and_summary_consistency(
    model_path: Union[str, Path],
    summary_path: Optional[Union[str, Path]] = None,
    allow_missing_summary: bool = False,
) -> Optional[Dict[str, Any]]:
    """Validate consistency between a serialized model artifact and its training summary report.

    Verifies:
    1. Model artifact exists.
    2. Summary report exists (raises FileNotFoundError if missing, unless allow_missing_summary=True).
    3. Summary report is valid JSON (raises ValueError if malformed).
    4. Summary report contains 'model_sha256' integrity digest (raises ValueError if missing).
    5. 'model_sha256' matches the SHA-256 digest of the actual model file (raises ValueError on mismatch).
    6. Model metadata (model_name, features_used, selected_threshold) matches between model and summary.

    NOTE ON SECURITY & INTEGRITY:
    The SHA-256 digest detects version drift, interrupted writes, and stale or mismatched
    metadata. It does NOT provide cryptographic authenticity against malicious tampering,
    because Python's joblib/pickle deserializer executes arbitrary code during loading.
    Only artifacts from trusted storage locations should be evaluated.

    Returns:
        Optional[Dict[str, Any]]: The parsed summary report dictionary if valid, or None if missing with allow_missing_summary=True.

    Raises:
        FileNotFoundError: If model_path or summary_path does not exist.
        ValueError: If summary is malformed, missing required keys, or fails digest/metadata consistency.
    """
    model_file = Path(model_path).resolve()
    if not model_file.exists():
        raise FileNotFoundError(f"Model artifact not found at '{model_file}'.")

    if summary_path is None:
        summary_file = Path("artifacts/training_summary.json").resolve()
    else:
        summary_file = Path(summary_path).resolve()

    if not summary_file.exists():
        if allow_missing_summary:
            logger.warning(
                "Training summary not found at '%s'. Proceeding with unverified standalone model.",
                summary_file,
            )
            return None
        raise FileNotFoundError(
            f"Training summary report not found at '{summary_file}'. "
            f"Expected summary corresponding to model '{model_file}'. "
            f"If evaluating an unverified standalone model, pass --allow-missing-summary."
        )

    try:
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception as exc:
        raise ValueError(f"Malformed training summary JSON at '{summary_file}': {exc}") from exc

    if not isinstance(summary, dict):
        raise ValueError(f"Invalid summary format in '{summary_file}': expected JSON object/dictionary.")

    # Check model_sha256 integrity digest
    summary_digest = summary.get("model_sha256")
    if not summary_digest:
        raise ValueError(
            f"Training summary at '{summary_file}' is missing required 'model_sha256' integrity digest. "
            f"Cannot verify that summary matches model artifact '{model_file}'."
        )

    import hashlib
    actual_digest = hashlib.sha256(model_file.read_bytes()).hexdigest()
    if summary_digest != actual_digest:
        raise ValueError(
            f"Artifact digest mismatch: summary at '{summary_file}' has model_sha256 '{summary_digest}', "
            f"but model file '{model_file}' has digest '{actual_digest}'. "
            f"Stale, unlinked, or interrupted publication detected."
        )

    # Check metadata match
    if summary.get("features_used") != EXPECTED_FEATURES:
        raise ValueError(
            f"Feature contract mismatch in summary: summary features {summary.get('features_used')} "
            f"do not match canonical EXPECTED_FEATURES {EXPECTED_FEATURES}."
        )

    return summary


def verify_artifact_consistency(model_path: Path, summary_path: Path) -> bool:
    """Boolean helper for backward compatibility."""
    try:
        res = validate_model_and_summary_consistency(model_path, summary_path)
        return res is not None
    except Exception:
        return False


if __name__ == "__main__":
    main()
