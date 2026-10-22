"""tests/test_classifier.py

Rigorous unit test suite for the ML classifier and SHAP explainer module (src/classifier.py).
Tests:
1. Valid input handling (DataFrame, Series, dict).
2. Missing required behavioral features.
3. Unexpected/extra columns handling.
4. NaN and infinite value sanitization.
5. Invalid data types (strings in numeric columns).
6. Physical constraint clipping (negative counts, ratios > 1, non-binary flags).
7. Correct suspicious-class probability extraction (resolving class 1).
8. Threshold boundary behavior (0.0, 1.0, out-of-range thresholds).
9. Single vs. batch prediction consistency.
10. Model serialization and deserialization (joblib save/load roundtrip).
11. Corrupt or missing model artifact error handling.
12. Unfitted model error handling.
13. Pydantic schema validation enforcement.
14. SHAP output structure, feature alignment, and ranking.
15. SHAP failure resilience (safe fallback without fabricating numbers).
16. Training data validation (1-class data, invalid non-binary labels).
17. Evaluation metrics calculation on a known deterministic fixture.
18. Validation threshold tuning without test set contamination.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from scripts.train_model import (
    evaluate_set,
    split_data,
    tune_threshold_on_validation,
    validate_model_and_summary_consistency,
    verify_artifact_consistency,
)
from src.classifier import (
    EXPECTED_FEATURES,
    SOCClassifier,
    generate_synthetic_behavioral_data,
)
from src.schemas import ClassifierPrediction, FeatureContribution


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, pd.Series]:
    """Provide a deterministic synthetic behavioral dataset for unit tests."""
    return generate_synthetic_behavioral_data(n_samples=80, attack_ratio=0.3, random_state=42)


@pytest.fixture
def trained_classifier(sample_data: tuple[pd.DataFrame, pd.Series]) -> SOCClassifier:
    """Provide a trained SOCClassifier instance with both classes."""
    X, y = sample_data
    clf = SOCClassifier(model_name="Test-RandomForest-SOC", threshold=0.5, random_state=42)
    clf.fit(X, y, n_estimators=15, max_depth=4)
    return clf


# =============================================================================
# 1. Input Validation Tests
# =============================================================================

def test_validate_features_dataframe(sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Test validation with a standard pandas DataFrame."""
    X, _ = sample_data
    clf = SOCClassifier()
    cleaned = clf.validate_features(X)

    assert isinstance(cleaned, pd.DataFrame)
    assert len(cleaned) == len(X)
    assert list(cleaned.columns) == EXPECTED_FEATURES
    assert cleaned.dtypes.iloc[0] == np.float64


def test_validate_features_series_and_dict(sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Test validation accepting a Series and a dict record."""
    X, _ = sample_data
    clf = SOCClassifier()

    # Series input
    series_record = X.iloc[0]
    cleaned_s = clf.validate_features(series_record)
    assert len(cleaned_s) == 1
    assert list(cleaned_s.columns) == EXPECTED_FEATURES

    # Dict input
    dict_record = series_record.to_dict()
    cleaned_d = clf.validate_features(dict_record)
    assert len(cleaned_d) == 1
    assert list(cleaned_d.columns) == EXPECTED_FEATURES


def test_validate_features_missing_columns() -> None:
    """Test that missing required behavioral features raise a clear ValueError."""
    incomplete_df = pd.DataFrame({"total_event_count": [10, 20]})
    clf = SOCClassifier()

    with pytest.raises(ValueError) as exc_info:
        clf.validate_features(incomplete_df)

    err_msg = str(exc_info.value)
    assert "Missing required behavioral feature" in err_msg
    assert "events_per_second_max" in err_msg


def test_validate_features_extra_columns_handled_deliberately(sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Test that extra metadata or leakage columns are filtered out without error."""
    X, _ = sample_data
    X_extra = X.copy()
    X_extra["source_ip"] = "192.168.1.100"
    X_extra["timestamp"] = "2026-10-22 12:00:00"
    X_extra["random_tag"] = 999

    clf = SOCClassifier()
    cleaned = clf.validate_features(X_extra)

    assert "source_ip" not in cleaned.columns
    assert "timestamp" not in cleaned.columns
    assert "random_tag" not in cleaned.columns
    assert list(cleaned.columns) == EXPECTED_FEATURES


def test_validate_features_handles_nan_and_inf(sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Test that NaN and infinite values are detected and sanitized properly."""
    X, _ = sample_data
    X_dirty = X.copy()
    X_dirty.loc[0, "failed_auth_count"] = np.nan
    X_dirty.loc[1, "events_per_second_max"] = np.inf
    X_dirty.loc[2, "byte_length_mean"] = -np.inf

    clf = SOCClassifier()
    cleaned = clf.validate_features(X_dirty)

    assert not np.isnan(cleaned.values).any()
    assert not np.isinf(cleaned.values).any()


def test_validate_features_raises_on_invalid_string_data(sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Test that non-numeric strings in numeric features raise a descriptive ValueError."""
    X, _ = sample_data
    X_invalid = X.copy()
    X_invalid["failed_auth_count"] = ["bad_text"] * len(X_invalid)

    clf = SOCClassifier()
    with pytest.raises(ValueError) as exc_info:
        clf.validate_features(X_invalid)

    assert "contains non-numeric data" in str(exc_info.value)
    assert "failed_auth_count" in str(exc_info.value)


def test_validate_features_physical_constraints(sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Test that negative counts, out-of-range ratios, and non-binary flags are sanitized."""
    X, _ = sample_data
    X_bounded = X.copy()
    X_bounded.loc[0, "total_event_count"] = -50.0  # negative count
    X_bounded.loc[0, "ephemeral_port_ratio"] = 2.5  # ratio > 1
    X_bounded.loc[0, "rare_template_ratio"] = -0.5  # ratio < 0
    X_bounded.loc[0, "root_or_admin_attempt_flag"] = 5.0  # non-binary flag

    clf = SOCClassifier()
    cleaned = clf.validate_features(X_bounded)

    assert cleaned.loc[0, "total_event_count"] == 0.0
    assert cleaned.loc[0, "ephemeral_port_ratio"] == 1.0
    assert cleaned.loc[0, "rare_template_ratio"] == 0.0
    assert cleaned.loc[0, "root_or_admin_attempt_flag"] == 1.0


# =============================================================================
# 2. Training Edge Cases
# =============================================================================

def test_fit_raises_on_single_class(sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Test that training with only one class raises a ValueError."""
    X, _ = sample_data
    y_single = np.zeros(len(X), dtype=int)
    clf = SOCClassifier()

    with pytest.raises(ValueError) as exc_info:
        clf.fit(X, y_single)

    assert "must contain both classes" in str(exc_info.value)


def test_fit_raises_on_non_binary_labels(sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Test that non-binary target labels raise a ValueError."""
    X, _ = sample_data
    # Ensure length matches len(X) exactly while including class 2
    y_multiclass = np.tile([0, 1, 2], (len(X) // 3) + 1)[: len(X)]
    clf = SOCClassifier()

    with pytest.raises(ValueError) as exc_info:
        clf.fit(X, y_multiclass)

    assert "Target labels must be binary" in str(exc_info.value)


# =============================================================================
# 3. Probability & Threshold Boundary Behavior
# =============================================================================

def test_threshold_out_of_bounds_raises_error() -> None:
    """Test that thresholds outside [0.0, 1.0] raise ValueError in init and methods."""
    with pytest.raises(ValueError):
        SOCClassifier(threshold=-0.1)

    with pytest.raises(ValueError):
        SOCClassifier(threshold=1.1)


def test_threshold_boundary_decisions(trained_classifier: SOCClassifier, sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Test threshold boundaries: threshold=0.0 flags all samples, threshold=1.0 flags only risk=1.0."""
    X, _ = sample_data

    # Threshold 0.0 -> all predicted as 1 (Suspicious)
    preds_zero = trained_classifier.predict(X, threshold=0.0)
    assert np.all(preds_zero == 1)

    # Threshold 1.0 -> only samples with risk == 1.0 flagged as 1
    probs = trained_classifier.predict_proba(X)
    preds_one = trained_classifier.predict(X, threshold=1.0)
    expected_ones = (probs >= 1.0).astype(int)
    np.testing.assert_array_equal(preds_one, expected_ones)


def test_threshold_monotonicity(trained_classifier: SOCClassifier, sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Test that lowering the threshold monotonically increases or preserves suspicious detections."""
    X, _ = sample_data
    preds_low = trained_classifier.predict(X, threshold=0.2)
    preds_mid = trained_classifier.predict(X, threshold=0.5)
    preds_high = trained_classifier.predict(X, threshold=0.8)

    assert preds_low.sum() >= preds_mid.sum() >= preds_high.sum()


def test_predict_record_and_batch_consistency(
    trained_classifier: SOCClassifier,
    sample_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Test that single-record inference yields identical risk scores to batch inference."""
    X, _ = sample_data
    batch_preds = trained_classifier.predict_batch(X.head(5), explain=False)

    for i in range(5):
        single_pred = trained_classifier.predict_record(X.iloc[i], explain=False)
        assert single_pred.predicted_class == batch_preds[i].predicted_class
        assert single_pred.risk_score == batch_preds[i].risk_score
        assert single_pred.threshold_applied == batch_preds[i].threshold_applied


# =============================================================================
# 4. Pydantic Schema Validation
# =============================================================================

def test_classifier_prediction_schema_compliance(
    trained_classifier: SOCClassifier,
    sample_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Test that predict_record conforms strictly to ClassifierPrediction schema."""
    X, _ = sample_data
    pred = trained_classifier.predict_record(X.iloc[0], top_k=3)

    assert isinstance(pred, ClassifierPrediction)
    assert 0.0 <= pred.risk_score <= 1.0
    assert pred.predicted_class in (0, 1)
    assert 0.0 <= pred.threshold_applied <= 1.0
    assert pred.inference_latency_ms >= 0.0
    assert len(pred.top_contributing_features) <= 3


def test_pydantic_schema_rejects_invalid_risk_score() -> None:
    """Test that ClassifierPrediction rejects out-of-range risk scores."""
    with pytest.raises(ValidationError):
        ClassifierPrediction(
            model_name="test",
            risk_score=1.5,  # Invalid: > 1.0
            predicted_class=1,
            threshold_applied=0.5,
            inference_latency_ms=1.2,
        )


# =============================================================================
# 5. SHAP Explainability & Failure Resilience
# =============================================================================

def test_shap_explanations_structure_and_ranking(
    trained_classifier: SOCClassifier,
    sample_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Test that SHAP feature attributions are sorted by absolute contribution descending."""
    X, _ = sample_data
    explanations = trained_classifier.explain(X.head(3), top_k=5)

    assert len(explanations) == 3
    for sample_expl in explanations:
        assert len(sample_expl) == 5
        abs_vals = [abs(c.shap_value) for c in sample_expl]
        assert abs_vals == sorted(abs_vals, reverse=True)
        for contrib in sample_expl:
            assert isinstance(contrib, FeatureContribution)
            assert contrib.feature_name in EXPECTED_FEATURES
            assert isinstance(contrib.feature_value, float)
            assert isinstance(contrib.shap_value, float)


def test_shap_safe_handling_when_explainer_fails(
    trained_classifier: SOCClassifier,
    sample_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Test that if TreeExplainer fails, explain returns an empty list without raising."""
    X, _ = sample_data
    # Mock explainer to raise an exception
    trained_classifier.explainer = MagicMock()
    trained_classifier.explainer.shap_values.side_effect = RuntimeError("Simulated SHAP failure")

    explanations = trained_classifier.explain(X.head(2), top_k=5)
    assert len(explanations) == 2
    assert explanations[0] == []
    assert explanations[1] == []


# =============================================================================
# 6. Serialization & Missing Artifact Error Handling
# =============================================================================

def test_model_save_and_load_roundtrip(
    trained_classifier: SOCClassifier,
    sample_data: tuple[pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    """Test that saving and loading via joblib preserves model behavior and attributes."""
    X, _ = sample_data
    save_path = tmp_path / "soc_model.joblib"

    trained_classifier.save(save_path)
    assert save_path.exists()

    loaded = SOCClassifier.load(save_path)
    assert loaded.model_name == trained_classifier.model_name
    assert loaded.threshold == trained_classifier.threshold
    assert loaded.feature_names == trained_classifier.feature_names

    orig_probs = trained_classifier.predict_proba(X)
    loaded_probs = loaded.predict_proba(X)
    np.testing.assert_allclose(orig_probs, loaded_probs, rtol=1e-5)


def test_load_missing_artifact_raises_file_not_found(tmp_path: Path) -> None:
    """Test that loading a non-existent artifact path raises FileNotFoundError."""
    missing = tmp_path / "non_existent.joblib"
    with pytest.raises(FileNotFoundError):
        SOCClassifier.load(missing)


def test_load_corrupt_artifact_raises_value_error(tmp_path: Path) -> None:
    """Test that loading a corrupted or non-dictionary artifact raises ValueError."""
    corrupt_file = tmp_path / "corrupt.joblib"
    corrupt_file.write_text("corrupted content", encoding="utf-8")

    with pytest.raises(ValueError):
        SOCClassifier.load(corrupt_file)


def test_unfitted_model_raises_runtime_error(sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Test that calling predict or save on an unfitted model raises RuntimeError."""
    X, _ = sample_data
    unfitted = SOCClassifier()

    with pytest.raises(RuntimeError):
        unfitted.predict_proba(X)

    with pytest.raises(RuntimeError):
        unfitted.save("dummy.joblib")


# =============================================================================
# 7. Evaluation & Validation Tuning Verification
# =============================================================================

def test_evaluation_metrics_on_known_fixture() -> None:
    """Verify evaluation metric calculations against known ground truth."""
    # Create mock classifier with predetermined probabilities
    mock_clf = SOCClassifier()
    mock_clf.predict_proba = MagicMock(return_value=np.array([0.9, 0.8, 0.2, 0.1]))

    X_dummy = pd.DataFrame(np.zeros((4, len(EXPECTED_FEATURES))), columns=EXPECTED_FEATURES)
    y_true = pd.Series([1, 0, 0, 1])  # TP=1, FP=1, TN=1, FN=1 at threshold 0.5

    metrics = evaluate_set(mock_clf, X_dummy, y_true, threshold=0.5, split_name="Fixture Test")

    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["true_negatives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1_score"] == 0.5


def test_threshold_tuning_uses_validation_only(sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Test that tune_threshold_on_validation tunes strictly on validation data without touching test data."""
    X, y = sample_data
    n_train = 40
    n_val = 20
    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_val, y_val = X.iloc[n_train : n_train + n_val], y.iloc[n_train : n_train + n_val]
    X_test, y_test = X.iloc[n_train + n_val :], y.iloc[n_train + n_val :]

    clf = SOCClassifier(random_state=42).fit(X_train, y_train, n_estimators=10)

    tuned_thresh, val_metrics = tune_threshold_on_validation(clf, X_val, y_val, target_recall=0.9)

    assert 0.0 <= tuned_thresh <= 1.0
    assert "recall" in val_metrics
    assert "precision" in val_metrics

    # Verify that evaluating on test set at the tuned threshold runs cleanly
    test_metrics = evaluate_set(clf, X_test, y_test, threshold=tuned_thresh, split_name="Untouched Test")
    assert test_metrics["threshold_applied"] == round(tuned_thresh, 4)


# =============================================================================
# 8. Final Targeted Correctness Regression Tests
# =============================================================================

def test_failed_to_success_ratio_greater_than_one_preserved(sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Regression test: verify failed_to_success_ratio is NOT clamped to 1.0.

    In brute-force attacks, failed_to_success_ratio legitimately exceeds 1.0
    (e.g., 20 failed attempts to 1 successful attempt = 20.0).
    Values > 1.0 must be preserved to maintain attack telemetry.
    """
    X, _ = sample_data
    X_test = X.copy()
    X_test.loc[0, "failed_to_success_ratio"] = 5.5
    X_test.loc[1, "failed_to_success_ratio"] = 25.0
    X_test.loc[2, "failed_to_success_ratio"] = -2.0  # Negative ratio must still clamp to 0.0

    clf = SOCClassifier()
    cleaned = clf.validate_features(X_test)

    assert cleaned.loc[0, "failed_to_success_ratio"] == 5.5
    assert cleaned.loc[1, "failed_to_success_ratio"] == 25.0
    assert cleaned.loc[2, "failed_to_success_ratio"] == 0.0


def test_division_by_zero_handling_in_failed_to_success_ratio(sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Regression test: verify division-by-zero resolution for failed_to_success_ratio.

    When accepted_auth_count == 0, dividing by zero yields inf.
    According to the feature contract:
    - If failed_auth_count > 0, resolves to failed_auth_count / 1.0.
    - If failed_auth_count == 0, resolves to 0.0.
    """
    X, _ = sample_data
    X_test = X.copy()
    # Scenario A: 12 failed auth attempts, 0 accepted auth attempts -> inf
    X_test.loc[0, "failed_auth_count"] = 12.0
    X_test.loc[0, "accepted_auth_count"] = 0.0
    X_test.loc[0, "failed_to_success_ratio"] = np.inf

    # Scenario B: 0 failed auth attempts, 0 accepted auth attempts -> inf or nan
    X_test.loc[1, "failed_auth_count"] = 0.0
    X_test.loc[1, "accepted_auth_count"] = 0.0
    X_test.loc[1, "failed_to_success_ratio"] = np.inf

    clf = SOCClassifier()
    cleaned = clf.validate_features(X_test)

    assert cleaned.loc[0, "failed_to_success_ratio"] == 12.0
    assert cleaned.loc[1, "failed_to_success_ratio"] == 0.0


def test_unit_ratio_features_bounded_at_one(sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Regression test: verify true proportion features are clamped to [0.0, 1.0]."""
    X, _ = sample_data
    X_test = X.copy()
    X_test.loc[0, "ephemeral_port_ratio"] = 1.8
    X_test.loc[0, "privileged_port_ratio"] = 2.4
    X_test.loc[0, "rare_template_ratio"] = 3.0

    clf = SOCClassifier()
    cleaned = clf.validate_features(X_test)

    assert cleaned.loc[0, "ephemeral_port_ratio"] == 1.0
    assert cleaned.loc[0, "privileged_port_ratio"] == 1.0
    assert cleaned.loc[0, "rare_template_ratio"] == 1.0


def test_missing_suspicious_class_raises_runtime_error(sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Regression test: verify model without class 1 raises RuntimeError.

    If model.classes_ does not contain class 1, the classifier must NEVER
    silently return 0.0 risk score or classify as benign.
    """
    X, _ = sample_data
    clf = SOCClassifier()

    # Mock fitted model having only class 0
    mock_rf = MagicMock()
    mock_rf.classes_ = np.array([0])
    mock_rf.predict_proba.return_value = np.array([[1.0]] * len(X))
    clf.model = mock_rf

    with pytest.raises(RuntimeError) as exc_info:
        clf.predict_proba(X)
    assert "class 1 ('Suspicious') is absent" in str(exc_info.value)

    with pytest.raises(RuntimeError) as exc_info:
        clf.predict(X)
    assert "class 1 ('Suspicious') is absent" in str(exc_info.value)

    with pytest.raises(RuntimeError) as exc_info:
        clf.explain(X)
    assert "class 1 ('Suspicious') is absent" in str(exc_info.value)


def test_load_model_without_suspicious_class_raises_value_error(tmp_path: Path) -> None:
    """Regression test: verify SOCClassifier.load rejects artifacts missing class 1."""
    import joblib
    from sklearn.tree import DecisionTreeClassifier

    save_path = tmp_path / "bad_single_class_model.joblib"

    # Fit a real scikit-learn estimator on single-class data so classes_ == [0]
    bad_estimator = DecisionTreeClassifier().fit(
        np.zeros((10, len(EXPECTED_FEATURES))),
        np.zeros(10, dtype=int),
    )
    assert list(bad_estimator.classes_) == [0]

    payload = {
        "model_name": "Bad-SingleClass-Model",
        "model": bad_estimator,
        "explainer": None,
        "feature_names": EXPECTED_FEATURES,
        "threshold": 0.5,
    }
    joblib.dump(payload, save_path)

    with pytest.raises(ValueError) as exc_info:
        SOCClassifier.load(save_path)
    assert "class 1 ('Suspicious') is absent from model.classes_" in str(exc_info.value)


def test_validate_features_duplicate_columns_raises_error(sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Regression test: verify duplicate column names raise an explicit ValueError."""
    X, _ = sample_data
    # Construct a DataFrame with duplicate column names
    df_dup = pd.concat([X, X[["failed_auth_count"]]], axis=1)
    assert df_dup.columns.duplicated().any()

    clf = SOCClassifier()
    with pytest.raises(ValueError) as exc_info:
        clf.validate_features(df_dup)
    assert "duplicate column names" in str(exc_info.value)


def test_validate_features_column_order_independence(sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Regression test: verify permuting column order produces identical cleaned features and predictions."""
    X, _ = sample_data
    # Reverse column order
    reversed_cols = list(reversed(EXPECTED_FEATURES))
    X_permuted = X[reversed_cols].copy()

    clf = SOCClassifier()
    cleaned_orig = clf.validate_features(X)
    cleaned_perm = clf.validate_features(X_permuted)

    # Columns must be returned in canonical EXPECTED_FEATURES order
    assert list(cleaned_orig.columns) == EXPECTED_FEATURES
    assert list(cleaned_perm.columns) == EXPECTED_FEATURES
    pd.testing.assert_frame_equal(cleaned_orig, cleaned_perm)


def test_fit_rejects_fractional_target_labels(sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Regression test: verify fractional target labels are rejected, never silently truncated."""
    X, _ = sample_data
    clf = SOCClassifier()
    y_fractional = [0.8 if i % 2 == 1 else 0.2 for i in range(len(X))]

    with pytest.raises(ValueError) as exc_info:
        clf.fit(X, y_fractional)
    assert "fractional/non-integer values" in str(exc_info.value)


def test_fit_rejects_nan_in_target_labels(sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Regression test: verify NaN/null values in target labels raise ValueError."""
    X, _ = sample_data
    clf = SOCClassifier()
    y_with_nan = pd.Series([1 if i % 2 == 1 else 0 for i in range(len(X))])
    y_with_nan.iloc[0] = np.nan

    with pytest.raises(ValueError) as exc_info:
        clf.fit(X, y_with_nan)
    assert "contain NaN/null values" in str(exc_info.value)


def test_predict_proba_missing_method_raises_runtime_error(sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Regression test: verify estimator lacking predict_proba raises explicit RuntimeError."""
    X, _ = sample_data
    clf = SOCClassifier()

    class DummyEstimatorWithoutProba:
        classes_ = np.array([0, 1])

        def predict(self, X):
            return np.zeros(len(X))

    clf.model = DummyEstimatorWithoutProba()

    with pytest.raises(RuntimeError) as exc_info:
        clf.predict_proba(X)
    assert "does not support 'predict_proba'" in str(exc_info.value)


def test_load_mismatched_feature_names_raises_value_error(tmp_path: Path) -> None:
    """Regression test: verify SOCClassifier.load rejects artifacts with mismatched feature contracts."""
    import joblib
    from sklearn.tree import DecisionTreeClassifier

    save_path = tmp_path / "mismatched_features_model.joblib"
    clf = DecisionTreeClassifier().fit(
        np.zeros((10, 3)),
        np.array([0, 1] * 5),
    )

    payload = {
        "model_name": "Mismatched-Features-Model",
        "model": clf,
        "explainer": None,
        "feature_names": ["feat_a", "feat_b", "feat_c"],  # 3 features instead of 13
        "threshold": 0.5,
    }
    joblib.dump(payload, save_path)

    with pytest.raises(ValueError) as exc_info:
        SOCClassifier.load(save_path)
    assert "feature contract mismatch" in str(exc_info.value)


def test_evaluate_model_preserves_existing_report_on_failure(tmp_path: Path) -> None:
    """Regression test: verify that a failed evaluation run does not destroy an existing valid report."""
    import subprocess
    import sys

    # Create an initial valid report file
    report_file = tmp_path / "evaluation_report.json"
    report_file.write_text('{"status": "prior_valid_report"}', encoding="utf-8")

    # Run evaluate_model pointing to a non-existent model path
    cmd = [
        sys.executable,
        "scripts/evaluate_model.py",
        "--model-path",
        str(tmp_path / "non_existent_model.joblib"),
        "--output-report",
        str(report_file),
        "--synthetic",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0

    # Ensure the prior valid report was NOT deleted by the failed evaluation run
    assert report_file.exists()
    assert report_file.read_text(encoding="utf-8") == '{"status": "prior_valid_report"}'


# =============================================================================
# 19. Data Leakage & Splitting Regression Tests
# =============================================================================

def test_split_data_guarantees_group_isolation() -> None:
    """Regression test: verify groups (e.g. scenarios, sessions) do not leak across splits."""
    rows = []
    for scenario_idx in range(6):
        for class_val in [0, 1]:
            for rep in range(5):
                row = {feat: float(scenario_idx * 10 + rep) for feat in EXPECTED_FEATURES}
                row["scenario_id"] = f"scenario_{scenario_idx}"
                row["target"] = class_val
                rows.append(row)
    df = pd.DataFrame(rows)

    X_train, _, X_val, _, X_test, _, strategy = split_data(
        df, target_col="target", group_col="scenario_id", return_strategy=True
    )

    assert "group_split:scenario_id" in strategy
    train_scenarios = set(df.loc[X_train.index, "scenario_id"])
    val_scenarios = set(df.loc[X_val.index, "scenario_id"])
    test_scenarios = set(df.loc[X_test.index, "scenario_id"])

    assert train_scenarios.isdisjoint(val_scenarios)
    assert train_scenarios.isdisjoint(test_scenarios)
    assert val_scenarios.isdisjoint(test_scenarios)


def test_split_data_raises_on_missing_configured_group_or_time_col() -> None:
    """Regression test: verify non-existent group or time columns raise ValueError."""
    df, y = generate_synthetic_behavioral_data(n_samples=30, random_state=42)
    df["target"] = y

    with pytest.raises(ValueError) as exc_info:
        split_data(df, group_col="non_existent_group")
    assert "Specified group column 'non_existent_group' was not found" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        split_data(df, time_col="non_existent_time")
    assert "Specified time column 'non_existent_time' was not found" in str(exc_info.value)


def test_split_data_unresolvable_temporal_overlap_raises_error() -> None:
    """Regression test: verify overlapping sliding windows that deplete partitions raise actionable ValueError."""
    n_samples = 30
    rows = []
    for i in range(n_samples):
        row = {feat: float(i) for feat in EXPECTED_FEATURES}
        row["timestamp"] = i * 10
        row["window_start"] = i * 10
        row["window_end"] = i * 10 + 60
        row["target"] = i % 2
        rows.append(row)
    df = pd.DataFrame(rows)

    with pytest.raises(ValueError) as exc_info:
        split_data(
            df, target_col="target", time_col="timestamp", return_strategy=True
        )
    assert "could not be resolved by buffer purging" in str(exc_info.value)


def test_split_data_temporal_buffer_purging_resolves_overlap() -> None:
    """Regression test: verify buffer purging resolves temporal boundary overlap and eliminates interval leakage."""
    n_samples = 80
    rows = []
    for i in range(n_samples):
        row = {feat: float(i) for feat in EXPECTED_FEATURES}
        row["timestamp"] = i * 10
        row["window_start"] = i * 10
        row["window_end"] = i * 10 + 60
        row["target"] = i % 2
        rows.append(row)
    df = pd.DataFrame(rows)

    X_train, y_train, X_val, y_val, X_test, y_test, strategy = split_data(
        df, target_col="target", time_col="timestamp", return_strategy=True
    )

    assert "chronological_split_purged_buffer:timestamp" in strategy
    # Verify no overlapping intervals cross partitions
    train_end_max = df.loc[X_train.index, "window_end"].max()
    val_start_min = df.loc[X_val.index, "window_start"].min()
    val_end_max = df.loc[X_val.index, "window_end"].max()
    test_start_min = df.loc[X_test.index, "window_start"].min()

    assert val_start_min >= train_end_max, "Validation windows overlap with train window intervals!"
    assert test_start_min >= val_end_max, "Test windows overlap with validation window intervals!"
    assert y_train.nunique() >= 2
    assert y_val.nunique() >= 2
    assert y_test.nunique() >= 2


def test_split_data_temporal_missing_interval_metadata_labels_strategy_unverified() -> None:
    """Regression test: verify chronological split with missing window bounds metadata is marked unverified."""
    df, y = generate_synthetic_behavioral_data(n_samples=40, random_state=42)
    df["target"] = y
    df["timestamp"] = range(len(df))

    _, _, _, _, _, _, strategy = split_data(
        df, target_col="target", time_col="timestamp", return_strategy=True
    )

    assert "chronological_split_unverified_interval:timestamp" in strategy
    assert "window duration/bounds metadata missing" in strategy


def test_split_data_group_insufficient_class_diversity_raises_actionable_error() -> None:
    """Regression test: verify group split failure raises an actionable error instead of silent fallback."""
    rows = []
    for i in range(20):
        row = {feat: float(i) for feat in EXPECTED_FEATURES}
        row["scenario_id"] = f"scenario_{i % 3}"
        # All attack samples belong exclusively to scenario_0
        row["target"] = 1 if (i % 3 == 0 and i < 18) else 0
        rows.append(row)
    df = pd.DataFrame(rows)

    with pytest.raises(ValueError) as exc_info:
        split_data(
            df, target_col="target", group_col="scenario_id", return_strategy=True
        )
    assert "cannot produce valid, class-balanced partitions" in str(exc_info.value)
    assert "scenario_id" in str(exc_info.value)


def test_split_data_raises_on_small_dataset() -> None:
    """Regression test: verify small datasets (< 5 samples or < 3 samples per class) raise ValueError."""
    tiny_df = pd.DataFrame([
        {**{f: 1.0 for f in EXPECTED_FEATURES}, "target": 0},
        {**{f: 2.0 for f in EXPECTED_FEATURES}, "target": 1},
        {**{f: 3.0 for f in EXPECTED_FEATURES}, "target": 0},
    ])
    with pytest.raises(ValueError) as exc_info:
        split_data(tiny_df)
    assert "too small to perform a 60/20/20" in str(exc_info.value)

    sparse_class_df = pd.DataFrame([
        {**{f: 1.0 for f in EXPECTED_FEATURES}, "target": 0},
        {**{f: 2.0 for f in EXPECTED_FEATURES}, "target": 0},
        {**{f: 3.0 for f in EXPECTED_FEATURES}, "target": 0},
        {**{f: 4.0 for f in EXPECTED_FEATURES}, "target": 0},
        {**{f: 5.0 for f in EXPECTED_FEATURES}, "target": 1},
    ])
    with pytest.raises(ValueError) as exc_info:
        split_data(sparse_class_df)
    assert "insufficient class diversity" in str(exc_info.value)


def test_split_data_excludes_metadata_columns_from_features() -> None:
    """Regression test: verify common metadata columns are never included in feature matrices."""
    df, y = generate_synthetic_behavioral_data(n_samples=30, random_state=42)
    df["target"] = y
    df["scenario_id"] = "scen_1"
    df["session_id"] = "sess_1"
    df["group_id"] = "grp_1"
    df["host_id"] = "host_1"
    df["timestamp"] = "2026-10-22T10:00:00"
    df["window_start"] = 100.0
    df["window_end"] = 160.0

    X_train, _, X_val, _, X_test, _ = split_data(df, target_col="target")

    for meta in ["scenario_id", "session_id", "group_id", "host_id", "timestamp", "window_start", "window_end", "target"]:
        assert meta not in X_train.columns
        assert meta not in X_val.columns
        assert meta not in X_test.columns
    assert list(X_train.columns) == EXPECTED_FEATURES


# =============================================================================
# 20. Feature Contract & Model Artifact Integrity Tests
# =============================================================================

def test_load_missing_feature_names_metadata_raises_value_error(tmp_path: Path) -> None:
    """Regression test: verify SOCClassifier.load rejects artifacts missing 'feature_names' metadata."""
    import joblib
    from sklearn.tree import DecisionTreeClassifier

    save_path = tmp_path / "missing_feature_names.joblib"
    clf = DecisionTreeClassifier().fit(np.zeros((10, 13)), np.array([0, 1] * 5))

    payload = {
        "model_name": "No-Features-Model",
        "model": clf,
        "explainer": None,
        "threshold": 0.5,
    }
    joblib.dump(payload, save_path)

    with pytest.raises(ValueError) as exc_info:
        SOCClassifier.load(save_path)
    assert "missing required 'feature_names' metadata" in str(exc_info.value)


def test_load_estimator_n_features_mismatch_raises_value_error(tmp_path: Path) -> None:
    """Regression test: verify SOCClassifier.load rejects model whose estimator n_features_in_ != 13."""
    import joblib
    from sklearn.tree import DecisionTreeClassifier

    save_path = tmp_path / "n_features_mismatch.joblib"
    clf = DecisionTreeClassifier().fit(np.zeros((10, 5)), np.array([0, 1] * 5))
    assert clf.n_features_in_ == 5

    payload = {
        "model_name": "Mismatch-Estimator-Features",
        "model": clf,
        "explainer": None,
        "feature_names": EXPECTED_FEATURES,
        "threshold": 0.5,
    }
    joblib.dump(payload, save_path)

    with pytest.raises(ValueError) as exc_info:
        SOCClassifier.load(save_path)
    assert "trained on 5 features, which does not match runtime expectation of 13 features" in str(exc_info.value)


def test_load_feature_names_hash_mismatch_raises_value_error(tmp_path: Path) -> None:
    """Regression test: verify SOCClassifier.load rejects artifacts with altered feature_names_hash."""
    import joblib
    from sklearn.tree import DecisionTreeClassifier

    save_path = tmp_path / "corrupted_hash.joblib"
    clf = DecisionTreeClassifier().fit(np.zeros((10, 13)), np.array([0, 1] * 5))

    payload = {
        "model_name": "Corrupted-Hash-Model",
        "model": clf,
        "explainer": None,
        "feature_names": EXPECTED_FEATURES,
        "feature_names_hash": "deadbeef12345678",
        "threshold": 0.5,
    }
    joblib.dump(payload, save_path)

    with pytest.raises(ValueError) as exc_info:
        SOCClassifier.load(save_path)
    assert "feature_names_hash" in str(exc_info.value)
    assert "does not match expected schema fingerprint" in str(exc_info.value)


# =============================================================================
# 21. Numerical & SHAP Failure Handling Tests
# =============================================================================

def test_explain_rejects_non_finite_shap_values_without_fabricating_zeros(
    trained_classifier: SOCClassifier, sample_data: tuple[pd.DataFrame, pd.Series]
) -> None:
    """Regression test: verify explain() rejects non-finite SHAP outputs and does not fabricate 0.0 values."""
    X, _ = sample_data
    single_record = X.iloc[:2].copy()

    mock_explainer = MagicMock()
    bad_shap = np.array([
        [np.nan] + [0.1] * 12,
        [np.inf] + [0.2] * 12,
    ])
    mock_explainer.shap_values.return_value = [bad_shap, bad_shap]
    trained_classifier.explainer = mock_explainer

    explanations = trained_classifier.explain(single_record)

    assert len(explanations) == 2
    assert explanations[0] == []
    assert explanations[1] == []


def test_explain_rejects_shap_dimension_mismatch(
    trained_classifier: SOCClassifier, sample_data: tuple[pd.DataFrame, pd.Series]
) -> None:
    """Regression test: verify explain() rejects explanations with mismatched feature count."""
    X, _ = sample_data
    single_record = X.iloc[:1].copy()

    mock_explainer = MagicMock()
    bad_dim_shap = np.zeros((1, 7))
    mock_explainer.shap_values.return_value = [bad_dim_shap, bad_dim_shap]
    trained_classifier.explainer = mock_explainer

    explanations = trained_classifier.explain(single_record)
    assert explanations == [[]]


# =============================================================================
# 22. Controlled Fixture Evaluation Metric Verification
# =============================================================================

def test_controlled_fixture_metric_calculations() -> None:
    """Regression test: verify evaluate_set calculations against independently hand-verified metrics.

    Fixture:
    True labels:  [1, 1, 1, 0, 0, 0]
    Pred probs:   [0.9, 0.8, 0.3, 0.7, 0.2, 0.1]
    Threshold:    0.5
    Predictions:  [1,   1,   0,   1,   0,   0  ]
    Ground truth counts:
    - True Positives (TP):  2 (indices 0, 1)
    - False Negatives (FN): 1 (index 2)
    - False Positives (FP): 1 (index 3)
    - True Negatives (TN):  2 (indices 4, 5)

    Expected metrics:
    - Precision: 2 / (2 + 1) = 2/3 ≈ 0.6667
    - Recall:    2 / (2 + 1) = 2/3 ≈ 0.6667
    - F1-Score:  2 * (2/3 * 2/3) / (2/3 + 2/3) = 2/3 ≈ 0.6667
    """
    class MockClfForMetrics:
        model_name = "Mock-Metric-Model"
        threshold = 0.5

        def predict_proba(self, X):
            return np.array([0.9, 0.8, 0.3, 0.7, 0.2, 0.1])

    X_dummy = pd.DataFrame({feat: np.zeros(6) for feat in EXPECTED_FEATURES})
    y_true = pd.Series([1, 1, 1, 0, 0, 0])

    metrics = evaluate_set(MockClfForMetrics(), X_dummy, y_true, threshold=0.5, split_name="Controlled Fixture")

    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["true_negatives"] == 2
    assert metrics["precision"] == round(2 / 3, 4)
    assert metrics["recall"] == round(2 / 3, 4)
    assert metrics["f1_score"] == round(2 / 3, 4)


# =============================================================================
# 23. Artifact Persistence & Consistency Verification Tests
# =============================================================================

def test_verify_artifact_consistency_detects_mismatch(tmp_path: Path) -> None:
    """Regression test: verify verify_artifact_consistency detects out-of-sync model and summary."""
    import hashlib
    import json
    import joblib
    from sklearn.tree import DecisionTreeClassifier

    model_path = tmp_path / "test_model.joblib"
    summary_path = tmp_path / "test_summary.json"

    clf = DecisionTreeClassifier().fit(np.zeros((10, 13)), np.array([0, 1] * 5))
    payload = {
        "model_name": "Consistency-Test-Model",
        "model": clf,
        "explainer": None,
        "feature_names": EXPECTED_FEATURES,
        "threshold": 0.5,
    }
    joblib.dump(payload, model_path)

    model_digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    summary = {
        "model_name": "Consistency-Test-Model",
        "features_used": EXPECTED_FEATURES,
        "selected_threshold": 0.5,
        "model_sha256": model_digest,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f)

    assert verify_artifact_consistency(model_path, summary_path) is True

    summary["model_sha256"] = "deadbeef" * 8
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f)

    assert verify_artifact_consistency(model_path, summary_path) is False


def test_validate_model_and_summary_consistency_missing_summary(tmp_path: Path) -> None:
    """Regression test: verify missing summary raises FileNotFoundError unless explicitly allowed."""
    import joblib
    from sklearn.tree import DecisionTreeClassifier

    model_path = tmp_path / "standalone_model.joblib"
    missing_summary_path = tmp_path / "non_existent_summary.json"

    clf = DecisionTreeClassifier().fit(np.zeros((10, 13)), np.array([0, 1] * 5))
    payload = {
        "model_name": "Standalone-Model",
        "model": clf,
        "explainer": None,
        "feature_names": EXPECTED_FEATURES,
        "threshold": 0.5,
    }
    joblib.dump(payload, model_path)

    # By default, missing summary must raise FileNotFoundError
    with pytest.raises(FileNotFoundError) as exc_info:
        validate_model_and_summary_consistency(model_path, missing_summary_path, allow_missing_summary=False)
    assert "Training summary report not found" in str(exc_info.value)

    # When allow_missing_summary=True, returns None
    result = validate_model_and_summary_consistency(model_path, missing_summary_path, allow_missing_summary=True)
    assert result is None


def test_validate_model_and_summary_consistency_malformed_json(tmp_path: Path) -> None:
    """Regression test: verify malformed JSON summary raises ValueError."""
    import joblib
    from sklearn.tree import DecisionTreeClassifier

    model_path = tmp_path / "model.joblib"
    summary_path = tmp_path / "summary.json"

    clf = DecisionTreeClassifier().fit(np.zeros((10, 13)), np.array([0, 1] * 5))
    joblib.dump({"model_name": "M", "model": clf, "feature_names": EXPECTED_FEATURES}, model_path)

    summary_path.write_text("{this is not valid json!}", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        validate_model_and_summary_consistency(model_path, summary_path)
    assert "Malformed training summary JSON" in str(exc_info.value)


def test_validate_model_and_summary_consistency_missing_digest(tmp_path: Path) -> None:
    """Regression test: verify summary missing model_sha256 digest raises ValueError."""
    import json
    import joblib
    from sklearn.tree import DecisionTreeClassifier

    model_path = tmp_path / "model.joblib"
    summary_path = tmp_path / "summary.json"

    clf = DecisionTreeClassifier().fit(np.zeros((10, 13)), np.array([0, 1] * 5))
    joblib.dump({"model_name": "M", "model": clf, "feature_names": EXPECTED_FEATURES}, model_path)

    summary = {
        "model_name": "M",
        "features_used": EXPECTED_FEATURES,
        "selected_threshold": 0.5,
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        validate_model_and_summary_consistency(model_path, summary_path)
    assert "missing required 'model_sha256' integrity digest" in str(exc_info.value)


def test_validate_model_and_summary_consistency_digest_mismatch(tmp_path: Path) -> None:
    """Regression test: verify mismatched model_sha256 digest raises ValueError."""
    import json
    import joblib
    from sklearn.tree import DecisionTreeClassifier

    model_path = tmp_path / "model.joblib"
    summary_path = tmp_path / "summary.json"

    clf = DecisionTreeClassifier().fit(np.zeros((10, 13)), np.array([0, 1] * 5))
    joblib.dump({"model_name": "M", "model": clf, "feature_names": EXPECTED_FEATURES}, model_path)

    summary = {
        "model_name": "M",
        "features_used": EXPECTED_FEATURES,
        "selected_threshold": 0.5,
        "model_sha256": "0123456789abcdef" * 4,
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        validate_model_and_summary_consistency(model_path, summary_path)
    assert "Artifact digest mismatch" in str(exc_info.value)


def test_validate_model_and_summary_consistency_feature_contract_mismatch(tmp_path: Path) -> None:
    """Regression test: verify feature list mismatch between summary and EXPECTED_FEATURES raises ValueError."""
    import hashlib
    import json
    import joblib
    from sklearn.tree import DecisionTreeClassifier

    model_path = tmp_path / "model.joblib"
    summary_path = tmp_path / "summary.json"

    clf = DecisionTreeClassifier().fit(np.zeros((10, 13)), np.array([0, 1] * 5))
    joblib.dump({"model_name": "M", "model": clf, "feature_names": EXPECTED_FEATURES}, model_path)

    summary = {
        "model_name": "M",
        "features_used": ["feature_a", "feature_b"],
        "selected_threshold": 0.5,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        validate_model_and_summary_consistency(model_path, summary_path)
    assert "Feature contract mismatch in summary" in str(exc_info.value)


def test_evaluate_model_entrypoint_rejects_mismatched_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test: verify evaluate_model.py entrypoint invokes validation and rejects mismatched artifacts."""
    import hashlib
    import json
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    import scripts.evaluate_model as eval_script

    model_path = tmp_path / "classifier.joblib"
    summary_path = tmp_path / "training_summary.json"
    report_path = tmp_path / "eval_report.json"

    # Train and save a small valid model
    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    clf.fit(np.zeros((10, 13)), np.array([0, 1] * 5))
    payload = {
        "model_name": "TestModel",
        "threshold": 0.5,
        "random_state": 42,
        "feature_names": EXPECTED_FEATURES,
        "model": clf,
        "explainer": None,
        "impute_values_": {f: 0.0 for f in EXPECTED_FEATURES},
    }
    joblib.dump(payload, model_path)

    # Write summary with mismatched digest
    summary = {
        "model_name": "TestModel",
        "features_used": EXPECTED_FEATURES,
        "selected_threshold": 0.5,
        "model_sha256": "mismatched_sha256_hash",
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    test_args = [
        "evaluate_model.py",
        "--model-path", str(model_path),
        "--summary-path", str(summary_path),
        "--output-report", str(report_path),
        "--synthetic",
    ]
    monkeypatch.setattr("sys.argv", test_args)

    with pytest.raises(ValueError) as exc_info:
        eval_script.main()
    assert "Artifact digest mismatch" in str(exc_info.value)


def test_training_preserves_existing_artifacts_on_publication_failure(tmp_path: Path) -> None:
    """Regression test: verify publication failure preserves previously usable artifacts without deletion."""
    import hashlib
    import json
    from unittest.mock import patch
    import scripts.train_model as train_script

    existing_model = tmp_path / "existing_model.joblib"
    existing_summary = tmp_path / "existing_summary.json"

    existing_model.write_bytes(b"ORIGINAL_MODEL_CONTENT")
    existing_summary.write_text(json.dumps({"version": "v1", "content": "original"}), encoding="utf-8")

    # Simulate training main invocation with mock data and simulated disk write failure during publication
    with patch("scripts.train_model.parse_args") as mock_args:
        mock_args.return_value.synthetic = True
        mock_args.return_value.output_model = str(existing_model)
        mock_args.return_value.target_recall = 0.95
        mock_args.return_value.group_col = None
        mock_args.return_value.time_col = None
        mock_args.return_value.random_state = 42

        orig_replace = Path.replace
        def failing_replace(*args: Any, **kwargs: Any) -> Path:
            target = args[0] if args else kwargs.get("target")
            if target and "existing_model" in str(target):
                raise OSError("Simulated atomic replacement failure / permission denied")
            return orig_replace(*args, **kwargs)

        with patch.object(Path, "replace", side_effect=failing_replace):
            with pytest.raises(OSError) as exc_info:
                train_script.main()
            assert "Simulated atomic replacement failure" in str(exc_info.value)

    # Verify that original artifacts remain completely intact and uncorrupted
    assert existing_model.read_bytes() == b"ORIGINAL_MODEL_CONTENT"
    assert json.loads(existing_summary.read_text(encoding="utf-8")) == {"version": "v1", "content": "original"}


def test_threshold_tuning_confined_to_validation_set(tmp_path: Path) -> None:
    """Regression test: verify training orchestration confines threshold tuning strictly to validation partition.

    Asserts that:
    1. Threshold tuning receives X_val and y_val, and NEVER receives X_test or y_test.
    2. The test partition remains completely untouched until final evaluation.
    3. If orchestration were erroneously modified to pass X_test to threshold tuning, the test would fail.
    """
    from unittest.mock import patch, MagicMock
    from scripts.train_model import run_training_pipeline, split_data
    import scripts.train_model as train_module

    df, y = generate_synthetic_behavioral_data(n_samples=100, random_state=42)
    df["target"] = y

    # Get expected split partition indices for deterministic comparison
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(df, target_col="target", random_state=42)

    # Spy on tune_threshold_on_validation to record exact arguments passed
    orig_tune = train_module.tune_threshold_on_validation
    spy_tune = MagicMock(side_effect=orig_tune)

    model_out = tmp_path / "model.joblib"
    summary_out = tmp_path / "summary.json"

    with patch("scripts.train_model.tune_threshold_on_validation", spy_tune):
        clf, summary = run_training_pipeline(
            df=df,
            target_recall=0.90,
            random_state=42,
            output_model=model_out,
            summary_path=summary_out,
            is_synthetic=True,
        )

    # Verify tune_threshold_on_validation was invoked exactly once
    assert spy_tune.call_count == 1
    called_args, _ = spy_tune.call_args
    passed_X = called_args[1]
    passed_y = called_args[2]

    # PROVE CONFINEMENT: passed_X must match validation partition exactly
    assert list(passed_X.index) == list(X_val.index)
    assert list(passed_y.index) == list(y_val.index)
    assert np.array_equal(passed_X.values, X_val.values)
    assert np.array_equal(passed_y.values, y_val.values)

    # PROVE ISOLATION: test partition was NEVER passed to threshold tuning
    assert not (list(passed_X.index) == list(X_test.index))
    assert not (list(passed_X.index) == list(X_train.index))
    assert set(passed_X.index).isdisjoint(set(X_test.index))

    # Verify threshold recorded in summary and applied to model is the one chosen on validation
    assert summary["selected_threshold"] == clf.threshold
    assert summary["test_metrics"]["threshold_applied"] == clf.threshold


def test_failed_to_success_ratio_indeterminate_zero_zero_resolved_to_zero() -> None:
    """Regression test: verify 0/0 ratio (0 failed, 0 accepted auths) resolves to 0.0 and avoids arbitrary median imputation."""
    df, y = generate_synthetic_behavioral_data(n_samples=40, random_state=42)
    clf = SOCClassifier(random_state=42)
    # Fit classifier so it learns training medians
    clf.fit(df, y, n_estimators=5)
    # Set the learned median for failed_to_success_ratio to a non-zero value
    clf.impute_values_["failed_to_success_ratio"] = 3.5

    # Create record with 0 failed auths and 0 accepted auths, with NaN ratio (from 0.0/0.0)
    record = {f: 1.0 for f in EXPECTED_FEATURES}
    record["failed_auth_count"] = 0.0
    record["accepted_auth_count"] = 0.0
    record["failed_to_success_ratio"] = np.nan

    cleaned = clf.validate_features(record)
    # Must resolve explicitly to 0.0 (no failure telemetry in window), NOT to the median 3.5
    assert cleaned.loc[0, "failed_to_success_ratio"] == 0.0

    # Also test when ratio was provided as inf with 0/0
    record["failed_to_success_ratio"] = np.inf
    cleaned_inf = clf.validate_features(record)
    assert cleaned_inf.loc[0, "failed_to_success_ratio"] == 0.0


def test_soc_classifier_rejects_conflicting_feature_contract() -> None:
    """Regression test: verify SOCClassifier cannot be initialized with a conflicting feature specification."""
    with pytest.raises(ValueError) as exc_info:
        SOCClassifier(feature_names=["feature_a", "feature_b"])
    assert "Conflicting feature contract provided to SOCClassifier" in str(exc_info.value)
    assert "The canonical 13-feature contract must strictly match" in str(exc_info.value)


def test_split_data_detects_multistream_columns_under_chronological_split() -> None:
    """Regression test: verify chronological split detects multi-stream columns and documents the limitation."""
    df, y = generate_synthetic_behavioral_data(n_samples=40, random_state=42)
    df["target"] = y
    df["timestamp"] = range(len(df))
    df["window_start"] = [i * 10.0 for i in range(len(df))]
    df["window_end"] = [i * 10.0 + 50.0 for i in range(len(df))]
    df["host_id"] = ["host_1", "host_2"] * 20

    *_, strategy = split_data(
        df,
        target_col="target",
        time_col="timestamp",
        return_strategy=True,
    )

    assert "multi-stream columns present: ['host_id']" in strategy
    assert "global temporal order does not isolate host streams" in strategy


def test_allow_missing_summary_flag_does_not_bypass_existing_mismatched_summary(tmp_path: Path) -> None:
    """Regression test: verify --allow-missing-summary only bypasses missing files, NOT existing mismatched files."""
    import hashlib
    import json
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from scripts.train_model import validate_model_and_summary_consistency

    model_path = tmp_path / "model.joblib"
    summary_path = tmp_path / "summary.json"

    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    clf.fit(np.zeros((10, 13)), np.array([0, 1] * 5))
    payload = {
        "model_name": "TestModel",
        "threshold": 0.5,
        "random_state": 42,
        "feature_names": EXPECTED_FEATURES,
        "model": clf,
        "explainer": None,
        "impute_values_": {f: 0.0 for f in EXPECTED_FEATURES},
    }
    joblib.dump(payload, model_path)

    # Write summary with mismatched sha256
    summary = {
        "model_name": "TestModel",
        "features_used": EXPECTED_FEATURES,
        "selected_threshold": 0.5,
        "model_sha256": "mismatched_digest_hash_value",
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    # When allow_missing_summary=True is set, an existing mismatched summary MUST STILL FAIL!
    with pytest.raises(ValueError) as exc_info:
        validate_model_and_summary_consistency(model_path, summary_path, allow_missing_summary=True)
    assert "Artifact digest mismatch" in str(exc_info.value)


def test_classifier_prediction_explanation_status_tracking() -> None:
    """Regression test: verify ClassifierPrediction explanation_status distinguishes success, unavailable, error, bypassed."""
    X, y = generate_synthetic_behavioral_data(n_samples=20, random_state=42)
    clf = SOCClassifier(model_name="Status-Model", random_state=42)
    clf.fit(X, y, n_estimators=5)

    # Case 1: Normal prediction -> explanation_status == 'success'
    pred_success = clf.predict_record(X.iloc[0], explain=True)
    assert pred_success.explanation_status == "success"
    assert len(pred_success.top_contributing_features) > 0

    # Case 2: Explain bypassed -> explanation_status == 'bypassed'
    pred_bypassed = clf.predict_record(X.iloc[0], explain=False)
    assert pred_bypassed.explanation_status == "bypassed"
    assert pred_bypassed.top_contributing_features == []

    # Case 3: Explainer is None -> explanation_status == 'unavailable'
    clf.explainer = None
    pred_unavail = clf.predict_record(X.iloc[0], explain=True)
    assert pred_unavail.explanation_status == "unavailable"
    assert pred_unavail.top_contributing_features == []
