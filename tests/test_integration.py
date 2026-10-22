"""tests/test_integration.py

Integration tests for the end-to-end Explainable SOC Assistant pipeline.
Validates end-to-end data flow: log streaming -> parsing -> ML classification ->
SHAP feature attribution -> GenAI incident triage formulation -> state store persistence.
"""


from pathlib import Path

import pytest
from src.classifier import EXPECTED_FEATURES, SOCClassifier, generate_synthetic_behavioral_data
from src.schemas import ClassifierPrediction, FeatureContribution


def test_classifier_to_prediction_schema_integration() -> None:
    """Test integration between classifier output and Pydantic prediction schema contract."""
    X, y = generate_synthetic_behavioral_data(n_samples=40, random_state=42)
    clf = SOCClassifier(model_name="Integration-SOC-Model", threshold=0.4, random_state=42)
    clf.fit(X, y, n_estimators=10)

    # 1. Single record prediction integration
    record = X.iloc[0].to_dict()
    prediction = clf.predict_record(record, top_k=3, explain=True)

    assert isinstance(prediction, ClassifierPrediction)
    assert prediction.model_name == "Integration-SOC-Model"
    assert 0.0 <= prediction.risk_score <= 1.0
    assert prediction.predicted_class in (0, 1)
    assert len(prediction.top_contributing_features) <= 3
    for contrib in prediction.top_contributing_features:
        assert isinstance(contrib, FeatureContribution)
        assert contrib.feature_name in EXPECTED_FEATURES

    # 2. Batch predictions integration
    batch_predictions = clf.predict_batch(X.head(5), top_k=3, explain=True)
    assert len(batch_predictions) == 5
    for pred in batch_predictions:
        assert isinstance(pred, ClassifierPrediction)
        assert 0.0 <= pred.risk_score <= 1.0


@pytest.mark.skip(reason="Pending Member 1 streamer/parser and Member 3 LLM agent implementation")
def test_integration_pipeline_placeholder() -> None:
    """Placeholder for full end-to-end pipeline integration."""
    pass


def test_end_to_end_training_to_evaluation_pipeline_integration(tmp_path: Path) -> None:
    """Integration test: exercise full pipeline from split, training, publication, consistency check to evaluation."""
    import hashlib
    import json
    import pandas as pd
    from scripts.train_model import (
        split_data,
        tune_threshold_on_validation,
        evaluate_set,
        validate_model_and_summary_consistency,
    )

    # 1. Prepare deterministic group-structured behavioral data
    rows = []
    for scen_idx in range(6):
        for rep in range(15):
            row = {feat: float(scen_idx * 5 + rep) for feat in EXPECTED_FEATURES}
            row["scenario_id"] = f"scen_{scen_idx}"
            row["target"] = 1 if rep % 3 == 0 else 0
            rows.append(row)
    df = pd.DataFrame(rows)

    # 2. Strict group split
    X_train, y_train, X_val, y_val, X_test, y_test, strategy = split_data(
        df, target_col="target", group_col="scenario_id", return_strategy=True
    )
    assert "group_split:scenario_id" in strategy

    # 3. Fit classifier and tune threshold on validation data only
    clf = SOCClassifier(model_name="Pipeline-Integration-Model", random_state=42)
    clf.fit(X_train, y_train, n_estimators=10)
    best_thresh, val_metrics = tune_threshold_on_validation(clf, X_val, y_val, target_recall=0.80)
    clf.threshold = best_thresh

    # 4. Evaluate untouched test set
    test_metrics = evaluate_set(clf, X_test, y_test, threshold=best_thresh, split_name="Integration Test")
    assert 0.0 <= test_metrics["f1_score"] <= 1.0

    # 5. Staged publication with SHA-256 digest
    model_path = tmp_path / "model.joblib"
    summary_path = tmp_path / "summary.json"
    clf.save(model_path)
    model_digest = hashlib.sha256(model_path.read_bytes()).hexdigest()

    summary_data = {
        "model_name": clf.model_name,
        "features_used": EXPECTED_FEATURES,
        "split_strategy_used": strategy,
        "selected_threshold": best_thresh,
        "model_sha256": model_digest,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")

    # 6. Verify consistency gate prior to downstream consumption
    verified_summary = validate_model_and_summary_consistency(model_path, summary_path)
    assert verified_summary is not None
    assert verified_summary["model_sha256"] == model_digest
    assert verified_summary["split_strategy_used"] == strategy

    # 7. Deserialization and downstream prediction
    loaded_clf = SOCClassifier.load(model_path)
    pred = loaded_clf.predict_record(X_test.iloc[0], top_k=3)
    assert isinstance(pred, ClassifierPrediction)
    assert pred.model_name == clf.model_name
