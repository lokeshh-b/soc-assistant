"""scripts/evaluate_model.py

Model evaluation and explainability validation script.
Loads a serialized SOCClassifier artifact, evaluates classification performance
on test data across multiple thresholds, computes PR-AUC / ROC-AUC, analyzes
false-positive / false-negative trade-offs, and generates SHAP global attribution summaries.

Usage:
    # Demonstration evaluation using synthetic data:
    python scripts/evaluate_model.py --model-path models/classifier.joblib --synthetic

    # Real evaluation on Member 1's test set:
    python scripts/evaluate_model.py --model-path models/classifier.joblib --data-path data/processed/test_features.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path when invoked directly as a standalone script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
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
logger = logging.getLogger("evaluate_model")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained SOC Classifier and validate SHAP explanations.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/classifier.joblib",
        help="Path to serialized SOCClassifier joblib artifact.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/test_features.csv",
        help="Path to test dataset CSV from Member 1.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Explicit flag required to run evaluation in demonstration mode with synthetic data.",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="artifacts/evaluation_report.json",
        help="Path to save output JSON evaluation report.",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default="artifacts/training_summary.json",
        help="Path to training summary report to validate consistency against model.",
    )
    parser.add_argument(
        "--allow-missing-summary",
        action="store_true",
        help="Allow evaluating standalone model without verifying training summary consistency.",
    )
    return parser.parse_args()


def evaluate_threshold_sweep(
    classifier: SOCClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    thresholds: List[float],
) -> List[Dict[str, Any]]:
    """Evaluate performance across a sweep of candidate decision thresholds.

    Demonstrates to the SOC analyst the trade-off between Recall (catching attacks)
    and Precision (minimizing false alarm alert fatigue).
    """
    probs = classifier.predict_proba(X_test)
    results = []

    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        results.append(
            {
                "threshold": round(float(thresh), 2),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1_score": round(float(f1), 4),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_negatives": int(tn),
            }
        )

    return results


def main() -> None:
    """Execute model evaluation and SHAP attribution reporting."""
    args = parse_args()

    # Staged report path: preserve any existing report until the new evaluation succeeds completely
    report_path = Path(args.output_report)

    model_file = Path(args.model_path)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Trained model artifact not found at '{model_file}'.\n"
            f"Please run the training script first:\n"
            f"    python scripts/train_model.py --synthetic"
        )

    # 1. Validate artifact consistency before loading model
    from scripts.train_model import validate_model_and_summary_consistency

    summary_data = validate_model_and_summary_consistency(
        model_path=model_file,
        summary_path=args.summary_path,
        allow_missing_summary=args.allow_missing_summary,
    )
    if summary_data is not None:
        logger.info(
            "Artifact consistency verified: model '%s' matches summary '%s' (SHA-256: %s...)",
            model_file,
            args.summary_path,
            summary_data["model_sha256"][:12],
        )
    else:
        logger.warning(
            "Evaluation proceeding with unverified standalone model (missing summary allowed by flag)."
        )

    # 2. Load trained classifier artifact
    logger.info("Loading model artifact from '%s'...", model_file)
    classifier = SOCClassifier.load(model_file)
    logger.info(
        "Successfully loaded model '%s' (configured threshold = %.2f)",
        classifier.model_name,
        classifier.threshold,
    )

    # 3. Ingest evaluation data / Strict validation check
    data_file = Path(args.data_path)
    if args.synthetic:
        logger.warning(
            "========================================================================\n"
            "[DEMONSTRATION RUN: SYNTHETIC TEST DATASET]\n"
            "Flag '--synthetic' was provided. Using synthetic test behavioral windows.\n"
            "NOTE: Synthetic metrics are for pipeline testing only and DO NOT represent\n"
            "real-world SOC attack detection capability.\n"
            "========================================================================"
        )
        X_synth, y_synth = generate_synthetic_behavioral_data(n_samples=200, random_state=99)
        df = X_synth.copy()
        df["target"] = y_synth
        is_synthetic = True
    else:
        # Real test data requested
        if not data_file.exists():
            raise FileNotFoundError(
                f"Evaluation dataset file not found at '{data_file}'.\n"
                f"Member 1 has not yet produced the test dataset, or the path is incorrect.\n"
                f"To run evaluation in demonstration mode using synthetic data, explicitly pass '--synthetic':\n"
                f"    python scripts/evaluate_model.py --synthetic"
            )

        logger.info("Loading real evaluation dataset from '%s'", data_file)
        if data_file.suffix == ".parquet":
            df = pd.read_parquet(data_file)
        else:
            df = pd.read_csv(data_file)
        is_synthetic = False

    X_test = df[EXPECTED_FEATURES]
    y_test = df["target"].astype(int)

    # 3. Overall Probability Metrics (ROC-AUC & PR-AUC)
    test_probs = classifier.predict_proba(X_test)
    try:
        roc_auc = float(roc_auc_score(y_test, test_probs))
    except ValueError:
        logger.warning("ROC-AUC could not be calculated (requires both classes in test set). Setting to NaN.")
        roc_auc = float("nan")

    try:
        pr_auc = float(average_precision_score(y_test, test_probs))
    except ValueError:
        logger.warning("PR-AUC could not be calculated. Setting to NaN.")
        pr_auc = float("nan")

    # 4. Threshold Sweep & Selected Threshold Evaluation
    sweep_results = evaluate_threshold_sweep(
        classifier,
        X_test,
        y_test,
        thresholds=[0.20, 0.35, 0.50, 0.65, 0.80],
    )

    # Active model threshold performance
    active_preds = classifier.predict(X_test)
    active_cm = confusion_matrix(y_test, active_preds, labels=[0, 1])
    tn, fp, fn, tp = active_cm.ravel()

    active_precision = float(precision_score(y_test, active_preds, zero_division=0))
    active_recall = float(recall_score(y_test, active_preds, zero_division=0))
    active_f1 = float(f1_score(y_test, active_preds, zero_division=0))

    logger.info("==================================================")
    logger.info("EVALUATION SUMMARY: %s", classifier.model_name)
    logger.info("Data source: %s", "Synthetic Demonstration" if is_synthetic else "Real Test Dataset")
    logger.info("==================================================")
    logger.info("  Configured Threshold: %.2f", classifier.threshold)
    logger.info("  Precision:            %.4f", active_precision)
    logger.info("  Recall:               %.4f", active_recall)
    logger.info("  F1-Score:             %.4f", active_f1)
    logger.info("  ROC-AUC:              %.4f", roc_auc)
    logger.info("  PR-AUC (Avg Prec):    %.4f", pr_auc)
    logger.info("  True Positives (TP):  %d", tp)
    logger.info("  False Positives (FP): %d (Analyst Alert Fatigue)", fp)
    logger.info("  False Negatives (FN): %d (Missed Intrusions)", fn)
    logger.info("  True Negatives (TN):  %d", tn)
    logger.info("--------------------------------------------------")
    logger.info("Threshold Sweep Analysis:")
    for row in sweep_results:
        logger.info(
            "  Thresh=%.2f -> Rec=%.4f, Prec=%.4f, F1=%.4f, FP=%d, FN=%d",
            row["threshold"],
            row["recall"],
            row["precision"],
            row["f1_score"],
            row["false_positives"],
            row["false_negatives"],
        )
    logger.info("==================================================")

    # 5. SHAP Global Feature Importance Computation
    logger.info("Computing global SHAP feature attribution statistics on sample...")
    eval_sample_size = min(50, len(X_test))
    sample_explanations = classifier.explain(X_test.iloc[:eval_sample_size], top_k=5)

    # Aggregate average absolute SHAP values across test samples
    feature_impact_sums: Dict[str, float] = {f: 0.0 for f in EXPECTED_FEATURES}
    feature_counts: Dict[str, int] = {f: 0 for f in EXPECTED_FEATURES}

    for sample_expl in sample_explanations:
        for contrib in sample_expl:
            feature_impact_sums[contrib.feature_name] += abs(contrib.shap_value)
            feature_counts[contrib.feature_name] += 1

    ranked_global_features = sorted(
        [
            {
                "feature_name": feat,
                "mean_abs_shap": round(
                    feature_impact_sums[feat] / max(feature_counts[feat], 1),
                    6,
                ),
            }
            for feat in EXPECTED_FEATURES
            if feature_counts[feat] > 0
        ],
        key=lambda x: x["mean_abs_shap"],
        reverse=True,
    )

    logger.info("Top Contributing Features (Global Mean |SHAP| on Samples):")
    for rank, item in enumerate(ranked_global_features[:5], start=1):
        logger.info("  %d. %s (mean |SHAP| = %.6f)", rank, item["feature_name"], item["mean_abs_shap"])

    # 6. Save Evaluation Report Atomically
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_report = report_path.with_name(f".{report_path.name}.tmp")

    report_payload = {
        "model_name": classifier.model_name,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "is_synthetic_demonstration": is_synthetic,
        "data_source": "synthetic_demonstration" if is_synthetic else str(data_file),
        "model_sha256": summary_data.get("model_sha256") if summary_data else None,
        "training_summary_verified": bool(summary_data is not None),
        "split_strategy_used": summary_data.get("split_strategy_used") if summary_data else "unverified_standalone",
        "num_test_samples": len(X_test),
        "configured_threshold": classifier.threshold,
        "metrics": {
            "precision": round(active_precision, 4),
            "recall": round(active_recall, 4),
            "f1_score": round(active_f1, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "confusion_matrix": {
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_negatives": int(tn),
            },
        },
        "threshold_sweep": sweep_results,
        "top_global_shap_features": ranked_global_features,
    }

    try:
        with open(tmp_report, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, indent=2)
        tmp_report.replace(report_path)
        logger.info("Saved complete evaluation report to '%s'", report_path)
    finally:
        if tmp_report.exists():
            try:
                tmp_report.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()
