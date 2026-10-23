"""tests/test_classifier.py

Feature contract and schema compatibility tests for the ML classifier module.
Validates that Member 1's feature matrices exactly conform to the 13-feature contract
expected by Member 2 for training and SHAP explainability.
"""

import csv
import sys
from pathlib import Path

# Ensure workspace root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.schemas import FEATURE_COLUMNS, FeatureWindow


def test_13_feature_columns_contract() -> None:
    """Verify that FEATURE_COLUMNS defines the exact 13 required features."""
    expected_features = [
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
    assert FEATURE_COLUMNS == expected_features
    assert len(FEATURE_COLUMNS) == 13


def test_processed_feature_matrix_headers() -> None:
    """Verify that exported feature CSV files strictly match FEATURE_COLUMNS."""
    train_csv = ROOT_DIR / "data" / "processed" / "train_features.csv"
    if train_csv.is_file():
        with train_csv.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            expected_header = ["window_id"] + FEATURE_COLUMNS
            assert header == expected_header
            # Verify each data row has exactly 14 values (window_id + 13 features)
            for row in reader:
                assert len(row) == 14
                for val in row[1:]:
                    # Ensure numerical convertible
                    float(val)
