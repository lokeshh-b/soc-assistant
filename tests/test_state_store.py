"""tests/test_state_store.py

Unit tests for the state store module (src/state_store.py).
Verifies:
1. Database initialization and table creation.
2. Saving and retrieving window telemetry metrics.
3. Saving and retrieving enriched incident records with SHAP and LLM payloads.
4. Human-in-the-loop analyst review submission and audit timestamping.
5. Error handling for non-existent incident IDs.
6. Conflict handling: technical payload updates vs. review audit preservation.
7. Dashboard metrics aggregation and Noise Reduction Ratio (NRR) computation.
"""

# pyrefly: ignore[missing-import]
import pytest

from src.schemas import (
    AnalystReviewSubmission,
    IncidentRecord,
    ShapFeatureImpact,
    Tier2LLMAnalysis,
    WindowMetricsRecord,
)
from src.state_store import StateStore


@pytest.fixture
def store() -> StateStore:
    """Create an isolated, in-memory StateStore for each test."""
    test_store = StateStore(db_path=":memory:")
    test_store.init_db()
    yield test_store
    test_store.close()


def test_init_db_creates_tables(store: StateStore) -> None:
    """Verify that required tables and columns are initialized."""
    with store.cursor() as cur:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row["name"] for row in cur.fetchall()]
        assert "window_metrics" in tables
        assert "incidents" in tables

        # Verify defensive migration for analyst_notes
        cur.execute("PRAGMA table_info(incidents);")
        columns = [row["name"] for row in cur.fetchall()]
        assert "analyst_notes" in columns
        assert "hallucination_flag" in columns


def test_save_and_get_window_metrics(store: StateStore) -> None:
    """Verify upserting and reading window telemetry."""
    metric = WindowMetricsRecord(
        window_id="win_test_001",
        timestamp_end="2026-10-22T21:00:00Z",
        total_raw_logs=150,
        risk_score=0.88,
        forwarded_to_llm=1,
        tier1_latency_ms=10.5,
        ground_truth_malicious=1,
        ground_truth_mitre="T1110",
    )
    store.save_window_metrics(metric)

    with store.cursor() as cur:
        cur.execute("SELECT * FROM window_metrics WHERE window_id = ?", ("win_test_001",))
        row = cur.fetchone()
        assert row is not None
        assert row["total_raw_logs"] == 150
        assert row["risk_score"] == 0.88
        assert row["forwarded_to_llm"] == 1


def test_save_and_get_incident(store: StateStore) -> None:
    """Verify saving a full incident and deserializing back into IncidentRecord."""
    incident = IncidentRecord(
        window_id="win_test_002",
        created_at="2026-10-22T21:05:00Z",
        source_ip="192.168.1.50",
        target_host="srv-web-01",
        occurrence_count=2,
        risk_score=0.91,
        shap_summary=[
            ShapFeatureImpact(feature="failed_auth_count_10s", value=35, shap_impact=0.45),
            ShapFeatureImpact(feature="port_22_ratio", value=0.90, shap_impact=0.20),
        ],
        raw_logs=["Oct 22 sshd: Failed password for root from 192.168.1.50"],
        llm_analysis=Tier2LLMAnalysis(
            evidence_status="Sufficient",
            incident_summary="SSH brute force attempt detected against root.",
            threat_category="Brute Force",
            mitre_attack_id="T1110",
            recommended_actions=["Block IP 192.168.1.50"],
        ),
        tier2_latency_ms=1150.0,
        analyst_decision="PENDING",
        hallucination_flag=0,
    )
    store.save_incident(incident)

    loaded = store.get_incident("win_test_002")
    assert loaded is not None
    assert loaded.window_id == "win_test_002"
    assert loaded.source_ip == "192.168.1.50"
    assert loaded.risk_score == 0.91
    assert len(loaded.shap_summary) == 2
    assert loaded.shap_summary[0].feature == "failed_auth_count_10s"
    assert loaded.llm_analysis is not None
    assert loaded.llm_analysis.mitre_attack_id == "T1110"
    assert loaded.analyst_decision == "PENDING"
    assert not loaded.is_hallucination


def test_submit_analyst_review(store: StateStore) -> None:
    """Verify human-in-the-loop review updates decision, notes, hallucination, and reviewed_at."""
    incident = IncidentRecord(
        window_id="win_test_003",
        source_ip="10.0.0.1",
        target_host="srv-app-01",
        risk_score=0.85,
    )
    store.save_incident(incident)

    review = AnalystReviewSubmission(
        window_id="win_test_003",
        decision="APPROVED",
        hallucination_flag=True,
        analyst_notes="Confirmed attacker IP. LLM asserted incorrect port, flagged hallucination.",
    )
    updated = store.submit_review(review)

    assert updated.analyst_decision == "APPROVED"
    assert updated.hallucination_flag == 1
    assert updated.is_hallucination is True
    assert updated.analyst_notes == "Confirmed attacker IP. LLM asserted incorrect port, flagged hallucination."
    assert updated.reviewed_at is not None


def test_submit_review_missing_incident_raises(store: StateStore) -> None:
    """Verify KeyError is raised when reviewing a non-existent incident."""
    review = AnalystReviewSubmission(
        window_id="win_non_existent",
        decision="REJECTED",
    )
    with pytest.raises(KeyError, match="not found"):
        store.submit_review(review)


def test_conflict_preserves_analyst_review(store: StateStore) -> None:
    """Verify that pipeline re-saves update technical fields but preserve analyst reviews."""
    incident = IncidentRecord(
        window_id="win_test_004",
        source_ip="10.0.0.2",
        target_host="srv-db-01",
        risk_score=0.80,
        occurrence_count=1,
    )
    store.save_incident(incident)

    # Analyst reviews the alert
    store.submit_review(
        AnalystReviewSubmission(
            window_id="win_test_004",
            decision="APPROVED",
            hallucination_flag=False,
            analyst_notes="Verified true incident.",
        )
    )

    # Pipeline re-runs and saves the same window_id with updated risk and occurrence
    updated_incident = IncidentRecord(
        window_id="win_test_004",
        source_ip="10.0.0.2",
        target_host="srv-db-01",
        risk_score=0.95,
        occurrence_count=3,
        tier2_latency_ms=900.0,
    )
    store.save_incident(updated_incident)

    reloaded = store.get_incident("win_test_004")
    assert reloaded is not None
    # Technical fields updated
    assert reloaded.risk_score == 0.95
    assert reloaded.occurrence_count == 3
    # Analyst review preserved
    assert reloaded.analyst_decision == "APPROVED"
    assert reloaded.hallucination_flag == 0
    assert reloaded.analyst_notes == "Verified true incident."


def test_dashboard_metrics_and_nrr(store: StateStore) -> None:
    """Verify aggregated metrics and Noise Reduction Ratio (NRR) computation."""
    # Empty DB check
    empty_metrics = store.get_dashboard_metrics()
    assert empty_metrics["total_windows"] == 0
    assert empty_metrics["total_incidents"] == 0
    assert empty_metrics["noise_reduction_ratio"] == 0.0

    # Add 4 windows: 3 dropped as noise (forwarded_to_llm=0 or 2), 1 sent to Tier 2 (forwarded_to_llm=1)
    for i in range(3):
        store.save_window_metrics(
            WindowMetricsRecord(
                window_id=f"win_benign_{i}",
                timestamp_end="2026-10-22T21:00:00Z",
                total_raw_logs=100,
                risk_score=0.1,
                forwarded_to_llm=0,
                tier1_latency_ms=5.0,
            )
        )
    store.save_window_metrics(
        WindowMetricsRecord(
            window_id="win_threat_1",
            timestamp_end="2026-10-22T21:00:00Z",
            total_raw_logs=100,
            risk_score=0.9,
            forwarded_to_llm=1,
            tier1_latency_ms=5.0,
        )
    )

    metrics = store.get_dashboard_metrics()
    assert metrics["total_windows"] == 4
    assert metrics["forwarded_windows"] == 1
    # NRR = 1.0 - (1 / 4) = 0.75
    assert metrics["noise_reduction_ratio"] == 0.75
