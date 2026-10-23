"""scripts/demo_genai_triage.py

Practical offline demonstration of the Member 3 GenAI Triage Module (src/llm_agent.py).
Demonstrates:
1. Scenario 1: Active Incident Triage (Sufficient Evidence & SHAP Attributions)
2. Scenario 2: Decoupled ML Prediction Safety (High Risk Score with Missing/Insufficient Evidence)
3. Scenario 3: Pipeline Alert Dictionary Ingestion & JSON Serialization

Note: Operates completely offline in deterministic mock mode.
No external network requests or Gemini API keys are required.
"""

import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_agent import analyze_incident
from src.schemas import IncidentAnalysisReport


def run_demo() -> None:
    print("=" * 80)
    print("EXPLAINABLE SOC ASSISTANT - GENAI INCIDENT TRIAGE MODULE DEMO")
    print("Member 3: Offline Deterministic Triage Demonstration")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Scenario 1: Sufficient Evidence (Active Brute Force Incident)
    # -------------------------------------------------------------------------
    print("\n[Scenario 1] Ingesting Corroborated Security Alert with SHAP Explanations...")
    synthetic_logs = (
        "2026-10-22T20:15:01Z host=auth-srv-01 sshd[4102]: Failed password for invalid user admin from 198.51.100.22 port 52331 ssh2\n"
        "2026-10-22T20:15:03Z host=auth-srv-01 sshd[4105]: Failed password for invalid user root from 198.51.100.22 port 52334 ssh2\n"
        "2026-10-22T20:15:05Z host=auth-srv-01 sshd[4109]: Failed password for invalid user admin from 198.51.100.22 port 52338 ssh2\n"
        "2026-10-22T20:15:08Z host=auth-srv-01 sshd[4114]: Maximum authentication attempts exceeded for root from 198.51.100.22"
    )
    shap_factors = {
        "failed_auth_count": 0.88,
        "source_ip_entropy": 0.65,
        "auth_failure_rate": 0.94,
    }

    report1: IncidentAnalysisReport = analyze_incident(
        classifier_prediction="ANOMALY",
        risk_score=0.92,
        threshold=0.50,
        shap_contributions=shap_factors,
        feature_values={"failed_logins_last_5m": 42, "unique_users": 2},
        log_evidence=synthetic_logs,
        mock_mode=True,
    )

    print("Validated Triage Output (JSON):")
    print(report1.model_dump_json(indent=2))

    # Field contract checks
    assert report1.evidence_status == "Sufficient"
    assert report1.threat_category == "Brute Force"
    assert report1.mitre_attack_id == "T1110"
    assert len(report1.recommended_actions) > 0

    # -------------------------------------------------------------------------
    # Scenario 2: Insufficient Evidence Safety (Prediction Decoupling)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[Scenario 2] High ML Anomaly Score Without Corroborating Logs (Safety Check)...")
    print("Telemetry: classifier_prediction='MALICIOUS', risk_score=0.99, log_evidence=''")

    report2: IncidentAnalysisReport = analyze_incident(
        classifier_prediction="MALICIOUS",
        risk_score=0.99,
        threshold=0.50,
        shap_contributions={"outbound_payload_bytes": 0.95},
        log_evidence="",  # Missing evidence
        mock_mode=True,
    )

    print("Validated Triage Output (JSON):")
    print(report2.model_dump_json(indent=2))

    # Invariant checks
    assert report2.evidence_status == "Insufficient Data"
    assert report2.threat_category == "Unknown / Insufficient Data"
    assert report2.mitre_attack_id is None
    assert report2.recommended_actions == []

    # -------------------------------------------------------------------------
    # Scenario 3: Pipeline Alert Dictionary Ingestion & Serialization
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[Scenario 3] Ingesting Streamer Alert Dictionary & Round-Trip Serialization...")
    pipeline_alert = {
        "classifier_prediction": "ANOMALY",
        "risk_score": 0.88,
        "threshold": 0.50,
        "shap_contributions": {"remote_service_install_count": 0.79},
        "log_evidence": "PsExec service installed remotely on srv-db-02 via internal SMB session",
    }

    report3: IncidentAnalysisReport = analyze_incident(pipeline_alert, mock_mode=True)
    serialized_dict = report3.model_dump()
    print("Serialized Python Dict for Pipeline State Store:")
    print(json.dumps(serialized_dict, indent=2))

    # Re-hydrate from dict
    restored = IncidentAnalysisReport.model_validate(serialized_dict)
    assert restored == report3
    print("\nState Store Re-hydration Check: PASSED (lossless roundtrip)")

    print("\n" + "=" * 80)
    print("DEMO COMPLETE: All contracts verified successfully offline.")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
