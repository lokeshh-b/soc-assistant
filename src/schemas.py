"""src/schemas.py

Pydantic data models and schemas defining data contracts across the pipeline.
Includes schemas for raw log windows, Tier 1 ML classifications, SHAP attributions,
Tier 2 GenAI structured analyses, SQLite persistence records, and analyst HITL reviews.

Architectural Notes & Assumptions:
- Reuses the Tier2LLMAnalysis contract specified in the project Master Plan.
- ShapFeatureImpact supports both numerical float and signed string representations
  (e.g., '+0.41') from Tier 1 ML explainer outputs without forcing premature type casting.
- raw_logs supports strings or serialized log lines matching the Tier 1 payload contract.
- SQLite persistence models map directly to 'incidents' and 'window_metrics' tables,
  with explicit type handling for SQLite compatibility (e.g., hallucination_flag INTEGER 0/1).
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Union
# pyrefly: ignore[missing-import]
from pydantic import    BaseModel, Field


# =============================================================================
# 1. SHAP & Tier 1 Feature Attributions (ML Classifier -> GenAI & UI)
# =============================================================================

class ShapFeatureImpact(BaseModel):
    """Represents a single feature's attribution to the anomaly score.
    
    Compatible with Tier 1 ML outputs where shap_impact may be provided as
    a signed float or formatted string (e.g. '+0.41').
    """
    feature: str = Field(description="Name of the behavioral log feature (e.g. failed_auth_count_10s).")
    value: Any = Field(description="Observed feature value in the window (int, float, str, etc.).")
    shap_impact: Union[float, str] = Field(
        description="Calculated SHAP impact value (float or signed string like '+0.41')."
    )


# =============================================================================
# 2. Schema B: Tier 2 LLM Agent Structured Output Contract (Master Plan Spec)
# =============================================================================

class Tier2LLMAnalysis(BaseModel):
    """Constrained Pydantic output contract for the GenAI triage agent (src/llm_agent.py).
    
    Enforces strict token-level schema compliance and an explicit abstention
    mechanism ('Insufficient Data') to eliminate hallucinations.
    """
    evidence_status: Literal["Sufficient", "Insufficient Data"] = Field(
        description="Output 'Insufficient Data' if logs and SHAP features do not support a verifiable conclusion."
    )
    incident_summary: str = Field(
        description="Strict 2-sentence explanation of the anomaly citing specific SHAP features and log events."
    )
    threat_category: Literal[
        "Brute Force",
        "Privilege Escalation",
        "Lateral Movement",
        "Command and Control",
        "Exfiltration",
        "Execution",
        "Unknown / Insufficient Data",
    ] = Field(description="Primary attack vector classification.")
    mitre_attack_id: str = Field(
        description="Predicted MITRE ATT&CK technique ID (e.g. T1110) or 'None' if insufficient data."
    )
    recommended_actions: List[str] = Field(
        default_factory=list,
        description="1 to 3 concrete host-specific mitigation steps. Empty list if insufficient data.",
    )


# =============================================================================
# 3. Schema C: SQLite Incidents & State Store Model
# =============================================================================

class IncidentRecord(BaseModel):
    """Complete enriched incident record matching the SQLite 'incidents' table.
    
    Used by src/state_store.py for persistence and src/dashboard_app.py for rendering.
    """
    window_id: str = Field(description="Unique window identifier (e.g. win_20261015_0042).")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC ISO-8601 creation timestamp."
    )
    source_ip: str = Field(description="Originating IP address flagged in the window.")
    target_host: str = Field(description="Target host name or server identifier.")
    occurrence_count: int = Field(default=1, ge=1, description="Number of coalesced occurrences for this entity.")
    risk_score: float = Field(ge=0.0, le=1.0, description="Calibrated ML threat probability score [0.0 - 1.0].")
    shap_summary: List[ShapFeatureImpact] = Field(
        default_factory=list,
        description="Parsed SHAP feature attributions (stored in DB as shap_summary_json)."
    )
    raw_logs: List[str] = Field(
        default_factory=list,
        description="Sample raw log strings associated with the window (stored in DB as raw_logs_json)."
    )
    llm_analysis: Optional[Tier2LLMAnalysis] = Field(
        default=None,
        description="Enriched GenAI incident analysis (stored in DB as llm_analysis_json)."
    )
    tier2_latency_ms: float = Field(default=0.0, ge=0.0, description="Inference latency of GenAI analysis in ms.")
    analyst_decision: Literal["PENDING", "APPROVED", "REJECTED"] = Field(
        default="PENDING",
        description="Human-in-the-loop analyst triage decision."
    )
    hallucination_flag: int = Field(
        default=0,
        ge=0,
        le=1,
        description="0 = Grounded in SHAP/logs; 1 = Hallucinated/unsupported claim flagged by analyst."
    )
    reviewed_at: Optional[str] = Field(
        default=None,
        description="UTC ISO-8601 timestamp when analyst completed the review."
    )
    analyst_notes: Optional[str] = Field(
        default=None,
        description="Operational triage notes or justification recorded by the analyst."
    )

    @property
    def is_hallucination(self) -> bool:
        """Convenience boolean property for UI toggles."""
        return self.hallucination_flag == 1


# =============================================================================
# 4. Human-In-The-Loop (HITL) Review Submission Contract
# =============================================================================

class AnalystReviewSubmission(BaseModel):
    """Payload sent from the Streamlit UI to src/state_store.py when an analyst reviews an alert.
    
    A submitted review represents a completed analyst verdict ('APPROVED' or 'REJECTED').
    """
    window_id: str = Field(description="Unique window ID of the alert being reviewed.")
    decision: Literal["APPROVED", "REJECTED"] = Field(description="Completed analyst verdict.")
    hallucination_flag: bool = Field(
        default=False,
        description="True if analyst determined the LLM summary asserted claims unsupported by logs/SHAP."
    )
    analyst_notes: Optional[str] = Field(
        default=None,
        description="Optional qualitative triage notes."
    )


# =============================================================================
# 5. Pipeline Telemetry Model (SQLite 'window_metrics' table)
# =============================================================================

class WindowMetricsRecord(BaseModel):
    """Telemetry record for tracking filtering efficiency, latency, and ground truth."""
    window_id: str = Field(description="Unique window identifier.")
    timestamp_end: str = Field(description="UTC ISO-8601 end of window timestamp.")
    total_raw_logs: int = Field(ge=0, description="Total raw log events ingested in window.")
    risk_score: float = Field(ge=0.0, le=1.0, description="ML anomaly risk score [0.0 - 1.0].")
    forwarded_to_llm: Literal[0, 1, 2] = Field(
        description="0 = Filtered Benign, 1 = Sent to Tier 2 LLM, 2 = Coalesced Duplicate."
    )
    tier1_latency_ms: float = Field(ge=0.0, description="Tier 1 inference latency in milliseconds.")
    ground_truth_malicious: int = Field(default=0, ge=0, le=1, description="0 = Benign, 1 = Malicious benchmark label.")
    ground_truth_mitre: Optional[str] = Field(default=None, description="Ground truth MITRE technique ID.")
