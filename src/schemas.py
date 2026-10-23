"""src/schemas.py

Pydantic data models and schemas defining data contracts across the pipeline.
Includes schemas for raw log records, normalized events, alert classifications,
SHAP feature attributions, and GenAI incident triage summaries.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:
    # Graceful standard-library fallback when pydantic is not yet installed
    class BaseModel:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            # Set default values from class attributes
            for key, val in self.__class__.__dict__.items():
                if not key.startswith("_") and not callable(val):
                    setattr(self, key, val)
            for k, v in kwargs.items():
                setattr(self, k, v)

        def dict(self) -> Dict[str, Any]:
            return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

        def model_dump(self) -> Dict[str, Any]:
            return self.dict()

        def __repr__(self) -> str:
            attrs = ", ".join(f"{k}={v!r}" for k, v in self.dict().items())
            return f"{self.__class__.__name__}({attrs})"

    def Field(default: Any = None, default_factory: Optional[Callable[[], Any]] = None, description: str = "", **kwargs: Any) -> Any:  # type: ignore[no-redef]
        if default_factory is not None:
            return default_factory()
        if default is ...:
            return None
        return default



# -----------------------------------------------------------------------------
# Official Label Contract: Confirmed Attack (1), Confirmed Normal (0), Unknown (-1)
# -----------------------------------------------------------------------------
LABEL_ATTACK: int = 1
LABEL_NORMAL: int = 0
LABEL_UNKNOWN: int = -1

# -----------------------------------------------------------------------------
# Official 13-Feature Contract for Member 2 (ML Classifier & SHAP)
# -----------------------------------------------------------------------------
FEATURE_COLUMNS: List[str] = [
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


class RawLogRecord(BaseModel):
    """Raw unprocessed log entry ingested from disk or stream."""

    raw_text: str = Field(..., description="Original raw line or record text")
    source_host: str = Field("unknown", description="Host or endpoint producing the log")
    log_file: str = Field("unknown", description="Log source path or filename")
    sequence: str = Field("unknown", description="Scenario execution sequence identifier")
    line_number: Optional[int] = Field(None, description="Line number in source file")
    received_at: datetime = Field(default_factory=datetime.utcnow, description="Time ingested")


class RejectedRecord(BaseModel):
    """Log record rejected during parsing or normalization with specific reason."""

    raw_text: str = Field(..., description="Original rejected line text")
    source_host: str = Field("unknown", description="Originating host")
    log_file: str = Field("unknown", description="Source log file identifier")
    sequence: str = Field("unknown", description="Sequence identifier")
    line_number: Optional[int] = Field(None, description="Line number in source file")
    rejection_reason: str = Field(..., description="Diagnostic reason for rejection")


class NormalizedEvent(BaseModel):
    """Normalized security event with canonical fields extracted by parser."""

    timestamp: float = Field(..., description="Unix epoch timestamp in seconds")
    datetime_utc: datetime = Field(..., description="Parsed UTC datetime")
    host: str = Field("unknown", description="Originating host name or IP")
    source_file: str = Field("unknown", description="Source log file identifier")
    sequence: str = Field("unknown", description="Scenario execution sequence identifier")
    log_format: str = Field("generic", description="Detected log format (e.g., audit, auth, syslog, web, json)")
    raw_message: str = Field(..., description="Raw message payload")
    message_length: int = Field(..., description="Character/byte length of log message")

    # Extracted behavioral security attributes
    user: Optional[str] = Field(None, description="Associated username or user identity")
    auth_status: Optional[str] = Field(None, description="Authentication outcome ('failed', 'accepted', 'none')")
    is_root_or_admin: bool = Field(False, description="Whether event references root, admin, or privileged UID")
    dest_port: Optional[int] = Field(None, description="Extracted network destination port")
    is_privileged_port: Optional[bool] = Field(None, description="True if port < 1024")
    is_ephemeral_port: Optional[bool] = Field(None, description="True if port >= 1024")

    # Template mining attributes
    drain_template_id: Optional[str] = Field(None, description="Drain log template identifier")
    drain_template: Optional[str] = Field(None, description="Generalized log template string with wildcards")


class FeatureWindow(BaseModel):
    """10-second behavioral log window matching the ML classifier contract."""

    window_id: str = Field(..., description="Unique deterministic identifier for the window")
    group_key: str = Field("all", description="Grouping key (host, sequence, or 'all')")
    sequence: Optional[str] = Field(None, description="Scenario sequence identity for leakage-free splitting")
    start_timestamp: float = Field(..., description="Epoch start of 10s window (inclusive)")
    end_timestamp: float = Field(..., description="Epoch end of 10s window (exclusive)")

    # 13 Machine Learning Features
    total_event_count: int = Field(0, description="Total number of log events in the window")
    events_per_second_max: float = Field(0.0, description="Peak event arrival rate per 1-second sub-interval")
    byte_length_mean: float = Field(0.0, description="Average message character length across window")
    failed_auth_count: int = Field(0, description="Number of failed/rejected authentication attempts")
    accepted_auth_count: int = Field(0, description="Number of successful/accepted authentication events")
    failed_to_success_ratio: float = Field(0.0, description="Ratio of failed to accepted auth: failed / (accepted + 1e-5)")
    distinct_usernames_count: int = Field(0, description="Number of distinct usernames seen in window")
    root_or_admin_attempt_flag: int = Field(0, description="1 if any root/admin/privileged activity detected, 0 otherwise")
    distinct_dest_ports_count: int = Field(0, description="Number of distinct destination ports seen in window")
    ephemeral_port_ratio: float = Field(0.0, description="Fraction of destination ports >= 1024")
    privileged_port_ratio: float = Field(0.0, description="Fraction of destination ports < 1024")
    unique_drain_templates_count: int = Field(0, description="Number of distinct Drain log templates in window")
    rare_template_ratio: float = Field(0.0, description="Fraction of events belonging to infrequent templates")

    # Metadata and Ground-Truth Provenance (Kept strictly separate from ML features)
    is_attack: Optional[int] = Field(None, description="Ground truth label: 1=attack, 0=normal, -1=unknown/ambiguous")
    tactics: List[str] = Field(default_factory=list, description="MITRE ATT&CK tactics active in window")
    techniques: List[str] = Field(default_factory=list, description="MITRE ATT&CK technique IDs active in window")
    scenario: Optional[str] = Field(None, description="Associated attack scenario identifier")
    step_id: Optional[str] = Field(None, description="Associated attack step identifier")
    events: Optional[List[NormalizedEvent]] = Field(default=None, exclude=True, description="Underlying normalized events for zero-leakage training fitting")

    def to_feature_vector(self) -> List[float]:
        """Return the exact 13-feature values in the order defined by FEATURE_COLUMNS."""
        return [
            float(self.total_event_count),
            float(self.events_per_second_max),
            float(self.byte_length_mean),
            float(self.failed_auth_count),
            float(self.accepted_auth_count),
            float(self.failed_to_success_ratio),
            float(self.distinct_usernames_count),
            float(self.root_or_admin_attempt_flag),
            float(self.distinct_dest_ports_count),
            float(self.ephemeral_port_ratio),
            float(self.privileged_port_ratio),
            float(self.unique_drain_templates_count),
            float(self.rare_template_ratio),
        ]

    def to_feature_dict(self) -> Dict[str, float]:
        """Return a dictionary of the 13 features matching FEATURE_COLUMNS."""
        return {col: val for col, val in zip(FEATURE_COLUMNS, self.to_feature_vector())}
