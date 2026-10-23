"""src/parser.py

Log parsing and normalization module for the Explainable SOC Assistant.
Responsible for extracting, cleaning, and validating security logs into
structured schemas, handling timestamp normalization, behavioral security
attribute extraction, and Drain-style log template mining.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Pattern, Tuple

from src.schemas import NormalizedEvent, RawLogRecord


# -----------------------------------------------------------------------------
# Regular Expressions for Timestamp & Field Extraction
# -----------------------------------------------------------------------------

# Linux Auditd: msg=audit(1758565786.382:4701):
RE_AUDIT_TS: Pattern[str] = re.compile(r"audit\((\d+(?:\.\d+)?):(\d+)\)")

# Apache/Nginx Combined Access Log: [22/Sep/2025:18:31:35 +0000]
RE_APACHE_TS: Pattern[str] = re.compile(r"\[(\d{2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2}\s+[+\-]\d{4})\]")

# Apache Error Log: [Wed Sep 17 13:48:01.937755 2025] or [Sep 17 13:48:01 2025]
RE_APACHE_ERROR_TS: Pattern[str] = re.compile(
    r"\[(?:[A-Za-z]{3}\s+)?([A-Za-z]{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?\s+\d{4})\]"
)

# Suricata fast.log: 12/11/2025-14:28:03.251218
RE_SURICATA_FAST_TS: Pattern[str] = re.compile(r"\b(\d{2}/\d{2}/\d{4}-\d{2}:\d{2}:\d{2}(?:\.\d+)?)\b")

# ISO 8601 / RFC 3339: 2025-09-22T18:31:35.536828+00:00 or 2025-09-22 18:31:35
RE_ISO_TS: Pattern[str] = re.compile(r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+\-]\d{2}:?\d{2})?)")

# BSD Syslog: Sep 22 18:31:35 (requires dataset reference year, e.g. 2025)
RE_BSD_SYSLOG: Pattern[str] = re.compile(
    r"^([A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+([a-zA-Z0-9_\-\.]+)\s+([^:]+):\s*(.*)$"
)

# Authentication Indicators
RE_AUTH_FAIL: Pattern[str] = re.compile(
    r"(?:failed|failure|authentication failure|invalid user|refused|denied|fatal:|incorrect password)",
    re.IGNORECASE,
)
RE_AUTH_ACCEPT: Pattern[str] = re.compile(
    r"(?:accepted\s+(?:password|publickey)|session\s+opened|authentication\s+successful|login\s+successful)",
    re.IGNORECASE,
)

# User Extraction Patterns
RE_USER_FOR_INVALID: Pattern[str] = re.compile(r"for\s+invalid\s+user\s+([a-zA-Z0-9_\-\.]+)", re.IGNORECASE)
RE_USER_FOR: Pattern[str] = re.compile(r"for\s+([a-zA-Z0-9_\-\.]+)\s+from", re.IGNORECASE)
RE_USER_SESSION: Pattern[str] = re.compile(r"session\s+opened\s+for\s+user\s+([a-zA-Z0-9_\-\.]+)", re.IGNORECASE)
RE_USER_KV: Pattern[str] = re.compile(r"\b(?:user|username|acct|auid|uid)=[\"']?([a-zA-Z0-9_\-\.]+)[\"']?", re.IGNORECASE)

# Network Port Indicators (Strict Destination Port isolation)
RE_PORT_EXPLICIT: Pattern[str] = re.compile(r"\bport\s+(\d{1,5})\b", re.IGNORECASE)
RE_PORT_DPT: Pattern[str] = re.compile(r"\b(?:DPT|dest_port|dport)[:=](\d{1,5})\b", re.IGNORECASE)
RE_PORT_ARROW: Pattern[str] = re.compile(r"->\s*[^:\s]+:(\d{1,5})\b")


# Masking Patterns for Drain-Style Template Mining
RE_MASK_IP: Pattern[str] = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
RE_MASK_HEX: Pattern[str] = re.compile(r"\b0x[0-9a-fA-F]+\b|\b[0-9a-fA-F]{8,}\b")
RE_MASK_UUID: Pattern[str] = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
RE_MASK_NUM: Pattern[str] = re.compile(r"\b\d+\b")
RE_MASK_PATH: Pattern[str] = re.compile(r"(?:/[a-zA-Z0-9_\-\.]+){2,}")
RE_MASK_URL: Pattern[str] = re.compile(r"https?://\S+")

# CAM-LDS Plausible Timestamp Bounds (Sept 2025 - Dec 2025; quarantine anything after Jan 1 2026)
MIN_PLAUSIBLE_TIMESTAMP: float = 1735689600.0  # 2025-01-01 00:00:00 UTC
MAX_PLAUSIBLE_TIMESTAMP: float = 1767225600.0  # 2026-01-01 00:00:00 UTC


class DrainTemplateMiner:
    """Lightweight, efficient Drain-inspired log template miner with anti-leakage freeze."""

    def __init__(self, rare_threshold_ratio: float = 0.05) -> None:
        self.rare_threshold_ratio = rare_threshold_ratio
        self.template_counts: Dict[str, int] = {}
        self.total_templates_processed: int = 0
        self.frozen: bool = False

    def freeze(self) -> None:
        """Freeze frequency counts so evaluation sets do not leak into training vocabulary."""
        self.frozen = True

    def unfreeze(self) -> None:
        """Unfreeze miner to allow further template learning."""
        self.frozen = False

    def fit(self, messages: List[str]) -> None:
        """Fit miner on training messages and freeze state."""
        for msg in messages:
            tpl = self.extract_template(msg)
            self.register_template(tpl)
        self.freeze()

    @staticmethod
    def extract_template(message: str) -> str:
        """Convert a raw log message into a generalized template using parameter masking."""
        text = RE_MASK_URL.sub("<URL>", message)
        text = RE_MASK_PATH.sub("<PATH>", text)
        text = RE_MASK_UUID.sub("<UUID>", text)
        text = RE_MASK_IP.sub("<IP>", text)
        text = RE_MASK_HEX.sub("<HEX>", text)
        text = RE_MASK_NUM.sub("<*>", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def compute_template_id(template: str) -> str:
        """Compute an 8-character deterministic hash identifier for a template."""
        return hashlib.md5(template.encode("utf-8")).hexdigest()[:8]

    def register_template(self, template: str) -> str:
        """Register template in vocabulary, update frequency counts (if not frozen), and return ID."""
        tid = self.compute_template_id(template)
        if not self.frozen:
            self.template_counts[tid] = self.template_counts.get(tid, 0) + 1
            self.total_templates_processed += 1
        return tid

    def is_rare(self, template_id: str) -> bool:
        """Determine if a template is rare based on frozen or training corpus frequency."""
        if self.total_templates_processed == 0:
            return False
        count = self.template_counts.get(template_id, 0)
        frequency = count / self.total_templates_processed
        return frequency < self.rare_threshold_ratio or count <= 1


class LogParser:
    """Multi-source security log parser with timestamp normalization and validation."""

    def __init__(
        self,
        default_year: int = 2025,
        template_miner: Optional[DrainTemplateMiner] = None,
        enforce_bounds: bool = True,
        min_timestamp: Optional[float] = None,
        max_timestamp: Optional[float] = None,
        learn_templates: bool = False,
    ) -> None:
        self.default_year = default_year
        self.template_miner = template_miner or DrainTemplateMiner()
        self.enforce_bounds = enforce_bounds
        # Default to dataset-specific bounds for CAM-LDS if not explicitly overridden
        self.min_timestamp = min_timestamp if min_timestamp is not None else (MIN_PLAUSIBLE_TIMESTAMP if enforce_bounds else None)
        self.max_timestamp = max_timestamp if max_timestamp is not None else (MAX_PLAUSIBLE_TIMESTAMP if enforce_bounds else None)
        self.learn_templates = learn_templates

    def _validate_epoch(self, epoch: float) -> Tuple[bool, Optional[str]]:
        """Validate parsed epoch against configured or sanity bounds."""
        if not self.enforce_bounds:
            return True, None
        if self.min_timestamp is not None and epoch < self.min_timestamp:
            return False, "suspicious_timestamp_out_of_bounds"
        if self.max_timestamp is not None and epoch > self.max_timestamp:
            return False, "suspicious_timestamp_out_of_bounds"
        return True, None

    def parse_timestamp(self, text: str) -> Tuple[Optional[float], Optional[datetime], str]:
        """Extract and normalize timestamp from log line.

        Returns:
            Tuple of (epoch_seconds, utc_datetime, detected_format)
        """
        # 1. Auditd log format: msg=audit(1758565786.382:4701):
        m_audit = RE_AUDIT_TS.search(text)
        if m_audit:
            try:
                epoch = float(m_audit.group(1))
                valid, reason = self._validate_epoch(epoch)
                if not valid:
                    return None, None, reason or "suspicious_timestamp_out_of_bounds"
                dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
                return epoch, dt, "audit"
            except (ValueError, OSError):
                pass

        # 2. Apache/Nginx combined access log: [22/Sep/2025:18:31:35 +0000]
        m_apache = RE_APACHE_TS.search(text)
        if m_apache:
            try:
                dt_str = m_apache.group(1)
                dt = datetime.strptime(dt_str, "%d/%b/%Y:%H:%M:%S %z")
                epoch = dt.timestamp()
                valid, reason = self._validate_epoch(epoch)
                if not valid:
                    return None, None, reason or "suspicious_timestamp_out_of_bounds"
                return epoch, dt.astimezone(timezone.utc), "web"
            except (ValueError, OSError):
                pass

        # 3. Apache error log: [Wed Sep 17 13:48:01.937755 2025] or [Sep 17 13:48:01 2025]
        m_apache_err = RE_APACHE_ERROR_TS.search(text)
        if m_apache_err:
            try:
                dt_str = m_apache_err.group(1)
                fmt = "%b %d %H:%M:%S.%f %Y" if "." in dt_str else "%b %d %H:%M:%S %Y"
                dt = datetime.strptime(dt_str, fmt).replace(tzinfo=timezone.utc)
                epoch = dt.timestamp()
                valid, reason = self._validate_epoch(epoch)
                if not valid:
                    return None, None, reason or "suspicious_timestamp_out_of_bounds"
                return epoch, dt, "apache_error"
            except (ValueError, OSError):
                pass

        # 4. Suricata fast.log: 12/11/2025-14:28:03.251218
        m_suricata = RE_SURICATA_FAST_TS.search(text)
        if m_suricata:
            try:
                ts_str = m_suricata.group(1)
                fmt = "%m/%d/%Y-%H:%M:%S.%f" if "." in ts_str else "%m/%d/%Y-%H:%M:%S"
                dt = datetime.strptime(ts_str, fmt).replace(tzinfo=timezone.utc)
                epoch = dt.timestamp()
                valid, reason = self._validate_epoch(epoch)
                if not valid:
                    return None, None, reason or "suspicious_timestamp_out_of_bounds"
                return epoch, dt, "suricata_fast"
            except (ValueError, OSError):
                pass

        # 5. ISO 8601 / RFC 3339 timestamps
        m_iso = RE_ISO_TS.search(text)
        if m_iso:
            try:
                dt_str = m_iso.group(1).replace(" ", "T")
                if dt_str.endswith("Z"):
                    dt_str = dt_str[:-1] + "+00:00"
                if "+" not in dt_str and "-" not in dt_str[10:]:
                    dt_str += "+00:00"
                dt = datetime.fromisoformat(dt_str)
                epoch = dt.timestamp()
                valid, reason = self._validate_epoch(epoch)
                if not valid:
                    return None, None, reason or "suspicious_timestamp_out_of_bounds"
                return epoch, dt.astimezone(timezone.utc), "iso"
            except (ValueError, OSError):
                pass

        # 6. BSD Syslog timestamp: Sep 22 18:31:35
        m_bsd = RE_BSD_SYSLOG.match(text)
        if m_bsd:
            try:
                ts_str = f"{self.default_year} {m_bsd.group(1)}"
                dt = datetime.strptime(ts_str, "%Y %b %d %H:%M:%S").replace(tzinfo=timezone.utc)
                epoch = dt.timestamp()
                valid, reason = self._validate_epoch(epoch)
                if not valid:
                    return None, None, reason or "suspicious_timestamp_out_of_bounds"
                return epoch, dt, "syslog"
            except (ValueError, OSError):
                pass

        return None, None, "unknown"

    def extract_security_attributes(self, text: str) -> Dict[str, Optional[object]]:
        """Extract authentication status, username, root/admin flags, and destination port."""
        # Auth outcome
        auth_status = "none"
        if RE_AUTH_FAIL.search(text):
            auth_status = "failed"
        elif RE_AUTH_ACCEPT.search(text):
            auth_status = "accepted"

        # User extraction
        user: Optional[str] = None
        m_u1 = RE_USER_FOR_INVALID.search(text)
        m_u2 = RE_USER_FOR.search(text)
        m_u3 = RE_USER_SESSION.search(text)
        m_u4 = RE_USER_KV.search(text)

        if m_u1:
            user = m_u1.group(1).strip()
        elif m_u2:
            user = m_u2.group(1).strip()
        elif m_u3:
            user = m_u3.group(1).strip()
        elif m_u4:
            user = m_u4.group(1).strip()

        # Root or admin attempt detection
        is_root_or_admin = False
        text_lower = text.lower()
        if user and user.lower() in {"root", "admin", "administrator"}:
            is_root_or_admin = True
        elif (
            "uid=0" in text
            or "euid=0" in text
            or "auid=0" in text
            or "sudo:" in text_lower
            or "su[" in text_lower
            or "/etc/shadow" in text_lower
            or "user=root" in text_lower
        ):
            is_root_or_admin = True

        # Network destination port extraction (strictly destination port, not source port)
        dest_port: Optional[int] = None
        m_p3 = RE_PORT_ARROW.search(text)  # src:port -> dst:port (highest precision)
        m_p2 = RE_PORT_DPT.search(text)    # DPT=port or dport=port
        m_p1 = RE_PORT_EXPLICIT.search(text)  # port 22

        raw_port = None
        if m_p3:
            raw_port = m_p3.group(1)
        elif m_p2:
            raw_port = m_p2.group(1)
        elif m_p1:
            raw_port = m_p1.group(1)

        if raw_port:
            try:
                p = int(raw_port)
                if 1 <= p <= 65535:
                    dest_port = p
            except ValueError:
                pass

        is_privileged_port = (dest_port < 1024) if dest_port is not None else None
        is_ephemeral_port = (dest_port >= 1024) if dest_port is not None else None

        return {
            "auth_status": auth_status,
            "user": user,
            "is_root_or_admin": is_root_or_admin,
            "dest_port": dest_port,
            "is_privileged_port": is_privileged_port,
            "is_ephemeral_port": is_ephemeral_port,
        }

    def parse_record_with_diagnostics(
        self,
        record: RawLogRecord,
        fallback_timestamp: Optional[float] = None,
    ) -> Tuple[Optional[NormalizedEvent], Optional[str]]:
        """Parse raw record and return (event, rejection_reason).

        Rejection reasons are logged for data quality auditing.
        """
        raw_text = record.raw_text.strip()
        if not raw_text:
            return None, "empty_log_line"

        # Check if line is JSON (Suricata eve.json or Wazuh alerts.json)
        if raw_text.startswith("{") and raw_text.endswith("}"):
            try:
                j = json.loads(raw_text)
                if isinstance(j, dict):
                    ev, rj = self._parse_json_record_with_diagnostics(j, record, fallback_timestamp)
                    return ev, rj
            except json.JSONDecodeError:
                pass  # Fall back to text parsing

        # Standard text parsing
        epoch, dt_utc, log_format = self.parse_timestamp(raw_text)
        if epoch is None or dt_utc is None:
            if fallback_timestamp is not None:
                epoch = fallback_timestamp
                dt_utc = datetime.fromtimestamp(epoch, tz=timezone.utc)
                log_format = "fallback"
            else:
                return None, f"unparseable_timestamp_{log_format}"

        # Extract host if present in BSD syslog format
        host = record.source_host
        m_bsd = RE_BSD_SYSLOG.match(raw_text)
        if m_bsd and host in {"unknown", ""}:
            host = m_bsd.group(2)

        # Extract security features
        sec_attrs = self.extract_security_attributes(raw_text)

        # Drain template mining (stateless ID extraction unless learn_templates is enabled)
        template = self.template_miner.extract_template(raw_text)
        if self.learn_templates and not self.template_miner.frozen:
            template_id = self.template_miner.register_template(template)
        else:
            template_id = self.template_miner.compute_template_id(template)

        event = NormalizedEvent(
            timestamp=epoch,
            datetime_utc=dt_utc,
            host=host or "unknown",
            source_file=record.log_file,
            sequence=record.sequence or "unknown",
            log_format=log_format,
            raw_message=raw_text,
            message_length=len(raw_text),
            user=sec_attrs["user"],  # type: ignore[arg-type]
            auth_status=sec_attrs["auth_status"],  # type: ignore[arg-type]
            is_root_or_admin=bool(sec_attrs["is_root_or_admin"]),
            dest_port=sec_attrs["dest_port"],  # type: ignore[arg-type]
            is_privileged_port=sec_attrs["is_privileged_port"],  # type: ignore[arg-type]
            is_ephemeral_port=sec_attrs["is_ephemeral_port"],  # type: ignore[arg-type]
            drain_template_id=template_id,
            drain_template=template,
        )
        return event, None

    def parse_record(
        self,
        record: RawLogRecord,
        fallback_timestamp: Optional[float] = None,
    ) -> Optional[NormalizedEvent]:
        """Backward-compatible parse method returning NormalizedEvent or None."""
        ev, _ = self.parse_record_with_diagnostics(record, fallback_timestamp)
        return ev

    def _parse_json_record_with_diagnostics(
        self,
        j: Dict[str, object],
        record: RawLogRecord,
        fallback_timestamp: Optional[float] = None,
    ) -> Tuple[Optional[NormalizedEvent], Optional[str]]:
        """Parse structured JSON security logs (Suricata / Wazuh)."""
        ts_val = j.get("timestamp") or j.get("@timestamp")
        epoch: Optional[float] = None
        dt_utc: Optional[datetime] = None

        if isinstance(ts_val, (int, float)):
            try:
                epoch = float(ts_val)
                dt_utc = datetime.fromtimestamp(epoch, tz=timezone.utc)
            except (ValueError, OSError, OverflowError):
                epoch = None
                dt_utc = None
        elif isinstance(ts_val, str):
            epoch, dt_utc, _ = self.parse_timestamp(ts_val)

        if epoch is None or dt_utc is None:
            if fallback_timestamp is not None:
                epoch = fallback_timestamp
                dt_utc = datetime.fromtimestamp(epoch, tz=timezone.utc)
            else:
                return None, "missing_or_invalid_json_timestamp"

        valid, reason = self._validate_epoch(epoch)
        if not valid:
            return None, reason or "suspicious_json_timestamp_out_of_bounds"

        # Host extraction
        host = record.source_host
        if host in {"unknown", ""}:
            if "host" in j and isinstance(j["host"], str):
                host = j["host"]
            elif "agent" in j and isinstance(j["agent"], dict):
                host = j["agent"].get("name", "unknown")

        raw_str = record.raw_text
        sec_attrs = self.extract_security_attributes(raw_str)

        # Port extraction from JSON fields if available
        dest_port = sec_attrs["dest_port"]
        for p_key in ("dest_port", "dport", "destination_port"):
            if p_key in j and isinstance(j[p_key], int):
                dest_port = j[p_key]
                break

        is_priv = (dest_port < 1024) if dest_port is not None else None
        is_ephem = (dest_port >= 1024) if dest_port is not None else None

        template = self.template_miner.extract_template(raw_str)
        if self.learn_templates and not self.template_miner.frozen:
            template_id = self.template_miner.register_template(template)
        else:
            template_id = self.template_miner.compute_template_id(template)

        event = NormalizedEvent(
            timestamp=epoch,
            datetime_utc=dt_utc,
            host=host or "unknown",
            source_file=record.log_file,
            sequence=record.sequence or "unknown",
            log_format="json",
            raw_message=raw_str,
            message_length=len(raw_str),
            user=sec_attrs["user"],  # type: ignore[arg-type]
            auth_status=sec_attrs["auth_status"],  # type: ignore[arg-type]
            is_root_or_admin=bool(sec_attrs["is_root_or_admin"]),
            dest_port=dest_port,  # type: ignore[arg-type]
            is_privileged_port=is_priv,
            is_ephemeral_port=is_ephem,
            drain_template_id=template_id,
            drain_template=template,
        )
        return event, None

    def _parse_json_record(
        self,
        j: Dict[str, object],
        record: RawLogRecord,
        fallback_timestamp: Optional[float] = None,
    ) -> Optional[NormalizedEvent]:
        """Backward-compatible JSON parse helper."""
        ev, _ = self._parse_json_record_with_diagnostics(j, record, fallback_timestamp)
        return ev
