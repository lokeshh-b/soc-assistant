"""src/state_store.py

State persistence and database management module for the Explainable SOC Assistant.
Manages storage and retrieval of window telemetry, classified alerts, SHAP explanations,
and human-in-the-loop analyst triage reviews using SQLite in Write-Ahead Logging (WAL) mode.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import threading
from typing import Any, Dict, Generator, List, Optional, Union

from src.schemas import (
    AnalystReviewSubmission,
    IncidentRecord,
    ShapFeatureImpact,
    Tier2LLMAnalysis,
    WindowMetricsRecord,
)


class StateStore:
    """Data Access Object (DAO) for the SOC Assistant SQLite state store.
    
    Concurrency & Synchronization:
    - In-Memory Mode (':memory:'): Multiple threads sharing the same StateStore instance are
      synchronized using an internal threading.RLock() around the shared connection, ensuring
      serialized access without race conditions.
    - File-Based Mode: Uses SQLite's Write-Ahead Logging (WAL) mode (PRAGMA journal_mode=WAL;),
      allowing concurrent readers and non-blocking background worker writes. Each thread/call
      acquires an isolated connection with standard transaction commit/rollback semantics.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize StateStore with a configurable database path.
        
        Args:
            db_path: Path to the SQLite database file, or ':memory:' for transient test stores.
                     Defaults to STATE_DATABASE_PATH env var or 'artifacts/pipeline_state.db'.
        """
        raw_path = str(db_path) if db_path is not None else os.getenv("STATE_DATABASE_PATH", "artifacts/pipeline_state.db")
        # Strip sqlite:/// prefix if standard URI format was provided
        if raw_path.startswith("sqlite:///"):
            raw_path = raw_path[len("sqlite:///"):]
        self.db_path: str = raw_path

        # Dedicated shared connection and reentrant lock for in-memory databases
        self._lock = threading.RLock()
        self._memory_conn: Optional[sqlite3.Connection] = None
        if self.db_path == ":memory:":
            self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_conn.row_factory = sqlite3.Row

    def get_connection(self) -> sqlite3.Connection:
        """Create and return a configured SQLite connection.
        
        Enforces Row factory and WAL mode for disk-based files.
        """
        if self.db_path == ":memory:":
            if self._memory_conn is None:
                self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
                self._memory_conn.row_factory = sqlite3.Row
            return self._memory_conn

        parent_dir = os.path.dirname(os.path.abspath(self.db_path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for write transactions. Automatically commits or rolls back.
        
        For in-memory stores (':memory:'), acquires a reentrant thread lock (self._lock)
        around the shared connection to prevent concurrent SQLite write collisions.
        """
        if self.db_path == ":memory:":
            with self._lock:
                conn = self.get_connection()
                with conn:
                    yield conn
        else:
            conn = self.get_connection()
            try:
                with conn:
                    yield conn
            finally:
                conn.close()

    @contextmanager
    def cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager for read-only queries with guaranteed resource cleanup.
        
        For in-memory stores (':memory:'), acquires a reentrant thread lock (self._lock)
        around the shared connection to prevent concurrent SQLite cursor race conditions.
        """
        if self.db_path == ":memory:":
            with self._lock:
                conn = self.get_connection()
                cur = conn.cursor()
                try:
                    yield cur
                finally:
                    cur.close()
        else:
            conn = self.get_connection()
            cur = conn.cursor()
            try:
                yield cur
            finally:
                cur.close()
                conn.close()

    def init_db(self) -> None:
        """Initialize database tables according to the project master plan.
        
        Creates 'window_metrics' and 'incidents' tables, enables WAL mode,
        and applies a defensive migration to add 'analyst_notes' if missing.
        """
        with self.transaction() as conn:
            cur = conn.cursor()

            # 1. Window metrics table (telemetry, latency, filtering performance)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS window_metrics (
                    window_id TEXT PRIMARY KEY,
                    timestamp_end TEXT NOT NULL,
                    total_raw_logs INTEGER NOT NULL,
                    risk_score REAL NOT NULL,
                    forwarded_to_llm INTEGER NOT NULL,
                    tier1_latency_ms REAL NOT NULL,
                    ground_truth_malicious INTEGER NOT NULL,
                    ground_truth_mitre TEXT
                );
                """
            )

            # 2. Incidents table (enriched alerts, SHAP attributions, LLM analysis, reviews)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS incidents (
                    window_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    source_ip TEXT NOT NULL,
                    target_host TEXT NOT NULL,
                    occurrence_count INTEGER DEFAULT 1,
                    risk_score REAL NOT NULL,
                    shap_summary_json TEXT NOT NULL,
                    raw_logs_json TEXT NOT NULL,
                    llm_analysis_json TEXT NOT NULL,
                    tier2_latency_ms REAL NOT NULL,
                    analyst_decision TEXT DEFAULT 'PENDING',
                    hallucination_flag INTEGER DEFAULT 0,
                    reviewed_at TEXT,
                    analyst_notes TEXT
                );
                """
            )

            # 3. Defensive migration check: ensure analyst_notes exists on legacy schemas
            cur.execute("PRAGMA table_info(incidents);")
            columns = [row["name"] if isinstance(row, sqlite3.Row) else row[1] for row in cur.fetchall()]
            if "analyst_notes" not in columns:
                cur.execute("ALTER TABLE incidents ADD COLUMN analyst_notes TEXT;")

    def save_window_metrics(self, metrics: WindowMetricsRecord) -> None:
        """Insert or update a window metrics telemetry record."""
        with self.transaction() as conn:
            conn.cursor().execute(
                """
                INSERT INTO window_metrics (
                    window_id, timestamp_end, total_raw_logs, risk_score,
                    forwarded_to_llm, tier1_latency_ms, ground_truth_malicious, ground_truth_mitre
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(window_id) DO UPDATE SET
                    timestamp_end = excluded.timestamp_end,
                    total_raw_logs = excluded.total_raw_logs,
                    risk_score = excluded.risk_score,
                    forwarded_to_llm = excluded.forwarded_to_llm,
                    tier1_latency_ms = excluded.tier1_latency_ms,
                    ground_truth_malicious = excluded.ground_truth_malicious,
                    ground_truth_mitre = excluded.ground_truth_mitre;
                """,
                (
                    metrics.window_id,
                    metrics.timestamp_end,
                    metrics.total_raw_logs,
                    metrics.risk_score,
                    metrics.forwarded_to_llm,
                    metrics.tier1_latency_ms,
                    metrics.ground_truth_malicious,
                    metrics.ground_truth_mitre,
                ),
            )

    def save_incident(self, incident: IncidentRecord) -> None:
        """Save a new incident or update technical payload details for an existing window.
        
        Serializes SHAP attributions, raw logs, and LLM analysis to JSON.
        Explicitly stores hallucination_flag as integer (0 or 1).
        
        Conflict Handling (Duplicate window_id):
        When an incident with an existing window_id is re-saved (e.g. coalesced duplicate or re-ingestion):
        - Technical fields UPDATED: occurrence_count, risk_score, shap_summary_json, raw_logs_json,
          llm_analysis_json, tier2_latency_ms.
        - Human-in-the-loop audit fields PRESERVED: analyst_decision, hallucination_flag,
          analyst_notes, reviewed_at. These analyst audit fields are strictly preserved from the
          existing row and will NEVER be overwritten by pipeline re-saves.
        """
        shap_data = [
            item.model_dump() if hasattr(item, "model_dump") else item.dict()
            for item in incident.shap_summary
        ]
        shap_summary_json = json.dumps(shap_data)

        raw_logs_json = json.dumps(incident.raw_logs)

        if incident.llm_analysis:
            llm_data = (
                incident.llm_analysis.model_dump()
                if hasattr(incident.llm_analysis, "model_dump")
                else incident.llm_analysis.dict()
            )
            llm_analysis_json = json.dumps(llm_data)
        else:
            llm_analysis_json = "{}"

        hallucination_int = 1 if incident.hallucination_flag == 1 else 0

        with self.transaction() as conn:
            conn.cursor().execute(
                """
                INSERT INTO incidents (
                    window_id, created_at, source_ip, target_host, occurrence_count,
                    risk_score, shap_summary_json, raw_logs_json, llm_analysis_json,
                    tier2_latency_ms, analyst_decision, hallucination_flag,
                    reviewed_at, analyst_notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(window_id) DO UPDATE SET
                    occurrence_count = excluded.occurrence_count,
                    risk_score = excluded.risk_score,
                    shap_summary_json = excluded.shap_summary_json,
                    raw_logs_json = excluded.raw_logs_json,
                    llm_analysis_json = excluded.llm_analysis_json,
                    tier2_latency_ms = excluded.tier2_latency_ms;
                """,
                (
                    incident.window_id,
                    incident.created_at,
                    incident.source_ip,
                    incident.target_host,
                    incident.occurrence_count,
                    incident.risk_score,
                    shap_summary_json,
                    raw_logs_json,
                    llm_analysis_json,
                    incident.tier2_latency_ms,
                    incident.analyst_decision,
                    hallucination_int,
                    incident.reviewed_at,
                    incident.analyst_notes,
                ),
            )

    def _row_to_incident(self, row: sqlite3.Row) -> IncidentRecord:
        """Helper to deserialize an SQLite row back into a typed IncidentRecord."""
        shap_raw = json.loads(row["shap_summary_json"] or "[]")
        shap_summary = [ShapFeatureImpact(**item) for item in shap_raw]

        raw_logs = json.loads(row["raw_logs_json"] or "[]")

        llm_raw = json.loads(row["llm_analysis_json"] or "{}")
        llm_analysis = Tier2LLMAnalysis(**llm_raw) if llm_raw and llm_raw != {} else None

        row_keys = row.keys()
        analyst_notes = row["analyst_notes"] if "analyst_notes" in row_keys else None

        return IncidentRecord(
            window_id=row["window_id"],
            created_at=row["created_at"],
            source_ip=row["source_ip"],
            target_host=row["target_host"],
            occurrence_count=int(row["occurrence_count"]),
            risk_score=float(row["risk_score"]),
            shap_summary=shap_summary,
            raw_logs=raw_logs,
            llm_analysis=llm_analysis,
            tier2_latency_ms=float(row["tier2_latency_ms"]),
            analyst_decision=row["analyst_decision"],
            hallucination_flag=int(row["hallucination_flag"] or 0),
            reviewed_at=row["reviewed_at"],
            analyst_notes=analyst_notes,
        )

    def get_incident(self, window_id: str) -> Optional[IncidentRecord]:
        """Retrieve a single incident by its window_id."""
        with self.cursor() as cur:
            cur.execute("SELECT * FROM incidents WHERE window_id = ?", (window_id,))
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_incident(row)

    def get_incidents(
        self,
        decision: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[IncidentRecord]:
        """Retrieve incidents with optional decision filter ('PENDING', 'APPROVED', 'REJECTED').
        
        Ordered by creation time descending (most recent first).
        """
        query = "SELECT * FROM incidents"
        params: List[Any] = []
        if decision:
            query += " WHERE analyst_decision = ?"
            params.append(decision)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self.cursor() as cur:
            cur.execute(query, tuple(params))
            rows = cur.fetchall()
            return [self._row_to_incident(row) for row in rows]

    def submit_review(self, review: AnalystReviewSubmission) -> IncidentRecord:
        """Submit a human-in-the-loop analyst review for an incident.
        
        Updates analyst_decision, hallucination_flag, analyst_notes, and reviewed_at.
        Raises KeyError if window_id is not found in database.
        """
        reviewed_at = datetime.now(timezone.utc).isoformat()
        hallucination_int = 1 if review.hallucination_flag else 0

        with self.transaction() as conn:
            cur = conn.cursor()
            cur.execute("SELECT window_id FROM incidents WHERE window_id = ?", (review.window_id,))
            if not cur.fetchone():
                raise KeyError(f"Incident with window_id '{review.window_id}' not found.")

            cur.execute(
                """
                UPDATE incidents
                SET analyst_decision = ?,
                    hallucination_flag = ?,
                    analyst_notes = ?,
                    reviewed_at = ?
                WHERE window_id = ?
                """,
                (review.decision, hallucination_int, review.analyst_notes, reviewed_at, review.window_id),
            )

        incident = self.get_incident(review.window_id)
        if incident is None:
            raise KeyError(f"Incident with window_id '{review.window_id}' could not be reloaded.")
        return incident

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Return aggregated operational and research metrics for the analyst dashboard."""
        with self.cursor() as cur:
            # Aggregate incident statistics
            cur.execute(
                """
                SELECT 
                    COUNT(*) AS total_incidents,
                    SUM(CASE WHEN analyst_decision = 'PENDING' THEN 1 ELSE 0 END) AS pending_count,
                    SUM(CASE WHEN analyst_decision = 'APPROVED' THEN 1 ELSE 0 END) AS approved_count,
                    SUM(CASE WHEN analyst_decision = 'REJECTED' THEN 1 ELSE 0 END) AS rejected_count,
                    SUM(CASE WHEN hallucination_flag = 1 THEN 1 ELSE 0 END) AS hallucination_count,
                    AVG(risk_score) AS avg_risk_score,
                    SUM(CASE WHEN risk_score >= 0.75 THEN 1 ELSE 0 END) AS high_risk_count,
                    AVG(tier2_latency_ms) AS avg_tier2_latency_ms
                FROM incidents;
                """
            )
            inc_row = cur.fetchone()

            # Aggregate window metrics statistics and Tier 1 filtering efficiency
            cur.execute(
                """
                SELECT 
                    COUNT(*) AS total_windows,
                    SUM(CASE WHEN forwarded_to_llm = 1 THEN 1 ELSE 0 END) AS forwarded_windows,
                    AVG(tier1_latency_ms) AS avg_tier1_latency_ms,
                    SUM(total_raw_logs) AS total_logs_ingested
                FROM window_metrics;
                """
            )
            win_row = cur.fetchone()

        total_incidents = (inc_row["total_incidents"] if inc_row else 0) or 0
        total_windows = (win_row["total_windows"] if win_row else 0) or 0
        forwarded_windows = (win_row["forwarded_windows"] if win_row else 0) or 0

        # Noise Reduction Ratio (NRR):
        # Master Plan Section 8.1 definition: NRR = 1.0 - (N_forwarded / N_total)
        # Measures Tier 1 filtering rate: proportion of log windows suppressed (benign/coalesced)
        # rather than forwarded to Tier 2 GenAI and analyst queue.
        # - Handles empty database cleanly (0.0).
        # - Strictly bounded within [0.0, 1.0].
        if total_windows > 0:
            raw_nrr = 1.0 - (float(forwarded_windows) / float(total_windows))
            nrr = max(0.0, min(1.0, raw_nrr))
        else:
            nrr = 0.0

        return {
            "total_incidents": total_incidents,
            "pending_count": (inc_row["pending_count"] if inc_row else 0) or 0,
            "approved_count": (inc_row["approved_count"] if inc_row else 0) or 0,
            "rejected_count": (inc_row["rejected_count"] if inc_row else 0) or 0,
            "hallucination_count": (inc_row["hallucination_count"] if inc_row else 0) or 0,
            "avg_risk_score": round(float(inc_row["avg_risk_score"] or 0.0), 3) if inc_row else 0.0,
            "high_risk_count": (inc_row["high_risk_count"] if inc_row else 0) or 0,
            "avg_tier2_latency_ms": round(float(inc_row["avg_tier2_latency_ms"] or 0.0), 2) if inc_row else 0.0,
            "total_windows": total_windows,
            "forwarded_windows": forwarded_windows,
            "avg_tier1_latency_ms": round(float(win_row["avg_tier1_latency_ms"] or 0.0), 2) if win_row else 0.0,
            "total_logs_ingested": (win_row["total_logs_ingested"] if win_row else 0) or 0,
            "noise_reduction_ratio": round(float(nrr), 4),
        }

    def close(self) -> None:
        """Close connection resources if an in-memory database is active."""
        with self._lock:
            if self._memory_conn is not None:
                self._memory_conn.close()
                self._memory_conn = None
