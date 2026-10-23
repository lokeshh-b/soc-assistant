"""src/streamer.py

Log streaming and batching simulator module for the Explainable SOC Assistant.
Emulates real-time security event ingestion by reading historical or synthetic logs,
slicing them into 10-second behavioral time windows, computing the 13-feature ML contract,
and associating ground-truth attack intervals without data leakage.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

from src.parser import DrainTemplateMiner, LogParser
from src.schemas import FEATURE_COLUMNS, FeatureWindow, NormalizedEvent, RawLogRecord


class AttackInterval:
    """Represents a time interval during which a specific cyber attack technique is active."""

    def __init__(
        self,
        scenario: str,
        event_id: str,
        start_ts: float,
        end_ts: float,
        tactics: List[str],
        techniques: List[str],
        cmd: str = "",
    ) -> None:
        self.scenario = scenario
        self.event_id = event_id
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.tactics = tactics
        self.techniques = techniques
        self.cmd = cmd

    def overlaps(self, window_start: float, window_end: float) -> bool:
        """Return True if window [window_start, window_end) overlaps with [start_ts, end_ts]."""
        return max(window_start, self.start_ts) < min(window_end, self.end_ts)


class AttackTruthStore:
    """Stores ground-truth attack execution intervals from attack_times.csv."""

    def __init__(self, intervals: Optional[List[AttackInterval]] = None) -> None:
        self.intervals: List[AttackInterval] = intervals or []
        self.scenario_intervals: Dict[str, List[AttackInterval]] = defaultdict(list)
        for iv in self.intervals:
            self.scenario_intervals[iv.scenario].append(iv)
        self.known_scenarios = set(self.scenario_intervals.keys())

    @classmethod
    def from_csv(cls, csv_path: Union[str, Path]) -> AttackTruthStore:
        """Load attack intervals from CAM-LDS attack_times.csv, safely skipping malformed rows."""
        p = Path(csv_path)
        if not p.is_file():
            return cls([])

        intervals: List[AttackInterval] = []
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                try:
                    s_str = row.get("start")
                    e_str = row.get("end")
                    if not s_str or not e_str:
                        continue
                    start_ts = float(s_str)
                    end_ts = float(e_str)

                    scenario = row.get("scenario", "unknown")
                    event_id = row.get("event_id", "0")
                    tactics_raw = row.get("tactics", "")
                    techniques_raw = row.get("techniques", "")
                    cmd = row.get("cmd", "")

                    tactics = [t.strip() for t in tactics_raw.split("_") if t.strip()]
                    techniques = [t.strip() for t in techniques_raw.split("_") if t.strip()]

                    intervals.append(
                        AttackInterval(
                            scenario=scenario,
                            event_id=event_id,
                            start_ts=start_ts,
                            end_ts=end_ts,
                            tactics=tactics,
                            techniques=techniques,
                            cmd=cmd,
                        )
                    )
                except (ValueError, KeyError, TypeError):
                    continue

        return cls(intervals)

    def resolve_scenario(self, sequence: Optional[str]) -> Optional[str]:
        """Extract scenario identity from sequence name (e.g. 1_cron_localaccount-17_18_19_33)."""
        if not sequence or sequence in {"unknown", "all", "general"}:
            return None
        for sc in sorted(self.known_scenarios, key=len, reverse=True):
            if sequence == sc or sequence.startswith(sc + "-") or sequence.startswith(sc + "_"):
                return sc
        return None

    def query_window(
        self,
        window_start: float,
        window_end: float,
        scenario: Optional[str] = None,
        sequence: Optional[str] = None,
    ) -> Tuple[int, List[str], List[str], Optional[str], Optional[str]]:
        """Query if a window intersects any attack interval for its specific scenario.

        Returns:
            (is_attack, tactics, techniques, scenario, step_id)
            is_attack: 1 (Confirmed Attack), 0 (Confirmed Normal), -1 (Unknown/Unannotated)
        """
        resolved_sc = scenario or self.resolve_scenario(sequence)

        # If a sequence is identified but has no ground truth in the catalog, it is Unknown
        if (sequence and sequence not in {"unknown", "all", "general"}) or scenario:
            if resolved_sc is None or resolved_sc not in self.known_scenarios:
                return -1, [], [], (scenario or sequence), None
            target_intervals = self.scenario_intervals.get(resolved_sc, [])
        else:
            # Fallback for synthetic/legacy testing without sequence tracking
            target_intervals = self.intervals

        overlapping = [iv for iv in target_intervals if iv.overlaps(window_start, window_end)]
        if not overlapping:
            # Confirmed normal window for an annotated scenario
            return 0, [], [], resolved_sc, None

        all_tactics = sorted(list({t for iv in overlapping for t in iv.tactics}))
        all_techniques = sorted(list({t for iv in overlapping for t in iv.techniques}))
        scenario_out = overlapping[0].scenario
        step_id = overlapping[0].event_id

        return 1, all_tactics, all_techniques, scenario_out, step_id


class WindowAggregator:
    """Aggregates normalized events into 10-second behavioral windows."""

    def __init__(
        self,
        window_duration: float = 10.0,
        stride: Optional[float] = None,
        truth_store: Optional[AttackTruthStore] = None,
        template_miner: Optional[DrainTemplateMiner] = None,
    ) -> None:
        self.window_duration = float(window_duration)
        self.stride = float(stride if stride is not None else window_duration)
        self.truth_store = truth_store or AttackTruthStore()
        self.template_miner = template_miner

    def compute_features(
        self,
        events: List[NormalizedEvent],
        window_start: float,
        window_end: float,
        group_key: str = "all",
        sequence: Optional[str] = None,
        template_miner: Optional[DrainTemplateMiner] = None,
        attach_events: bool = True,
    ) -> FeatureWindow:
        """Compute the official 13-feature contract for a list of events in a 10s window."""
        window_id = f"{group_key}_{int(window_start)}_{int(window_end)}"
        total_event_count = len(events)
        win_seq = sequence or (events[0].sequence if events else None)

        if total_event_count == 0:
            is_attack, tactics, techniques, sc, step_id = self.truth_store.query_window(
                window_start, window_end, sequence=win_seq
            )
            return FeatureWindow(
                window_id=window_id,
                group_key=group_key,
                sequence=win_seq,
                start_timestamp=window_start,
                end_timestamp=window_end,
                is_attack=is_attack,
                tactics=tactics,
                techniques=techniques,
                scenario=sc,
                step_id=step_id,
                events=[] if attach_events else None,
            )

        # 1. Peak event arrival rate (max events in any 1-second sub-bucket)
        one_sec_buckets: Dict[int, int] = defaultdict(int)
        for ev in events:
            bucket = int(ev.timestamp)
            one_sec_buckets[bucket] += 1
        events_per_second_max = float(max(one_sec_buckets.values())) if one_sec_buckets else 0.0

        # 2. Mean message length
        byte_length_mean = float(sum(ev.message_length for ev in events) / total_event_count)

        # 3. Authentication metrics
        failed_auth_count = sum(1 for ev in events if ev.auth_status == "failed")
        accepted_auth_count = sum(1 for ev in events if ev.auth_status == "accepted")
        # Documented ratio policy: direct division if accepted > 0, monotonic failure count if accepted == 0
        if accepted_auth_count > 0:
            failed_to_success_ratio = float(failed_auth_count / accepted_auth_count)
        else:
            failed_to_success_ratio = float(failed_auth_count)

        # 4. User identity & privilege metrics
        distinct_users = {ev.user for ev in events if ev.user}
        distinct_usernames_count = len(distinct_users)
        root_or_admin_attempt_flag = 1 if any(ev.is_root_or_admin for ev in events) else 0

        # 5. Network port metrics
        ports = [ev.dest_port for ev in events if ev.dest_port is not None]
        distinct_dest_ports_count = len(set(ports))
        total_ports = len(ports)
        if total_ports > 0:
            ephemeral_port_ratio = float(sum(1 for p in ports if p >= 1024) / total_ports)
            privileged_port_ratio = float(sum(1 for p in ports if p < 1024) / total_ports)
        else:
            ephemeral_port_ratio = 0.0
            privileged_port_ratio = 0.0

        # 6. Drain template metrics (evaluated against frozen or training template miner)
        template_ids = [ev.drain_template_id for ev in events if ev.drain_template_id]
        unique_drain_templates_count = len(set(template_ids))

        rare_template_events = 0
        active_miner = template_miner or self.template_miner
        if active_miner:
            for tid in template_ids:
                if active_miner.is_rare(tid):
                    rare_template_events += 1
        rare_template_ratio = float(rare_template_events / total_event_count) if total_event_count > 0 else 0.0

        # Ground-truth attack association scoped to sequence/scenario
        is_attack, tactics, techniques, sc, step_id = self.truth_store.query_window(
            window_start, window_end, sequence=win_seq
        )

        return FeatureWindow(
            window_id=window_id,
            group_key=group_key,
            sequence=win_seq,
            start_timestamp=window_start,
            end_timestamp=window_end,
            total_event_count=total_event_count,
            events_per_second_max=events_per_second_max,
            byte_length_mean=byte_length_mean,
            failed_auth_count=failed_auth_count,
            accepted_auth_count=accepted_auth_count,
            failed_to_success_ratio=failed_to_success_ratio,
            distinct_usernames_count=distinct_usernames_count,
            root_or_admin_attempt_flag=root_or_admin_attempt_flag,
            distinct_dest_ports_count=distinct_dest_ports_count,
            ephemeral_port_ratio=ephemeral_port_ratio,
            privileged_port_ratio=privileged_port_ratio,
            unique_drain_templates_count=unique_drain_templates_count,
            rare_template_ratio=rare_template_ratio,
            is_attack=is_attack,
            tactics=tactics,
            techniques=techniques,
            scenario=sc,
            step_id=step_id,
            events=events if attach_events else None,
        )

    def aggregate_stream(
        self,
        events: Iterable[NormalizedEvent],
        group_by_host: bool = False,
        group_by: Optional[str] = None,
    ) -> Iterator[FeatureWindow]:
        """Aggregate an ordered stream of events into sliding/tumbling windows.

        group_by options: 'sequence', 'host', or 'all' (default: 'all', or 'host' if group_by_host=True)
        """
        # Separate by group if required
        mode = group_by or ("host" if group_by_host else "all")
        grouped_events: Dict[str, List[NormalizedEvent]] = defaultdict(list)
        for ev in events:
            if mode == "host":
                key = ev.host
            elif mode == "sequence":
                key = ev.sequence
            else:
                key = "all"
            grouped_events[key].append(ev)

        for key, ev_list in grouped_events.items():
            if not ev_list:
                continue

            # Ensure chronological order
            ev_list.sort(key=lambda x: x.timestamp)
            first_ts = ev_list[0].timestamp
            last_ts = ev_list[-1].timestamp

            # Align window start to integer multiple of window_duration
            window_start = math.floor(first_ts / self.window_duration) * self.window_duration

            idx = 0
            n = len(ev_list)

            while window_start <= last_ts:
                window_end = window_start + self.window_duration

                # Collect all events falling in [window_start, window_end)
                win_events: List[NormalizedEvent] = []
                while idx < n and ev_list[idx].timestamp < window_start:
                    idx += 1

                scan_idx = idx
                while scan_idx < n and ev_list[scan_idx].timestamp < window_end:
                    win_events.append(ev_list[scan_idx])
                    scan_idx += 1

                # Yield window feature model
                if win_events:
                    win_seq = win_events[0].sequence if win_events else None
                    yield self.compute_features(
                        win_events, window_start, window_end, group_key=key, sequence=win_seq
                    )
                    window_start += self.stride
                else:
                    # Skip empty temporal voids directly to next event
                    if idx < n and ev_list[idx].timestamp >= window_end:
                        window_start = math.floor(ev_list[idx].timestamp / self.stride) * self.stride
                    else:
                        window_start += self.stride


class LogStreamer:
    """Reads raw security logs from files, lines, or archives and streams normalized events."""

    def __init__(
        self,
        parser: Optional[LogParser] = None,
        default_host: str = "unknown",
        default_sequence: str = "unknown",
    ) -> None:
        self.parser = parser or LogParser()
        self.default_host = default_host
        self.default_sequence = default_sequence

    def stream_file(
        self,
        file_path: Union[str, Path],
        host: Optional[str] = None,
        sequence: Optional[str] = None,
    ) -> Iterator[NormalizedEvent]:
        """Read a log file line by line and stream normalized events."""
        p = Path(file_path)
        if not p.is_file():
            return

        assigned_host = host or self.default_host
        assigned_sequence = sequence or self.default_sequence
        log_name = p.name

        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line_no, line in enumerate(f, start=1):
                raw_rec = RawLogRecord(
                    raw_text=line.rstrip("\n"),
                    source_host=assigned_host,
                    log_file=log_name,
                    sequence=assigned_sequence,
                    line_number=line_no,
                )
                event = self.parser.parse_record(raw_rec)
                if event is not None:
                    yield event

    def stream_lines(
        self,
        lines: Iterable[str],
        host: Optional[str] = None,
        sequence: Optional[str] = None,
        log_name: str = "stream.log",
    ) -> Iterator[NormalizedEvent]:
        """Stream normalized events from an iterable of raw strings."""
        assigned_host = host or self.default_host
        assigned_sequence = sequence or self.default_sequence
        for line_no, line in enumerate(lines, start=1):
            raw_rec = RawLogRecord(
                raw_text=line.rstrip("\n"),
                source_host=assigned_host,
                log_file=log_name,
                sequence=assigned_sequence,
                line_number=line_no,
            )
            event = self.parser.parse_record(raw_rec)
            if event is not None:
                yield event
