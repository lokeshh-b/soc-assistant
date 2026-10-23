"""tests/test_streamer.py

Unit tests for the log streamer and windowing module (src/streamer.py).
Verifies 10-second behavioral window slicing, the 13-feature ML contract, peak event rate,
ground-truth attack interval association, and zero-division protection.
"""

import sys
from pathlib import Path

# Ensure workspace root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.parser import DrainTemplateMiner, LogParser
from src.schemas import FEATURE_COLUMNS, NormalizedEvent, RawLogRecord
from src.streamer import AttackInterval, AttackTruthStore, WindowAggregator


def test_window_aggregator_13_feature_contract() -> None:
    """Verify that WindowAggregator produces the exact 13 features contract."""
    miner = DrainTemplateMiner()
    parser = LogParser(template_miner=miner)
    aggregator = WindowAggregator(window_duration=10.0, stride=10.0, template_miner=miner)

    # Create 5 synthetic events inside a single 10s window [1000.0, 1010.0)
    lines = [
        "Sep 22 18:31:01 videoserver sshd[10]: Failed password for root from 192.42.1.174 port 48212 ssh2",
        "Sep 22 18:31:01 videoserver sshd[11]: Failed password for root from 192.42.1.174 port 48213 ssh2",
        "Sep 22 18:31:02 videoserver sshd[12]: Failed password for admin from 192.42.1.174 port 48214 ssh2",
        "Sep 22 18:31:05 videoserver sshd[13]: Accepted password for root from 192.42.1.174 port 22 ssh2",
        '192.42.1.174 - - [22/Sep/2025:18:31:08 +0000] "GET /index.html HTTP/1.1" 200 1200',
    ]

    events = []
    base_ts = 1758565860.0  # Align to a 10s boundary
    for i, l in enumerate(lines):
        rec = RawLogRecord(raw_text=l, source_host="videoserver", log_file="test.log")
        ev = parser.parse_record(rec, fallback_timestamp=base_ts + i)
        if ev:
            events.append(ev)

    windows = list(aggregator.aggregate_stream(events))
    assert len(windows) == 1

    w = windows[0]
    vec = w.to_feature_vector()
    assert len(vec) == 13
    assert len(FEATURE_COLUMNS) == 13

    # Check feature values
    assert w.total_event_count == 5
    assert w.events_per_second_max >= 1.0
    assert w.byte_length_mean > 0.0
    assert w.failed_auth_count == 3
    assert w.accepted_auth_count == 1
    assert abs(w.failed_to_success_ratio - 3.0) < 0.01
    assert w.distinct_usernames_count >= 2  # 'root' and 'admin'
    assert w.root_or_admin_attempt_flag == 1
    assert w.distinct_dest_ports_count >= 3
    assert w.ephemeral_port_ratio > 0.0  # ephemeral ports >= 1024
    assert w.privileged_port_ratio > 0.0  # port 22 < 1024
    assert w.unique_drain_templates_count >= 2


def test_window_aggregator_attack_overlap() -> None:
    """Verify ground-truth interval intersection and label association."""
    # Active attack interval: [100.0, 150.0]
    attack = AttackInterval(
        scenario="scenario_1_exploit",
        event_id="step_5",
        start_ts=100.0,
        end_ts=150.0,
        tactics=["initial-access"],
        techniques=["T1190"],
    )
    truth_store = AttackTruthStore([attack])
    aggregator = WindowAggregator(window_duration=10.0, stride=10.0, truth_store=truth_store)

    # 1. Normal window [50.0, 60.0) -> No overlap
    w_normal = aggregator.compute_features([], 50.0, 60.0)
    assert w_normal.is_attack == 0
    assert len(w_normal.tactics) == 0

    # 2. Attack window [105.0, 115.0) -> Full overlap
    w_attack = aggregator.compute_features([], 105.0, 115.0)
    assert w_attack.is_attack == 1
    assert "initial-access" in w_attack.tactics
    assert "T1190" in w_attack.techniques
    assert w_attack.scenario == "scenario_1_exploit"
    assert w_attack.step_id == "step_5"


def test_window_aggregator_zero_division_safety() -> None:
    """Verify that empty windows do not cause ZeroDivisionError and have valid defaults."""
    aggregator = WindowAggregator(window_duration=10.0)
    w = aggregator.compute_features([], 200.0, 210.0)

    assert w.total_event_count == 0
    assert w.failed_to_success_ratio == 0.0
    assert w.ephemeral_port_ratio == 0.0
    assert w.privileged_port_ratio == 0.0
    assert w.rare_template_ratio == 0.0
    assert len(w.to_feature_vector()) == 13


def test_sequence_scenario_isolation_in_truth_store() -> None:
    """Verify that attack intervals in scenario A do NOT match events from unrelated scenario B."""
    attack = AttackInterval(
        scenario="1_cron_localaccount",
        event_id="step_17",
        start_ts=1000.0,
        end_ts=1020.0,
        tactics=["execution"],
        techniques=["T1059"],
    )
    truth_store = AttackTruthStore([attack])

    # 1. Matching scenario -> Confirmed Attack (1)
    status_match, _, _, sc_match, _ = truth_store.query_window(
        1005.0, 1015.0, sequence="1_cron_localaccount-17"
    )
    assert status_match == 1
    assert sc_match == "1_cron_localaccount"

    # 2. Unannotated / different scenario -> Unknown (-1), NEVER Normal (0) or Attack (1)
    status_other, _, _, _, _ = truth_store.query_window(
        1005.0, 1015.0, sequence="3_ssh_healthcheck-1"
    )
    assert status_other == -1

    # 3. Same annotated scenario, but outside the attack interval -> Confirmed Normal (0)
    status_normal, _, _, sc_norm, _ = truth_store.query_window(
        1050.0, 1060.0, sequence="1_cron_localaccount-17"
    )
    assert status_normal == 0
    assert sc_norm == "1_cron_localaccount"


def test_unknown_labels_not_normal_by_default() -> None:
    """Verify that windows from unannotated sequences are never silently coerced to normal (0)."""
    truth_store = AttackTruthStore([])
    aggregator = WindowAggregator(window_duration=10.0, truth_store=truth_store)

    # Empty truth store with unknown sequence
    w = aggregator.compute_features([], 100.0, 110.0, sequence="unannotated_scenario_99")
    assert w.is_attack == -1  # Explicit Unknown contract


def test_interval_boundary_conditions() -> None:
    """Verify exact interval boundary overlap: [start, end) semantics."""
    attack = AttackInterval(
        scenario="1_cron_localaccount",
        event_id="step_1",
        start_ts=100.0,
        end_ts=110.0,
        tactics=["persistence"],
        techniques=["T1053"],
    )
    truth_store = AttackTruthStore([attack])

    # Window [90.0, 100.0) ends exactly at attack start -> No overlap (0)
    assert truth_store.query_window(90.0, 100.0, scenario="1_cron_localaccount")[0] == 0

    # Window [95.0, 105.0) overlaps [100.0, 105.0) -> Attack (1)
    assert truth_store.query_window(95.0, 105.0, scenario="1_cron_localaccount")[0] == 1

    # Window [110.0, 120.0) starts exactly at attack end -> No overlap (0)
    assert truth_store.query_window(110.0, 120.0, scenario="1_cron_localaccount")[0] == 0


def test_duplicate_and_out_of_order_events() -> None:
    """Verify that WindowAggregator sorts events chronologically and handles duplicates."""
    aggregator = WindowAggregator(window_duration=10.0)
    parser = LogParser()

    # Events arrived out of order: 108.0, 101.0, 105.0, 101.0 (duplicate)
    timestamps = [108.0, 101.0, 105.0, 101.0]
    events = [
        NormalizedEvent(
            timestamp=ts,
            datetime_utc=parser.parse_timestamp(f"Sep 22 18:31:00")[1],  # dummy
            host="videoserver",
            source_file="test.log",
            log_format="syslog",
            raw_message=f"event at {ts}",
            message_length=20,
        )
        for ts in timestamps
    ]

    windows = list(aggregator.aggregate_stream(events))
    assert len(windows) == 1
    w = windows[0]
    assert w.total_event_count == 4
    assert w.start_timestamp == 100.0
    assert w.end_timestamp == 110.0


def test_empty_windows_and_large_time_gaps() -> None:
    """Verify that large time gaps are skipped without generating millions of empty windows."""
    aggregator = WindowAggregator(window_duration=10.0, stride=10.0)
    parser = LogParser()

    # Event 1 at t=100.0, Event 2 at t=50000.0 (almost 14 hours later)
    ev1 = NormalizedEvent(
        timestamp=100.0,
        datetime_utc=parser.parse_timestamp("Sep 22 18:31:00")[1],
        host="h",
        source_file="f.log",
        raw_message="m1",
        message_length=10,
    )
    ev2 = NormalizedEvent(
        timestamp=50000.0,
        datetime_utc=parser.parse_timestamp("Sep 22 18:31:00")[1],
        host="h",
        source_file="f.log",
        raw_message="m2",
        message_length=10,
    )

    windows = list(aggregator.aggregate_stream([ev1, ev2]))
    # Must only generate 2 non-empty windows, NOT 5000 empty windows!
    assert len(windows) == 2
    assert windows[0].start_timestamp == 100.0
    assert windows[1].start_timestamp == 50000.0


def test_grouping_by_sequence_and_host() -> None:
    """Verify grouping events by sequence vs host vs global."""
    aggregator = WindowAggregator(window_duration=10.0)
    parser = LogParser()

    ev1 = NormalizedEvent(
        timestamp=100.0,
        datetime_utc=parser.parse_timestamp("Sep 22 18:31:00")[1],
        host="hostA",
        sequence="seq1",
        source_file="f.log",
        raw_message="m1",
        message_length=10,
    )
    ev2 = NormalizedEvent(
        timestamp=100.0,
        datetime_utc=parser.parse_timestamp("Sep 22 18:31:00")[1],
        host="hostB",
        sequence="seq2",
        source_file="f.log",
        raw_message="m2",
        message_length=10,
    )

    # Global grouping (all in 1 window)
    wins_all = list(aggregator.aggregate_stream([ev1, ev2], group_by="all"))
    assert len(wins_all) == 1
    assert wins_all[0].total_event_count == 2

    # Group by sequence -> 2 distinct windows
    wins_seq = list(aggregator.aggregate_stream([ev1, ev2], group_by="sequence"))
    assert len(wins_seq) == 2
    keys = {w.group_key for w in wins_seq}
    assert keys == {"seq1", "seq2"}

    # Group by host -> 2 distinct windows
    wins_host = list(aggregator.aggregate_stream([ev1, ev2], group_by="host"))
    assert len(wins_host) == 2
    host_keys = {w.group_key for w in wins_host}
    assert host_keys == {"hostA", "hostB"}
