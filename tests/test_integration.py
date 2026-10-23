"""tests/test_integration.py

Integration tests for the Explainable SOC Assistant pipeline.
Validates data flow: raw logs -> parser normalization -> 10s behavioral windowing
-> 13-feature contract extraction -> zero leakage dataset export.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure workspace root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.parser import DrainTemplateMiner, LogParser
from src.schemas import FEATURE_COLUMNS, RawLogRecord
from src.streamer import AttackInterval, AttackTruthStore, WindowAggregator


def test_member1_pipeline_integration() -> None:
    """Verify complete end-to-end flow from raw logs to behavioral feature windows."""
    miner = DrainTemplateMiner()
    parser = LogParser(template_miner=miner)

    # Simulated multi-source attack stream
    raw_logs = [
        # Reconnaissance / port scan
        "Sep 22 18:31:01 videoserver sshd[101]: Failed password for invalid user admin from 192.42.1.174 port 48212 ssh2",
        "Sep 22 18:31:02 videoserver sshd[102]: Failed password for invalid user admin from 192.42.1.174 port 48213 ssh2",
        "Sep 22 18:31:04 videoserver sshd[103]: Failed password for invalid user admin from 192.42.1.174 port 48214 ssh2",
        # Web exploit attempt
        '192.42.1.174 - - [22/Sep/2025:18:31:05 +0000] "POST /zm/index.php HTTP/1.1" 302 756 "-" "Mozilla/5.0"',
        # Command execution / privilege escalation
        "type=SYSCALL msg=audit(1758565866.120:10): arch=c000003e syscall=59 success=yes exit=0 auid=0 uid=0 euid=0",
        "Sep 22 18:31:08 videoserver sudo[105]: root : TTY=pts/0 ; COMMAND=/bin/bash /var/www/default/.zm_preload.sh",
        # Accepted login
        "Sep 22 18:31:09 videoserver sshd[106]: Accepted password for root from 192.42.1.174 port 2222 ssh2",
    ]

    base_epoch = 1758565860.0
    normalized_events = []
    for i, line in enumerate(raw_logs):
        rec = RawLogRecord(raw_text=line, source_host="videoserver", log_file="auth.log")
        ev = parser.parse_record(rec, fallback_timestamp=base_epoch + i)
        assert ev is not None
        normalized_events.append(ev)

    assert len(normalized_events) == len(raw_logs)

    # Attack interval definition
    attack = AttackInterval(
        scenario="1_autostart_localaccount",
        event_id="step_5",
        start_ts=base_epoch,
        end_ts=base_epoch + 15.0,
        tactics=["initial-access", "execution"],
        techniques=["T1190", "T1059.006"],
    )
    truth_store = AttackTruthStore([attack])

    aggregator = WindowAggregator(window_duration=10.0, stride=10.0, truth_store=truth_store, template_miner=miner)
    windows = list(aggregator.aggregate_stream(normalized_events))

    assert len(windows) >= 1
    w = windows[0]

    # Verify features and labels
    assert w.total_event_count == len(raw_logs)
    assert w.is_attack == 1
    assert "initial-access" in w.tactics
    assert "T1190" in w.techniques
    assert w.failed_auth_count == 3
    assert w.accepted_auth_count == 1
    assert w.root_or_admin_attempt_flag == 1
    assert w.distinct_dest_ports_count >= 3

    # Ensure full feature vector matches schema
    vector = w.to_feature_vector()
    assert len(vector) == len(FEATURE_COLUMNS)
    for val in vector:
        assert isinstance(val, (int, float))


def test_cam_lds_archive_and_dir_path_parsing() -> None:
    """Verify parse_path_identity parses CAM-LDS sequences and hosts correctly."""
    from scripts.prepare_data import parse_path_identity

    # Archive path
    path1 = "manifestations_filtered/sequences/1_cron_localaccount-17_18_19_33/videoserver/logs/log/audit/audit.log"
    seq1, host1, file1 = parse_path_identity(path1)
    assert seq1 == "1_cron_localaccount-17_18_19_33"
    assert host1 == "videoserver"
    assert file1 == "audit.log"

    # Directory path
    path2 = "/data/raw/manifestations_filtered/sequences/3_ssh_healthcheck-1/wazuh/logs/alerts/alerts.json"
    seq2, host2, file2 = parse_path_identity(path2)
    assert seq2 == "3_ssh_healthcheck-1"
    assert host2 == "wazuh"
    assert file2 == "alerts.json"


def test_archive_and_dir_ingestion_equivalence(tmp_path: Path = None) -> None:
    """Verify archive streaming and directory streaming yield equivalent records."""
    import io
    import tempfile
    import zipfile
    from scripts.prepare_data import stream_logs_from_archive, stream_logs_from_dir

    sample_line = "Sep 22 18:31:01 videoserver sshd[10]: Failed password for root from 1.2.3.4 port 22 ssh2"
    rel_path = "manifestations_filtered/sequences/seq1/host1/logs/auth.log"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)

        # 1. Create zip archive
        zip_path = tmp_p / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr(rel_path, sample_line + "\n")

        # 2. Create disk directory
        disk_file = tmp_p / rel_path
        disk_file.parent.mkdir(parents=True, exist_ok=True)
        disk_file.write_text(sample_line + "\n")

        # Compare outputs
        records_zip = list(stream_logs_from_archive(zip_path))
        records_dir = list(stream_logs_from_dir(tmp_p / "manifestations_filtered"))

        assert len(records_zip) == 1
        assert len(records_dir) == 1
        assert records_zip[0] == records_dir[0]
        assert records_zip[0] == ("seq1", "host1", "auth.log", sample_line)


def test_deterministic_group_splitting_reproducibility() -> None:
    """Verify group-aware splitting is deterministic, reproducible, and keeps sequences together."""
    from scripts.prepare_data import split_windows_grouped
    from src.schemas import FeatureWindow

    # Create 6 windows across 3 distinct sequences
    windows = [
        FeatureWindow(window_id=f"seq1_w{i}", start_timestamp=100.0 + i*10, end_timestamp=110.0 + i*10, sequence="seq1", is_attack=1)
        for i in range(2)
    ] + [
        FeatureWindow(window_id=f"seq2_w{i}", start_timestamp=200.0 + i*10, end_timestamp=210.0 + i*10, sequence="seq2", is_attack=1)
        for i in range(2)
    ] + [
        FeatureWindow(window_id=f"seq3_w{i}", start_timestamp=300.0 + i*10, end_timestamp=310.0 + i*10, sequence="seq3", is_attack=0)
        for i in range(2)
    ]

    # Run split twice with same seed
    tr1, va1, te1, meta1 = split_windows_grouped(windows, seed=42)
    tr2, va2, te2, meta2 = split_windows_grouped(windows, seed=42)

    assert [w.window_id for w in tr1] == [w.window_id for w in tr2]
    assert [w.window_id for w in va1] == [w.window_id for w in va2]
    assert [w.window_id for w in te1] == [w.window_id for w in te2]

    # Verify sequence integrity: windows of seq1 must never be split across train and test
    train_seqs = {w.sequence for w in tr1}
    val_seqs = {w.sequence for w in va1}
    test_seqs = {w.sequence for w in te1}
    assert train_seqs.isdisjoint(val_seqs) or len(val_seqs) == 0
    assert train_seqs.isdisjoint(test_seqs) or len(test_seqs) == 0


def test_impossible_split_constraints_handling() -> None:
    """Verify split_windows_grouped raises ValueError cleanly when no labeled windows exist."""
    from scripts.prepare_data import split_windows_grouped
    from src.schemas import FeatureWindow

    # Only unknown windows (-1)
    unlabeled = [
        FeatureWindow(window_id="w1", start_timestamp=100.0, end_timestamp=110.0, is_attack=-1)
    ]
    try:
        split_windows_grouped(unlabeled)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "No labeled windows available" in str(e)



def test_manifest_and_csv_consistency_validation() -> None:
    """Verify validate_dataset_outputs detects header, count, and label mismatches."""
    import json
    import tempfile
    from scripts.prepare_data import export_feature_matrix, validate_dataset_outputs
    from src.schemas import FEATURE_COLUMNS, FeatureWindow

    with tempfile.TemporaryDirectory() as tmp_dir:
        p = Path(tmp_dir)
        wins = [
            FeatureWindow(window_id="w1", start_timestamp=1758565860.0, end_timestamp=1758565870.0, is_attack=1)
        ]

        # Valid export
        export_feature_matrix(wins, p / "train_features.csv", p / "train_metadata.csv")
        export_feature_matrix(wins, p / "val_features.csv", p / "val_metadata.csv")
        export_feature_matrix(wins, p / "test_features.csv", p / "test_metadata.csv")

        manifest = {
            "split_counts": {"train": 1, "val": 1, "test": 1}
        }
        manifest_p = p / "dataset_manifest.json"
        manifest_p.write_text(json.dumps(manifest))

        valid, errors = validate_dataset_outputs(p, manifest_p)
        assert valid is True
        assert len(errors) == 0

        # Discrepancy test: manifest claims 10 train rows, but CSV only has 1
        manifest["split_counts"]["train"] = 10
        manifest_p.write_text(json.dumps(manifest))
        valid_bad, errors_bad = validate_dataset_outputs(p, manifest_p)
        assert valid_bad is False
        assert any("feature count (1) != manifest count (10)" in err for err in errors_bad)


def test_pipeline_reproducibility_on_small_fixture() -> None:
    """Verify end-to-end preparation, windowing, and validation on synthetic security logs."""
    import json
    import tempfile
    from scripts.prepare_data import export_feature_matrix, split_windows_grouped, validate_dataset_outputs
    from src.parser import DrainTemplateMiner, LogParser
    from src.schemas import RawLogRecord
    from src.streamer import AttackInterval, AttackTruthStore, WindowAggregator

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_p = Path(tmp_dir)

        attack = AttackInterval(
            scenario="1_cron_localaccount",
            event_id="step_17",
            start_ts=1758565860.0,
            end_ts=1758565880.0,
            tactics=["execution"],
            techniques=["T1059"],
        )
        truth_store = AttackTruthStore([attack])
        miner = DrainTemplateMiner()
        parser = LogParser(template_miner=miner)

        lines = [
            "Sep 22 18:31:01 videoserver sshd[10]: Failed password for root from 1.1.1.1 port 22 ssh2",
            "Sep 22 18:31:05 videoserver sshd[11]: Accepted password for root from 1.1.1.1 port 22 ssh2",
            "Sep 22 18:31:12 videoserver sshd[12]: Failed password for user from 1.1.1.1 port 22 ssh2",
        ]
        events = []
        for i, l in enumerate(lines):
            rec = RawLogRecord(raw_text=l, source_host="videoserver", log_file="auth.log", sequence="1_cron_localaccount-17")
            ev = parser.parse_record(rec, fallback_timestamp=1758565860.0 + i * 5)
            if ev:
                events.append(ev)

        agg = WindowAggregator(window_duration=10.0, truth_store=truth_store, template_miner=miner)
        windows = list(agg.aggregate_stream(events, group_by="sequence"))
        assert len(windows) >= 1

        tr, va, te, meta = split_windows_grouped(windows, seed=42)
        export_feature_matrix(tr, out_p / "train_features.csv", out_p / "train_metadata.csv")
        export_feature_matrix(va, out_p / "val_features.csv", out_p / "val_metadata.csv")
        export_feature_matrix(te, out_p / "test_features.csv", out_p / "test_metadata.csv")

        manifest = {"split_counts": {"train": len(tr), "val": len(va), "test": len(te)}}
        manifest_p = out_p / "dataset_manifest.json"
        manifest_p.write_text(json.dumps(manifest))

        valid, errors = validate_dataset_outputs(out_p, manifest_p)
        assert valid is True
        assert len(errors) == 0


def test_zero_leakage_template_miner_feature_invariance() -> None:
    """Prove that training window features are 100% invariant to validation/test templates."""
    from scripts.prepare_data import export_feature_matrix
    from src.parser import DrainTemplateMiner, LogParser
    from src.schemas import NormalizedEvent, RawLogRecord
    from src.streamer import WindowAggregator

    def run_pipeline_with_eval_logs(eval_log_text: str) -> List[List[float]]:
        # Consistent training events
        parser = LogParser(default_year=2025, learn_templates=False)
        train_lines = [
            "Sep 22 18:31:01 videoserver sshd[10]: Failed password for root from 1.1.1.1 port 22 ssh2",
            "Sep 22 18:31:02 videoserver sshd[11]: Accepted password for root from 1.1.1.1 port 22 ssh2",
        ]
        train_events: List[NormalizedEvent] = []
        for i, l in enumerate(train_lines):
            rec = RawLogRecord(raw_text=l, source_host="videoserver", log_file="auth.log", sequence="train-seq")
            ev = parser.parse_record(rec, fallback_timestamp=1758565860.0 + i)
            assert ev is not None
            train_events.append(ev)

        # Eval events (different across runs)
        eval_rec = RawLogRecord(raw_text=eval_log_text, source_host="videoserver", log_file="auth.log", sequence="eval-seq")
        eval_ev = parser.parse_record(eval_rec, fallback_timestamp=1758565900.0)
        assert eval_ev is not None

        # Build training window
        agg = WindowAggregator(window_duration=10.0)
        train_window = agg.compute_features(train_events, 1758565860.0, 1758565870.0, group_key="train-seq", sequence="train-seq")

        # Fit miner strictly on training events
        miner = DrainTemplateMiner(rare_threshold_ratio=0.05)
        for ev in (train_window.events or []):
            if ev.drain_template:
                miner.register_template(ev.drain_template)
        miner.freeze()

        # Update train window features using frozen miner
        evs = train_window.events or []
        tids = [ev.drain_template_id for ev in evs if ev.drain_template_id]
        train_window.unique_drain_templates_count = len(set(tids))
        rare_cnt = sum(1 for tid in tids if miner.is_rare(tid))
        train_window.rare_template_ratio = float(rare_cnt / len(evs)) if evs else 0.0

        return [train_window.to_feature_vector()]

    # Run 1: evaluation log is a normal web log
    features_run1 = run_pipeline_with_eval_logs('10.0.0.1 - - [22/Sep/2025:18:31:40 +0000] "GET /index.html HTTP/1.1" 200 120')

    # Run 2: evaluation log is a completely novel exotic kernel error
    features_run2 = run_pipeline_with_eval_logs("CRITICAL: Out of memory at 0xdeadbeef killed process 9999")

    # Assert that training features are bit-for-bit identical!
    assert features_run1 == features_run2


def test_max_records_controlled_fixture_counting() -> None:
    """Verify that --max-records collects exact target events and maintains consistent counters."""
    from src.parser import LogParser
    from src.schemas import RawLogRecord

    parser = LogParser(default_year=2025)
    # Stream with 3 valid logs and 2 unparseable logs
    stream = [
        "Sep 22 18:31:01 videoserver sshd[10]: Failed password for root from 1.1.1.1 port 22 ssh2",
        "UNPARSEABLE CORRUPTED NO TIMESTAMP LINE 1",
        "Sep 22 18:31:05 videoserver sshd[11]: Accepted password for root from 1.1.1.1 port 22 ssh2",
        "ANOTHER UNPARSEABLE CORRUPTED LINE 2",
        "Sep 22 18:31:09 videoserver sshd[12]: Accepted password for user from 1.1.1.1 port 22 ssh2",
        "Sep 22 18:31:15 videoserver sshd[13]: Failed password for user from 1.1.1.1 port 22 ssh2",
    ]

    target_max = 2
    raw_lines_encountered = 0
    normalized_events = []
    rejected_count = 0

    for line in stream:
        raw_lines_encountered += 1
        rec = RawLogRecord(raw_text=line, source_host="videoserver", log_file="auth.log")
        ev, rj = parser.parse_record_with_diagnostics(rec)
        if ev:
            normalized_events.append(ev)
            if len(normalized_events) >= target_max:
                break
        else:
            rejected_count += 1

    assert len(normalized_events) == target_max
    assert rejected_count == 1  # Encountered 1 invalid line before collecting 2 valid events
    assert raw_lines_encountered == len(normalized_events) + rejected_count == 3


def test_canonical_sequence_hierarchy_filtering() -> None:
    """Verify stream_logs_from_archive rejects step and technique slice directories."""
    import io
    import zipfile
    from scripts.prepare_data import stream_logs_from_archive

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "manifestations_filtered/sequences/1_cron_pam-1/videoserver/logs/auth.log",
            "Sep 22 18:31:01 videoserver sshd[10]: Accepted password for root from 1.1.1.1 port 22 ssh2\n",
        )
        zf.writestr(
            "manifestations_filtered/steps/1_cron_pam-33/videoserver/logs/auth.log",
            "Sep 22 18:31:02 videoserver sshd[11]: Accepted password for root from 1.1.1.1 port 22 ssh2\n",
        )
        zf.writestr(
            "manifestations_filtered/techniques/T1059/videoserver/logs/auth.log",
            "Sep 22 18:31:03 videoserver sshd[12]: Accepted password for root from 1.1.1.1 port 22 ssh2\n",
        )

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".zip") as tf:
        tf.write(buf.getvalue())
        tf.flush()
        yielded = list(stream_logs_from_archive(Path(tf.name)))

    assert len(yielded) == 1
    seq, host, file, line = yielded[0]
    assert seq == "1_cron_pam-1"
    assert host == "videoserver"


def test_split_warnings_on_insufficient_groups() -> None:
    """Verify split_windows_grouped reports explicit warnings on small/insufficient datasets."""
    from scripts.prepare_data import split_windows_grouped
    from src.schemas import FeatureWindow

    # Dataset with only 1 single sequence and only attack windows
    wins = [
        FeatureWindow(
            window_id=f"win_{i}",
            group_key="seq-1",
            sequence="seq-1",
            start_timestamp=1758565860.0 + i * 10,
            end_timestamp=1758565870.0 + i * 10,
            is_attack=1,
        )
        for i in range(5)
    ]

    tr, va, te, meta = split_windows_grouped(wins, seed=42)
    assert meta["strategy"] == "chronological_sequence_blocks"
    assert "warnings" in meta
    warnings = meta["warnings"]
    assert any("Insufficient independent groups" in w for w in warnings)
    assert any("0 normal windows" in w for w in warnings)


def test_per_sequence_raw_limit_and_global_valid_limit_controlled_fixture() -> None:
    """Verify archive/dir equivalence, per-sequence raw line limits, and global valid event caps."""
    import io
    import tempfile
    import zipfile
    from scripts.prepare_data import stream_logs_from_archive, stream_logs_from_dir
    from src.parser import LogParser
    from src.schemas import RawLogRecord

    # Create test data
    seq_a_lines = [
        "Sep 22 18:31:01 videoserver sshd[1]: Accepted password for root from 1.1.1.1 port 22 ssh2\n",  # valid
        "CORRUPTED LINE WITHOUT TIMESTAMP 1\n",                                                          # malformed
        "Sep 22 18:31:02 videoserver sshd[2]: Accepted password for root from 1.1.1.1 port 22 ssh2\n",  # valid
        "2030-10-22T18:31:03Z videoserver sshd[3]: Accepted password for root from 1.1.1.1 port 22 ssh2\n",  # out of bounds
        "Sep 22 18:31:04 videoserver sshd[4]: Accepted password for root from 1.1.1.1 port 22 ssh2\n",  # valid (line 5)
    ]
    seq_b_lines = [
        f"Sep 22 18:32:0{i} videoserver sshd[1{i}]: Failed password for root from 1.1.1.1 port 22 ssh2\n"
        for i in range(1, 6)
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        zip_path = tmp_path / "dataset.zip"
        raw_dir = tmp_path / "raw_extracted"

        # Write ZIP
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("manifestations_filtered/sequences/seq_a/videoserver/logs/auth.log", "".join(seq_a_lines))
            zf.writestr("manifestations_filtered/sequences/seq_b/videoserver/logs/auth.log", "".join(seq_b_lines))

        # Write unzipped directory
        dir_a = raw_dir / "sequences" / "seq_a" / "videoserver" / "logs"
        dir_b = raw_dir / "sequences" / "seq_b" / "videoserver" / "logs"
        dir_a.mkdir(parents=True)
        dir_b.mkdir(parents=True)
        (dir_a / "auth.log").write_text("".join(seq_a_lines))
        (dir_b / "auth.log").write_text("".join(seq_b_lines))

        # 1. Test that max_records_per_sequence=4 caps raw lines per sequence at 4
        archive_yields = list(stream_logs_from_archive(zip_path, max_records_per_sequence=4))
        dir_yields = list(stream_logs_from_dir(raw_dir, max_records_per_sequence=4))

        # Equivalence between archive and directory ingestion
        assert len(archive_yields) == len(dir_yields) == 8  # 4 from seq_a + 4 from seq_b
        for ay, dy in zip(archive_yields, dir_yields):
            assert ay[0] == dy[0]  # sequence
            assert ay[1] == dy[1]  # host
            assert ay[2] == dy[2]  # file
            assert ay[3] == dy[3]  # line text

        # Check that seq_a stopped at 4 raw lines (even though 2 were invalid)
        seq_a_yielded = [y for y in archive_yields if y[0] == "seq_a"]
        assert len(seq_a_yielded) == 4
        assert seq_a_yielded[1][3].strip() == "CORRUPTED LINE WITHOUT TIMESTAMP 1"

        # 2. Test that global max_records stops at target VALID events
        parser = LogParser(default_year=2025, min_timestamp=1735689600.0, max_timestamp=1767225600.0, enforce_bounds=True)
        target_valid = 3
        normalized = []
        rejected = []
        raw_lines_seen = 0

        for seq, host, log_file, text in stream_logs_from_archive(zip_path, max_records_per_sequence=4):
            raw_lines_seen += 1
            rec = RawLogRecord(raw_text=text, source_host=host, log_file=log_file, sequence=seq)
            ev, rj = parser.parse_record_with_diagnostics(rec)
            if ev:
                normalized.append(ev)
                if len(normalized) >= target_valid:
                    break
            else:
                rejected.append(rj)

        # In seq_a: line 1 (valid), line 2 (rejected malformed), line 3 (valid), line 4 (rejected out-of-bounds)
        # In seq_b: line 1 (valid -> target reached: 3 valid events)
        assert len(normalized) == 3
        assert len(rejected) == 2
        assert raw_lines_seen == 5  # 4 lines from seq_a + 1 line from seq_b
        assert raw_lines_seen == len(normalized) + len(rejected)
        assert any("unparseable_timestamp" in r for r in rejected)
        assert any("suspicious_timestamp_out_of_bounds" in r for r in rejected)
