"""tests/test_parser.py

Unit tests for the log parsing and normalization module (src/parser.py).
Verifies timestamp extraction, schema conversion, error handling, malformed log handling,
security attribute extraction, and Drain template mining.
"""

import sys
from pathlib import Path

# Ensure workspace root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.parser import DrainTemplateMiner, LogParser
from src.schemas import RawLogRecord


def test_audit_log_parsing() -> None:
    """Verify parsing of Linux Auditd log records."""
    parser = LogParser()
    raw = (
        "type=SYSCALL msg=audit(1758565786.382:4701): arch=c000003e syscall=59 "
        "success=yes exit=0 a0=7fe4cb2e7152 a1=7ffc1def0bd0 a2=7ffc1def3878 a3=8 "
        "items=3 ppid=2948 pid=2977 auid=0 uid=0 gid=0 euid=0"
    )
    rec = RawLogRecord(raw_text=raw, source_host="videoserver", log_file="audit.log")
    ev = parser.parse_record(rec)

    assert ev is not None
    assert ev.log_format == "audit"
    assert abs(ev.timestamp - 1758565786.382) < 0.001
    assert ev.is_root_or_admin is True
    assert ev.message_length == len(raw)
    assert ev.drain_template_id is not None


def test_auth_log_failed_ssh() -> None:
    """Verify parsing of SSH authentication failure records."""
    parser = LogParser(default_year=2025)
    raw = "Sep 22 18:31:35 videoserver sshd[28491]: Failed password for invalid user admin from 192.42.1.174 port 48212 ssh2"
    rec = RawLogRecord(raw_text=raw, source_host="videoserver", log_file="auth.log")
    ev = parser.parse_record(rec)

    assert ev is not None
    assert ev.auth_status == "failed"
    assert ev.user == "admin"
    assert ev.is_root_or_admin is True
    assert ev.dest_port == 48212
    assert ev.is_ephemeral_port is True
    assert ev.is_privileged_port is False


def test_auth_log_accepted_ssh() -> None:
    """Verify parsing of SSH accepted authentication records."""
    parser = LogParser(default_year=2025)
    raw = "Sep 22 18:35:10 videoserver sshd[28500]: Accepted password for root from 192.42.1.174 port 22 ssh2"
    rec = RawLogRecord(raw_text=raw, source_host="videoserver", log_file="auth.log")
    ev = parser.parse_record(rec)

    assert ev is not None
    assert ev.auth_status == "accepted"
    assert ev.user == "root"
    assert ev.is_root_or_admin is True
    assert ev.dest_port == 22
    assert ev.is_privileged_port is True
    assert ev.is_ephemeral_port is False


def test_web_access_log() -> None:
    """Verify parsing of Apache Combined web access log lines."""
    parser = LogParser()
    raw = '192.42.1.174 - - [22/Sep/2025:18:31:35 +0000] "GET /zm/index.php HTTP/1.1" 404 341 "-" "Mozilla/5.0"'
    rec = RawLogRecord(raw_text=raw, source_host="videoserver", log_file="access.log")
    ev = parser.parse_record(rec)

    assert ev is not None
    assert ev.log_format == "web"
    assert ev.timestamp > 0
    assert ev.host == "videoserver"


def test_suricata_json_alert() -> None:
    """Verify parsing of Suricata eve.json structured IDS alerts."""
    parser = LogParser()
    raw = '{"timestamp": "2025-09-22T18:33:26.202763+0000", "event_type": "alert", "src_ip": "192.42.1.174", "dest_port": 80, "alert": {"signature": "ET EXPLOIT Zoneminder RCE"}}'
    rec = RawLogRecord(raw_text=raw, source_host="firewall", log_file="eve.json")
    ev = parser.parse_record(rec)

    assert ev is not None
    assert ev.log_format == "json"
    assert ev.dest_port == 80
    assert ev.is_privileged_port is True
    assert ev.host == "firewall"


def test_malformed_and_empty_logs() -> None:
    """Verify robust handling of empty, whitespace, and malformed log lines."""
    parser = LogParser()

    # Empty string
    assert parser.parse_record(RawLogRecord(raw_text="")) is None
    # Whitespace string
    assert parser.parse_record(RawLogRecord(raw_text="   \n\t  ")) is None

    # Corrupt string without recognized timestamp rejected if no fallback
    corrupt = "CORRUPTED_RAW_BINARY_DATA_WITH_NO_STRUCTURE"
    rec = RawLogRecord(raw_text=corrupt, source_host="unknown", log_file="corrupt.log")
    assert parser.parse_record(rec) is None

    # Corrupt string with fallback timestamp works
    ev = parser.parse_record(rec, fallback_timestamp=1758565786.0)
    assert ev is not None
    assert ev.timestamp == 1758565786.0
    assert ev.log_format == "fallback"
    assert ev.raw_message == corrupt


def test_wazuh_iso_timestamp() -> None:
    """Verify Wazuh ISO 8601 timestamp with +0000 offset."""
    parser = LogParser()
    raw = '{"timestamp":"2025-12-11T14:28:04.494+0000","rule":{"description":"Suricata alert"}}'
    rec = RawLogRecord(raw_text=raw, source_host="wazuh", log_file="alerts.json")
    ev = parser.parse_record(rec)

    assert ev is not None
    assert abs(ev.timestamp - 1765463284.494) < 0.001
    assert ev.log_format == "json"
    assert ev.datetime_utc.year == 2025
    assert ev.datetime_utc.month == 12
    assert ev.datetime_utc.day == 11
    assert ev.datetime_utc.hour == 14
    assert ev.datetime_utc.minute == 28


def test_bsd_syslog_timestamp() -> None:
    """Verify BSD syslog timestamp parsing with default reference year."""
    parser = LogParser(default_year=2025)
    line = "Dec 11 14:28:09 inetfw dnsmasq[736]: query[A] changelogs.ubuntu.com from 172.17.100.122"
    epoch, dt, fmt = parser.parse_timestamp(line)

    assert fmt == "syslog"
    assert epoch == 1765463289.0
    assert dt is not None
    assert dt.year == 2025
    assert dt.month == 12
    assert dt.day == 11
    assert dt.hour == 14
    assert dt.minute == 28
    assert dt.second == 9

    rec = RawLogRecord(raw_text=line, source_host="unknown", log_file="syslog")
    ev = parser.parse_record(rec)
    assert ev is not None
    assert ev.host == "inetfw"
    assert ev.timestamp == 1765463289.0


def test_suricata_fast_log_timestamp() -> None:
    """Verify Suricata fast.log timestamp format (%m/%d/%Y-%H:%M:%S.%f)."""
    parser = LogParser()
    line = (
        "12/11/2025-14:28:03.251218  [**] [1:2001219:20] ET SCAN Potential SSH Scan "
        "[**] [Classification: Attempted Information Leak] [Priority: 2] {TCP} 192.42.1.174:56754 -> 172.17.100.122:22"
    )
    epoch, dt, fmt = parser.parse_timestamp(line)

    assert fmt == "suricata_fast"
    assert abs(epoch - 1765463283.251218) < 0.000001
    assert dt is not None
    assert dt.year == 2025
    assert dt.month == 12
    assert dt.day == 11
    assert dt.hour == 14
    assert dt.minute == 28
    assert dt.second == 3
    assert dt.microsecond == 251218

    # Also test parse_record extracts destination port 22
    rec = RawLogRecord(raw_text=line, source_host="inetfw", log_file="fast.log")
    ev = parser.parse_record(rec)
    assert ev is not None
    assert ev.dest_port == 22
    assert ev.is_privileged_port is True
    assert ev.is_ephemeral_port is False


def test_unparseable_text_log_rejected_without_fallback() -> None:
    """Verify unparseable text logs are rejected (None) when no fallback timestamp is given."""
    parser = LogParser()
    rec = RawLogRecord(raw_text="A completely arbitrary log line without timestamps", source_host="h", log_file="f.log")

    # Must return None, NOT datetime.now()
    assert parser.parse_record(rec) is None
    assert parser.parse_record(rec, fallback_timestamp=None) is None


def test_unparseable_json_log_rejected_without_fallback() -> None:
    """Verify JSON logs without usable timestamp or @timestamp are rejected (None)."""
    parser = LogParser()

    # 1. Missing timestamp / @timestamp entirely
    rec1 = RawLogRecord(raw_text='{"user": "alice", "action": "login"}', source_host="h", log_file="f.json")
    assert parser.parse_record(rec1) is None

    # 2. Corrupt / unparseable timestamp string
    rec2 = RawLogRecord(raw_text='{"timestamp": "not_a_valid_timestamp", "user": "bob"}', source_host="h", log_file="f.json")
    assert parser.parse_record(rec2) is None

    # 3. Arbitrary timestamp elsewhere in JSON (not at top level timestamp / @timestamp)
    rec3 = RawLogRecord(raw_text='{"event": {"timestamp": "2025-12-11T14:28:03Z"}}', source_host="h", log_file="f.json")
    assert parser.parse_record(rec3) is None

    # 4. Attackmate JSON with only start-datetime (no top-level timestamp / @timestamp)
    rec4 = RawLogRecord(raw_text='{"start-datetime": "2025-12-11T14:28:03.325538", "type": "shell"}', source_host="h", log_file="attackmate.json")
    assert parser.parse_record(rec4) is None


def test_explicit_fallback_timestamps_still_work() -> None:
    """Verify that explicit fallback_timestamp is preserved and works for text and JSON."""
    parser = LogParser()
    fb_ts = 1758565786.5

    # Text fallback
    rec_text = RawLogRecord(raw_text="No timestamp message", source_host="server", log_file="app.log")
    ev_text = parser.parse_record(rec_text, fallback_timestamp=fb_ts)
    assert ev_text is not None
    assert ev_text.timestamp == fb_ts
    assert ev_text.log_format == "fallback"

    # JSON fallback
    rec_json = RawLogRecord(raw_text='{"status": "ok", "service": "health"}', source_host="server", log_file="health.json")
    ev_json = parser.parse_record(rec_json, fallback_timestamp=fb_ts)
    assert ev_json is not None
    assert ev_json.timestamp == fb_ts
    assert ev_json.log_format == "json"


def test_drain_template_miner() -> None:
    """Verify parameter masking and template frequency tracking in DrainTemplateMiner."""
    miner = DrainTemplateMiner(rare_threshold_ratio=0.1)

    t1 = miner.extract_template("Failed password for root from 192.168.1.1 port 2222")
    t2 = miner.extract_template("Failed password for root from 10.0.0.5 port 3333")

    # IPs and Ports should be masked identically
    assert t1 == t2
    assert "<IP>" in t1

    tid1 = miner.register_template(t1)
    tid2 = miner.register_template(t2)
    assert tid1 == tid2

    # Rare template
    rare_raw = "CRITICAL_KERNEL_PANIC at address 0xdeadbeef occurred"
    t_rare = miner.extract_template(rare_raw)
    tid_rare = miner.register_template(t_rare)

    assert miner.is_rare(tid_rare) is True


def test_apache_error_log_timestamp() -> None:
    """Verify parsing of Apache error log timestamps with fractional and integer seconds."""
    parser = LogParser()
    raw = "[Wed Sep 17 13:48:01.937755 2025] [mpm_prefork:notice] [pid 602] AH00171: Graceful restart requested"
    rec = RawLogRecord(raw_text=raw, source_host="videoserver", log_file="error.log")
    ev = parser.parse_record(rec)

    assert ev is not None
    assert ev.log_format == "apache_error"
    assert ev.datetime_utc.year == 2025
    assert ev.datetime_utc.month == 9
    assert ev.datetime_utc.day == 17
    assert ev.datetime_utc.hour == 13
    assert ev.datetime_utc.minute == 48
    assert ev.datetime_utc.second == 1
    assert ev.datetime_utc.microsecond == 937755


def test_future_timestamp_rejected() -> None:
    """Verify timestamps dated in the future (e.g. 2026) are rejected by bounds checking."""
    parser = LogParser(enforce_bounds=True)
    # October 2026 log
    future_log = "2026-10-22T18:26:40+00:00 videoserver kernel: Out of memory"
    rec = RawLogRecord(raw_text=future_log, source_host="videoserver", log_file="syslog")
    ev, reason = parser.parse_record_with_diagnostics(rec)

    assert ev is None
    assert "suspicious" in reason or "out_of_bounds" in reason


def test_drain_template_miner_freeze_prevents_leakage() -> None:
    """Verify that freezing template miner locks vocabulary and prevents test frequency leakage."""
    miner = DrainTemplateMiner(rare_threshold_ratio=0.2)
    # Training logs
    train_logs = [
        "User admin login from 10.0.0.1",
        "User admin login from 10.0.0.2",
        "User admin login from 10.0.0.3",
    ]
    for l in train_logs:
        miner.register_template(miner.extract_template(l))

    assert miner.total_templates_processed == 3
    login_tid = miner.compute_template_id(miner.extract_template(train_logs[0]))
    assert miner.is_rare(login_tid) is False  # frequent in train

    # Freeze miner before evaluation
    miner.freeze()

    # Incoming test / validation logs with a new template
    test_log = "Kernel panic at address 0xdeadbeef"
    test_tid = miner.register_template(miner.extract_template(test_log))

    # Total templates count MUST NOT change after freeze
    assert miner.total_templates_processed == 3
    assert test_tid not in miner.template_counts
    # Unseen test template is correctly classified as rare
    assert miner.is_rare(test_tid) is True


def test_timezone_offsets_normalization() -> None:
    """Verify ISO and web timestamps with timezone offsets convert to correct UTC epoch."""
    parser = LogParser()

    # Web log with +0200
    line_web = '10.0.0.1 - - [22/Sep/2025:20:31:35 +0200] "GET /test HTTP/1.1" 200 100'
    epoch1, dt1, _ = parser.parse_timestamp(line_web)
    assert dt1 is not None and epoch1 is not None
    assert dt1.hour == 18  # 20:31 +0200 is 18:31 UTC

    # ISO with -0400
    line_iso = "2025-09-22T14:31:35-04:00 server: test"
    epoch2, dt2, _ = parser.parse_timestamp(line_iso)
    assert dt2 is not None and epoch2 is not None
    assert dt2.hour == 18  # 14:31 -0400 is 18:31 UTC
    assert abs(epoch1 - epoch2) < 1.0


def test_parser_rejection_diagnostics() -> None:
    """Verify parse_record_with_diagnostics returns informative diagnostic reasons."""
    parser = LogParser()

    # Empty
    rec_empty = RawLogRecord(raw_text="")
    ev, r1 = parser.parse_record_with_diagnostics(rec_empty)
    assert ev is None and r1 == "empty_log_line"

    # Missing timestamp text
    rec_notime = RawLogRecord(raw_text="No timestamp log payload")
    ev, r2 = parser.parse_record_with_diagnostics(rec_notime)
    assert ev is None and "unparseable_timestamp" in r2

    # Malformed JSON timestamp
    rec_json = RawLogRecord(raw_text='{"status": 200, "user": "guest"}')
    ev, r3 = parser.parse_record_with_diagnostics(rec_json)
    assert ev is None and "missing_or_invalid_json_timestamp" in r3


def test_configurable_timestamp_bounds_and_policies() -> None:
    """Verify that LogParser obeys custom bounds and enforce_bounds policies."""
    # 1. Custom bounds: only allow June 2025 to August 2025
    min_epoch = 1748736000.0  # 2025-06-01
    max_epoch = 1756684800.0  # 2025-09-01
    bounded_parser = LogParser(min_timestamp=min_epoch, max_timestamp=max_epoch, enforce_bounds=True)

    # Valid event in July 2025
    log_july = "2025-07-15T12:00:00Z host: test"
    ep_july, _, _ = bounded_parser.parse_timestamp(log_july)
    assert ep_july is not None

    # Event in September 2025 (above max)
    log_sept = "2025-09-15T12:00:00Z host: test"
    ep_sept, _, r_sept = bounded_parser.parse_timestamp(log_sept)
    assert ep_sept is None
    assert r_sept == "suspicious_timestamp_out_of_bounds"

    # 2. Relaxed policy: enforce_bounds=False accepts historical or future valid timestamps
    unbounded_parser = LogParser(enforce_bounds=False)
    log_2026 = "2026-10-22T18:26:40Z host: test"
    ep_2026, dt_2026, _ = unbounded_parser.parse_timestamp(log_2026)
    assert ep_2026 is not None
    assert dt_2026.year == 2026
