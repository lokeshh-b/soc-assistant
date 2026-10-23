"""scripts/inspect_dataset.py

Dataset inspection utility script for the Explainable SOC Assistant.
Analyzes raw security log datasets, computes label distributions, missing value
statistics, column summaries, and validates dataset integrity before preprocessing.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.parser import DrainTemplateMiner, LogParser
from src.schemas import RawLogRecord



def inspect_attack_times(csv_path: Path) -> Dict[str, object]:
    """Inspect attack execution intervals and MITRE ATT&CK technique distributions."""
    print(f"\n[+] Inspecting Ground-Truth Intervals: {csv_path}")
    if not csv_path.is_file():
        print(f"    [-] File not found: {csv_path}")
        return {}

    scenarios = Counter()
    tactics_count = Counter()
    techniques_count = Counter()
    durations = []
    total_events = 0

    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            total_events += 1
            scenarios[row.get("scenario", "unknown")] += 1
            t_raw = row.get("tactics", "")
            for t in t_raw.split("_"):
                if t.strip():
                    tactics_count[t.strip()] += 1

            tech_raw = row.get("techniques", "")
            for tech in tech_raw.split("_"):
                if tech.strip():
                    techniques_count[tech.strip()] += 1

            try:
                s_val = row.get("start")
                e_val = row.get("end")
                if s_val is not None and e_val is not None and s_val != "" and e_val != "":
                    s = float(s_val)
                    e = float(e_val)
                    if e >= s > 0:
                        durations.append(e - s)
            except (ValueError, TypeError):
                pass


    print(f"    Total Attack Steps Recorded: {total_events}")
    print(f"    Distinct Scenarios: {len(scenarios)}")
    print("    Top Scenarios:")
    for sc, count in scenarios.most_common(5):
        print(f"      - {sc}: {count} steps")

    print(f"    Distinct MITRE ATT&CK Tactics: {len(tactics_count)}")
    for tac, count in tactics_count.most_common(8):
        print(f"      - {tac}: {count}")

    print(f"    Distinct MITRE ATT&CK Techniques: {len(techniques_count)}")
    for tech, count in techniques_count.most_common(8):
        print(f"      - {tech}: {count}")

    if durations:
        avg_dur = sum(durations) / len(durations)
        print(f"    Attack Step Duration: Mean={avg_dur:.1f}s, Min={min(durations):.1f}s, Max={max(durations):.1f}s")

    return {
        "total_attack_steps": total_events,
        "scenarios": dict(scenarios),
        "tactics": dict(tactics_count),
        "techniques": dict(techniques_count),
    }


def inspect_archive(archive_path: Path) -> Dict[str, object]:
    """Inspect files, log formats, and sizes inside raw dataset archive."""
    print(f"\n[+] Inspecting Dataset Archive: {archive_path}")
    if not archive_path.is_file():
        print(f"    [-] File not found: {archive_path}")
        return {}

    archive_size_mb = archive_path.stat().st_size / (1024 * 1024)
    print(f"    Archive Size: {archive_size_mb:.2f} MB")

    if not zipfile.is_zipfile(archive_path):
        print("    [-] File is not a valid zip archive (download may be in progress)")
        return {"status": "in_progress", "size_mb": archive_size_mb}

    log_types = Counter()
    hosts = Counter()
    total_uncompressed_bytes = 0
    total_files = 0

    with zipfile.ZipFile(archive_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            total_files += 1
            total_uncompressed_bytes += info.file_size
            name = info.filename
            parts = name.split("/")

            # Determine log type
            log_name = parts[-1]
            if "audit" in log_name:
                log_types["audit"] += 1
            elif "auth" in log_name:
                log_types["auth"] += 1
            elif "syslog" in log_name or "messages" in log_name:
                log_types["syslog"] += 1
            elif "access" in log_name:
                log_types["web_access"] += 1
            elif "eve" in log_name or "suricata" in log_name:
                log_types["suricata_ids"] += 1
            elif "alert" in log_name or "wazuh" in log_name:
                log_types["wazuh_alerts"] += 1
            else:
                log_types["other"] += 1

            # Extract host name if structured as <step>/<host>/logs/...
            if len(parts) >= 3:
                hosts[parts[2]] += 1

    uncomp_mb = total_uncompressed_bytes / (1024 * 1024)
    print(f"    Total Files in Archive: {total_files}")
    print(f"    Uncompressed Total Size: {uncomp_mb:.2f} MB (Compression ratio: {uncomp_mb / max(archive_size_mb, 0.1):.1f}x)")
    print("    Log Types Breakdown:")
    for ltype, count in log_types.most_common():
        print(f"      - {ltype}: {count} files")

    return {
        "status": "valid",
        "archive_size_mb": archive_size_mb,
        "uncompressed_mb": uncomp_mb,
        "total_files": total_files,
        "log_types": dict(log_types),
    }


def inspect_parsing_quality(sample_lines: List[str]) -> Dict[str, object]:
    """Test sample log records against LogParser and compute quality metrics."""
    print(f"\n[+] Testing Parser Data Quality on {len(sample_lines)} Sample Records:")
    miner = DrainTemplateMiner()
    parser = LogParser(template_miner=miner)

    parsed_count = 0
    format_counts = Counter()
    auth_counts = Counter()
    admin_count = 0
    ports_detected = 0

    for i, line in enumerate(sample_lines):
        rec = RawLogRecord(raw_text=line, source_host="test-host", log_file="sample.log", line_number=i + 1)
        ev = parser.parse_record(rec)
        if ev:
            parsed_count += 1
            format_counts[ev.log_format] += 1
            auth_counts[ev.auth_status or "none"] += 1
            if ev.is_root_or_admin:
                admin_count += 1
            if ev.dest_port is not None:
                ports_detected += 1

    success_rate = (parsed_count / len(sample_lines)) * 100 if sample_lines else 0
    print(f"    Parsing Success Rate: {success_rate:.1f}% ({parsed_count}/{len(sample_lines)})")
    print("    Detected Formats:")
    for fmt, cnt in format_counts.most_common():
        print(f"      - {fmt}: {cnt}")
    print("    Auth Status Distribution:")
    for a, cnt in auth_counts.most_common():
        print(f"      - {a}: {cnt}")
    print(f"    Privileged/Root Indicators Detected: {admin_count}")
    print(f"    Network Ports Extracted: {ports_detected}")

    return {
        "sample_size": len(sample_lines),
        "success_rate": success_rate,
        "formats": dict(format_counts),
        "auth_distribution": dict(auth_counts),
        "admin_flags": admin_count,
        "ports_detected": ports_detected,
    }


def main() -> None:
    """Inspect raw log datasets and print descriptive statistics."""
    parser = argparse.ArgumentParser(description="CAM-LDS Dataset Inspection Tool")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw data directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print("=" * 70)
    print("EXPLAINABLE SOC ASSISTANT - DATASET INSPECTION & VALIDATION")
    print("=" * 70)

    # 1. Ground truth intervals
    csv_file = data_dir / "attack_times.csv"
    inspect_attack_times(csv_file)

    # 2. Dataset archive
    archive_file = data_dir / "manifestations_filtered.zip"
    inspect_archive(archive_file)

    # 3. Validation on sample multi-source log lines
    sample_records = [
        # Linux Auditd
        "type=SYSCALL msg=audit(1758565786.382:4701): arch=c000003e syscall=59 success=yes exit=0 a0=7fe4cb2e7152 a1=7ffc1def0bd0 a2=7ffc1def3878 a3=8 items=3 ppid=2948 pid=2977 auid=0 uid=0 gid=0 euid=0",
        # SSH failed login
        "Sep 22 18:31:35 videoserver sshd[28491]: Failed password for invalid user admin from 192.42.1.174 port 48212 ssh2",
        # SSH accepted login
        "Sep 22 18:35:10 videoserver sshd[28500]: Accepted password for root from 192.42.1.174 port 2222 ssh2",
        # Sudo execution
        "Sep 22 18:48:50 videoserver sudo[29112]: root : TTY=pts/0 ; PWD=/tmp ; USER=root ; COMMAND=/bin/bash /var/www/default/.zm_preload.sh",
        # Web access log
        '192.42.1.174 - - [22/Sep/2025:18:31:35 +0000] "GET /zm/index.php HTTP/1.1" 404 341 "-" "Fuzz Faster U Fool v2.1.0-dev"',
        # Suricata JSON alert
        '{"timestamp": "2025-09-22T18:33:26.202763+0000", "event_type": "alert", "src_ip": "192.42.1.174", "dest_port": 80, "alert": {"signature": "ET EXPLOIT Zoneminder Snapshot RCE"}}',
        # Wazuh alert JSON
        '{"timestamp": "2025-09-22T18:58:46.027074+0000", "agent": {"name": "videoserver"}, "rule": {"description": "Access to shadow file attempted"}}',
        # Idle cron activity
        "Sep 22 19:00:01 videoserver CRON[29410]: (root) CMD (/usr/sbin/logrotate /etc/logrotate.conf)",
    ]
    inspect_parsing_quality(sample_records)
    print("\n[+] Dataset inspection completed successfully.\n")


if __name__ == "__main__":
    main()
