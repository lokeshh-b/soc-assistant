"""scripts/prepare_data.py

Data preparation and feature engineering script for the Explainable SOC Assistant.
Ingests raw security logs from the CAM-LDS dataset (or archive), parses and normalizes
events, aggregates them into 10-second behavioral windows matching the 13-feature ML contract,
performs leakage-free dataset splitting, and exports validated datasets to data/processed/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.parser import DrainTemplateMiner, LogParser
from src.schemas import (
    FEATURE_COLUMNS,
    LABEL_ATTACK,
    LABEL_NORMAL,
    LABEL_UNKNOWN,
    FeatureWindow,
    NormalizedEvent,
    RawLogRecord,
    RejectedRecord,
)
from src.streamer import AttackTruthStore, WindowAggregator


def parse_path_identity(path_str: str) -> Tuple[str, str, str]:
    """Extract (sequence, host, log_file) from CAM-LDS folder structure."""
    parts = path_str.replace("\\", "/").split("/")
    log_file = parts[-1]
    sequence = "unknown"
    host = "unknown"

    if "sequences" in parts:
        idx = parts.index("sequences")
        if idx + 1 < len(parts):
            sequence = parts[idx + 1]
        if idx + 2 < len(parts):
            host = parts[idx + 2]
    elif "steps" in parts:
        idx = parts.index("steps")
        if idx + 1 < len(parts):
            sequence = parts[idx + 1]
        if idx + 2 < len(parts):
            host = parts[idx + 2]

    return sequence, host, log_file


def stream_logs_from_archive(
    archive_path: Path,
    annotated_scenarios: Optional[Set[str]] = None,
    max_records: Optional[int] = None,
    max_records_per_sequence: Optional[int] = None,
) -> Iterator[Tuple[str, str, str, str]]:
    """Stream raw log lines directly from dataset zip archive without extracting to disk.

    Yields:
        (sequence, host, log_file, line_text)
    """
    if not archive_path.is_file() or not zipfile.is_zipfile(archive_path):
        return

    records_yielded = 0
    seq_records_count: Counter[str] = Counter()

    with zipfile.ZipFile(archive_path, "r") as zf:
        namelist = sorted(zf.namelist())
        has_sequences_hierarchy = any("/sequences/" in n or n.startswith("sequences/") for n in namelist)

        # If annotated_scenarios given, prioritize sequences matching known scenarios
        if annotated_scenarios:
            def sequence_priority(name: str) -> int:
                seq, _, _ = parse_path_identity(name)
                for sc in annotated_scenarios:
                    if seq == sc or seq.startswith(sc + "-") or seq.startswith(sc + "_"):
                        return 0
                return 1
            namelist.sort(key=sequence_priority)

        for name in namelist:
            if name.endswith("/") or "attacker" in name.lower():
                continue
            if any(name.endswith(ext) for ext in (".pcap", ".swp", ".gz", ".zip", ".tar", ".bin", ".iso")):
                continue

            # Restrict to canonical sequence hierarchy to avoid duplicate logs from steps/ or techniques/ slices
            normalized_name = name.replace("\\", "/")
            if has_sequences_hierarchy and ("/sequences/" not in normalized_name and not normalized_name.startswith("sequences/")):
                continue

            sequence, host, log_file = parse_path_identity(name)
            if host in {"attacker", "unknown"} and "attacker" in name.lower():
                continue

            if max_records_per_sequence and seq_records_count[sequence] >= max_records_per_sequence:
                continue

            try:
                with zf.open(name, "r") as f:
                    for line_bytes in f:
                        line = line_bytes.decode("utf-8", errors="ignore").rstrip("\r\n")
                        if line.strip():
                            yield (sequence, host, log_file, line)
                            records_yielded += 1
                            seq_records_count[sequence] += 1
                            if max_records and records_yielded >= max_records:
                                return
                            if max_records_per_sequence and seq_records_count[sequence] >= max_records_per_sequence:
                                break
            except Exception:
                continue


def stream_logs_from_dir(
    raw_dir: Path,
    annotated_scenarios: Optional[Set[str]] = None,
    max_records: Optional[int] = None,
    max_records_per_sequence: Optional[int] = None,
) -> Iterator[Tuple[str, str, str, str]]:
    """Stream raw log lines from an unzipped directory structure."""
    records_yielded = 0
    seq_records_count: Counter[str] = Counter()
    all_files = sorted([p for p in raw_dir.rglob("*") if p.is_file()])
    has_sequences_hierarchy = any("sequences" in p.parts for p in all_files)

    if annotated_scenarios:
        def path_priority(p: Path) -> int:
            seq, _, _ = parse_path_identity(str(p))
            for sc in annotated_scenarios:
                if seq == sc or seq.startswith(sc + "-") or seq.startswith(sc + "_"):
                    return 0
            return 1
        all_files.sort(key=path_priority)

    for log_path in all_files:
        if "attacker" in str(log_path).lower():
            continue
        if log_path.suffix in {".pcap", ".swp", ".gz", ".zip", ".tar", ".bin", ".iso"}:
            continue

        # Restrict to canonical sequences folder if present
        if has_sequences_hierarchy and "sequences" not in log_path.parts:
            continue

        sequence, host, log_file = parse_path_identity(str(log_path))
        if host == "attacker":
            continue

        if max_records_per_sequence and seq_records_count[sequence] >= max_records_per_sequence:
            continue

        try:
            with log_path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.rstrip("\r\n")
                    if line.strip():
                        yield (sequence, host, log_file, line)
                        records_yielded += 1
                        seq_records_count[sequence] += 1
                        if max_records and records_yielded >= max_records:
                            return
                        if max_records_per_sequence and seq_records_count[sequence] >= max_records_per_sequence:
                            break
        except Exception:
            continue



def export_feature_matrix(
    windows: List[FeatureWindow],
    output_csv: Path,
    meta_csv: Optional[Path] = None,
) -> None:
    """Export 13-feature ML matrix and separate ground-truth metadata to CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["window_id"] + FEATURE_COLUMNS)
        for w in windows:
            writer.writerow([w.window_id] + w.to_feature_vector())

    if meta_csv:
        with meta_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "window_id",
                "group_key",
                "sequence",
                "start_timestamp",
                "end_timestamp",
                "is_attack",
                "tactics",
                "techniques",
                "scenario",
                "step_id",
            ])
            for w in windows:
                writer.writerow([
                    w.window_id,
                    w.group_key,
                    w.sequence or "",
                    w.start_timestamp,
                    w.end_timestamp,
                    w.is_attack if w.is_attack is not None else -1,
                    ";".join(w.tactics),
                    ";".join(w.techniques),
                    w.scenario or "",
                    w.step_id or "",
                ])


def split_windows_grouped(
    windows: List[FeatureWindow],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[FeatureWindow], List[FeatureWindow], List[FeatureWindow], Dict[str, object]]:
    """Group-aware dataset splitting to prevent data leakage across related windows.

    Keeps all windows from the same sequence/scenario together in the same split.
    """
    # Exclude unknown windows from supervised splits
    labeled_windows = [w for w in windows if w.is_attack in (LABEL_ATTACK, LABEL_NORMAL)]
    if not labeled_windows:
        raise ValueError("No labeled windows available for dataset splitting.")

    # Group windows by sequence (or scenario)
    groups: Dict[str, List[FeatureWindow]] = defaultdict(list)
    for w in labeled_windows:
        grp = w.sequence or w.scenario or w.group_key or "default"
        groups[grp].append(w)

    group_keys = sorted(groups.keys())
    rng = random.Random(seed)

    # Classify groups by presence of attacks
    attack_groups = [k for k in group_keys if any(w.is_attack == LABEL_ATTACK for w in groups[k])]
    normal_only_groups = [k for k in group_keys if all(w.is_attack == LABEL_NORMAL for w in groups[k])]

    rng.shuffle(attack_groups)
    rng.shuffle(normal_only_groups)

    # If we have multiple independent groups, distribute across splits
    if len(group_keys) >= 3 and len(attack_groups) >= 2:
        split_strategy = "group_stratified"
        train_groups: List[str] = []
        val_groups: List[str] = []
        test_groups: List[str] = []

        def partition_keys(keys: List[str]) -> Tuple[List[str], List[str], List[str]]:
            n = len(keys)
            n_tr = max(1, int(round(n * train_ratio)))
            n_va = max(1, int(round(n * val_ratio))) if n >= 3 else 0
            if n_tr + n_va >= n and n >= 3:
                n_tr = n - 2
                n_va = 1
            return keys[:n_tr], keys[n_tr : n_tr + n_va], keys[n_tr + n_va :]

        tr_atk, va_atk, te_atk = partition_keys(attack_groups)
        tr_norm, va_norm, te_norm = partition_keys(normal_only_groups)

        train_groups.extend(tr_atk + tr_norm)
        val_groups.extend(va_atk + va_norm)
        test_groups.extend(te_atk + te_norm)

        train_wins = [w for k in train_groups for w in groups[k]]
        val_wins = [w for k in val_groups for w in groups[k]]
        test_wins = [w for k in test_groups for w in groups[k]]
    else:
        # Fallback to chronological non-overlapping blocks when independent groups are scarce
        split_strategy = "chronological_sequence_blocks"
        windows_sorted = sorted(labeled_windows, key=lambda w: w.start_timestamp)
        n = len(windows_sorted)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio)) if n > 2 else 0

        train_wins = windows_sorted[:n_train]
        val_wins = windows_sorted[n_train : n_train + n_val]
        test_wins = windows_sorted[n_train + n_val :]

    # Sort each partition chronologically
    train_wins.sort(key=lambda w: w.start_timestamp)
    val_wins.sort(key=lambda w: w.start_timestamp)
    test_wins.sort(key=lambda w: w.start_timestamp)

    warnings: List[str] = []
    if split_strategy != "group_stratified":
        warnings.append("Insufficient independent groups (need >= 3 total and >= 2 attack groups); used chronological split.")

    for s_name, s_wins in [("train", train_wins), ("val", val_wins), ("test", test_wins)]:
        if not s_wins:
            warnings.append(f"{s_name} split is empty.")
            continue
        has_atk = any(w.is_attack == LABEL_ATTACK for w in s_wins)
        has_norm = any(w.is_attack == LABEL_NORMAL for w in s_wins)
        if not has_atk:
            warnings.append(f"{s_name} split contains 0 attack windows.")
        if not has_norm:
            warnings.append(f"{s_name} split contains 0 normal windows.")

    train_seqs = {w.sequence for w in train_wins if w.sequence}
    val_seqs = {w.sequence for w in val_wins if w.sequence}
    test_seqs = {w.sequence for w in test_wins if w.sequence}

    overlap_tr_va = len(train_seqs & val_seqs)
    overlap_tr_te = len(train_seqs & test_seqs)
    overlap_va_te = len(val_seqs & test_seqs)

    meta = {
        "strategy": split_strategy,
        "seed": seed,
        "total_groups": len(groups),
        "attack_groups": len(attack_groups),
        "normal_groups": len(normal_only_groups),
        "group_overlaps": {
            "train_val": overlap_tr_va,
            "train_test": overlap_tr_te,
            "val_test": overlap_va_te,
        },
        "warnings": warnings,
    }
    return train_wins, val_wins, test_wins, meta


def validate_dataset_outputs(out_dir: Path, manifest_path: Path) -> Tuple[bool, List[str]]:
    """Validate that exported CSV datasets and manifest are internally consistent and clean."""
    errors: List[str] = []
    if not manifest_path.is_file():
        errors.append(f"Manifest not found: {manifest_path}")
        return False, errors

    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        errors.append(f"Could not parse manifest JSON: {e}")
        return False, errors

    splits = ["train", "val", "test"]
    split_counts = manifest.get("split_counts", {})

    total_csv_rows = 0
    for split in splits:
        f_csv = out_dir / f"{split}_features.csv"
        m_csv = out_dir / f"{split}_metadata.csv"

        if not f_csv.is_file():
            errors.append(f"Missing feature CSV: {f_csv}")
            continue
        if not m_csv.is_file():
            errors.append(f"Missing metadata CSV: {m_csv}")
            continue

        with f_csv.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header != ["window_id"] + FEATURE_COLUMNS:
                errors.append(f"{f_csv.name} header mismatch: {header}")
            rows = list(reader)
            count = len(rows)
            total_csv_rows += count

            expected = split_counts.get(split)
            if expected is not None and count != expected:
                errors.append(f"{split} feature count ({count}) != manifest count ({expected})")

            # Validate numeric feature values and bounds
            for r_idx, row in enumerate(rows):
                if len(row) != len(FEATURE_COLUMNS) + 1:
                    errors.append(f"{f_csv.name} row {r_idx} invalid column count: {len(row)}")
                    break
                for v in row[1:]:
                    try:
                        fv = float(v)
                        if math.isnan(fv) or math.isinf(fv):
                            errors.append(f"{f_csv.name} row {r_idx} contains NaN/Inf: {fv}")
                    except ValueError:
                        errors.append(f"{f_csv.name} row {r_idx} non-numeric value: {v}")

        # Metadata validation
        with m_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            m_rows = list(reader)
            if len(m_rows) != count:
                errors.append(f"{m_csv.name} row count ({len(m_rows)}) != feature rows ({count})")
            for r in m_rows:
                lbl = r.get("is_attack")
                if lbl not in {"0", "1"}:
                    errors.append(f"{m_csv.name} contains invalid supervised label: {lbl}")
                try:
                    ts = float(r.get("start_timestamp", 0))
                    # Validate timestamp bounds: 2025 range
                    if ts > 1767225600.0 or ts < 1735689600.0:
                        errors.append(f"{m_csv.name} suspicious timestamp out of 2025 bounds: {ts}")
                except ValueError:
                    errors.append(f"{m_csv.name} non-numeric start_timestamp")

    return len(errors) == 0, errors


def main() -> None:
    """Preprocess raw logs and generate processed datasets."""
    parser = argparse.ArgumentParser(description="CAM-LDS Feature Engineering Pipeline")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Directory containing raw data")
    parser.add_argument("--out-dir", type=str, default="data/processed", help="Output directory for processed data")
    parser.add_argument("--window-sec", type=float, default=10.0, help="Window duration in seconds (default: 10.0)")
    parser.add_argument("--stride-sec", type=float, default=10.0, help="Window stride in seconds (default: 10.0)")
    parser.add_argument("--max-records", type=int, default=None, help="Optional max log records to process")
    parser.add_argument("--max-records-per-sequence", type=int, default=300, help="Optional max log records per sequence (default: 300)")
    parser.add_argument("--group-by", type=str, default="sequence", choices=["sequence", "host", "all"], help="Grouping mode")
    parser.add_argument("--group-by-host", action="store_true", help="Legacy flag to group windows per host")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for group splitting")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grouping_mode = "host" if args.group_by_host else args.group_by

    print("=" * 70)
    print("MEMBER 1: CAM-LDS DATA PREPARATION & BEHAVIORAL WINDOWING")
    print("=" * 70)
    print(f"Window Duration: {args.window_sec}s | Stride: {args.stride_sec}s | Grouping: {grouping_mode}")
    print(f"Feature Contract: {len(FEATURE_COLUMNS)} Features -> {FEATURE_COLUMNS}")

    # 1. Load Ground-Truth Attack Truth Store
    attack_csv = data_dir / "attack_times.csv"
    truth_store = AttackTruthStore.from_csv(attack_csv)
    print(f"[+] Loaded {len(truth_store.intervals)} attack intervals across {len(truth_store.known_scenarios)} scenarios from {attack_csv}")

    # 2. Initialize Parser with stateless template extraction (Anti-Leakage)
    log_parser = LogParser(default_year=2025, learn_templates=False)

    # 3. Stream and Parse Normalized Events
    archive_zip = data_dir / "manifestations_filtered.zip"
    unzipped_dir = data_dir / "manifestations_filtered"

    event_generator: Iterator[Tuple[str, str, str, str]]
    source_name = "synthetic"
    archive_size_bytes = 0

    if unzipped_dir.is_dir():
        print(f"[+] Ingesting from directory: {unzipped_dir}")
        source_name = str(unzipped_dir)
        event_generator = stream_logs_from_dir(
            unzipped_dir,
            annotated_scenarios=truth_store.known_scenarios,
            max_records=None,
            max_records_per_sequence=args.max_records_per_sequence,
        )
    elif archive_zip.is_file() and zipfile.is_zipfile(archive_zip):
        archive_size_bytes = archive_zip.stat().st_size
        print(f"[+] Ingesting directly from archive: {archive_zip} ({archive_size_bytes / (1024*1024):.2f} MB)")
        source_name = str(archive_zip)
        event_generator = stream_logs_from_archive(
            archive_zip,
            annotated_scenarios=truth_store.known_scenarios,
            max_records=None,
            max_records_per_sequence=args.max_records_per_sequence,
        )

    else:
        print("[-] No raw dataset archive or directory found. Falling back to synthetic security logs for validation.")
        synthetic_lines = [
            ("1_cron_localaccount-17", "videoserver", "auth.log", f"Sep 24 07:47:{i:02d} videoserver sshd[100]: Failed password for root from 192.42.1.174 port {2200+i} ssh2")
            for i in range(30)
        ] + [
            ("1_cron_localaccount-17", "videoserver", "auth.log", f"Sep 24 07:48:{i:02d} videoserver sshd[200]: Accepted password for root from 192.42.1.174 port 22 ssh2")
            for i in range(15)
        ] + [
            ("1_autostart_pam-24", "videoserver", "auth.log", f"Oct 14 11:00:{i:02d} videoserver sshd[300]: Failed password for admin from 192.42.1.174 port 2222 ssh2")
            for i in range(25)
        ]
        event_generator = iter(synthetic_lines)

    normalized_events: List[NormalizedEvent] = []
    rejected_records: List[RejectedRecord] = []
    rejection_counts: Counter[str] = Counter()
    raw_lines_encountered = 0

    print("[+] Normalizing log stream with diagnostic tracking...")
    for sequence, host, log_file, raw_text in event_generator:
        raw_lines_encountered += 1
        raw_rec = RawLogRecord(
            raw_text=raw_text,
            source_host=host,
            log_file=log_file,
            sequence=sequence,
            line_number=raw_lines_encountered,
        )
        ev, rejection_reason = log_parser.parse_record_with_diagnostics(raw_rec)
        if ev:
            normalized_events.append(ev)
            if len(normalized_events) % 10000 == 0:
                print(f"    Ingested {len(normalized_events):,} valid events...")
            if args.max_records and len(normalized_events) >= args.max_records:
                print(f"    Reached target normalized events limit: {args.max_records}")
                break
        else:
            rejection_counts[rejection_reason or "unknown"] += 1
            if len(rejected_records) < 500:
                rejected_records.append(
                    RejectedRecord(
                        raw_text=raw_text[:120],
                        source_host=host,
                        log_file=log_file,
                        sequence=sequence,
                        line_number=raw_lines_encountered,
                        rejection_reason=rejection_reason or "unknown",
                    )
                )

    acceptance_rate = (len(normalized_events) / max(raw_lines_encountered, 1)) * 100
    print(f"[+] Total Normalized Events Ingested: {len(normalized_events):,} / {raw_lines_encountered:,} ({acceptance_rate:.2f}%)")
    if rejection_counts:
        print(f"    Rejected lines breakdown: {dict(rejection_counts)}")

    if not normalized_events:
        print("[-] No valid events could be parsed. Exiting with failure.")
        sys.exit(1)

    # 4. Aggregate Events into 10-Second Windows
    print(f"[+] Generating 10-second behavioral windows (group_by={grouping_mode})...")
    aggregator = WindowAggregator(
        window_duration=args.window_sec,
        stride=args.stride_sec,
        truth_store=truth_store,
    )

    all_windows = list(aggregator.aggregate_stream(normalized_events, group_by=grouping_mode))
    print(f"[+] Total Behavioral Windows Generated: {len(all_windows):,}")

    attack_wins_total = sum(1 for w in all_windows if w.is_attack == LABEL_ATTACK)
    normal_wins_total = sum(1 for w in all_windows if w.is_attack == LABEL_NORMAL)
    unknown_wins_total = sum(1 for w in all_windows if w.is_attack == LABEL_UNKNOWN)

    print(f"    Class Breakdown: Attack={attack_wins_total} | Normal={normal_wins_total} | Unknown/Unannotated={unknown_wins_total}")

    # Exclude unknown windows from supervised ML splits to protect Member 2 classifier contract
    supervised_windows = [w for w in all_windows if w.is_attack in (LABEL_ATTACK, LABEL_NORMAL)]
    print(f"[+] Retaining {len(supervised_windows):,} supervised windows ({unknown_wins_total:,} unknown windows excluded)")

    if not supervised_windows:
        print("[-] No supervised windows with confirmed ground-truth labels. Exiting.")
        sys.exit(1)

    # 5. Group-Aware Leakage-Free Dataset Split
    train_wins, val_wins, test_wins, split_meta = split_windows_grouped(
        supervised_windows,
        train_ratio=0.70,
        val_ratio=0.15,
        seed=args.seed,
    )
    print(f"[+] Leakage-Free Split ({split_meta['strategy']}): Train={len(train_wins):,} | Val={len(val_wins):,} | Test={len(test_wins):,}")

    # 6. Fit DrainTemplateMiner strictly on Training Partition and freeze to prevent leakage
    print("[+] Fitting log template miner strictly on training partition (Anti-Leakage Guarantee)...")
    train_miner = DrainTemplateMiner(rare_threshold_ratio=0.05)
    train_event_occurrences = 0
    for w in train_wins:
        for ev in (w.events or []):
            if ev.drain_template:
                train_miner.register_template(ev.drain_template)
                train_event_occurrences += 1
    train_miner.freeze()
    print(f"    Frozen training vocabulary: {len(train_miner.template_counts)} templates ({train_event_occurrences} training event occurrences)")

    # Recompute template features for each split using the frozen training miner
    def apply_frozen_miner_to_windows(wins: List[FeatureWindow], miner: DrainTemplateMiner) -> None:
        for w in wins:
            evs = w.events or []
            if evs:
                tids = [ev.drain_template_id for ev in evs if ev.drain_template_id]
                w.unique_drain_templates_count = len(set(tids))
                rare_cnt = sum(1 for tid in tids if miner.is_rare(tid))
                w.rare_template_ratio = float(rare_cnt / len(evs)) if evs else 0.0

    apply_frozen_miner_to_windows(train_wins, train_miner)
    apply_frozen_miner_to_windows(val_wins, train_miner)
    apply_frozen_miner_to_windows(test_wins, train_miner)

    # 7. Export Feature Matrices and Companion Metadata
    print(f"[+] Exporting processed CSV datasets to {out_dir}/...")
    export_feature_matrix(train_wins, out_dir / "train_features.csv", out_dir / "train_metadata.csv")
    export_feature_matrix(val_wins, out_dir / "val_features.csv", out_dir / "val_metadata.csv")
    export_feature_matrix(test_wins, out_dir / "test_features.csv", out_dir / "test_metadata.csv")

    # 8. Generate Manifest Summary
    def get_class_dist(wins: List[FeatureWindow]) -> Dict[str, int]:
        return {
            "total": len(wins),
            "attack": sum(1 for w in wins if w.is_attack == LABEL_ATTACK),
            "normal": sum(1 for w in wins if w.is_attack == LABEL_NORMAL),
        }

    manifest = {
        "dataset_name": "CAM-LDS Behavioral Feature Windows",
        "pipeline_version": "2.0.0-repaired",
        "source": source_name,
        "archive_size_bytes": archive_size_bytes,
        "window_duration_seconds": args.window_sec,
        "stride_seconds": args.stride_sec,
        "grouping_mode": grouping_mode,
        "raw_lines_encountered": raw_lines_encountered,
        "normalized_events_count": len(normalized_events),
        "acceptance_rate_percent": round(acceptance_rate, 2),
        "rejection_counts": dict(rejection_counts),
        "total_windows_generated": len(all_windows),
        "excluded_unknown_windows": unknown_wins_total,
        "retained_supervised_windows": len(supervised_windows),
        "class_distribution_total": {
            "attack_windows": attack_wins_total,
            "normal_windows": normal_wins_total,
            "unknown_windows": unknown_wins_total,
            "attack_percentage": round((attack_wins_total / max(len(supervised_windows), 1)) * 100, 2),
        },
        "split_policy": split_meta,
        "split_counts": {
            "train": len(train_wins),
            "val": len(val_wins),
            "test": len(test_wins),
        },
        "split_distributions": {
            "train": get_class_dist(train_wins),
            "val": get_class_dist(val_wins),
            "test": get_class_dist(test_wins),
        },
        "feature_columns": FEATURE_COLUMNS,
    }

    manifest_path = out_dir / "dataset_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[+] Saved manifest to {manifest_path}")

    # 9. Automated Data Quality Verification
    print("[+] Running automated output validation check...")
    valid, errors = validate_dataset_outputs(out_dir, manifest_path)
    if not valid:
        print("[-] DATA VALIDATION FAILED:")
        for err in errors:
            print(f"    - {err}")
        sys.exit(1)
    else:
        print("[+] Output validation check passed: CSVs and manifest are 100% consistent!")

    print("\n[+] Data pipeline completed successfully for Member 2 handoff!\n")


if __name__ == "__main__":
    main()
