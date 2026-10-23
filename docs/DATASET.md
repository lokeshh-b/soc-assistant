# CAM-LDS Dataset & Member 1 Data Pipeline Specification

## 1. Dataset Provenance & Source Verification

- **Dataset Name:** Cyber Attack Manifestations Log Data Set (CAM-LDS)
- **Creators:** Max Landauer, Wolfgang Hotwagner, Thorina Boenke, Florian Skopik, Markus Wurzenberger
- **Institution:** Austrian Institute of Technology (AIT)
- **Official DOI / Zenodo Record:** [10.5281/zenodo.18861762](https://zenodo.org/records/18861762) (Version 2)
- **Official Repository:** [GitHub ait-aecid/attack-manifestations-interpretation](https://github.com/ait-aecid/attack-manifestations-interpretation)
- **Associated Publication:** *CAM-LDS: Cyber Attack Manifestations for Automatic Interpretation of System Logs and Security Alerts*, International Journal of Information Security, 25(5), 2026.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Physical Archive:** `data/raw/manifestations_filtered.zip` (203.87 MB compressed; 4.23 GB uncompressed, containing 17,242 multi-source Linux log files).
- **Ground Truth Metadata:** `data/raw/attack_times.csv` (157 recorded attack steps across scenarios, start/end timestamps, tactics, techniques).

---

## 2. Ingested Log Sources & Formats

The dataset mirrors an enterprise Linux environment (*AttackBed*) across workstations, web servers, video surveillance nodes, and firewalls:

| Log Type | Source Files | Extracted Attributes |
| :--- | :--- | :--- |
| **Linux Auditd** | `audit.log` | Timestamp, syscall ID, user (`uid`, `auid`, `euid`), process execution, root flag (`uid=0`). |
| **Authentication** | `auth.log` | Timestamp, SSH/Sudo events, user, auth outcome (`failed`, `accepted`), port, root attempt. |
| **Syslog / Daemons** | `syslog`, `messages` | Timestamp, host, daemon/process name, event payload, Drain template. |
| **Apache Web Server** | `access.log` | Timestamp, client IP, HTTP method, URL path, response status, user agent. |
| **Suricata / Wazuh IDS** | `eve.json`, `alerts.json` | Timestamp, signature description, source IP, destination port, alert severity. |

---

## 3. Behavioral 10-Second Windowing Convention

To capture temporal security dynamics without alert fatigue, raw logs are aggregated into discrete behavioral windows:
- **Window Duration:** Exactly `10.0` seconds (e.g., `[t, t + 10.0)`).
- **Stride:** `10.0` seconds (tumbling window by default) or configurable sliding stride (e.g., `5.0` seconds).
- **Grouping:** Grouped per sequence (`group_by="sequence"`), host (`group_by="host"`), or globally (`group_by="all"`).
- **Boundary Semantics:** Lower boundary is inclusive, upper boundary is exclusive:
  $$\text{event} \in \text{window} \iff w_{\text{start}} \le \text{event.timestamp} < w_{\text{end}}$$
- **Ground-Truth Labeling & Contract:**
  - `LABEL_ATTACK = 1`: Window temporally overlaps with a verified ground-truth attack interval in `attack_times.csv` within the matching scenario:
    $$\max(w_{\text{start}}, a_{\text{start}}) < \min(w_{\text{end}}, a_{\text{end}}) \quad \text{and} \quad \text{scenario}(w) = \text{scenario}(a)$$
  - `LABEL_NORMAL = 0`: Window is within an annotated scenario but outside any recorded attack interval.
  - `LABEL_UNKNOWN = -1`: Window belongs to a sequence without ground-truth attack annotations in `attack_times.csv`. To protect classifier integrity, unknown windows are excluded from supervised train/val/test splits rather than being falsely labeled normal.

---

## 4. The Official 13-Feature Contract for Member 2

The feature vector produced by Member 1 strictly follows the 13-column schema defined in `src/schemas.py::FEATURE_COLUMNS`:

| # | Feature Name | Type | Description |
| :-: | :--- | :--- | :--- |
| 1 | `total_event_count` | `float` | Total number of security events in the 10-second window. |
| 2 | `events_per_second_max` | `float` | Peak event arrival rate in any 1-second sub-interval within the window. |
| 3 | `byte_length_mean` | `float` | Average character length of log messages in the window. |
| 4 | `failed_auth_count` | `float` | Count of failed authentication / rejection events. |
| 5 | `accepted_auth_count` | `float` | Count of successful authentication / login events. |
| 6 | `failed_to_success_ratio` | `float` | Ratio: If `accepted_auth_count > 0`, $\text{failed\_auth\_count} / \text{accepted\_auth\_count}$; if `accepted_auth_count == 0`, equals `failed_auth_count`. Avoids numerical distortions from arbitrary epsilon additions. |
| 7 | `distinct_usernames_count` | `float` | Count of distinct usernames observed in the window. |
| 8 | `root_or_admin_attempt_flag` | `float` | Binary flag (`1.0` or `0.0`) if root, admin, or UID 0 activity occurred. |
| 9 | `distinct_dest_ports_count` | `float` | Count of unique destination network ports observed. |
| 10 | `ephemeral_port_ratio` | `float` | Fraction of destination ports in ephemeral range ($\ge 1024$). |
| 11 | `privileged_port_ratio` | `float` | Fraction of destination ports in privileged range ($< 1024$). |
| 12 | `unique_drain_templates_count` | `float` | Number of distinct Drain log templates mined in the window. |
| 13 | `rare_template_ratio` | `float` | Fraction of events in window belonging to rare templates (frequency $\le 2$ in train fit). |

---

## 5. Anti-Leakage & Reproducibility Guarantees

1. **Grouped Sequence & Scenario Splitting:**
   - Windows from the same attack sequence or scenario are kept together in the same split partition. This strictly prevents temporal autocorrelation leakage and ensures independent evaluation.
   - The splitting algorithm (`split_windows_grouped`) balances attack and normal representation across train (70%), validation (15%), and test (15%) partitions deterministically.
2. **Template Vocabulary Isolation:**
   - The `DrainTemplateMiner` frequency tables and rare-template vocabulary are fitted strictly on training events (`miner.freeze()`).
   - Validation and test windows are transformed using the frozen training vocabulary without updating internal template counts.
3. **Storage Separation:**
   - Machine learning feature matrices (`train_features.csv`, `val_features.csv`, `test_features.csv`) contain only `window_id` and the 13 feature columns.
   - Ground truth labels (`is_attack`), MITRE tactics, MITRE techniques, and scenario names are stored in companion metadata files (`train_metadata.csv`, `val_metadata.csv`, `test_metadata.csv`).
4. **Strict Timestamp & Record Validation:**
   - BSD syslog dates default to reference year 2025 (matching CAM-LDS ground truth), preventing wall-clock year drift.
   - Apache error logs (`[%a %b %d %H:%M:%S.%f %Y]`) are natively parsed.
   - All parsed timestamps are validated against plausible bounds (`1735689600.0` to `1767225600.0`). Unparseable records are quarantined with rejection diagnostics rather than silently coerced to `time.time()`.

---

## 6. Dataset Regeneration Instructions

To deterministically regenerate the processed feature matrices, metadata, and manifest from raw files, run:

```bash
python3 scripts/prepare_data.py \
  --data-dir data/raw \
  --out-dir data/processed \
  --window-sec 10.0 \
  --stride-sec 10.0 \
  --group-by sequence \
  --max-records 20000 \
  --max-records-per-sequence 300 \
  --seed 42
```

### Command Line Arguments:
- `--data-dir`: Directory containing `manifestations_filtered.zip` and `attack_times.csv` (default: `data/raw`).
- `--out-dir`: Output directory for processed feature CSVs, metadata CSVs, and `dataset_manifest.json` (default: `data/processed`).
- `--window-sec`: Window aggregation duration in seconds (default: `10.0`).
- `--stride-sec`: Window stride in seconds (default: `10.0` for tumbling windows).
- `--group-by`: Aggregation grouping key; set to `sequence` to ensure windows are aggregated within isolated sequence boundaries.
- `--max-records`: Total valid normalized events to ingest across all sequences (default: `20000`).
- `--max-records-per-sequence`: Maximum raw lines read per sequence file (default: `300`) to guarantee balanced multi-scenario representation.
- `--seed`: Deterministic random seed for group-based train/val/test splitting (default: `42`).

To inspect and validate the generated dataset against the manifest and feature contract, run:
```bash
python3 scripts/inspect_dataset.py --data-dir data/processed
```

---

## 7. Class Distribution & Test-Set Evaluation Limitations

### Supervised Split Summary:
- **Train:** 126 windows (102 attack [80.95%], 24 normal [19.05%]) across 60 sequence groups.
- **Validation:** 45 windows (42 attack [93.33%], 3 normal [6.67%]) across 13 sequence groups.
- **Test:** 44 windows (43 attack [97.73%], 1 normal [2.27%]) across 13 sequence groups.
- **Quarantined Unknowns:** 135 windows from unannotated sequences excluded from supervised datasets.

### Critical Evaluation Limitation for Downstream Modeling (Member 2):
The test set contains **only 1 normal window** out of 44 total windows.
- **Consequence for False-Positive Rate (FPR) & Specificity:**
  - With a single negative sample ($TN + FP = 1$), Specificity ($\frac{TN}{TN + FP}$) can only ever evaluate to $1.0$ (if normal is predicted correctly) or $0.0$ (if falsely flagged as attack).
  - False Positive Rate ($\frac{FP}{TN + FP}$) can only ever evaluate to $0.0$ or $1.0$.
  - It is statistically impossible to establish confidence intervals, measure low-probability false alarm rates, or plot a meaningful ROC curve from the test set alone.
- **Downstream Modeling Guidance:**
  - Member 2 must evaluate **detection recall and precision** on the test set.
  - Evaluation of **specificity, false-positive rate, and ROC AUC** must be conducted via **k-fold group cross-validation** or pooled evaluation across splits, rather than relying on the single test normal sample as an indicator of production false alarms.
