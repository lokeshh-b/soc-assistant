# soc-assistant: Explainable SOC Assistant

ML and GenAI-assisted SOC alert triage using behavioral log windows, SHAP explanations, and Generative AI.

## Project Overview

Modern Security Operations Centers (SOCs) face alert fatigue and complex multi-source log streams. This project implements an **Explainable SOC Assistant** designed to:
- Ingest and parse behavioral security log windows.
- Classify anomalous activities and score potential threats using machine learning.
- Provide local and global interpretability using **SHAP** (SHapley Additive exPlanations) so analysts understand *why* an alert was flagged.
- Leverage **Generative AI** agents to translate technical feature attributions into natural-language incident summaries and actionable triage recommendations.
- Present a real-time interactive investigation dashboard for tier-1/tier-2 SOC analysts.

## Architecture & Modular Design

```text
Log Sources (CAM-LDS / Syslog)
            │
            ▼
   [ Streamer & Parser ]
            │
            ▼
    [ ML Classifier ] ──► [ SHAP Explainer ]
            │                     │
            └──────────┬──────────┘
                       ▼
              [ GenAI Agent ]
                       │
                       ▼
         [ State Store & DB ]
                       │
                       ▼
            [ Streamlit Dashboard ]
```

- **Separation of Concerns**: The machine learning classifier and SHAP components are strictly separated from the GenAI agent and presentation layers, ensuring modularity and reproducibility.

## Project Structure

```text
explainable-soc-assistant/
├── README.md               # Project documentation and setup guide
├── requirements.txt        # Initial core project dependencies
├── .gitignore              # Git ignore rules for data, models, and caches
├── .env.example            # Template for environment variables (no secrets)
├── app.py                  # Main application entrypoint
├── pipeline_worker.py      # Background stream consumption and triage worker
├── src/                    # Application source code
│   ├── __init__.py
│   ├── schemas.py          # Pydantic data models for logs and alerts
│   ├── parser.py           # Log normalization and feature extraction
│   ├── streamer.py         # Log stream simulator and windowing
│   ├── classifier.py       # ML anomaly detection and alert scoring
│   ├── llm_agent.py        # GenAI contextual explanations and triage summaries
│   ├── state_store.py      # Persistence and alert lifecycle tracking
│   └── dashboard_app.py    # Streamlit analyst UI
├── scripts/                # Utility and ML workflow scripts
│   ├── inspect_dataset.py  # Dataset exploration and stats
│   ├── prepare_data.py     # Preprocessing and feature engineering
│   ├── train_model.py      # Model training and artifact serialization
│   └── evaluate_model.py   # Metrics evaluation and SHAP analysis
├── data/                   # Data directories (gitignored)
│   ├── raw/                # Raw log files (e.g., CAM-LDS)
│   └── processed/          # Processed feature matrices
├── models/                 # Saved model checkpoints (gitignored)
├── artifacts/              # Local SQLite databases, logs, reports (gitignored)
└── tests/                  # Unit and integration test suites
    ├── test_parser.py
    ├── test_streamer.py
    ├── test_classifier.py
    ├── test_state_store.py
    └── test_integration.py
```

## Setup Instructions

### 1. Prerequisites
- Python 3.10+ installed
- Virtual environment tool (`venv` or `conda`)

### 2. Environment Setup

Clone the repository and navigate into the workspace:

```bash
cd soc-assistant
```

Create and activate a virtual environment:

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Install the initial core dependencies:

```bash
pip install -r requirements.txt
```

### 4. Configuration

Copy the example environment configuration:

```bash
cp .env.example .env
```

Edit `.env` to configure your API keys (e.g. Gemini API key for GenAI triage) and local paths.

### 5. Running Tests

Run the test suite using `pytest`:

```bash
pytest
```

## GenAI Incident Triage Module (`src/llm_agent.py`)

The GenAI Incident Triage module synthesizes raw log evidence, upstream ML anomaly predictions, risk scores, and local SHAP feature attributions into structured, explainable security incident reports for SOC analysts.

### Data Contract & Pydantic Schema

Every triage invocation returns an `IncidentAnalysisReport` (defined in `src/schemas.py`) strictly validated against Pydantic v2:

| Field | Type | Description |
| :--- | :--- | :--- |
| `evidence_status` | `Literal["Sufficient", "Insufficient Data"]` | Indicates whether corroborating log evidence confirms an active incident. |
| `incident_summary` | `str` | Grounded, factual narrative explaining the event and SHAP feature contributions. |
| `threat_category` | `ThreatCategory` | Restricted to: `Brute Force`, `Privilege Escalation`, `Lateral Movement`, `Command and Control`, `Exfiltration`, `Execution`, `Unknown / Insufficient Data`. |
| `mitre_attack_id` | `Optional[str]` | Associated MITRE ATT&CK technique ID (e.g. `T1110`, `T1068`) if evidence is sufficient; strictly `None` when insufficient. |
| `recommended_actions` | `List[str]` | Advisory remediation and containment steps (strictly `[]` when evidence is insufficient). |

### Security Guarantees & Triage Principles

1. **Prediction Decoupling**: Suspicious ML predictions or high anomaly scores alone are **never** treated as proof of an attack. Corroborating evidence in raw logs is mandatory.
2. **Untrusted Log Boundary & Prompt Injection Mitigation**: All log contents are treated as untrusted and quarantined inside `<untrusted_log_evidence>` XML boundaries. Adversarial instructions embedded in logs (e.g. prompt injection, command overrides) and delimiter breakout tags are neutralized and escaped prior to LLM evaluation. Note: These measures serve as defense-in-depth mitigations rather than an impenetrable security boundary; raw logs remain untrusted.
3. **Insufficient Evidence Invariants**: If log evidence is missing, blank, or insufficient to corroborate an attack:
   - `evidence_status` is `"Insufficient Data"`
   - `threat_category` is `"Unknown / Insufficient Data"`
   - `mitre_attack_id` is `None`
   - `recommended_actions` is `[]` (empty list)
4. **No Action Execution**: Recommended actions are strictly advisory guidance for human SOC tier-1/tier-2 analysts. The module contains zero execution logic and never executes system commands.
5. **No Invented Telemetry**: The module never fabricates hostnames, IP addresses, user credentials, scores, or SHAP contributions not present in the input telemetry.

### Environment Configuration & Offline Mock Mode

Configure the following variables in `.env`:

```bash
# Google Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here
LLM_MODEL=gemini-1.5-flash
LLM_TEMPERATURE=0.2

# Deterministic Offline Mock Mode (Zero network calls, offline testing)
MOCK_LLM=true
```

#### Deterministic Offline Mock Mode
When running in testing or CI environments without external API credentials or internet access:
- Set `MOCK_LLM=true` in `.env`, or
- Pass `mock_mode=True` directly to `analyze_incident(...)`.

In mock mode, the module performs deterministic, rule-based classification over sanitized evidence and SHAP values with zero external HTTP requests.

### Testing Instructions

Run all project tests:
```bash
pytest
```

Run the GenAI triage module test suite with verbose output:
```bash
pytest tests/test_llm_agent.py -v
```

The test suite covers:
- Valid schema instantiation and report generation
- Strict taxonomy enforcement and rejection of unapproved threat categories
- Missing required fields rejection and extra fields prohibition (`extra="forbid"`)
- Insufficient evidence handling and ML prediction decoupling
- Malformed JSON and model response recovery
- Prompt injection neutralization and delimiter enforcement
- Deterministic offline mock mode across all attack classifications
- Safe handling of missing credentials and placeholder keys
- Upstream HTTP errors (e.g. 500) and network timeout handling
- Advisory-only action verification (zero command execution)

### Practical Offline Demo Script

Run the self-contained offline demonstration script (requires no API keys or network calls):

```bash
python scripts/demo_genai_triage.py
```

This runs three synthetic scenarios:
1. Active brute force security incident with SHAP factor explanations.
2. High ML anomaly score without corroborating logs (demonstrating anti-hallucination and evidence-decoupling safety).
3. Pipeline alert dictionary ingestion and lossless round-trip serialization.

### Team Integration Contract & Usage Example

Teammates implementing the pipeline worker, classifier, and state store can call `analyze_incident()` directly using an alert dictionary or keyword arguments:

```python
from src.llm_agent import analyze_incident
from src.schemas import IncidentAnalysisReport

# Example: Unified alert payload passed from the pipeline worker
alert_payload = {
    "classifier_prediction": "ANOMALY",  # str, int, or dict from src/classifier.py
    "risk_score": 0.92,                   # float or numpy.float64
    "threshold": 0.50,                    # operating decision threshold
    "shap_contributions": {               # dict or pandas.Series of local SHAP values
        "failed_auth_count": 0.88,
        "source_ip_entropy": 0.65
    },
    "feature_values": {                   # raw or aggregated feature vector
        "failed_logins_last_5m": 42
    },
    "log_evidence": (                     # str, list of event dicts, or pandas.DataFrame
        "Failed password for invalid user admin from 198.51.100.22 port 52331 ssh2\n"
        "Maximum authentication attempts exceeded for admin from 198.51.100.22"
    ),
}

# Execute triage (mock_mode=True for offline/test environments; None reads MOCK_LLM env var)
report: IncidentAnalysisReport = analyze_incident(alert_payload, mock_mode=True)

# 1. Access validated fields
print(report.evidence_status)     # "Sufficient" or "Insufficient Data"
print(report.threat_category)     # e.g. "Brute Force" (strictly from allowed taxonomy)
print(report.mitre_attack_id)     # e.g. "T1110" (None when Insufficient Data)
print(report.incident_summary)    # Grounded narrative referencing SHAP factors
print(report.recommended_actions) # Advisory steps for Tier-1/2 analysts

# 2. Serialize for state store / database persistence
state_dict = report.model_dump()           # Python dictionary
json_payload = report.model_dump_json()    # JSON string

# 3. Re-hydrate from database record (e.g. SQLite row in src/state_store.py)
restored_report = IncidentAnalysisReport.model_validate(state_dict)
```

### Team Integration Status

| Component | File | Implementation Status | Integration Contract / Role |
| :--- | :--- | :--- | :--- |
| **GenAI Incident Triage** | `src/llm_agent.py` | **Implemented** (Member 3) | Generates validated `IncidentAnalysisReport`. Tested offline & live, unpacks alert dicts and Pydantic containers. |
| **Data Contracts & Schemas** | `src/schemas.py` | **Implemented** (Member 3) | Defines `IncidentAnalysisReport`, `IncidentTriageInput`, and `ThreatCategory`. |
| **Offline Demo Script** | `scripts/demo_genai_triage.py` | **Implemented** (Member 3) | Standalone mock demonstration of triage scenarios and serialization. |
| **ML Threat Classifier** | `src/classifier.py` | *Pending Implementation* | Upstream provider of predictions, risk scores, and SHAP explanation vectors. |
| **Log Parser & Normalizer** | `src/parser.py` | *Pending Implementation* | Upstream provider of normalized event windows for `log_evidence`. |
| **Log Streamer Simulator** | `src/streamer.py` | *Pending Implementation* | Emulates log ingestion and sliding windows for the pipeline. |
| **Pipeline Worker Loop** | `pipeline_worker.py` | *Pending Implementation* | Orchestrates Streamer -> Parser -> Classifier -> GenAI Triage -> State Store. |
| **State Persistence Store** | `src/state_store.py` | *Pending Implementation* | Persists serialized reports (`report.model_dump()`) to SQLite (`STATE_DATABASE_URL`). |
| **Analyst Dashboard UI** | `src/dashboard_app.py` | *Pending Implementation* | Streamlit UI displaying alerts, SHAP plots, and GenAI triage narratives. |

### Scope and Limitations
- **API Availability**: Live inference requires outbound network connectivity to the Google Generative Language API and a valid Gemini API key. In offline/CI environments, use `mock_mode=True` or `MOCK_LLM=true`.
- **Taxonomy Scope**: Threat classification is restricted to the 6 predefined MITRE ATT&CK categories. Unrecognized activity safely defaults to `Unknown / Insufficient Data`.
- **Prompt Injection Boundary**: Delimiter isolation and regex filtering mitigate adversarial manipulation from host logs, but prompt constraints on LLMs are defense-in-depth mitigations rather than an impenetrable cryptographic boundary. Recommendations must remain strictly advisory.
- **Human-in-the-Loop**: All outputs are decision-support recommendations intended for human analyst validation and authorization.
