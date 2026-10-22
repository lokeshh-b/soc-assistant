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
