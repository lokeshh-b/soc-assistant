"""pipeline_worker.py

Background worker process for continuous log ingestion and alert triage pipeline.
Consumes streaming security events, coordinates parsing, model inference,
SHAP explanation generation, and state store persistence.
"""


def run_worker() -> None:
    """Execute the streaming triage worker loop."""
    pass


if __name__ == "__main__":
    run_worker()
