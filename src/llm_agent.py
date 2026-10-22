"""src/llm_agent.py

Generative AI triage assistance module.
Interfaces with LLM providers to synthesize security context, interpret SHAP feature
attributions, generate human-readable alert narratives, and recommend remediation steps.

Architectural Note:
Consumes structured outputs from src/classifier.py without altering model inference logic.
"""
