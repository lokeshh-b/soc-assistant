"""src/classifier.py

Machine Learning classification and SHAP explainability module.
Responsible for loading trained ML models, performing threat/anomaly inference
on log feature windows, and computing local/global SHAP feature attributions.

Architectural Note:
Kept strictly independent from GenAI agents and dashboard UI presentation logic.
"""
