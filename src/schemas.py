"""src/schemas.py

Pydantic data models and schemas defining data contracts across the pipeline.
Includes schemas for raw log records, normalized events, alert classifications,
SHAP feature attributions, and GenAI incident triage summaries.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class FeatureContribution(BaseModel):
    """Represents a single feature's local contribution to a prediction via SHAP.

    Attributes:
        feature_name: Name of the behavioral feature (e.g. 'failed_auth_count').
        feature_value: Observed numeric value of the feature in the window.
        shap_value: SHAP attribution value relative to the suspicious class.
                    Positive values push toward suspicious; negative toward benign.
    """
    feature_name: str = Field(..., description="Name of the behavioral feature")
    feature_value: float = Field(..., description="Observed numeric value of the feature")
    shap_value: float = Field(..., description="SHAP attribution value relative to the suspicious class")


class ClassifierPrediction(BaseModel):
    """Standardized output schema produced by the ML Classifier module.

    Coordinated contract shared with Member 3 (GenAI Agent) and Member 4 (Dashboard & State Store).

    Attributes:
        model_name: Identifier of the trained model producing the prediction.
        risk_score: Continuous probability of the suspicious class in [0.0, 1.0].
        predicted_class: Discrete binary decision (0 = Benign, 1 = Suspicious).
        threshold_applied: Decision threshold used to map risk_score to predicted_class.
        top_contributing_features: Ordered list of top SHAP feature contributions.
        inference_latency_ms: Milliseconds elapsed during feature parsing and inference.
    """
    model_name: str = Field(..., description="Name/version of the classification model")
    risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Suspicious-class probability score between 0.0 and 1.0",
    )
    predicted_class: int = Field(
        ...,
        description="Binary classification label: 0 for Benign, 1 for Suspicious",
    )
    threshold_applied: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Classification decision threshold used to determine predicted_class",
    )
    top_contributing_features: List[FeatureContribution] = Field(
        default_factory=list,
        description="Top local feature attributions explaining the risk score",
    )
    explanation_status: Optional[str] = Field(
        default="success",
        description="Status of SHAP feature explanation: 'success', 'unavailable', 'error', or 'bypassed'",
    )
    inference_latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Inference execution latency in milliseconds",
    )
