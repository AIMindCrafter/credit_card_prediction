"""
schemas.py
~~~~~~~~~~
Pydantic request/response models for the Fraud Detection API.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------
class TransactionRequest(BaseModel):
    """Single transaction for fraud prediction."""

    # Time (seconds elapsed since first transaction in dataset)
    Time: float = Field(..., description="Seconds elapsed since first transaction", example=0.0)

    # PCA-transformed features (V1–V28)
    V1: float = Field(..., example=-1.3598071336738)
    V2: float = Field(..., example=-0.0727811733098497)
    V3: float = Field(..., example=2.53634673796914)
    V4: float = Field(..., example=1.37815522427443)
    V5: float = Field(..., example=-0.338320769942518)
    V6: float = Field(..., example=0.462387777762292)
    V7: float = Field(..., example=0.239598554061257)
    V8: float = Field(..., example=0.0986979012610507)
    V9: float = Field(..., example=0.363786969611213)
    V10: float = Field(..., example=0.0907941719789316)
    V11: float = Field(..., example=-0.551599533260813)
    V12: float = Field(..., example=-0.617800855762348)
    V13: float = Field(..., example=-0.991389847235408)
    V14: float = Field(..., example=-0.311169353699879)
    V15: float = Field(..., example=1.46817697209427)
    V16: float = Field(..., example=-0.470400525259478)
    V17: float = Field(..., example=0.207971241929242)
    V18: float = Field(..., example=0.0257905801985591)
    V19: float = Field(..., example=0.403992960255733)
    V20: float = Field(..., example=0.251412098239705)
    V21: float = Field(..., example=-0.018306777944153)
    V22: float = Field(..., example=0.277837575558899)
    V23: float = Field(..., example=-0.110473910188767)
    V24: float = Field(..., example=0.0669280749146731)
    V25: float = Field(..., example=0.128539358273528)
    V26: float = Field(..., example=-0.189114843888824)
    V27: float = Field(..., example=0.133558376740387)
    V28: float = Field(..., example=-0.0210530534538215)

    # Transaction amount (unscaled)
    Amount: float = Field(..., ge=0.0, description="Transaction amount in USD", example=149.62)

    model_config = {"json_schema_extra": {"example": {
        "Time": 0.0, "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
        "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36,
        "V10": 0.09, "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31,
        "V15": 1.47, "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40,
        "V20": 0.25, "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
        "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02, "Amount": 149.62,
    }}}


class BatchRequest(BaseModel):
    """Batch of transactions for bulk fraud prediction."""
    transactions: List[TransactionRequest] = Field(
        ..., min_length=1, max_length=1000,
        description="List of transactions (max 1000)"
    )


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------
class PredictionResponse(BaseModel):
    """Response for a single transaction prediction."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    is_fraud: bool = Field(..., description="True if the transaction is predicted as fraudulent")
    fraud_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of fraud")
    confidence: str = Field(..., description="Confidence level: HIGH / MEDIUM / LOW")
    label: str = Field(..., description="Human-readable label: LEGITIMATE or FRAUDULENT")
    latency_ms: Optional[float] = Field(None, description="Inference latency in milliseconds")
    model_version: Optional[str] = Field(None, description="Model version used for prediction")

    @field_validator("confidence", mode="before")
    @classmethod
    def compute_confidence(cls, v, info):
        if hasattr(info, "data") and "fraud_probability" in info.data:
            p = info.data["fraud_probability"]
            if p >= 0.8 or p <= 0.2:
                return "HIGH"
            elif p >= 0.6 or p <= 0.4:
                return "MEDIUM"
        return v or "LOW"


class BatchPredictionItem(BaseModel):
    index: int
    is_fraud: bool
    fraud_probability: float
    label: str
    confidence: str


class BatchPredictionResponse(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    total: int
    fraud_count: int
    fraud_rate_pct: float
    predictions: List[BatchPredictionItem]
    latency_ms: Optional[float] = None
    model_version: Optional[str] = None


# ---------------------------------------------------------------------------
# Health / Status
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ReadinessResponse(BaseModel):
    status: str
    model_loaded: bool
    mlflow_reachable: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
