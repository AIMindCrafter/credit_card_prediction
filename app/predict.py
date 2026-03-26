"""
predict.py
~~~~~~~~~~
Prediction router — single & batch transaction fraud detection endpoints.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import joblib
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.schemas import (
    BatchPredictionItem,
    BatchPredictionResponse,
    BatchRequest,
    PredictionResponse,
    TransactionRequest,
)
from src.preprocessing import FEATURE_COLUMNS

router = APIRouter(prefix="/predict", tags=["Prediction"])


def get_model(request: Request) -> Any:
    """Dependency: retrieve the loaded model from app state."""
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Service is starting up.",
        )
    return model


def get_scaler(request: Request) -> Any:
    """Dependency: retrieve the loaded scaler from app state."""
    return getattr(request.app.state, "scaler", None)


def get_model_version(request: Request) -> str | None:
    return getattr(request.app.state, "model_version", None)


def _predict_proba(model, scaler, record: dict) -> float:
    """
    Run single-record inference and return fraud probability.

    Handles three cases:
    1. sklearn Pipeline  — has internal steps (scaler included) — call predict_proba on raw df
    2. Plain classifier + external scaler — scale then predict
    3. Plain classifier + no scaler — predict on raw features
    """
    import pandas as pd
    from sklearn.pipeline import Pipeline

    df = pd.DataFrame([record])[FEATURE_COLUMNS]

    is_pipeline = isinstance(model, Pipeline) or hasattr(model, "steps")

    if is_pipeline:
        # Pipeline handles its own preprocessing
        prob = float(model.predict_proba(df.values)[0, 1])
    elif scaler is not None:
        scale_cols = ["Amount", "Time"]
        df[scale_cols] = scaler.transform(df[scale_cols])
        prob = float(model.predict_proba(df.values)[0, 1])
    else:
        prob = float(model.predict_proba(df.values)[0, 1])

    return prob


def _confidence(prob: float) -> str:
    if prob >= 0.8 or prob <= 0.2:
        return "HIGH"
    elif prob >= 0.6 or prob <= 0.4:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Single Prediction
# ---------------------------------------------------------------------------
@router.post(
    "",
    response_model=PredictionResponse,
    summary="Predict fraud for a single transaction",
    response_description="Fraud prediction result with probability and confidence",
)
async def predict_single(
    transaction: TransactionRequest,
    request: Request,
    model=Depends(get_model),
    scaler=Depends(get_scaler),
    model_version=Depends(get_model_version),
) -> PredictionResponse:
    """
    Run real-time fraud detection on a **single** credit card transaction.

    - **Time**: seconds elapsed since first transaction
    - **V1–V28**: PCA-anonymized features
    - **Amount**: transaction amount in USD

    Returns fraud probability, binary label, and confidence tier.
    """
    t0 = time.perf_counter()
    record = transaction.model_dump()

    prob = _predict_proba(model, scaler, record)
    is_fraud = prob >= 0.5
    latency_ms = (time.perf_counter() - t0) * 1000

    return PredictionResponse(
        request_id=str(uuid.uuid4()),
        is_fraud=is_fraud,
        fraud_probability=round(prob, 6),
        confidence=_confidence(prob),
        label="FRAUDULENT" if is_fraud else "LEGITIMATE",
        latency_ms=round(latency_ms, 2),
        model_version=model_version,
    )


# ---------------------------------------------------------------------------
# Batch Prediction
# ---------------------------------------------------------------------------
@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    summary="Predict fraud for multiple transactions",
    response_description="Batch fraud prediction results",
)
async def predict_batch(
    batch: BatchRequest,
    request: Request,
    model=Depends(get_model),
    scaler=Depends(get_scaler),
    model_version=Depends(get_model_version),
) -> BatchPredictionResponse:
    """
    Run fraud detection on a **batch** of up to 1,000 transactions.

    Returns individual predictions plus aggregate statistics.
    """
    t0 = time.perf_counter()

    import pandas as pd
    from sklearn.pipeline import Pipeline
    records = [t.model_dump() for t in batch.transactions]
    df = pd.DataFrame(records)[FEATURE_COLUMNS]

    is_pipeline = isinstance(model, Pipeline) or hasattr(model, "steps")

    if is_pipeline:
        probs = model.predict_proba(df.values)[:, 1]
    elif scaler is not None:
        scale_cols = ["Amount", "Time"]
        df[scale_cols] = scaler.transform(df[scale_cols])
        probs = model.predict_proba(df.values)[:, 1]
    else:
        probs = model.predict_proba(df.values)[:, 1]

    latency_ms = (time.perf_counter() - t0) * 1000

    predictions = []
    for i, prob in enumerate(probs):
        is_fraud = bool(prob >= 0.5)
        predictions.append(BatchPredictionItem(
            index=i,
            is_fraud=is_fraud,
            fraud_probability=round(float(prob), 6),
            label="FRAUDULENT" if is_fraud else "LEGITIMATE",
            confidence=_confidence(float(prob)),
        ))

    fraud_count = sum(1 for p in predictions if p.is_fraud)

    return BatchPredictionResponse(
        total=len(predictions),
        fraud_count=fraud_count,
        fraud_rate_pct=round(fraud_count / len(predictions) * 100, 4),
        predictions=predictions,
        latency_ms=round(latency_ms, 2),
        model_version=model_version,
    )
