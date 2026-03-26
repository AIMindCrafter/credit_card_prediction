"""
health.py
~~~~~~~~~
Health and readiness check endpoints.
"""

from __future__ import annotations

import os
from datetime import datetime

from fastapi import APIRouter, Request

from app.schemas import HealthResponse, ReadinessResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Returns 200 OK if the service is alive.",
)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=os.getenv("APP_VERSION", "1.0.0"),
        timestamp=datetime.utcnow(),
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description="Returns 200 if the service is ready to serve traffic (model loaded).",
)
async def readiness(request: Request) -> ReadinessResponse:
    model_loaded = getattr(request.app.state, "model", None) is not None

    # Try to reach MLflow tracking server
    mlflow_reachable = False
    try:
        import httpx
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{tracking_uri}/health")
            mlflow_reachable = response.status_code == 200
    except Exception:
        mlflow_reachable = False

    return ReadinessResponse(
        status="ready" if model_loaded else "not_ready",
        model_loaded=model_loaded,
        mlflow_reachable=mlflow_reachable,
        timestamp=datetime.utcnow(),
    )
