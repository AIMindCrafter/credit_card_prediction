"""
settings.py
~~~~~~~~~~~
Centralised configuration via Pydantic BaseSettings.
All values can be overridden with environment variables.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Application ─────────────────────────────────────────────────────────
    APP_NAME: str = "Credit Card Fraud Detection API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = (
        "Enterprise-grade REST API for real-time credit card fraud detection "
        "powered by machine learning and MLflow model registry."
    )
    DEBUG: bool = False
    WORKERS: int = 4
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── Security ────────────────────────────────────────────────────────────
    API_KEY: str = "changeme-secret-api-key"
    ALLOWED_ORIGINS: List[str] = ["*"]
    ALLOWED_HOSTS: List[str] = ["*"]

    # ── Model ───────────────────────────────────────────────────────────────
    MODEL_PATH: str = "complete_fraud_pipeline.joblib"
    MODEL_LOAD_FROM_REGISTRY: bool = False
    MLFLOW_MODEL_NAME: str = "fraud-detection"
    MLFLOW_MODEL_STAGE: str = "Production"

    # ── MLflow ──────────────────────────────────────────────────────────────
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "credit-card-fraud-detection"

    # ── Logging ─────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # "json" | "text"

    # ── Prediction ──────────────────────────────────────────────────────────
    FRAUD_THRESHOLD: float = 0.5
    BATCH_MAX_SIZE: int = 1000

    # ── Database (for audit log, optional) ──────────────────────────────────
    DATABASE_URL: str = "sqlite:///./fraud_audit.db"

    # ── Paths ───────────────────────────────────────────────────────────────
    MODELS_DIR: Path = Path("models")
    REPORTS_DIR: Path = Path("reports")
    DATA_DIR: Path = Path("data")


settings = Settings()
