"""
mlflow_utils.py
~~~~~~~~~~~~~~~
MLflow experiment tracking and model registry utilities.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
DEFAULT_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "credit-card-fraud-detection")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "fraud-detection")


def setup_mlflow(experiment_name: str = DEFAULT_EXPERIMENT) -> str:
    """Configure MLflow tracking URI and return the experiment ID."""
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info("Created MLflow experiment '%s' (id=%s)", experiment_name, experiment_id)
    else:
        experiment_id = experiment.experiment_id
        logger.info("Using MLflow experiment '%s' (id=%s)", experiment_name, experiment_id)
    mlflow.set_experiment(experiment_name)
    return experiment_id


def log_params(params: dict[str, Any]) -> None:
    """Log a dictionary of hyperparameters."""
    mlflow.log_params(params)
    logger.info("Logged %d params to MLflow", len(params))


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log a dictionary of evaluation metrics."""
    mlflow.log_metrics(metrics, step=step)
    logger.info("Logged metrics: %s", metrics)


def log_model(
    model,
    artifact_path: str = "model",
    model_name: str = MODEL_NAME,
    register: bool = True,
) -> str:
    """
    Log sklearn model to MLflow and optionally register it.

    Returns
    -------
    run_id : str
    """
    run_id = mlflow.active_run().info.run_id
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=artifact_path,
        registered_model_name=model_name if register else None,
    )
    logger.info("Model logged (run_id=%s, registered=%s)", run_id, register)
    return run_id


def log_artifact(local_path: str | Path) -> None:
    """Upload a local file as an artifact to the active MLflow run."""
    mlflow.log_artifact(str(local_path))
    logger.info("Artifact uploaded: %s", local_path)


def get_latest_model_version(model_name: str = MODEL_NAME) -> str | None:
    """Return the latest version number of a registered model, or None."""
    client = MlflowClient(tracking_uri=TRACKING_URI)
    try:
        versions = client.get_latest_versions(model_name, stages=["None", "Staging", "Production"])
        if versions:
            return versions[0].version
    except Exception as exc:
        logger.warning("Could not fetch model versions: %s", exc)
    return None


def load_model_from_registry(
    model_name: str = MODEL_NAME,
    stage: str = "Production",
):
    """Load the latest model from the MLflow Model Registry."""
    model_uri = f"models:/{model_name}/{stage}"
    logger.info("Loading model from registry: %s", model_uri)
    return mlflow.sklearn.load_model(model_uri)


def transition_model_stage(
    model_name: str = MODEL_NAME,
    version: str | None = None,
    stage: str = "Production",
) -> None:
    """Transition a model version to a given stage."""
    client = MlflowClient(tracking_uri=TRACKING_URI)
    if version is None:
        version = get_latest_model_version(model_name)
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=True,
    )
    logger.info("Model '%s' v%s → stage '%s'", model_name, version, stage)
