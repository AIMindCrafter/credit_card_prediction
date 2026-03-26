"""
preprocessing.py
~~~~~~~~~~~~~~~~
Reusable data loading, cleaning, and feature-engineering helpers
for the Credit Card Fraud Detection pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
TARGET_COLUMN = "Class"
SCALE_COLUMNS = ["Amount", "Time"]


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_data(filepath: str | Path) -> pd.DataFrame:
    """Load raw CSV dataset from *filepath* and return a DataFrame."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    logger.info("Loading dataset from %s", filepath)
    df = pd.read_csv(filepath)
    logger.info("Loaded %d rows × %d columns", *df.shape)
    return df


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def remove_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with any NaN values and log the count removed."""
    before = len(df)
    df = df.dropna()
    removed = before - len(df)
    if removed:
        logger.info("Removed %d rows with NaN values", removed)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        logger.info("Removed %d duplicate rows", removed)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Assert that all expected columns are present."""
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    logger.info("Schema validation passed.")


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
def scale_features(
    df: pd.DataFrame,
    scaler: RobustScaler | None = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, RobustScaler]:
    """
    Apply RobustScaler to Amount and Time columns.

    Parameters
    ----------
    df      : Input DataFrame
    scaler  : Existing fitted scaler (pass during inference)
    fit     : Whether to fit the scaler (True for training, False for inference)

    Returns
    -------
    df      : DataFrame with scaled columns
    scaler  : Fitted RobustScaler instance
    """
    df = df.copy()
    if scaler is None:
        scaler = RobustScaler()

    if fit:
        df[SCALE_COLUMNS] = scaler.fit_transform(df[SCALE_COLUMNS])
        logger.info("Scaler fitted and applied to %s", SCALE_COLUMNS)
    else:
        df[SCALE_COLUMNS] = scaler.transform(df[SCALE_COLUMNS])
        logger.info("Scaler applied to %s (inference mode)", SCALE_COLUMNS)

    return df, scaler


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------
def preprocess(
    df: pd.DataFrame,
    scaler: RobustScaler | None = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, RobustScaler]:
    """
    Full preprocessing pipeline: validate → clean → scale.

    Returns
    -------
    X       : Feature DataFrame
    y       : Target Series
    scaler  : Fitted scaler
    """
    validate_schema(df)
    df = remove_nulls(df)
    df = remove_duplicates(df)

    # Ensure Class is integer
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    df, scaler = scale_features(df, scaler=scaler, fit=fit)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    fraud_pct = y.mean() * 100
    logger.info(
        "Preprocessing complete — %d samples, %.4f%% fraud", len(X), fraud_pct
    )
    return X, y, scaler


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------
def preprocess_single(record: dict, scaler: RobustScaler) -> np.ndarray:
    """
    Transform a single transaction dict into a feature array for prediction.

    Parameters
    ----------
    record  : dict with keys matching FEATURE_COLUMNS
    scaler  : Fitted RobustScaler

    Returns
    -------
    np.ndarray of shape (1, 30)
    """
    df = pd.DataFrame([record])
    # Ensure columns are in correct order
    df = df[FEATURE_COLUMNS]
    df, _ = scale_features(df, scaler=scaler, fit=False)
    return df.values


def preprocess_batch(records: list[dict], scaler: RobustScaler) -> np.ndarray:
    """Transform a list of transaction dicts into a feature matrix."""
    df = pd.DataFrame(records)[FEATURE_COLUMNS]
    df, _ = scale_features(df, scaler=scaler, fit=False)
    return df.values
