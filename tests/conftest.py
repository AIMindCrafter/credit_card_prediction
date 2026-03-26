"""
conftest.py
~~~~~~~~~~~
Shared pytest fixtures for the fraud detection test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Make sure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Sample Data ─────────────────────────────────────────────────────────────
SAMPLE_TRANSACTION = {
    "Time": 0.0,
    "V1": -1.3598071336738,
    "V2": -0.0727811733098497,
    "V3": 2.53634673796914,
    "V4": 1.37815522427443,
    "V5": -0.338320769942518,
    "V6": 0.462387777762292,
    "V7": 0.239598554061257,
    "V8": 0.0986979012610507,
    "V9": 0.363786969611213,
    "V10": 0.0907941719789316,
    "V11": -0.551599533260813,
    "V12": -0.617800855762348,
    "V13": -0.991389847235408,
    "V14": -0.311169353699879,
    "V15": 1.46817697209427,
    "V16": -0.470400525259478,
    "V17": 0.207971241929242,
    "V18": 0.0257905801985591,
    "V19": 0.403992960255733,
    "V20": 0.251412098239705,
    "V21": -0.018306777944153,
    "V22": 0.277837575558899,
    "V23": -0.110473910188767,
    "V24": 0.0669280749146731,
    "V25": 0.128539358273528,
    "V26": -0.189114843888824,
    "V27": 0.133558376740387,
    "V28": -0.0210530534538215,
    "Amount": 149.62,
}

FRAUD_TRANSACTION = {**SAMPLE_TRANSACTION, "Amount": 9999.99, "V1": -5.0, "V3": -4.0}


@pytest.fixture
def sample_transaction() -> dict:
    return SAMPLE_TRANSACTION.copy()


@pytest.fixture
def fraud_transaction() -> dict:
    return FRAUD_TRANSACTION.copy()


@pytest.fixture
def mock_model():
    """A mock sklearn classifier that returns deterministic outputs."""
    model = MagicMock()
    model.predict.return_value = np.array([0])
    model.predict_proba.return_value = np.array([[0.95, 0.05]])
    return model


@pytest.fixture
def mock_scaler():
    """A mock RobustScaler that returns input unchanged."""
    scaler = MagicMock()
    scaler.transform.side_effect = lambda x: x
    return scaler


@pytest.fixture
def app_with_model(mock_model, mock_scaler):
    """FastAPI TestClient with mocked model and scaler pre-loaded."""
    from app.main import app

    app.state.model = mock_model
    app.state.scaler = mock_scaler
    app.state.model_version = "test-v1"

    with TestClient(app) as client:
        yield client


@pytest.fixture
def app_no_model():
    """FastAPI TestClient with no model loaded (testing 503 responses)."""
    from app.main import app

    app.state.model = None
    app.state.scaler = None
    app.state.model_version = None

    with TestClient(app) as client:
        yield client
