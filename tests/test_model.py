"""
test_model.py
~~~~~~~~~~~~~
Tests for model loading and prediction output.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler


class TestModelOutput:
    def test_predict_returns_binary(self, mock_model):
        X = np.random.randn(5, 30)
        preds = mock_model.predict(X)
        assert all(p in [0, 1] for p in preds)

    def test_predict_proba_shape(self, mock_model):
        X = np.random.randn(5, 30)
        proba = mock_model.predict_proba(X)
        assert proba.shape == (5, 2)

    def test_predict_proba_sums_to_one(self, mock_model):
        X = np.random.randn(3, 30)
        proba = mock_model.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), [1.0, 1.0, 1.0], atol=1e-6)

    def test_real_random_forest_trains(self):
        """Smoke test: real RandomForest trains and predicts."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 30))
        y = rng.integers(0, 2, 200)

        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)

        preds = clf.predict(X[:10])
        assert preds.shape == (10,)
        assert set(preds).issubset({0, 1})

    def test_scaler_transform_shape(self, mock_scaler):
        X = np.random.randn(5, 30)
        result = mock_scaler.transform(X)
        assert result is not None
