"""
test_api.py
~~~~~~~~~~~
Tests for FastAPI endpoints: health, readiness, predict, batch.
"""

from __future__ import annotations

import pytest


# ── Health / Readiness ───────────────────────────────────────────────────────
class TestHealth:
    def test_health_returns_200(self, app_with_model):
        resp = app_with_model.get("/health")
        assert resp.status_code == 200

    def test_health_body(self, app_with_model):
        body = app_with_model.get("/health").json()
        assert body["status"] == "ok"
        assert "version" in body
        assert "timestamp" in body

    def test_ready_with_model(self, app_with_model):
        resp = app_with_model.get("/ready")
        body = resp.json()
        assert resp.status_code == 200
        assert body["model_loaded"] is True

    def test_ready_without_model(self, app_no_model):
        body = app_no_model.get("/ready").json()
        assert body["model_loaded"] is False
        assert body["status"] == "not_ready"


# ── Root ─────────────────────────────────────────────────────────────────────
class TestRoot:
    def test_root(self, app_with_model):
        resp = app_with_model.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert "docs" in body


# ── Single Prediction ────────────────────────────────────────────────────────
class TestPredict:
    def test_predict_returns_200(self, app_with_model, sample_transaction):
        resp = app_with_model.post("/predict", json=sample_transaction)
        assert resp.status_code == 200

    def test_predict_response_structure(self, app_with_model, sample_transaction):
        body = app_with_model.post("/predict", json=sample_transaction).json()
        assert "is_fraud" in body
        assert "fraud_probability" in body
        assert "confidence" in body
        assert "label" in body
        assert "latency_ms" in body
        assert "request_id" in body

    def test_predict_label_matches_is_fraud(self, app_with_model, sample_transaction):
        body = app_with_model.post("/predict", json=sample_transaction).json()
        if body["is_fraud"]:
            assert body["label"] == "FRAUDULENT"
        else:
            assert body["label"] == "LEGITIMATE"

    def test_predict_confidence_enum(self, app_with_model, sample_transaction):
        body = app_with_model.post("/predict", json=sample_transaction).json()
        assert body["confidence"] in ("HIGH", "MEDIUM", "LOW")

    def test_predict_probability_range(self, app_with_model, sample_transaction):
        body = app_with_model.post("/predict", json=sample_transaction).json()
        assert 0.0 <= body["fraud_probability"] <= 1.0

    def test_predict_without_model_returns_503(self, app_no_model, sample_transaction):
        resp = app_no_model.post("/predict", json=sample_transaction)
        assert resp.status_code == 503

    def test_predict_missing_field_returns_422(self, app_with_model):
        bad = {"Time": 0.0, "Amount": 100.0}  # Missing V1–V28
        resp = app_with_model.post("/predict", json=bad)
        assert resp.status_code == 422

    def test_predict_negative_amount_returns_422(self, app_with_model, sample_transaction):
        bad = {**sample_transaction, "Amount": -10.0}
        resp = app_with_model.post("/predict", json=bad)
        assert resp.status_code == 422


# ── Batch Prediction ─────────────────────────────────────────────────────────
class TestBatchPredict:
    def test_batch_returns_200(self, app_with_model, sample_transaction):
        payload = {"transactions": [sample_transaction, sample_transaction]}
        resp = app_with_model.post("/predict/batch", json=payload)
        assert resp.status_code == 200

    def test_batch_response_structure(self, app_with_model, sample_transaction):
        payload = {"transactions": [sample_transaction]}
        body = app_with_model.post("/predict/batch", json=payload).json()
        assert body["total"] == 1
        assert "fraud_count" in body
        assert "fraud_rate_pct" in body
        assert "predictions" in body
        assert len(body["predictions"]) == 1

    def test_batch_empty_returns_422(self, app_with_model):
        resp = app_with_model.post("/predict/batch", json={"transactions": []})
        assert resp.status_code == 422

    def test_batch_multiple_transactions(self, app_with_model, sample_transaction):
        payload = {"transactions": [sample_transaction] * 5}
        body = app_with_model.post("/predict/batch", json=payload).json()
        assert body["total"] == 5
        assert len(body["predictions"]) == 5
