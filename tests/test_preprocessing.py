"""
test_preprocessing.py
~~~~~~~~~~~~~~~~~~~~~
Unit tests for the preprocessing pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    preprocess,
    preprocess_batch,
    preprocess_single,
    remove_duplicates,
    remove_nulls,
    scale_features,
    validate_schema,
)


# ── Helpers ──────────────────────────────────────────────────────────────────
def make_df(n_rows: int = 10, add_null: bool = False, add_dupe: bool = False) -> pd.DataFrame:
    """Create a minimal valid DataFrame."""
    data = {col: np.random.randn(n_rows) for col in FEATURE_COLUMNS}
    data["Amount"] = np.abs(data["Amount"])
    data[TARGET_COLUMN] = np.random.randint(0, 2, n_rows).astype(float)
    df = pd.DataFrame(data)
    if add_null:
        df.loc[0, "V1"] = np.nan
    if add_dupe:
        df.loc[n_rows - 1] = df.loc[0]
    return df


# ── Schema Validation ────────────────────────────────────────────────────────
class TestValidateSchema:
    def test_valid_schema(self):
        df = make_df()
        validate_schema(df)  # Should not raise

    def test_missing_column_raises(self):
        df = make_df()
        df = df.drop(columns=["V1"])
        with pytest.raises(ValueError, match="Missing columns"):
            validate_schema(df)


# ── Null Removal ─────────────────────────────────────────────────────────────
class TestRemoveNulls:
    def test_no_nulls(self):
        df = make_df(10)
        result = remove_nulls(df)
        assert len(result) == 10

    def test_removes_null_rows(self):
        df = make_df(10, add_null=True)
        result = remove_nulls(df)
        assert len(result) == 9
        assert result.isnull().sum().sum() == 0


# ── Duplicate Removal ────────────────────────────────────────────────────────
class TestRemoveDuplicates:
    def test_no_duplicates(self):
        df = make_df(10)
        result = remove_duplicates(df)
        assert len(result) == 10

    def test_removes_duplicates(self):
        df = make_df(10, add_dupe=True)
        result = remove_duplicates(df)
        assert len(result) < 10


# ── Scaling ───────────────────────────────────────────────────────────────────
class TestScaleFeatures:
    def test_fit_returns_scaler(self):
        df = make_df(50)
        scaled_df, scaler = scale_features(df, fit=True)
        assert scaler is not None

    def test_inference_uses_existing_scaler(self):
        df = make_df(50)
        _, scaler = scale_features(df, fit=True)
        df2 = make_df(10)
        scaled, _ = scale_features(df2, scaler=scaler, fit=False)
        assert scaled is not None


# ── Full Pipeline ─────────────────────────────────────────────────────────────
class TestPreprocess:
    def test_preprocess_returns_correct_shapes(self):
        df = make_df(50)
        X, y, scaler = preprocess(df)
        assert X.shape == (50, 30)
        assert len(y) == 50

    def test_preprocess_removes_nulls(self):
        df = make_df(50, add_null=True)
        X, y, _ = preprocess(df)
        assert len(X) == 49

    def test_target_is_int(self):
        df = make_df(20)
        _, y, _ = preprocess(df)
        assert y.dtype in (np.int32, np.int64)


# ── Inference Helpers ────────────────────────────────────────────────────────
class TestInferenceHelpers:
    def test_preprocess_single_shape(self):
        from sklearn.preprocessing import RobustScaler
        df = make_df(50)
        _, _, scaler = preprocess(df)
        from tests.conftest import SAMPLE_TRANSACTION
        result = preprocess_single(SAMPLE_TRANSACTION, scaler)
        assert result.shape == (1, 30)

    def test_preprocess_batch_shape(self):
        from sklearn.preprocessing import RobustScaler
        df = make_df(50)
        _, _, scaler = preprocess(df)
        from tests.conftest import SAMPLE_TRANSACTION
        result = preprocess_batch([SAMPLE_TRANSACTION, SAMPLE_TRANSACTION], scaler)
        assert result.shape == (2, 30)
