"""
train.py
~~~~~~~~
Production training script for Credit Card Fraud Detection.

Usage:
    python src/train.py --data creditcard.csv
    python src/train.py --data creditcard.csv --experiment my-exp --no-register
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, preprocess
from src.mlflow_utils import (
    log_artifact,
    log_metrics,
    log_model,
    log_params,
    setup_mlflow,
    transition_model_stage,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
CLASSIFIERS = {
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    ),
    "logistic_regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    ),
}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(
    data_path: str,
    experiment: str,
    model_type: str = "random_forest",
    test_size: float = 0.2,
    cv_folds: int = 5,
    register: bool = True,
) -> dict:
    """
    Full training run: load → preprocess → train → evaluate → log to MLflow.

    Returns
    -------
    dict of evaluation metrics
    """
    MODELS_DIR.mkdir(exist_ok=True)

    # ---- Setup MLflow ----
    setup_mlflow(experiment)

    with mlflow.start_run(run_name=f"train_{model_type}") as run:
        logger.info("MLflow run id: %s", run.info.run_id)

        # ---- 1. Load & Preprocess ----
        from src.preprocessing import load_data
        df = load_data(data_path)
        X, y, scaler = preprocess(df, fit=True)

        logger.info("Class distribution:\n%s", y.value_counts().to_string())

        # ---- 2. Train / Test Split ----
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        logger.info(
            "Split: train=%d, test=%d (%.0f%% / %.0f%%)",
            len(X_train), len(X_test),
            (1 - test_size) * 100, test_size * 100,
        )

        # ---- 3. Hyperparams ----
        clf = CLASSIFIERS[model_type]
        params = {
            "model_type": model_type,
            "test_size": test_size,
            "cv_folds": cv_folds,
            **clf.get_params(),
        }
        log_params(params)

        # ---- 4. Cross-Validation ----
        logger.info("Running %d-fold cross-validation…", cv_folds)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_roc_auc = cross_val_score(clf, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        cv_f1 = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        log_metrics({
            "cv_roc_auc_mean": float(cv_roc_auc.mean()),
            "cv_roc_auc_std": float(cv_roc_auc.std()),
            "cv_f1_mean": float(cv_f1.mean()),
            "cv_f1_std": float(cv_f1.std()),
        })
        logger.info("CV ROC-AUC: %.4f ± %.4f", cv_roc_auc.mean(), cv_roc_auc.std())

        # ---- 5. Final Fit ----
        clf.fit(X_train, y_train)

        # ---- 6. Evaluation ----
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        metrics = {
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "avg_precision": float(average_precision_score(y_test, y_prob)),
            "f1": float(f1_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
        }
        log_metrics(metrics)

        report = classification_report(y_test, y_pred, target_names=["Legit", "Fraud"])
        logger.info("\nClassification Report:\n%s", report)

        # ---- 7. Save Artifacts ----
        pipeline_path = MODELS_DIR / "fraud_pipeline.joblib"
        scaler_path = MODELS_DIR / "scaler.joblib"
        metrics_path = MODELS_DIR / "metrics.json"

        joblib.dump({"model": clf, "scaler": scaler}, pipeline_path)
        joblib.dump(scaler, scaler_path)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        log_artifact(pipeline_path)
        log_artifact(scaler_path)
        log_artifact(metrics_path)

        # ---- 8. Log Model to Registry ----
        log_model(clf, model_name="fraud-detection", register=register)

        if register:
            transition_model_stage(model_name="fraud-detection", stage="Production")

        logger.info("=" * 60)
        logger.info("Training complete.")
        logger.info("  ROC-AUC   : %.4f", metrics["roc_auc"])
        logger.info("  F1        : %.4f", metrics["f1"])
        logger.info("  Precision : %.4f", metrics["precision"])
        logger.info("  Recall    : %.4f", metrics["recall"])
        logger.info("=" * 60)

        return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Fraud Detection Model")
    parser.add_argument("--data", default="creditcard.csv", help="Path to CSV dataset")
    parser.add_argument(
        "--experiment",
        default="credit-card-fraud-detection",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--model",
        default="random_forest",
        choices=list(CLASSIFIERS.keys()),
        help="Model type to train",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds")
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Skip MLflow model registry registration",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_path=args.data,
        experiment=args.experiment,
        model_type=args.model,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        register=not args.no_register,
    )
