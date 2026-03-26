"""
evaluate.py
~~~~~~~~~~~
Standalone evaluation script — loads a saved model and generates
a full evaluation report (metrics, confusion matrix, ROC curve).

Usage:
    python src/evaluate.py --model models/fraud_pipeline.joblib --data creditcard.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocessing import preprocess, load_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports")


def evaluate(model_path: str, data_path: str, output_dir: Path = REPORTS_DIR) -> dict:
    """
    Load model, run evaluation on data, save reports.

    Returns
    -------
    dict of metrics
    """
    output_dir.mkdir(exist_ok=True)

    # Load model
    artifact = joblib.load(model_path)
    if isinstance(artifact, dict):
        model = artifact["model"]
        scaler = artifact.get("scaler")
    else:
        model = artifact
        scaler = None

    # Load and preprocess data
    df = load_data(data_path)
    X, y, _ = preprocess(df, scaler=scaler, fit=(scaler is None))

    # Predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # Metrics
    metrics = {
        "roc_auc": float(roc_auc_score(y, y_prob)),
        "avg_precision": float(average_precision_score(y, y_prob)),
        "f1": float(f1_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred)),
        "recall": float(recall_score(y, y_pred)),
        "n_samples": int(len(y)),
        "n_fraud": int(y.sum()),
        "fraud_rate_pct": float(y.mean() * 100),
    }

    # Print report
    print("\n" + "=" * 60)
    print("  CREDIT CARD FRAUD DETECTION — EVALUATION REPORT")
    print("=" * 60)
    print(f"  Samples     : {metrics['n_samples']:,}")
    print(f"  Fraud cases : {metrics['n_fraud']:,} ({metrics['fraud_rate_pct']:.4f}%)")
    print(f"  ROC-AUC     : {metrics['roc_auc']:.4f}")
    print(f"  Avg Prec    : {metrics['avg_precision']:.4f}")
    print(f"  F1          : {metrics['f1']:.4f}")
    print(f"  Precision   : {metrics['precision']:.4f}")
    print(f"  Recall      : {metrics['recall']:.4f}")
    print("=" * 60)
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=["Legitimate", "Fraud"]))

    # Save metrics JSON
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    # Confusion Matrix plot
    _plot_confusion_matrix(y, y_pred, output_dir)

    # ROC Curve plot
    _plot_roc_curve(y, y_prob, metrics["roc_auc"], output_dir)

    return metrics


def _plot_confusion_matrix(y_true, y_pred, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Legitimate", "Fraud"],
        colorbar=False, ax=ax
    )
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = output_dir / "confusion_matrix.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved confusion matrix → %s", path)


def _plot_roc_curve(y_true, y_prob, roc_auc: float, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax, name=f"Fraud Model (AUC={roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_title("ROC Curve — Fraud Detection", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = output_dir / "roc_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved ROC curve → %s", path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Fraud Detection Model")
    parser.add_argument("--model", default="models/fraud_pipeline.joblib")
    parser.add_argument("--data", default="creditcard.csv")
    parser.add_argument("--output-dir", default="reports")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.model, args.data, Path(args.output_dir))
