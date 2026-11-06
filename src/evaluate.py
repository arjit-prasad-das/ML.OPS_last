"""Evaluation module for scoring held-out test data and producing metrics and visualizations."""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, Any
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data import load_config, load_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model(config_path: str | Path = "configs/config.yaml") -> Dict[str, Any]:
    """Evaluate trained pipeline against held-out test dataset."""
    config = load_config(config_path)

    test_path = config["data"]["test_path"]
    model_path = Path(config["model"]["pipeline_output_path"])
    metrics_path = Path(config["reports"]["metrics_path"])
    plot_path = Path(config["reports"]["plot_path"])
    features = config["data"]["features"]
    target = config["data"]["target"]

    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run training first.")

    # 1. Load pipeline and test dataset
    pipeline = joblib.load(model_path)
    test_df = load_data(test_path)

    X_test = test_df[features]
    y_test = test_df[target]

    # 2. Predict on held-out test set
    predictions = pipeline.predict(X_test)

    # 3. Compute metrics
    mae = float(mean_absolute_error(y_test, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = float(r2_score(y_test, predictions))

    metrics = {
        "test_mae": round(mae, 4),
        "test_rmse": round(rmse, 4),
        "test_r2": round(r2, 4),
        "test_samples": len(y_test),
        "model_type": config["model"]["type"],
    }

    # 4. Save metrics to reports
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved evaluation metrics to: {metrics_path}")

    # 5. Generate actual vs. predicted plot
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, predictions, alpha=0.7, color="#1f77b4", edgecolors="k", label="Test Samples")
    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal Fit (y=x)")
    ax.set_title("Actual vs. Predicted House Prices", fontsize=14, fontweight="bold")
    ax.set_xlabel("Actual Price ($)", fontsize=12)
    ax.set_ylabel("Predicted Price ($)", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved evaluation plot to: {plot_path}")

    logger.info(f"Evaluation Results -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
    return metrics


if __name__ == "__main__":
    evaluate_model()
