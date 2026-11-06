"""Training module for building and saving the complete Scikit-learn inference pipeline.

Encapsulates feature scaling and regression in a single Pipeline object,
guaranteeing identical preprocessing during training and inference without data leakage.
"""

from __future__ import annotations
import logging
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_config, prepare_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_pipeline(n_estimators: int = 100, max_depth: int = 5, random_state: int = 42) -> Pipeline:
    """Construct an end-to-end Scikit-learn Pipeline combining preprocessing and estimator."""
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return pipeline


def train_model(config_path: str | Path = "configs/config.yaml") -> Pipeline:
    """Train the model pipeline and persist it to disk."""
    config = load_config(config_path)
    
    # 1. Prepare data splits (without data leakage)
    train_df, _ = prepare_data(config_path)

    features = config["data"]["features"]
    target = config["data"]["target"]

    X_train = train_df[features]
    y_train = train_df[target]

    # 2. Extract model hyperparameters
    n_estimators = int(config["model"].get("n_estimators", 100))
    raw_depth = config["model"].get("max_depth", 5)
    max_depth = None if str(raw_depth).lower() == "none" else int(raw_depth)
    random_state = int(config["model"].get("random_state", 42))

    # 3. Build & fit pipeline
    logger.info(f"Building pipeline with RandomForestRegressor (n_estimators={n_estimators}, max_depth={max_depth})...")
    pipeline = build_pipeline(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    pipeline.fit(X_train, y_train)
    logger.info("Pipeline training complete.")

    # 4. Save serialized pipeline
    output_path = Path(config["model"]["pipeline_output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    logger.info(f"Saved complete inference pipeline artifact to: {output_path}")

    return pipeline


def main() -> None:
    """Entry point for CLI execution: python -m src.train"""
    logger.info("Starting training run via CLI...")
    train_model()
    logger.info("CLI training run completed successfully.")


if __name__ == "__main__":
    main()
