"""Prediction module for batch and online inference using the saved pipeline."""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Union, List, Dict, Any
import joblib
import pandas as pd

from src.data import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXPECTED_FEATURES = ["area_sqft", "bedrooms", "bathrooms", "age_years", "distance_km"]


def load_model(model_path: str | Path = "models/house_price_pipeline.joblib") -> Any:
    """Load the trained inference pipeline from disk."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model pipeline not found at: {path}. Train the model first.")
    pipeline = joblib.load(path)
    return pipeline


def predict(
    model: Any,
    input_data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]
) -> List[float]:
    """Score input records using the loaded pipeline, preserving column order."""
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        df = pd.DataFrame(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise TypeError(f"Unsupported input type for prediction: {type(input_data)}")

    # Check for missing feature columns
    missing = [f for f in EXPECTED_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Input records missing required features: {missing}")

    # Reorder DataFrame strictly according to expected feature order
    df_ordered = df[EXPECTED_FEATURES]
    predictions = model.predict(df_ordered)
    return [round(float(p), 2) for p in predictions]


def score_sample_records(config_path: str | Path = "configs/config.yaml") -> List[Dict[str, Any]]:
    """Generate predictions for at least five distinct records and persist report."""
    config = load_config(config_path)
    model_path = config["model"]["pipeline_output_path"]
    predictions_path = Path(config["reports"]["predictions_path"])
    
    pipeline = load_model(model_path)

    sample_records = [
        {"area_sqft": 1050.0, "bedrooms": 2, "bathrooms": 1, "age_years": 10, "distance_km": 5.2},
        {"area_sqft": 1800.0, "bedrooms": 3, "bathrooms": 2, "age_years": 5, "distance_km": 12.0},
        {"area_sqft": 2400.0, "bedrooms": 4, "bathrooms": 3, "age_years": 15, "distance_km": 3.5},
        {"area_sqft": 850.0, "bedrooms": 1, "bathrooms": 1, "age_years": 25, "distance_km": 18.4},
        {"area_sqft": 3100.0, "bedrooms": 5, "bathrooms": 4, "age_years": 2, "distance_km": 2.0},
    ]

    predictions = predict(pipeline, sample_records)

    results = []
    for record, pred in zip(sample_records, predictions):
        results.append({
            "features": record,
            "predicted_price": pred
        })

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Persisted sample predictions to: {predictions_path}")
    return results


if __name__ == "__main__":
    sample_preds = score_sample_records()
    for i, item in enumerate(sample_preds, 1):
        print(f"Sample {i}: Features={item['features']} -> Predicted Price=${item['predicted_price']:,.2f}")
