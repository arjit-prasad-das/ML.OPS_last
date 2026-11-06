"""Verification script to compare local model predictions against containerized REST API predictions."""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import requests
import joblib
import pandas as pd

from src.predict import EXPECTED_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_comparison(
    api_url: str = "http://localhost:8000/predict",
    model_path: str = "models/house_price_pipeline.joblib",
    output_path: str = "reports/prediction_comparison.json"
) -> List[Dict[str, Any]]:
    """Compare predictions between local Scikit-learn Pipeline and Dockerized REST API."""
    local_pipeline = joblib.load(model_path)
    
    test_cases = [
        {"area_sqft": 1050.0, "bedrooms": 2, "bathrooms": 1, "age_years": 10, "distance_km": 5.2},
        {"area_sqft": 1800.0, "bedrooms": 3, "bathrooms": 2, "age_years": 5, "distance_km": 12.0},
        {"area_sqft": 2400.0, "bedrooms": 4, "bathrooms": 3, "age_years": 15, "distance_km": 3.5},
        {"area_sqft": 850.0, "bedrooms": 1, "bathrooms": 1, "age_years": 25, "distance_km": 18.4},
        {"area_sqft": 3100.0, "bedrooms": 5, "bathrooms": 4, "age_years": 2, "distance_km": 2.0},
    ]

    results = []
    logger.info("Executing Local vs. Docker Container inference comparison...")

    for i, payload in enumerate(test_cases, 1):
        # Local direct model inference
        df_local = pd.DataFrame([payload])[EXPECTED_FEATURES]
        local_pred = float(local_pipeline.predict(df_local)[0])
        local_pred_rounded = round(local_pred, 2)

        # Docker container REST API inference
        resp = requests.post(api_url, json=payload, timeout=5)
        if resp.status_code != 200:
            raise RuntimeError(f"Container API returned status {resp.status_code}: {resp.text}")
        docker_pred = float(resp.json()["predicted_price"])

        diff = abs(local_pred_rounded - docker_pred)
        match = diff < 0.01  # tolerance of 1 cent

        results.append({
            "test_case_id": i,
            "inputs": payload,
            "local_pipeline_prediction": local_pred_rounded,
            "docker_container_prediction": docker_pred,
            "absolute_difference": round(diff, 4),
            "consistent": match,
        })
        logger.info(
            f"Case {i}: Local=${local_pred_rounded:,.2f} | Docker=${docker_pred:,.2f} | Diff={diff:.4f} | Consistent={match}"
        )

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Comparison report persisted to {out_file}")

    return results


if __name__ == "__main__":
    run_comparison()
