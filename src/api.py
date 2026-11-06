"""FastAPI REST application for online house price prediction.

Exposes GET /health and POST /predict with Pydantic validation
and explicit error handling.
"""

from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Any, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
from pydantic import BaseModel, Field

from src.predict import EXPECTED_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Configurable model path via environment variable (useful for Docker & MLflow)
MODEL_PATH = os.getenv("MODEL_PATH", "models/house_price_pipeline.joblib")

# Global model container
model_pipeline: Any = None


def load_model_safely(path_str: str) -> Any:
    """Safely load serialized pipeline artifact."""
    model_file = Path(path_str)
    if not model_file.exists():
        logger.warning(f"Model file not found at {model_file}. API starting without loaded model.")
        return None
    try:
        pipeline = joblib.load(model_file)
        logger.info(f"Model successfully loaded from {model_file}")
        return pipeline
    except Exception as e:
        logger.error(f"Error loading model from {model_file}: {e}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event to manage model loading during server startup."""
    global model_pipeline
    model_pipeline = load_model_safely(MODEL_PATH)
    yield
    model_pipeline = None


app = FastAPI(
    title="House Price Prediction API",
    description="Production-grade REST API for serving house price predictions using a Scikit-learn Pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)


class HouseFeatures(BaseModel):
    """Input schema with strict validation rules and realistic documentation examples."""
    area_sqft: float = Field(
        ...,
        gt=0,
        le=50000,
        description="Property floor area in square feet (must be > 0).",
        examples=[1800.0],
    )
    bedrooms: int = Field(
        ...,
        gt=0,
        le=30,
        description="Total number of bedrooms (must be >= 1).",
        examples=[3],
    )
    bathrooms: int = Field(
        ...,
        gt=0,
        le=30,
        description="Total number of bathrooms (must be >= 1).",
        examples=[2],
    )
    age_years: int = Field(
        ...,
        ge=0,
        le=200,
        description="Age of property in years (must be >= 0).",
        examples=[5],
    )
    distance_km: float = Field(
        ...,
        ge=0.0,
        le=500.0,
        description="Distance to primary city center in kilometers (must be >= 0).",
        examples=[12.0],
    )


class PredictionResponse(BaseModel):
    """Output schema returned to API consumers."""
    predicted_price: float = Field(..., description="Estimated market price in dollars.")
    currency: str = Field(default="USD", description="Currency denomination.")
    status: str = Field(default="success", description="Inference status.")


@app.get("/health", status_code=status.HTTP_200_OK, tags=["Health"])
def health() -> Dict[str, Any]:
    """Health check endpoint reporting API and model readiness."""
    global model_pipeline
    # Lazily attempt reload if not yet loaded
    if model_pipeline is None:
        model_pipeline = load_model_safely(MODEL_PATH)

    is_ready = model_pipeline is not None
    return {
        "status": "ok" if is_ready else "degraded",
        "service": "House Price Prediction API",
        "version": "1.0.0",
        "model_loaded": is_ready,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
def predict(payload: HouseFeatures) -> PredictionResponse:
    """Predict house market value from provided structural and location features."""
    global model_pipeline
    if model_pipeline is None:
        model_pipeline = load_model_safely(MODEL_PATH)

    if model_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not currently available for inference. Ensure training has completed.",
        )

    try:
        # Construct DataFrame ensuring strict column ordering expected by pipeline
        input_data = {
            "area_sqft": [payload.area_sqft],
            "bedrooms": [payload.bedrooms],
            "bathrooms": [payload.bathrooms],
            "age_years": [payload.age_years],
            "distance_km": [payload.distance_km],
        }
        df = pd.DataFrame(input_data)[EXPECTED_FEATURES]
        prediction_val = float(model_pipeline.predict(df)[0])

        return PredictionResponse(
            predicted_price=round(prediction_val, 2),
            currency="USD",
            status="success",
        )
    except Exception as exc:
        logger.error(f"Inference error encountered: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating the prediction.",
        )
