"""Automated tests for FastAPI endpoints (Part B)."""

import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_health_endpoint():
    """Verify GET /health returns 200 OK and model is confirmed loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "service" in data


def test_predict_valid_payload():
    """Verify POST /predict returns 200 OK and valid numerical prediction."""
    payload = {
        "area_sqft": 1800.0,
        "bedrooms": 3,
        "bathrooms": 2,
        "age_years": 5,
        "distance_km": 12.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "predicted_price" in data
    assert isinstance(data["predicted_price"], float)
    assert data["predicted_price"] > 0
    assert data["currency"] == "USD"


def test_predict_missing_required_field():
    """Verify POST /predict with missing required field returns 422 Unprocessable Entity."""
    # Missing distance_km
    incomplete_payload = {
        "area_sqft": 1800.0,
        "bedrooms": 3,
        "bathrooms": 2,
        "age_years": 5
    }
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    # Ensure error indicates missing field
    assert any("distance_km" in str(err.get("loc")) for err in data["detail"])


def test_predict_wrong_data_type():
    """Verify POST /predict with invalid data type returns 422 Unprocessable Entity."""
    invalid_type_payload = {
        "area_sqft": "one thousand eight hundred",  # string instead of float
        "bedrooms": 3,
        "bathrooms": 2,
        "age_years": 5,
        "distance_km": 12.0
    }
    response = client.post("/predict", json=invalid_type_payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_predict_invalid_business_constraints():
    """Verify business constraints (e.g., negative area or 0 bedrooms) return 422."""
    negative_area_payload = {
        "area_sqft": -500.0,  # invalid: area must be > 0
        "bedrooms": 0,        # invalid: bedrooms must be > 0
        "bathrooms": 2,
        "age_years": 5,
        "distance_km": 12.0
    }
    response = client.post("/predict", json=negative_area_payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
