"""Unit tests for the ML Pipeline (Part A)."""

import pytest
import pandas as pd
from pathlib import Path
from src.data import load_config, load_data, clean_data, split_data, validate_schema
from src.predict import load_model, predict, EXPECTED_FEATURES


@pytest.fixture
def config():
    return load_config("configs/config.yaml")


@pytest.fixture
def model(config):
    model_path = config["model"]["pipeline_output_path"]
    return load_model(model_path)


def test_data_loading_and_schema(config):
    """Verify raw dataset exists, loads successfully, and contains expected features."""
    raw_path = config["data"]["raw_path"]
    expected_features = config["data"]["features"]
    target = config["data"]["target"]

    df = load_data(raw_path)
    assert not df.empty, "Dataset should not be empty."
    assert validate_schema(df, expected_features, target) is True
    assert len(df) >= 200, "Dataset should contain at least 200 rows."


def test_split_data_no_leakage():
    """Verify train/test split maintains strict separation without row leakage."""
    sample_df = pd.DataFrame({
        "area_sqft": [1000, 1500, 2000, 2500, 3000],
        "bedrooms": [2, 3, 3, 4, 4],
        "bathrooms": [1, 2, 2, 3, 3],
        "age_years": [5, 10, 15, 20, 25],
        "distance_km": [2.0, 4.0, 6.0, 8.0, 10.0],
        "price": [200000, 300000, 400000, 500000, 600000],
    })
    train_df, test_df = split_data(sample_df, test_size=0.4, random_state=42)
    assert len(train_df) == 3
    assert len(test_df) == 2
    # Ensure disjoint indices
    assert set(train_df.index).isdisjoint(set(test_df.index))


def test_pipeline_prediction_shape_and_type(model):
    """Verify inference pipeline produces numeric predictions of identical shape."""
    sample_input = [
        {"area_sqft": 1200.0, "bedrooms": 2, "bathrooms": 1, "age_years": 8, "distance_km": 5.0},
        {"area_sqft": 2200.0, "bedrooms": 3, "bathrooms": 2, "age_years": 3, "distance_km": 2.5},
    ]
    preds = predict(model, sample_input)
    assert len(preds) == len(sample_input), "Output prediction length must match input batch size."
    for p in preds:
        assert isinstance(p, float), "Prediction output must be float."
        assert p > 0, "Predicted house price must be positive."


def test_prediction_input_validation(model):
    """Verify pipeline rejects inputs missing mandatory feature columns."""
    invalid_input = [{"area_sqft": 1500.0, "bedrooms": 3}]  # missing bathrooms, age_years, distance_km
    with pytest.raises(ValueError, match="missing required features"):
        predict(model, invalid_input)
