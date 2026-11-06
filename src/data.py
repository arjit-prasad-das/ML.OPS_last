"""Data module for loading, validating, cleaning, and splitting data.

Ensures strict separation between raw and processed data and prevents
data leakage by performing split operations prior to pipeline fitting.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Tuple, List, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str | Path = "configs/config.yaml") -> dict:
    """Load configuration dictionary from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_data(filepath: str | Path) -> pd.DataFrame:
    """Load dataset from a CSV file with validation."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path}")
    
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Dataset at {path} is empty.")
    
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns from {path}")
    return df


def validate_schema(df: pd.DataFrame, expected_features: List[str], target_col: Optional[str] = None) -> bool:
    """Validate that required features and target columns exist in DataFrame."""
    required_cols = list(expected_features)
    if target_col:
        required_cols.append(target_col)
    
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")
    return True


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data by removing exact duplicates and dropping null rows."""
    initial_count = len(df)
    cleaned = df.drop_duplicates().copy()
    cleaned = cleaned.dropna()
    dropped = initial_count - len(cleaned)
    if dropped > 0:
        logger.info(f"Removed {dropped} duplicate or null rows during data cleaning.")
    return cleaned


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset into train and test sets without leaking test distribution."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    logger.info(f"Dataset split completed. Train rows: {len(train_df)}, Test rows: {len(test_df)}")
    return train_df, test_df


def prepare_data(config_path: str | Path = "configs/config.yaml") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Orchestrates loading, validation, cleaning, splitting and caching processed splits."""
    config = load_config(config_path)
    raw_path = config["data"]["raw_path"]
    train_path = Path(config["data"]["train_path"])
    test_path = Path(config["data"]["test_path"])
    expected_features = config["data"]["features"]
    target_col = config["data"]["target"]
    test_size = float(config["training"]["test_size"])
    random_state = int(config["training"]["random_state"])

    df = load_data(raw_path)
    validate_schema(df, expected_features, target_col)
    cleaned_df = clean_data(df)
    train_df, test_df = split_data(cleaned_df, test_size=test_size, random_state=random_state)

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info(f"Saved processed train split to {train_path} and test split to {test_path}")

    return train_df, test_df


if __name__ == "__main__":
    prepare_data()
