"""Tests for preprocessing module."""

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.preprocessing.pipeline import DataPreprocessor


@pytest.fixture
def config():
    """Create a test configuration."""
    return Config()


@pytest.fixture
def sample_pv_data():
    """Create sample PV data."""
    dates = pd.date_range("2024-01-01", periods=96, freq="15T")
    return pd.DataFrame({"timestamp": dates, "pv_power": np.random.uniform(0, 5, 96)}).set_index(
        "timestamp"
    )


@pytest.fixture
def sample_irradiance_data():
    """Create sample irradiance data."""
    dates = pd.date_range("2024-01-01", periods=96, freq="15T")
    return pd.DataFrame(
        {"timestamp": dates, "irradiance": np.random.uniform(0, 1000, 96)}
    ).set_index("timestamp")


@pytest.fixture
def sample_weather_data():
    """Create sample weather data (hourly)."""
    dates = pd.date_range("2024-01-01", periods=24, freq="H")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "temperature_2m": np.random.uniform(10, 30, 24),
            "relative_humidity_2m": np.random.uniform(30, 80, 24),
            "cloud_cover": np.random.uniform(0, 100, 24),
            "weather_code": np.random.choice([0, 1, 2, 3], 24),
        }
    ).set_index("timestamp")


def test_preprocessor_initialization(config):
    """Test preprocessor initialization."""
    preprocessor = DataPreprocessor(config)
    assert preprocessor.config == config
    assert preprocessor.df_combined is None
    assert preprocessor.feature_columns == []


def test_resample_weather_data(config, sample_weather_data):
    """Test weather data resampling."""
    preprocessor = DataPreprocessor(config)
    resampled = preprocessor.resample_weather_data(sample_weather_data)

    # Should have 4x the rows (hourly -> 15min)
    assert len(resampled) == len(sample_weather_data) * 4
    assert resampled.index.freq.freqstr == "15T"


def test_add_time_features(config, sample_pv_data):
    """Test time feature addition."""
    preprocessor = DataPreprocessor(config)
    df_with_time = preprocessor.add_time_features(sample_pv_data)

    # Check that time features were added
    assert "hour_sin" in df_with_time.columns
    assert "hour_cos" in df_with_time.columns
    assert "day_sin" in df_with_time.columns
    assert "day_cos" in df_with_time.columns

    # Check value ranges
    assert df_with_time["hour_sin"].between(-1, 1).all()
    assert df_with_time["hour_cos"].between(-1, 1).all()


def test_generate_labels_absolute(config, sample_pv_data):
    """Test label generation with absolute method."""
    preprocessor = DataPreprocessor(config)
    df_with_labels = preprocessor.generate_labels(sample_pv_data, threshold=1.0, method="absolute")

    assert "label" in df_with_labels.columns
    assert df_with_labels["label"].dtype == np.int64
    assert df_with_labels["label"].isin([0, 1]).all()


def test_generate_labels_relative(config, sample_pv_data):
    """Test label generation with relative method."""
    preprocessor = DataPreprocessor(config)
    df_with_labels = preprocessor.generate_labels(sample_pv_data, threshold=0.8, method="relative")

    assert "label" in df_with_labels.columns
    assert df_with_labels["label"].dtype == np.int64
    assert df_with_labels["label"].isin([0, 1]).all()


def test_compute_data_hash(config):
    """Test data hash computation."""
    preprocessor = DataPreprocessor(config)
    # This will compute hash even if files don't exist
    data_hash = preprocessor.compute_data_hash()
    assert isinstance(data_hash, str)
    assert len(data_hash) == 64  # SHA-256 hex digest length
