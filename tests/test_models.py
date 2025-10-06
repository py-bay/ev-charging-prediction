"""Tests for model modules."""

import numpy as np
import pandas as pd
import pytest

from ev_charge_forecasting.config.models import Config
from ev_charge_forecasting.models import LSTMModel, RandomForestModel


@pytest.fixture
def config():
    """Create a test configuration."""
    return Config()


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    n_samples = 200
    n_features = 5

    X_train = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    y_train = pd.Series(np.random.randint(0, 2, n_samples), name="label")

    X_test = pd.DataFrame(
        np.random.randn(50, n_features), columns=[f"feature_{i}" for i in range(n_features)]
    )
    y_test = pd.Series(np.random.randint(0, 2, 50), name="label")

    return X_train, X_test, y_train, y_test


def test_randomforest_initialization(config):
    """Test RandomForest model initialization."""
    model = RandomForestModel(config)
    assert model.config == config
    assert model.model is None
    assert model.feature_names is None


def test_randomforest_build_model(config):
    """Test RandomForest model building."""
    model = RandomForestModel(config)
    rf_model = model.build_model()

    assert rf_model is not None
    assert rf_model.n_estimators == config.models.randomforest.n_estimators


def test_randomforest_train(config, sample_data):
    """Test RandomForest model training."""
    X_train, X_test, y_train, y_test = sample_data

    model = RandomForestModel(config)
    train_info = model.train(X_train, y_train)

    assert model.model is not None
    assert model.feature_names == list(X_train.columns)
    assert train_info["model_type"] == "randomforest"
    assert train_info["n_features"] == len(X_train.columns)
    assert train_info["n_samples"] == len(X_train)


def test_randomforest_predict(config, sample_data):
    """Test RandomForest model predictions."""
    X_train, X_test, y_train, y_test = sample_data

    model = RandomForestModel(config)
    model.train(X_train, y_train)

    predictions = model.predict(X_test)

    assert len(predictions) == len(X_test)
    assert predictions.dtype in [np.int32, np.int64]
    assert set(predictions).issubset({0, 1})


def test_randomforest_predict_proba(config, sample_data):
    """Test RandomForest probability predictions."""
    X_train, X_test, y_train, y_test = sample_data

    model = RandomForestModel(config)
    model.train(X_train, y_train)

    proba = model.predict_proba(X_test)

    assert proba.shape == (len(X_test), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)  # Probabilities sum to 1


def test_lstm_initialization(config):
    """Test LSTM model initialization."""
    model = LSTMModel(config)
    assert model.config == config
    assert model.model is None
    assert model.feature_names is None


def test_lstm_create_sequences(config, sample_data):
    """Test LSTM sequence creation."""
    X_train, X_test, y_train, y_test = sample_data

    model = LSTMModel(config)
    X_seq, y_seq = model.create_sequences(X_train, y_train)

    sequence_length = config.models.lstm.sequence_length
    expected_samples = len(X_train) - sequence_length

    assert X_seq.shape[0] == expected_samples
    assert X_seq.shape[1] == sequence_length
    assert X_seq.shape[2] == X_train.shape[1]
    assert len(y_seq) == expected_samples


def test_lstm_train(config, sample_data):
    """Test LSTM model training."""
    X_train, X_test, y_train, y_test = sample_data

    # Use smaller config for faster testing
    config.models.lstm.epochs = 2
    config.models.lstm.batch_size = 16

    model = LSTMModel(config)
    train_info = model.train(X_train, y_train)

    assert model.model is not None
    assert model.feature_names == list(X_train.columns)
    assert train_info["model_type"] == "lstm"


def test_lstm_predict(config, sample_data):
    """Test LSTM model predictions."""
    X_train, X_test, y_train, y_test = sample_data

    # Use smaller config for faster testing
    config.models.lstm.epochs = 2
    config.models.lstm.batch_size = 16

    model = LSTMModel(config)
    model.train(X_train, y_train)

    predictions = model.predict(X_test)

    # Predictions will be fewer due to sequence length
    sequence_length = config.models.lstm.sequence_length
    expected_predictions = len(X_test) - sequence_length

    assert len(predictions) == expected_predictions
    assert predictions.dtype in [np.int32, np.int64]
    assert set(predictions).issubset({0, 1})
