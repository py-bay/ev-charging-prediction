"""Tests for evaluation module."""

import numpy as np
import pytest

from src.config import Config, load_config
from src.evaluation.metrics import ModelEvaluator


@pytest.fixture
def config() -> Config:
    """Create a test configuration."""
    return load_config()


@pytest.fixture
def sample_predictions():
    """Create sample predictions and labels."""
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1, 1, 0])
    y_proba = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.3, 0.9, 0.85, 0.6, 0.15])

    return y_true, y_pred, y_proba


def test_evaluator_initialization(config):
    """Test evaluator initialization."""
    evaluator = ModelEvaluator(config)
    assert evaluator.config == config


def test_compute_metrics(config, sample_predictions):
    """Test metrics computation."""
    y_true, y_pred, y_proba = sample_predictions

    evaluator = ModelEvaluator(config)
    metrics = evaluator.compute_metrics(y_true, y_pred, y_proba)

    # Check that all expected metrics are present
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    assert "confusion_matrix" in metrics

    # Check metric value ranges
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1_score"] <= 1

    # Check confusion matrix structure
    cm = np.array(metrics["confusion_matrix"])
    assert cm.shape == (2, 2)
    assert cm.sum() == len(y_true)


def test_compute_metrics_without_proba(config, sample_predictions):
    """Test metrics computation without probabilities."""
    y_true, y_pred, _ = sample_predictions

    evaluator = ModelEvaluator(config)
    metrics = evaluator.compute_metrics(y_true, y_pred, y_proba=None)

    # ROC AUC should not be computed
    assert "roc_auc" not in metrics or metrics["roc_auc"] is None

    # Other metrics should still be present
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics


def test_confusion_matrix_components(config, sample_predictions):
    """Test confusion matrix components extraction."""
    y_true, y_pred, _ = sample_predictions

    evaluator = ModelEvaluator(config)
    metrics = evaluator.compute_metrics(y_true, y_pred)

    # Check that confusion matrix components are present
    assert "true_negatives" in metrics
    assert "false_positives" in metrics
    assert "false_negatives" in metrics
    assert "true_positives" in metrics

    # Check that sum equals total samples
    total = (
        metrics["true_negatives"]
        + metrics["false_positives"]
        + metrics["false_negatives"]
        + metrics["true_positives"]
    )
    assert total == len(y_true)
