"""Metrics calculation for model evaluation."""

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class MetricsCalculator:
    """
    Handles computation of classification metrics.
    Separated from visualization for better modularity and testability.
    """

    @staticmethod
    def compute_classification_metrics(
        y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None
    ) -> Dict[str, Any]:
        """
        Compute standard classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities for positive class (optional)

        Returns:
            Dictionary containing all computed metrics
        """
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        }

        # Compute ROC AUC if probabilities are provided
        if y_proba is not None:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            except ValueError as e:
                logger.warning(f"Could not compute ROC AUC: {e}")
                metrics["roc_auc"] = None

        return metrics

    @staticmethod
    def compute_confusion_matrix_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Compute confusion matrix and its components.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary with confusion matrix and its components
        """
        cm = confusion_matrix(y_true, y_pred)

        metrics = {
            "confusion_matrix": cm.tolist(),
        }

        # Add confusion matrix components for binary classification
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["true_negatives"] = int(tn)
            metrics["false_positives"] = int(fp)
            metrics["false_negatives"] = int(fn)
            metrics["true_positives"] = int(tp)

            # Add derived metrics
            metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            metrics["false_positive_rate"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            metrics["false_negative_rate"] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

        return metrics

    @staticmethod
    def compute_all_metrics(
        y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None
    ) -> Dict[str, Any]:
        """
        Compute all available metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities for positive class (optional)

        Returns:
            Dictionary containing all metrics
        """
        # Get classification metrics
        metrics = MetricsCalculator.compute_classification_metrics(y_true, y_pred, y_proba)

        # Get confusion matrix metrics
        cm_metrics = MetricsCalculator.compute_confusion_matrix_metrics(y_true, y_pred)

        # Merge dictionaries
        metrics.update(cm_metrics)

        return metrics

    @staticmethod
    def save_metrics(
        metrics: Dict[str, Any], output_path: Path, model_name: str | None = None
    ) -> None:
        """
        Save metrics to JSON file.

        Args:
            metrics: Dictionary of metrics to save
            output_path: Path to save the JSON file
            model_name: Optional model name to include in saved data
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data to save
        data = {
            "metrics": metrics,
        }

        if model_name:
            data["model_name"] = model_name

        # Save to JSON
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved metrics to {output_path}")

    @staticmethod
    def log_metrics(metrics: Dict[str, Any]) -> None:
        """
        Log metrics to console in a formatted way.

        Args:
            metrics: Dictionary of metrics to log
        """
        logger.info("=" * 60)
        logger.info("Classification Metrics:")
        logger.info("=" * 60)

        # Core metrics
        if "accuracy" in metrics:
            logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        if "precision" in metrics:
            logger.info(f"Precision: {metrics['precision']:.4f}")
        if "recall" in metrics:
            logger.info(f"Recall:    {metrics['recall']:.4f}")
        if "f1_score" in metrics:
            logger.info(f"F1 Score:  {metrics['f1_score']:.4f}")
        if metrics.get("roc_auc") is not None:
            logger.info(f"ROC AUC:   {metrics['roc_auc']:.4f}")

        # Confusion matrix components
        if "true_positives" in metrics:
            logger.info("")
            logger.info("Confusion Matrix Components:")
            logger.info(f"  True Positives:  {metrics['true_positives']}")
            logger.info(f"  True Negatives:  {metrics['true_negatives']}")
            logger.info(f"  False Positives: {metrics['false_positives']}")
            logger.info(f"  False Negatives: {metrics['false_negatives']}")

        # Additional metrics
        if "specificity" in metrics:
            logger.info("")
            logger.info("Additional Metrics:")
            logger.info(f"  Specificity: {metrics['specificity']:.4f}")
            logger.info(f"  Sensitivity: {metrics['sensitivity']:.4f}")
            logger.info(f"  FPR: {metrics['false_positive_rate']:.4f}")
            logger.info(f"  FNR: {metrics['false_negative_rate']:.4f}")

        logger.info("=" * 60)

    @staticmethod
    def get_predictions_and_probas(
        model, X_test: pd.DataFrame, y_test: pd.Series
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Get predictions and probabilities from a model, handling LSTM sequence alignment.

        Args:
            model: Trained model instance
            X_test: Test features
            y_test: Test labels

        Returns:
            Tuple of (y_true_aligned, y_pred, y_proba_positive)
        """
        # Make predictions
        y_pred = model.predict(X_test)

        # Handle sequence-based models (LSTM) that may return fewer predictions
        if len(y_pred) < len(y_test):
            logger.warning(
                f"Model returned {len(y_pred)} predictions for {len(y_test)} samples. "
                "Aligning labels..."
            )
            y_true_aligned = y_test.iloc[-len(y_pred) :].values
        else:
            y_true_aligned = y_test.values

        # Get predicted probabilities
        y_proba_positive = None
        try:
            y_proba = model.predict_proba(X_test)
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                y_proba_positive = y_proba[:, 1]  # Probability of positive class
            elif y_proba.ndim == 2:
                y_proba_positive = y_proba[:, 0]
            else:
                y_proba_positive = y_proba
        except Exception as e:
            logger.warning(f"Could not get prediction probabilities: {e}")

        return y_true_aligned, y_pred, y_proba_positive
