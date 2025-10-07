"""Model evaluation and metrics computation."""

import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config.models import Config


class ModelEvaluator:
    """
    Handles model evaluation, metrics computation, and visualization.
    """

    def __init__(self, config: Config):
        """
        Initialize evaluator.

        Args:
            config: Configuration object
        """
        self.config = config

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> Dict[str, Any]:
        """
        Compute evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities for positive class (optional)

        Returns:
            Dictionary of metrics
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
        }

        # Compute ROC AUC if probabilities are provided
        if y_proba is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
            except ValueError as e:
                logger.warning(f"Could not compute ROC AUC: {e}")
                metrics["roc_auc"] = None

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        # Add confusion matrix components
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["true_negatives"] = int(tn)
            metrics["false_positives"] = int(fp)
            metrics["false_negatives"] = int(fn)
            metrics["true_positives"] = int(tp)

        return metrics

    def evaluate_model(
            self,
            model,
            x_test: pd.DataFrame,
            y_test: pd.Series,
            model_name: str,
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model.

        Args:
            model: Trained model (RandomForestModel or LSTMModel)
            x_test: Test features
            y_test: Test labels
            model_name: Name of the model

        Returns:
            Dictionary of evaluation results
        """
        logger.info("=" * 60)
        logger.info(f"Evaluating {model_name} model")
        logger.info("=" * 60)

        # Make predictions
        y_pred = model.predict(x_test)

        # Handle sequence-based models (LSTM) that may return fewer predictions
        if len(y_pred) < len(y_test):
            logger.warning(
                f"Model returned {len(y_pred)} predictions for {len(y_test)} samples. "
                "Aligning labels..."
            )
            y_test_aligned = y_test.iloc[-len(y_pred):]
        else:
            y_test_aligned = y_test

        # Get predicted probabilities
        try:
            y_proba = model.predict_proba(x_test)
            if y_proba.shape[1] == 2:
                y_proba_positive = y_proba[:, 1]  # Probability of positive class
            else:
                y_proba_positive = y_proba[:, 0]
        except Exception as e:
            logger.warning(f"Could not get prediction probabilities: {e}")
            y_proba_positive = None

        # Compute metrics
        metrics = self.compute_metrics(y_test_aligned.values, y_pred, y_proba_positive)

        # Log metrics
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1 Score:  {metrics['f1_score']:.4f}")
        if metrics.get("roc_auc") is not None:
            logger.info(f"ROC AUC:   {metrics['roc_auc']:.4f}")

        results = {
            "model_name": model_name,
            "metrics": metrics,
            "n_test_samples": len(y_test_aligned),
        }

        return results

    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, output_dir: Path) -> None:
        """
        Plot and save confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            output_dir: Output directory for plot
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
        )
        plt.title(f"Confusion Matrix - {model_name}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

        output_path = output_dir / f"{model_name}_confusion_matrix.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved confusion matrix to {output_path}")

    @staticmethod
    def plot_roc_curve(
            y_true: np.ndarray,
            y_proba: np.ndarray,
            model_name: str,
            output_dir: Path,
    ):
        """
        Plot and save ROC curve.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities for positive class
            model_name: Name of the model
            output_dir: Output directory for plot
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random classifier")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {model_name}")
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)

            output_path = output_dir / f"{model_name}_roc_curve.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Saved ROC curve to {output_path}")

        except Exception as e:
            logger.error(f"Failed to plot ROC curve: {e}")

    @staticmethod
    def plot_precision_recall_curve(
            y_true: np.ndarray,
            y_proba: np.ndarray,
            model_name: str,
            output_dir: Path,
    ):
        """
        Plot and save precision-recall curve.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities for positive class
            model_name: Name of the model
            output_dir: Output directory for plot
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color="blue", lw=2)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall Curve - {model_name}")
            plt.grid(alpha=0.3)

            output_path = output_dir / f"{model_name}_precision_recall_curve.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Saved precision-recall curve to {output_path}")

        except Exception as e:
            logger.error(f"Failed to plot precision-recall curve: {e}")

    @staticmethod
    def save_results(results: Dict[str, Any], output_dir: Path, model_name: str):
        """
        Save evaluation results to JSON.

        Args:
            results: Evaluation results dictionary
            output_dir: Output directory
            model_name: Name of the model
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{model_name}_results.json"

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved evaluation results to {output_path}")

    def generate_evaluation_report(
            self,
            model,
            x_test: pd.DataFrame,
            y_test: pd.Series,
            model_name: str,
    ) -> Dict[str, Any]:
        """
        Generate complete evaluation report with metrics and visualizations.

        Args:
            model: Trained model
            x_test: Test features
            y_test: Test labels
            model_name: Name of the model

        Returns:
            Evaluation results dictionary
        """
        # Evaluate model
        results = self.evaluate_model(model, x_test, y_test, model_name)

        # Make predictions for plotting
        y_pred = model.predict(x_test)

        # Align labels if needed (for LSTM)
        if len(y_pred) < len(y_test):
            y_test_aligned = y_test.iloc[-len(y_pred):]
        else:
            y_test_aligned = y_test

        # Get probabilities
        try:
            y_proba = model.predict_proba(x_test)
            if y_proba.shape[1] == 2:
                y_proba_positive = y_proba[:, 1]
            else:
                y_proba_positive = y_proba[:, 0]
        except Exception:
            y_proba_positive = None

        # Generate plots
        output_dir = self.config.output_paths.plots

        # Confusion matrix
        self.plot_confusion_matrix(y_test_aligned.values, y_pred, model_name, output_dir)

        # ROC curve
        if y_proba_positive is not None:
            self.plot_roc_curve(y_test_aligned.values, y_proba_positive, model_name, output_dir)
            self.plot_precision_recall_curve(
                y_test_aligned.values, y_proba_positive, model_name, output_dir
            )

        # Save results
        self.save_results(results, self.config.output_paths.results, model_name)

        logger.info("=" * 60)
        logger.info(f"Evaluation report for {model_name} completed")
        logger.info("=" * 60)

        return results
