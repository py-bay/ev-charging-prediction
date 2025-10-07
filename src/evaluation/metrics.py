"""Model evaluation orchestrator - combines metrics and visualization."""

from pathlib import Path
from typing import Any, Dict

import pandas as pd
from loguru import logger

from src.config.models import Config
from .metrics_calculator import MetricsCalculator
from .visualizer import EvaluationVisualizer


class ModelEvaluator:
    """
    High-level orchestrator for model evaluation.
    Combines metrics calculation and visualization in a clean interface.
    """

    def __init__(self, config: Config):
        """
        Initialize evaluator with configuration.

        Args:
            config: Configuration object
        """
        self.config = config
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = EvaluationVisualizer()

    def evaluate_model(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str,
        generate_plots: bool = True,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model with metrics and optional visualizations.

        Args:
            model: Trained model instance
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model for logging and file naming
            generate_plots: Whether to generate visualization plots
            save_results: Whether to save results to disk

        Returns:
            Dictionary containing evaluation results
        """
        logger.info("=" * 60)
        logger.info(f"Evaluating {model_name} model")
        logger.info("=" * 60)

        # Get predictions and probabilities
        y_true, y_pred, y_proba = self.metrics_calculator.get_predictions_and_probas(
            model, X_test, y_test
        )

        # Compute all metrics
        metrics = self.metrics_calculator.compute_all_metrics(y_true, y_pred, y_proba)

        # Log metrics to console
        self.metrics_calculator.log_metrics(metrics)

        # Prepare results dictionary
        results = {
            "model_name": model_name,
            "metrics": metrics,
            "n_test_samples": len(y_true),
        }

        # Save metrics to file
        if save_results:
            results_path = self.config.output_paths.results / f"{model_name}_results.json"
            self.metrics_calculator.save_metrics(metrics, results_path, model_name=model_name)

        # Generate plots
        if generate_plots:
            self._generate_plots(y_true, y_pred, y_proba, model_name)

        logger.info("=" * 60)
        logger.info(f"Evaluation of {model_name} completed")
        logger.info("=" * 60)

        return results

    def _generate_plots(self, y_true, y_pred, y_proba, model_name: str) -> None:
        """
        Generate all evaluation plots.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            model_name: Model name for file naming
        """
        output_dir = self.config.output_paths.plots

        # Generate all standard evaluation plots
        self.visualizer.plot_all_evaluation_curves(y_true, y_pred, y_proba, output_dir, model_name)

    def generate_evaluation_report(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str,
    ) -> Dict[str, Any]:
        """
        Generate complete evaluation report with metrics and visualizations.
        Convenience method that calls evaluate_model with all options enabled.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model

        Returns:
            Evaluation results dictionary
        """
        return self.evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            model_name=model_name,
            generate_plots=True,
            save_results=True,
        )

    # Legacy static methods for backward compatibility
    @staticmethod
    def compute_metrics(y_true, y_pred, y_proba=None):
        """Legacy method - calls MetricsCalculator.compute_all_metrics."""
        return MetricsCalculator.compute_all_metrics(y_true, y_pred, y_proba)

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, model_name, output_dir: Path):
        """Legacy method - calls EvaluationVisualizer.plot_confusion_matrix."""
        output_path = output_dir / f"{model_name}_confusion_matrix.png"
        EvaluationVisualizer.plot_confusion_matrix(y_true, y_pred, output_path, model_name)

    @staticmethod
    def plot_roc_curve(y_true, y_proba, model_name, output_dir: Path):
        """Legacy method - calls EvaluationVisualizer.plot_roc_curve."""
        output_path = output_dir / f"{model_name}_roc_curve.png"
        EvaluationVisualizer.plot_roc_curve(y_true, y_proba, output_path, model_name)

    @staticmethod
    def plot_precision_recall_curve(y_true, y_proba, model_name, output_dir: Path):
        """Legacy method - calls EvaluationVisualizer.plot_precision_recall_curve."""
        output_path = output_dir / f"{model_name}_precision_recall_curve.png"
        EvaluationVisualizer.plot_precision_recall_curve(y_true, y_proba, output_path, model_name)

    @staticmethod
    def save_results(results: Dict[str, Any], output_dir: Path, model_name: str):
        """Legacy method - calls MetricsCalculator.save_metrics."""
        output_path = output_dir / f"{model_name}_results.json"
        metrics = results.get("metrics", results)
        MetricsCalculator.save_metrics(metrics, output_path, model_name)
