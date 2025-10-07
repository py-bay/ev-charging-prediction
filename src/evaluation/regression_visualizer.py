"""Visualization tools for regression model evaluation."""

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger


class RegressionVisualizer:
    """Create visualizations for regression model evaluation."""

    @staticmethod
    def plot_predictions_vs_actual(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_path: Path,
        model_name: Optional[str] = None,
    ):
        """Plot predicted vs actual values scatter plot.

        Args:
            y_true: True values
            y_pred: Predicted values
            output_path: Path to save plot
            model_name: Optional model name for title
        """
        plt.figure(figsize=(10, 8))

        # Scatter plot
        plt.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors='k', linewidth=0.5)

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

        plt.xlabel('Actual PV Power (kW)', fontsize=12)
        plt.ylabel('Predicted PV Power (kW)', fontsize=12)
        title = 'Predicted vs Actual PV Power'
        if model_name:
            title = f'{model_name}: {title}'
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved predictions vs actual plot to {output_path}")

    @staticmethod
    def plot_residuals(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_path: Path,
        model_name: Optional[str] = None,
    ):
        """Plot residuals (errors) distribution and vs predicted values.

        Args:
            y_true: True values
            y_pred: Predicted values
            output_path: Path to save plot
            model_name: Optional model name for title
        """
        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Predicted PV Power (kW)', fontsize=12)
        axes[0].set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
        axes[0].set_title('Residual Plot', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Residuals distribution
        axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Residuals (kW)', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Residuals Distribution', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        if model_name:
            fig.suptitle(f'{model_name}: Residual Analysis', fontsize=16, fontweight='bold', y=1.02)

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved residuals plot to {output_path}")

    @staticmethod
    def plot_time_series_comparison(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_path: Path,
        model_name: Optional[str] = None,
        n_samples: int = 500,
    ):
        """Plot time series comparison of actual vs predicted (first n_samples).

        Args:
            y_true: True values
            y_pred: Predicted values
            output_path: Path to save plot
            model_name: Optional model name for title
            n_samples: Number of samples to plot
        """
        plt.figure(figsize=(16, 6))

        # Plot only first n_samples for clarity
        n = min(n_samples, len(y_true))
        x = np.arange(n)

        plt.plot(x, y_true[:n], label='Actual', color='blue', alpha=0.7, linewidth=1.5)
        plt.plot(x, y_pred[:n], label='Predicted', color='red', alpha=0.7, linewidth=1.5)

        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('PV Power (kW)', fontsize=12)
        title = f'Time Series Comparison (First {n} Samples)'
        if model_name:
            title = f'{model_name}: {title}'
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved time series comparison plot to {output_path}")

    @staticmethod
    def plot_error_distribution_by_magnitude(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_path: Path,
        model_name: Optional[str] = None,
        n_bins: int = 10,
    ):
        """Plot error distribution binned by actual value magnitude.

        Args:
            y_true: True values
            y_pred: Predicted values
            output_path: Path to save plot
            model_name: Optional model name for title
            n_bins: Number of bins for magnitude
        """
        residuals = y_true - y_pred
        abs_errors = np.abs(residuals)

        # Create bins based on actual values
        bins = np.linspace(y_true.min(), y_true.max(), n_bins + 1)
        bin_indices = np.digitize(y_true, bins)

        # Calculate mean absolute error per bin
        bin_centers = []
        mean_errors = []

        for i in range(1, n_bins + 1):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_centers.append((bins[i-1] + bins[i]) / 2)
                mean_errors.append(abs_errors[mask].mean())

        plt.figure(figsize=(12, 6))
        plt.bar(bin_centers, mean_errors, width=(bins[1] - bins[0]) * 0.8,
                edgecolor='black', alpha=0.7)
        plt.xlabel('Actual PV Power (kW)', fontsize=12)
        plt.ylabel('Mean Absolute Error (kW)', fontsize=12)
        title = 'Error Distribution by Power Magnitude'
        if model_name:
            title = f'{model_name}: {title}'
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved error distribution plot to {output_path}")

    @staticmethod
    def plot_all_regression_visualizations(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_dir: Path,
        model_name: str,
    ):
        """Generate all regression visualizations.

        Args:
            y_true: True values
            y_pred: Predicted values
            output_dir: Directory to save plots
            model_name: Model name for filenames and titles
        """
        logger.info(f"Generating regression visualizations for {model_name}")

        # Predictions vs Actual
        RegressionVisualizer.plot_predictions_vs_actual(
            y_true, y_pred,
            output_dir / f"{model_name}_pred_vs_actual.png",
            model_name
        )

        # Residuals
        RegressionVisualizer.plot_residuals(
            y_true, y_pred,
            output_dir / f"{model_name}_residuals.png",
            model_name
        )

        # Time series comparison
        RegressionVisualizer.plot_time_series_comparison(
            y_true, y_pred,
            output_dir / f"{model_name}_time_series.png",
            model_name
        )

        # Error by magnitude
        RegressionVisualizer.plot_error_distribution_by_magnitude(
            y_true, y_pred,
            output_dir / f"{model_name}_error_by_magnitude.png",
            model_name
        )

        logger.info(f"All visualizations saved to {output_dir}")
