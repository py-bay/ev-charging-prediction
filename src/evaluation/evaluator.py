"""Model evaluation with metrics and visualization."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


class ModelEvaluator:
    """Evaluates model predictions with metrics and visualizations."""

    def __init__(self, config):
        """Initialize evaluator with configuration.

        Args:
            config: Configuration object
        """
        self.config = config

    def calculate_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> dict:
        """Calculate regression metrics.

        Args:
            actual: Array of actual values
            predicted: Array of predicted values

        Returns:
            Dictionary with metrics
        """
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)
        max_error = np.max(np.abs(actual - predicted))

        # MAPE (avoid division by zero)
        mask = actual != 0
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

        metrics = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape,
            "max_error": max_error,
        }

        return metrics

    def plot_predicted_vs_actual(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        output_path: Path,
        dataset_name: str,
    ):
        """Create scatter plot of predicted vs actual values.

        Args:
            actual: Array of actual values
            predicted: Array of predicted values
            output_path: Directory to save plot
            dataset_name: Name of dataset
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        # Calculate R² for display
        r2 = r2_score(actual, predicted)

        # Scatter plot
        ax.scatter(actual, predicted, alpha=0.5, s=20, edgecolors="none")

        # Ideal line (y=x)
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            linewidth=2,
            label="Ideale Linie (y=x)",
        )

        # Linear regression line
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            actual, predicted
        )
        regression_line = slope * actual + intercept
        ax.plot(
            actual,
            regression_line,
            "b-",
            linewidth=2,
            label=f"Regression (R²={r2:.3f})",
        )

        # Labels
        ax.set_xlabel("Tatsächliche Produktion (W)", fontsize=14)
        ax.set_ylabel("Vorhergesagte Produktion (W)", fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        # Equal aspect ratio
        ax.set_aspect("equal", adjustable="box")

        plt.tight_layout()
        output_file = output_path / f"{dataset_name}_predicted_vs_actual.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved plot: {output_file}")

    def plot_timeseries(
        self,
        df: pd.DataFrame,
        output_path: Path,
        dataset_name: str,
        sample_size: int = 672,
    ):
        """Create time series plot of actual vs predicted.

        Args:
            df: DataFrame with timestamp, actual, predicted columns
            output_path: Directory to save plot
            dataset_name: Name of dataset
            sample_size: Number of samples to plot (672 = 1 week at 15min intervals)
        """
        # Sample random week if dataset is larger
        if len(df) > sample_size:
            start_idx = np.random.randint(0, len(df) - sample_size)
            df_sample = df.iloc[start_idx : start_idx + sample_size].copy()
        else:
            df_sample = df.copy()

        fig, ax = plt.subplots(figsize=(14, 6))

        # Convert timestamp to datetime if it's a string
        if df_sample["timestamp"].dtype == "object":
            df_sample["timestamp"] = pd.to_datetime(df_sample["timestamp"])

        # Plot lines
        ax.plot(
            df_sample["timestamp"],
            df_sample["actual"],
            label="Tatsächliche Produktion",
            linewidth=1.5,
            alpha=0.8,
        )
        ax.plot(
            df_sample["timestamp"],
            df_sample["predicted"],
            label="Vorhergesagte Produktion",
            linewidth=1.5,
            alpha=0.8,
        )

        # Labels
        ax.set_xlabel("Zeit", fontsize=14)
        ax.set_ylabel("Produktion (W)", fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        # Rotate x-axis labels
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        output_file = output_path / f"{dataset_name}_timeseries.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved plot: {output_file}")

    def plot_residuals(
        self,
        df: pd.DataFrame,
        output_path: Path,
        dataset_name: str,
    ):
        """Create residual plot over time.

        Args:
            df: DataFrame with timestamp, actual, predicted columns
            output_path: Directory to save plot
            dataset_name: Name of dataset
        """
        # Calculate residuals
        residuals = df["predicted"] - df["actual"]

        fig, ax = plt.subplots(figsize=(14, 6))

        # Convert timestamp to datetime if it's a string
        if df["timestamp"].dtype == "object":
            timestamps = pd.to_datetime(df["timestamp"])
        else:
            timestamps = df["timestamp"]

        # Plot residuals
        ax.scatter(timestamps, residuals, alpha=0.5, s=10, edgecolors="none")
        ax.axhline(y=0, color="r", linestyle="--", linewidth=2, label="Null-Linie")

        # Labels
        ax.set_xlabel("Zeit", fontsize=14)
        ax.set_ylabel("Residuen (Vorhersage - Tatsächlich) (W)", fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        # Rotate x-axis labels
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        output_file = output_path / f"{dataset_name}_residuals.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved plot: {output_file}")

    def plot_error_distribution(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        output_path: Path,
        dataset_name: str,
    ):
        """Create histogram of error distribution.

        Args:
            actual: Array of actual values
            predicted: Array of predicted values
            output_path: Directory to save plot
            dataset_name: Name of dataset
        """
        residuals = predicted - actual

        fig, ax = plt.subplots(figsize=(10, 6))

        # Histogram
        ax.hist(residuals, bins=50, edgecolor="black", alpha=0.7)

        # Vertical line at 0
        ax.axvline(x=0, color="r", linestyle="--", linewidth=2, label="Null-Linie")

        # Labels
        ax.set_xlabel("Residuen (Vorhersage - Tatsächlich) (W)", fontsize=14)
        ax.set_ylabel("Häufigkeit", fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        output_file = output_path / f"{dataset_name}_error_distribution.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved plot: {output_file}")

    def plot_error_vs_actual(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        output_path: Path,
        dataset_name: str,
    ):
        """Create plot of error vs actual values.

        Args:
            actual: Array of actual values
            predicted: Array of predicted values
            output_path: Directory to save plot
            dataset_name: Name of dataset
        """
        residuals = predicted - actual

        fig, ax = plt.subplots(figsize=(10, 8))

        # Scatter plot
        ax.scatter(actual, residuals, alpha=0.5, s=20, edgecolors="none")

        # Horizontal line at 0
        ax.axhline(y=0, color="r", linestyle="--", linewidth=2, label="Null-Linie")

        # Labels
        ax.set_xlabel("Tatsächliche Produktion (W)", fontsize=14)
        ax.set_ylabel("Residuen (Vorhersage - Tatsächlich) (W)", fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = output_path / f"{dataset_name}_error_vs_actual.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved plot: {output_file}")

    def evaluate_dataset(
        self,
        predictions_path: Path,
        output_path: Path,
        dataset_name: str,
    ) -> dict:
        """Complete evaluation pipeline for a dataset.

        Args:
            predictions_path: Path to predictions CSV file
            output_path: Directory to save results
            dataset_name: Name of dataset

        Returns:
            Dictionary with metrics
        """
        logger.info(f"Evaluating dataset: {dataset_name}")

        # Load predictions
        df = pd.read_csv(predictions_path)

        # Extract arrays
        actual = df["actual"].values
        predicted = df["predicted"].values

        # Calculate metrics
        metrics = self.calculate_metrics(actual, predicted)

        logger.info(f"Metrics for {dataset_name}:")
        logger.info(f"  MAE: {metrics['mae']:.2f} W")
        logger.info(f"  RMSE: {metrics['rmse']:.2f} W")
        logger.info(f"  R²: {metrics['r2']:.4f}")
        logger.info(f"  MAPE: {metrics['mape']:.2f} %")
        logger.info(f"  Max Error: {metrics['max_error']:.2f} W")

        # Create output directory for this dataset
        dataset_output_path = output_path / dataset_name
        dataset_output_path.mkdir(parents=True, exist_ok=True)

        # Generate all plots
        self.plot_predicted_vs_actual(actual, predicted, dataset_output_path, dataset_name)
        self.plot_timeseries(df, dataset_output_path, dataset_name)
        self.plot_residuals(df, dataset_output_path, dataset_name)
        self.plot_error_distribution(actual, predicted, dataset_output_path, dataset_name)
        self.plot_error_vs_actual(actual, predicted, dataset_output_path, dataset_name)

        # Save metrics to CSV
        metrics_df = pd.DataFrame([metrics])
        metrics_file = dataset_output_path / f"{dataset_name}_metrics.csv"
        metrics_df.to_csv(metrics_file, index=False)
        logger.info(f"Saved metrics: {metrics_file}")

        return metrics
