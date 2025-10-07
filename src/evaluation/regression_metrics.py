"""Regression evaluation metrics for solar forecasting models."""

from typing import Dict, Any

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class RegressionMetricsCalculator:
    """Calculate regression metrics (RMSE, MAE, R², MAPE)."""

    @staticmethod
    def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute all regression metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary with RMSE, MAE, R², MAPE
        """
        # RMSE: Root Mean Squared Error
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # MAE: Mean Absolute Error
        mae = mean_absolute_error(y_true, y_pred)

        # R²: Coefficient of Determination
        r2 = r2_score(y_true, y_pred)

        # MAPE: Mean Absolute Percentage Error
        # Avoid division by zero - only calculate for non-zero true values
        non_zero_mask = y_true != 0
        if non_zero_mask.sum() > 0:
            mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        else:
            mape = np.nan

        return {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "mape": float(mape),
        }

    @staticmethod
    def compute_additional_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Compute additional regression metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary with additional metrics
        """
        # Residuals
        residuals = y_true - y_pred

        # Mean Bias Error (MBE) - shows systematic over/under-prediction
        mbe = float(np.mean(residuals))

        # Standard deviation of residuals
        std_residuals = float(np.std(residuals))

        # Max error
        max_error = float(np.max(np.abs(residuals)))

        # Normalized RMSE (nRMSE) - RMSE / mean of true values
        mean_true = np.mean(y_true)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        if mean_true != 0:
            nrmse = float(rmse / mean_true * 100)  # in percentage
        else:
            nrmse = np.nan

        return {
            "mbe": mbe,
            "std_residuals": std_residuals,
            "max_error": max_error,
            "nrmse_percent": nrmse,
            "mean_true": float(mean_true),
            "mean_pred": float(np.mean(y_pred)),
            "std_true": float(np.std(y_true)),
            "std_pred": float(np.std(y_pred)),
        }

    @staticmethod
    def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Compute all regression metrics (standard + additional).

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Combined dictionary of all metrics
        """
        metrics = RegressionMetricsCalculator.compute_regression_metrics(y_true, y_pred)
        additional = RegressionMetricsCalculator.compute_additional_metrics(y_true, y_pred)

        return {**metrics, **additional}
