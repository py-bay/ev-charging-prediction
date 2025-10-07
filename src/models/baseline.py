"""Baseline forecasting models: GHI → Total PV power."""

from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.config import Config


class LinearRegressionBaseline:
    """Simple linear regression: GHI → Total PV power."""

    def __init__(self, config: Config):
        self.config = config
        self.model = LinearRegression()
        self.feature_name = "ghi"  # Global Horizontal Irradiance

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Train linear regression model.

        Args:
            X_train: Training features (only GHI column expected)
            y_train: Training target (total PV power)

        Returns:
            Training info dictionary
        """
        logger.info(f"Training Linear Regression Baseline (GHI → Total PV)")
        logger.info(f"Training samples: {len(X_train)}")

        self.model.fit(X_train, y_train)

        # Training metrics
        train_score = self.model.score(X_train, y_train)

        return {
            "model": "LinearRegressionBaseline",
            "n_samples": len(X_train),
            "r2_train": train_score,
            "coefficient": float(self.model.coef_[0]) if len(self.model.coef_) == 1 else self.model.coef_.tolist(),
            "intercept": float(self.model.intercept_),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict total PV power from GHI.

        Args:
            X: Input features (GHI)

        Returns:
            Predicted total PV power
        """
        return self.model.predict(X)

    def save(self, save_dir: Path, model_name: str):
        """Save model to disk.

        Args:
            save_dir: Directory to save model
            model_name: Name for saved model file
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / f"{model_name}.pkl"
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")

    def load(self, model_path: Path):
        """Load model from disk.

        Args:
            model_path: Path to saved model file
        """
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")


class RandomForestBaseline:
    """Random Forest baseline: GHI → Total PV power."""

    def __init__(self, config: Config):
        self.config = config
        rf_config = config.models.randomforest
        self.model = RandomForestRegressor(
            n_estimators=rf_config.n_estimators,
            max_depth=rf_config.max_depth,
            min_samples_split=rf_config.min_samples_split,
            min_samples_leaf=rf_config.min_samples_leaf,
            random_state=rf_config.random_state,
            n_jobs=-1,
        )
        self.feature_name = "ghi"

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Train Random Forest baseline model.

        Args:
            X_train: Training features (only GHI column expected)
            y_train: Training target (total PV power)

        Returns:
            Training info dictionary
        """
        logger.info(f"Training Random Forest Baseline (GHI → Total PV)")
        logger.info(f"Training samples: {len(X_train)}")

        self.model.fit(X_train, y_train)

        # Training metrics
        train_score = self.model.score(X_train, y_train)

        return {
            "model": "RandomForestBaseline",
            "n_samples": len(X_train),
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "r2_train": train_score,
            "feature_importance": float(self.model.feature_importances_[0]) if len(self.model.feature_importances_) == 1 else self.model.feature_importances_.tolist(),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict total PV power from GHI.

        Args:
            X: Input features (GHI)

        Returns:
            Predicted total PV power
        """
        return self.model.predict(X)

    def save(self, save_dir: Path, model_name: str):
        """Save model to disk.

        Args:
            save_dir: Directory to save model
            model_name: Name for saved model file
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / f"{model_name}.pkl"
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")

    def load(self, model_path: Path):
        """Load model from disk.

        Args:
            model_path: Path to saved model file
        """
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
