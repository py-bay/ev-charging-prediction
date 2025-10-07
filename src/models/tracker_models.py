"""Tracker-specific forecasting models: [GHI, DNI, DHI, Weather] → Individual tracker PV power."""

from pathlib import Path
from typing import Dict, Any, List

import joblib
import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestRegressor

from src.config import Config


class TrackerRandomForest:
    """Random Forest for individual tracker prediction with multiple features."""

    def __init__(self, config: Config, tracker_name: str, feature_names: List[str]):
        """Initialize tracker-specific Random Forest.

        Args:
            config: Configuration object
            tracker_name: Name of tracker (e.g., "Tracker1_south", "Tracker2_north")
            feature_names: List of feature names used for prediction
        """
        self.config = config
        self.tracker_name = tracker_name
        self.feature_names = feature_names

        rf_config = config.models.randomforest
        self.model = RandomForestRegressor(
            n_estimators=rf_config.n_estimators,
            max_depth=rf_config.max_depth,
            min_samples_split=rf_config.min_samples_split,
            min_samples_leaf=rf_config.min_samples_leaf,
            random_state=rf_config.random_state,
            n_jobs=-1,
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Train tracker-specific Random Forest model.

        Args:
            X_train: Training features [GHI, DNI, DHI, weather features]
            y_train: Training target (tracker PV power)

        Returns:
            Training info dictionary
        """
        logger.info(f"Training Random Forest for {self.tracker_name}")
        logger.info(f"Features: {self.feature_names}")
        logger.info(f"Training samples: {len(X_train)}")

        self.model.fit(X_train, y_train)

        # Training metrics
        train_score = self.model.score(X_train, y_train)

        # Feature importances
        feature_importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
        # Sort by importance
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

        logger.info("Feature importances:")
        for feat, importance in sorted_features[:10]:  # Top 10
            logger.info(f"  {feat}: {importance:.4f}")

        return {
            "model": f"TrackerRandomForest_{self.tracker_name}",
            "n_samples": len(X_train),
            "n_features": len(self.feature_names),
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "r2_train": train_score,
            "feature_importances": dict(sorted_features),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict tracker PV power.

        Args:
            X: Input features

        Returns:
            Predicted tracker PV power
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

        # Save model and metadata
        save_data = {
            "model": self.model,
            "tracker_name": self.tracker_name,
            "feature_names": self.feature_names,
        }
        joblib.dump(save_data, model_path)
        logger.info(f"Model saved to {model_path}")

    def load(self, model_path: Path):
        """Load model from disk.

        Args:
            model_path: Path to saved model file
        """
        save_data = joblib.load(model_path)
        self.model = save_data["model"]
        self.tracker_name = save_data["tracker_name"]
        self.feature_names = save_data["feature_names"]
        logger.info(f"Model loaded from {model_path}")


class AdvancedTrackerForecaster:
    """Combines two tracker-specific models and sums their predictions."""

    def __init__(self, tracker1_model: TrackerRandomForest, tracker2_model: TrackerRandomForest):
        """Initialize combined tracker forecaster.

        Args:
            tracker1_model: Trained model for Tracker 1 (south)
            tracker2_model: Trained model for Tracker 2 (north)
        """
        self.tracker1_model = tracker1_model
        self.tracker2_model = tracker2_model

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict total PV by summing tracker predictions.

        Args:
            X: Input features

        Returns:
            Tuple of (tracker1_pred, tracker2_pred, total_pred)
        """
        tracker1_pred = self.tracker1_model.predict(X)
        tracker2_pred = self.tracker2_model.predict(X)
        total_pred = tracker1_pred + tracker2_pred

        return tracker1_pred, tracker2_pred, total_pred

    def predict_total(self, X: np.ndarray) -> np.ndarray:
        """Predict total PV power (sum of trackers).

        Args:
            X: Input features

        Returns:
            Predicted total PV power
        """
        _, _, total_pred = self.predict(X)
        return total_pred
