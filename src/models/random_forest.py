"""Random Forest classifier implementation."""

import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier

from src.config.models import Config


class RandomForestModel:
    """
    Random Forest classifier for EV charging window prediction.
    """

    def __init__(self, config: Config):
        """
        Initialize Random Forest model.

        Args:
            config: Configuration object
        """
        self.config = config
        self.model: RandomForestClassifier | None = None
        self.feature_names: list[str] | None = None

    def build_model(self) -> RandomForestClassifier:
        """
        Build Random Forest classifier with configured hyperparameters.

        Returns:
            RandomForestClassifier instance
        """
        rf_config = self.config.models.randomforest

        model = RandomForestClassifier(
            n_estimators=rf_config.n_estimators,
            max_depth=rf_config.max_depth,
            min_samples_split=rf_config.min_samples_split,
            min_samples_leaf=rf_config.min_samples_leaf,
            random_state=rf_config.random_state,
            n_jobs=rf_config.n_jobs,
            verbose=0,
        )

        logger.info(f"Built RandomForestClassifier with config: {rf_config.model_dump()}")
        return model

    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Train the Random Forest model.

        Args:
            x_train: Training features
            y_train: Training labels

        Returns:
            Dictionary with training info
        """
        logger.info("=" * 60)
        logger.info("Training Random Forest model")
        logger.info("=" * 60)

        self.feature_names = list(x_train.columns)

        # Build model
        self.model = self.build_model()

        # Train
        logger.info(f"Training on {len(x_train)} samples with {len(self.feature_names)} features")
        self.model.fit(x_train, y_train)

        # Get feature importance's
        feature_importance = pd.DataFrame(
            {"feature": self.feature_names, "importance": self.model.feature_importances_}
        ).sort_values("importance", ascending=False)

        logger.info("Top 10 feature importances:")
        for idx, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        logger.info("Random Forest training completed")

        return {
            "model_type": "randomforest",
            "n_features": len(self.feature_names),
            "n_samples": len(x_train),
            "feature_importances": feature_importance.to_dict("records"),
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Predicted labels
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input features

        Returns:
            Predicted probabilities (shape: [n_samples, 2])
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict_proba(X)

    def save(self, output_dir: Path, model_name: str = "randomforest"):
        """
        Save trained model to disk.

        Args:
            output_dir: Output directory
            model_name: Model file name prefix
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"{model_name}.pkl"

        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "feature_names": self.feature_names,
                    "config": self.config.models.randomforest.model_dump(),
                },
                f,
            )

        logger.info(f"Saved Random Forest model to {model_path}")

    def load(self, model_path: Path):
        """
        Load trained model from disk.

        Args:
            model_path: Path to saved model file
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.feature_names = data["feature_names"]

        logger.info(f"Loaded Random Forest model from {model_path}")
