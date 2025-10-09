"""LSTM Predictor for making predictions on test data."""

import logging
from pathlib import Path

import pandas as pd
import torch

from src.models.dataset import SequenceDataset
from src.models.lstm_model import LSTMModel

logger = logging.getLogger(__name__)


class LSTMPredictor:
    """Makes predictions using trained LSTM models."""

    def __init__(self, config):
        """Initialize predictor with configuration.

        Args:
            config: Configuration object containing model parameters
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def load_model(self, model_path: Path, dataset_name: str):
        """Load trained model.

        Args:
            model_path: Directory containing saved models
            dataset_name: Name of the dataset (total, north, south)
        """
        checkpoint_path = model_path / f"lstm_{dataset_name}.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model not found: {checkpoint_path}")

        logger.info(f"Loading model from: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Initialize model with correct architecture
        self.model = LSTMModel(
            input_size=checkpoint["input_size"],
            hidden_size=self.config.model.hidden_size,
            num_layers=self.config.model.num_layers,
            dropout=self.config.model.dropout,
        )

        # Load model weights
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded successfully")
        logger.info(f"  Input size: {checkpoint['input_size']}")
        logger.info(f"  Device: {self.device}")

    def predict(self, test_path: Path, target_col: str) -> pd.DataFrame:
        """Make predictions on test data.

        Args:
            test_path: Path to test CSV file
            target_col: Name of target column

        Returns:
            DataFrame with columns: timestamp, actual, predicted
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        logger.info(f"Loading test data from: {test_path}")

        # Load test data
        test_df = pd.read_csv(test_path)

        # Create dataset (note: data should already be scaled from processing step)
        test_dataset = SequenceDataset(
            data=test_df,
            target_col=target_col,
            lookback_window=self.config.model.lookback_window,
        )

        logger.info(f"Making predictions on {len(test_dataset)} samples...")

        # Make predictions
        predictions = []
        actuals = []
        timestamps = []

        with torch.no_grad():
            for i in range(len(test_dataset)):
                X, y = test_dataset[i]

                # Add batch dimension
                X = X.unsqueeze(0).to(self.device)

                # Predict
                pred = self.model(X)

                # Store results
                predictions.append(pred.cpu().item())
                actuals.append(y.item())

                # Get timestamp for the predicted point
                # The prediction is for the point after the lookback window
                timestamp_idx = i + self.config.model.lookback_window
                if "timestamp" in test_df.columns:
                    timestamps.append(test_df.iloc[timestamp_idx]["timestamp"])
                else:
                    # If no timestamp column, use index
                    timestamps.append(timestamp_idx)

        # Create results dataframe
        results_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "actual": actuals,
                "predicted": predictions,
            }
        )

        logger.info(f"Predictions completed: {len(results_df)} samples")

        return results_df

    def save_predictions(
        self, predictions_df: pd.DataFrame, output_path: Path, dataset_name: str
    ):
        """Save predictions to CSV.

        Args:
            predictions_df: DataFrame with predictions
            output_path: Directory to save predictions
            dataset_name: Name of the dataset (total, north, south)
        """
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"predictions_{dataset_name}.csv"

        predictions_df.to_csv(output_file, index=False)
        logger.info(f"Predictions saved to: {output_file}")

    def run_prediction(
        self,
        model_path: Path,
        test_path: Path,
        output_path: Path,
        dataset_name: str,
        target_col: str,
    ) -> pd.DataFrame:
        """Complete prediction pipeline for a dataset.

        Args:
            model_path: Directory containing trained models
            test_path: Path to test CSV file
            output_path: Directory to save predictions
            dataset_name: Name of the dataset (total, north, south)
            target_col: Name of target column

        Returns:
            DataFrame with predictions
        """
        # Load model
        self.load_model(model_path, dataset_name)

        # Make predictions
        predictions_df = self.predict(test_path, target_col)

        # Save predictions
        self.save_predictions(predictions_df, output_path, dataset_name)

        return predictions_df
