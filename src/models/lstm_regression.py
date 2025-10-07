"""LSTM models for solar power regression forecasting."""

from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from src.config import Config


class LSTMRegressor(nn.Module):
    """LSTM network for regression."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.2):
        super(LSTMRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take last time step
        last_out = lstm_out[:, -1, :]
        output = self.fc(last_out)
        return output.squeeze()


class LSTMBaselineModel:
    """LSTM baseline model: GHI sequence -> Total PV."""

    def __init__(self, config: Config, lookback: int = 12):
        self.config = config
        self.lookback = lookback  # Number of time steps to look back (default: 12 * 15min = 3 hours)

        lstm_config = config.models.lstm
        self.hidden_size = lstm_config.hidden_size
        self.num_layers = lstm_config.num_layers
        self.dropout = lstm_config.dropout
        self.learning_rate = lstm_config.learning_rate
        self.batch_size = lstm_config.batch_size
        self.epochs = lstm_config.epochs
        self.patience = lstm_config.patience

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.feature_mean = None
        self.feature_std = None
        self.target_mean = None
        self.target_std = None

    def _create_sequences(self, X: np.ndarray, y: np.ndarray):
        """Create sequences for LSTM."""
        X_seq, y_seq = [], []
        for i in range(self.lookback, len(X)):
            X_seq.append(X[i - self.lookback : i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)

    def _normalize(self, X: np.ndarray, y: np.ndarray, fit: bool = False):
        """Normalize features and target."""
        if fit:
            self.feature_mean = X.mean(axis=0)
            self.feature_std = X.std(axis=0) + 1e-8
            self.target_mean = y.mean()
            self.target_std = y.std() + 1e-8

        X_norm = (X - self.feature_mean) / self.feature_std
        y_norm = (y - self.target_mean) / self.target_std
        return X_norm, y_norm

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Train LSTM model.

        Args:
            X_train: Training features (GHI)
            y_train: Training target (total PV power)

        Returns:
            Training info dictionary
        """
        logger.info(f"Training LSTM Baseline (GHI sequence -> Total PV)")
        logger.info(f"Lookback: {self.lookback} steps")
        logger.info(f"Training samples: {len(X_train)}")

        # Normalize
        X_norm, y_norm = self._normalize(X_train, y_train, fit=True)

        # Create sequences
        X_seq, y_seq = self._create_sequences(X_norm, y_norm)
        logger.info(f"Sequences created: {len(X_seq)} samples")

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).to(self.device)

        # Create model
        input_size = X_seq.shape[2]
        self.model = LSTMRegressor(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop with early stopping
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            num_batches = 0

            # Mini-batch training
            for i in range(0, len(X_tensor), self.batch_size):
                batch_X = X_tensor[i : i + self.batch_size]
                batch_y = y_tensor[i : i + self.batch_size]

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.4f}")

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        return {
            "model": "LSTMBaseline",
            "n_samples": len(X_seq),
            "lookback": self.lookback,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "final_loss": best_loss,
            "epochs_trained": epoch + 1,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict total PV power from GHI sequence.

        Args:
            X: Input features (GHI)

        Returns:
            Predicted total PV power
        """
        self.model.eval()

        # Normalize
        X_norm = (X - self.feature_mean) / self.feature_std

        # Create sequences
        X_seq, _ = self._create_sequences(X_norm, np.zeros(len(X_norm)))

        # Convert to tensor
        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        # Predict
        with torch.no_grad():
            y_pred_norm = self.model(X_tensor).cpu().numpy()

        # Denormalize
        y_pred = y_pred_norm * self.target_std + self.target_mean

        return y_pred

    def save(self, save_dir: Path, model_name: str):
        """Save model to disk."""
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / f"{model_name}.pt"

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "feature_mean": self.feature_mean,
                "feature_std": self.feature_std,
                "target_mean": self.target_mean,
                "target_std": self.target_std,
                "lookback": self.lookback,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
            },
            model_path,
        )
        logger.info(f"Model saved to {model_path}")

    def load(self, model_path: Path):
        """Load model from disk."""
        checkpoint = torch.load(model_path, map_location=self.device)

        self.lookback = checkpoint["lookback"]
        self.hidden_size = checkpoint["hidden_size"]
        self.num_layers = checkpoint["num_layers"]
        self.dropout = checkpoint["dropout"]
        self.feature_mean = checkpoint["feature_mean"]
        self.feature_std = checkpoint["feature_std"]
        self.target_mean = checkpoint["target_mean"]
        self.target_std = checkpoint["target_std"]

        input_size = len(self.feature_mean)
        self.model = LSTMRegressor(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        logger.info(f"Model loaded from {model_path}")


class LSTMTrackerModel:
    """LSTM for tracker-specific forecasting with multiple features."""

    def __init__(self, config: Config, tracker_name: str, feature_names: List[str], lookback: int = 12):
        self.config = config
        self.tracker_name = tracker_name
        self.feature_names = feature_names
        self.lookback = lookback

        lstm_config = config.models.lstm
        self.hidden_size = lstm_config.hidden_size
        self.num_layers = lstm_config.num_layers
        self.dropout = lstm_config.dropout
        self.learning_rate = lstm_config.learning_rate
        self.batch_size = lstm_config.batch_size
        self.epochs = lstm_config.epochs
        self.patience = lstm_config.patience

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.feature_mean = None
        self.feature_std = None
        self.target_mean = None
        self.target_std = None

    def _create_sequences(self, X: np.ndarray, y: np.ndarray):
        """Create sequences for LSTM."""
        X_seq, y_seq = [], []
        for i in range(self.lookback, len(X)):
            X_seq.append(X[i - self.lookback : i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)

    def _normalize(self, X: np.ndarray, y: np.ndarray, fit: bool = False):
        """Normalize features and target."""
        if fit:
            self.feature_mean = X.mean(axis=0)
            self.feature_std = X.std(axis=0) + 1e-8
            self.target_mean = y.mean()
            self.target_std = y.std() + 1e-8

        X_norm = (X - self.feature_mean) / self.feature_std
        y_norm = (y - self.target_mean) / self.target_std
        return X_norm, y_norm

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Train tracker-specific LSTM model."""
        logger.info(f"Training LSTM for {self.tracker_name}")
        logger.info(f"Features: {self.feature_names}")
        logger.info(f"Lookback: {self.lookback} steps")
        logger.info(f"Training samples: {len(X_train)}")

        # Normalize
        X_norm, y_norm = self._normalize(X_train, y_train, fit=True)

        # Create sequences
        X_seq, y_seq = self._create_sequences(X_norm, y_norm)
        logger.info(f"Sequences created: {len(X_seq)} samples")

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).to(self.device)

        # Create model
        input_size = X_seq.shape[2]
        self.model = LSTMRegressor(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            num_batches = 0

            for i in range(0, len(X_tensor), self.batch_size):
                batch_X = X_tensor[i : i + self.batch_size]
                batch_y = y_tensor[i : i + self.batch_size]

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.4f}")

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        return {
            "model": f"LSTMTracker_{self.tracker_name}",
            "n_samples": len(X_seq),
            "n_features": len(self.feature_names),
            "lookback": self.lookback,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "final_loss": best_loss,
            "epochs_trained": epoch + 1,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict tracker PV power."""
        self.model.eval()

        X_norm = (X - self.feature_mean) / self.feature_std
        X_seq, _ = self._create_sequences(X_norm, np.zeros(len(X_norm)))
        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        with torch.no_grad():
            y_pred_norm = self.model(X_tensor).cpu().numpy()

        y_pred = y_pred_norm * self.target_std + self.target_mean
        return y_pred

    def save(self, save_dir: Path, model_name: str):
        """Save model to disk."""
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / f"{model_name}.pt"

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "tracker_name": self.tracker_name,
                "feature_names": self.feature_names,
                "feature_mean": self.feature_mean,
                "feature_std": self.feature_std,
                "target_mean": self.target_mean,
                "target_std": self.target_std,
                "lookback": self.lookback,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
            },
            model_path,
        )
        logger.info(f"Model saved to {model_path}")

    def load(self, model_path: Path):
        """Load model from disk."""
        checkpoint = torch.load(model_path, map_location=self.device)

        self.tracker_name = checkpoint["tracker_name"]
        self.feature_names = checkpoint["feature_names"]
        self.lookback = checkpoint["lookback"]
        self.hidden_size = checkpoint["hidden_size"]
        self.num_layers = checkpoint["num_layers"]
        self.dropout = checkpoint["dropout"]
        self.feature_mean = checkpoint["feature_mean"]
        self.feature_std = checkpoint["feature_std"]
        self.target_mean = checkpoint["target_mean"]
        self.target_std = checkpoint["target_std"]

        input_size = len(self.feature_mean)
        self.model = LSTMRegressor(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        logger.info(f"Model loaded from {model_path}")
