"""LSTM classifier implementation using PyTorch."""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from ..config.models import Config


class LSTMClassifier(nn.Module):
    """
    LSTM neural network for binary classification.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.2):
        """
        Initialize LSTM classifier.

        Args:
            input_size: Number of input features
            hidden_size: Hidden layer size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(LSTMClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, sequence_length, input_size]

        Returns:
            Output tensor [batch_size, 1]
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Take the output from the last time step
        last_output = lstm_out[:, -1, :]

        # Apply dropout and fully connected layer
        out = self.dropout(last_output)
        out = self.fc(out)
        out = self.sigmoid(out)

        return out


class LSTMModel:
    """
    LSTM model wrapper for EV charging window prediction.
    """

    def __init__(self, config: Config):
        """
        Initialize LSTM model.

        Args:
            config: Configuration object
        """
        self.config = config
        self.model: LSTMClassifier | None = None
        self.feature_names: list[str] | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_sequences(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Create sequences for LSTM input.

        Args:
            X: Input features
            y: Labels (optional)

        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        sequence_length = self.config.models.lstm.sequence_length
        X_values = X.values
        n_samples = len(X_values)

        # Create sequences
        X_sequences = []
        y_sequences = [] if y is not None else None

        for i in range(sequence_length, n_samples):
            X_sequences.append(X_values[i - sequence_length : i])
            if y is not None:
                y_sequences.append(y.iloc[i])

        X_sequences = np.array(X_sequences)
        if y is not None:
            y_sequences = np.array(y_sequences)

        return X_sequences, y_sequences

    def build_model(self, input_size: int) -> LSTMClassifier:
        """
        Build LSTM classifier.

        Args:
            input_size: Number of input features

        Returns:
            LSTMClassifier instance
        """
        lstm_config = self.config.models.lstm

        model = LSTMClassifier(
            input_size=input_size,
            hidden_size=lstm_config.hidden_size,
            num_layers=lstm_config.num_layers,
            dropout=lstm_config.dropout,
        )

        model = model.to(self.device)
        logger.info(f"Built LSTM model on device: {self.device}")
        logger.info(f"Model config: {lstm_config.model_dump()}")

        return model

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Train the LSTM model.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Dictionary with training info
        """
        logger.info("=" * 60)
        logger.info("Training LSTM model")
        logger.info("=" * 60)

        self.feature_names = list(X_train.columns)
        lstm_config = self.config.models.lstm

        # Create sequences
        logger.info(f"Creating sequences with length={lstm_config.sequence_length}")
        X_seq, y_seq = self.create_sequences(X_train, y_train)
        logger.info(f"Created {len(X_seq)} sequences from {len(X_train)} samples")

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).unsqueeze(1).to(self.device)

        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=lstm_config.batch_size, shuffle=True)

        # Build model
        self.model = self.build_model(input_size=X_seq.shape[2])

        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=lstm_config.learning_rate,
            weight_decay=lstm_config.weight_decay,
        )

        # Training loop
        best_loss = float("inf")
        patience_counter = 0
        train_losses = []

        for epoch in range(lstm_config.epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # Calculate average loss
            avg_loss = epoch_loss / n_batches
            train_losses.append(avg_loss)

            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch [{epoch+1}/{lstm_config.epochs}], Loss: {avg_loss:.4f}")

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= lstm_config.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        logger.info(f"LSTM training completed. Best loss: {best_loss:.4f}")

        return {
            "model_type": "lstm",
            "n_features": len(self.feature_names),
            "n_sequences": len(X_seq),
            "epochs_trained": epoch + 1,
            "final_loss": avg_loss,
            "best_loss": best_loss,
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

        self.model.eval()

        # Create sequences
        X_seq, _ = self.create_sequences(X)

        if len(X_seq) == 0:
            logger.warning("Not enough data to create sequences for prediction")
            return np.array([])

        # Convert to tensor
        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = (outputs.cpu().numpy() > 0.5).astype(int).flatten()

        return predictions

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

        self.model.eval()

        # Create sequences
        X_seq, _ = self.create_sequences(X)

        if len(X_seq) == 0:
            logger.warning("Not enough data to create sequences for prediction")
            return np.array([])

        # Convert to tensor
        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(X_tensor).cpu().numpy()
            # Return [prob_class_0, prob_class_1]
            proba = np.hstack([1 - outputs, outputs])

        return proba

    def save(self, output_dir: Path, model_name: str = "lstm"):
        """
        Save trained model to disk.

        Args:
            output_dir: Output directory
            model_name: Model file name prefix
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"{model_name}.pt"

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "feature_names": self.feature_names,
                "config": self.config.models.lstm.model_dump(),
                "input_size": len(self.feature_names),
            },
            model_path,
        )

        logger.info(f"Saved LSTM model to {model_path}")

    def load(self, model_path: Path):
        """
        Load trained model from disk.

        Args:
            model_path: Path to saved model file
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)

        self.feature_names = checkpoint["feature_names"]
        input_size = checkpoint["input_size"]

        self.model = self.build_model(input_size)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        logger.info(f"Loaded LSTM model from {model_path}")
