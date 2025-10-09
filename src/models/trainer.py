"""LSTM model trainer with early stopping."""

import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from typing import Dict
from loguru import logger
from torch.utils.data import DataLoader

from src.config.models import Config
from src.models.lstm_model import LSTMModel
from src.models.dataset import SequenceDataset


class LSTMTrainer:
    """Trainer for LSTM model with early stopping."""

    def __init__(self, config: Config):
        """
        Initialize trainer.

        Args:
            config: Configuration object
        """
        self.config = config
        self.model_config = config.model

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Model will be initialized when loading data
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
        }

    def load_data(
        self,
        train_path: Path,
        test_path: Path,
        target_col: str,
    ) -> tuple:
        """
        Load and prepare data for training.

        Args:
            train_path: Path to training CSV
            test_path: Path to test CSV
            target_col: Name of target column

        Returns:
            Tuple of (train_loader, test_loader, input_size)
        """
        logger.info(f"Loading training data from {train_path}")
        train_df = pd.read_csv(train_path)

        logger.info(f"Loading test data from {test_path}")
        test_df = pd.read_csv(test_path)

        # Create datasets
        train_dataset = SequenceDataset(
            train_df,
            target_col=target_col,
            lookback_window=self.model_config.lookback_window,
        )

        test_dataset = SequenceDataset(
            test_df,
            target_col=target_col,
            lookback_window=self.model_config.lookback_window,
        )

        logger.info(f"Train sequences: {len(train_dataset)}")
        logger.info(f"Test sequences: {len(test_dataset)}")
        logger.info(f"Input features: {train_dataset.get_input_size()}")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.model_config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.model_config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        return train_loader, test_loader, train_dataset.get_input_size()

    def initialize_model(self, input_size: int):
        """
        Initialize model and optimizer.

        Args:
            input_size: Number of input features
        """
        logger.info("Initializing LSTM model...")
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.model_config.hidden_size,
            num_layers=self.model_config.num_layers,
            dropout=self.model_config.dropout,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.model_config.learning_rate,
        )

        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0

        for sequences, targets in train_loader:
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            outputs = self.model(sequences)
            loss = self.criterion(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate model.

        Args:
            val_loader: Validation data loader

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(sequences)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict:
        """
        Train model with early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader

        Returns:
            Training history dictionary
        """
        logger.info("Starting training...")
        logger.info(f"Epochs: {self.model_config.epochs}")
        logger.info(f"Batch size: {self.model_config.batch_size}")
        logger.info(f"Learning rate: {self.model_config.learning_rate}")
        logger.info(f"Early stopping patience: {self.model_config.patience}")

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(self.model_config.epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)

            # Validate
            val_loss = self.validate(val_loader)
            self.history["val_loss"].append(val_loss)

            logger.info(
                f"Epoch [{epoch+1}/{self.model_config.epochs}] "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                logger.info(f"New best validation loss: {best_val_loss:.6f}")
            else:
                patience_counter += 1
                logger.info(f"No improvement. Patience: {patience_counter}/{self.model_config.patience}")

                if patience_counter >= self.model_config.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info("Loaded best model weights")

        logger.success("Training completed!")
        logger.info(f"Best validation loss: {best_val_loss:.6f}")

        return self.history

    def save_model(self, output_path: Path, dataset_name: str):
        """
        Save trained model.

        Args:
            output_path: Directory to save model
            dataset_name: Name of dataset (e.g., "total", "north", "south")
        """
        output_path.mkdir(parents=True, exist_ok=True)
        model_file = output_path / f"lstm_{dataset_name}.pt"

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "input_size": self.model.input_size,  # Save input size for loading
            "model_config": {
                "hidden_size": self.model_config.hidden_size,
                "num_layers": self.model_config.num_layers,
                "dropout": self.model_config.dropout,
                "lookback_window": self.model_config.lookback_window,
            },
            "history": self.history,
        }, model_file)

        logger.info(f"Model saved: {model_file}")

    def load_model(self, model_path: Path, input_size: int):
        """
        Load trained model.

        Args:
            model_path: Path to model file
            input_size: Number of input features
        """
        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        self.initialize_model(input_size)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.history = checkpoint["history"]

        logger.info("Model loaded successfully")
