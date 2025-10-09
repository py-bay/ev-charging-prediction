"""Dataset for creating LSTM sequences."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Tuple


class SequenceDataset(Dataset):
    """PyTorch Dataset for creating time series sequences."""

    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        lookback_window: int,
        feature_cols: list = None,
    ):
        """
        Initialize sequence dataset.

        Args:
            data: DataFrame with features and target
            target_col: Name of the target column
            lookback_window: Number of past timesteps to use
            feature_cols: List of feature column names (if None, use all except target and utc)
        """
        self.lookback_window = lookback_window
        self.target_col = target_col

        # Identify feature columns
        if feature_cols is None:
            self.feature_cols = [col for col in data.columns if col not in [target_col, "utc"]]
        else:
            self.feature_cols = feature_cols

        # Extract features and target as numpy arrays
        self.features = data[self.feature_cols].values.astype(np.float32)
        self.targets = data[target_col].values.astype(np.float32)

        # Calculate number of valid sequences
        self.n_samples = len(self.features) - lookback_window

        if self.n_samples <= 0:
            raise ValueError(
                f"Dataset too small for lookback window {lookback_window}. "
                f"Need at least {lookback_window + 1} samples, got {len(self.features)}."
            )

    def __len__(self) -> int:
        """Return number of sequences."""
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sequence and its target.

        Args:
            idx: Index of the sequence

        Returns:
            Tuple of (sequence, target)
            - sequence: Tensor of shape (lookback_window, n_features)
            - target: Tensor of shape (1,)
        """
        # Get sequence of features
        seq_start = idx
        seq_end = idx + self.lookback_window
        sequence = self.features[seq_start:seq_end]

        # Get target (value at the end of the sequence)
        target = self.targets[seq_end]

        # Convert to tensors
        sequence = torch.from_numpy(sequence)
        target = torch.tensor([target], dtype=torch.float32)

        return sequence, target

    def get_feature_names(self) -> list:
        """Return list of feature column names."""
        return self.feature_cols

    def get_input_size(self) -> int:
        """Return number of input features."""
        return len(self.feature_cols)
