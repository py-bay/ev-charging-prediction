"""LSTM model for PV production forecasting."""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """LSTM model for time series forecasting."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        """
        Initialize LSTM model.

        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units in LSTM layers
            num_layers: Number of LSTM layers
            dropout: Dropout rate for regularization
        """
        super(LSTMModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # LSTM output
        # lstm_out: (batch_size, sequence_length, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state
        # h_n: (num_layers, batch_size, hidden_size)
        last_hidden = h_n[-1]  # (batch_size, hidden_size)

        # Fully connected layer
        out = self.fc(last_hidden)  # (batch_size, 1)

        return out
