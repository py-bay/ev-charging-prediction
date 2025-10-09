"""
Separate Tracker Modeling Comparison

Research Question:
Are the individual tracker errors with separate modeling smaller than those
implicitly derived from the baseline, even if the sum becomes worse?

Approach:
1. Baseline: Single LSTM predicts total PV only (no tracker knowledge)
   - Derive tracker values using historical ratio for evaluation
2. Separate: Two independent LSTMs predict North and South trackers
   - Accept: pred_north + pred_south ≠ true_total (errors add up)

Key Comparison:
- Tracker accuracy: Baseline (derived) vs. Separate (predicted)
- Total accuracy: Baseline (perfect sum) vs. Separate (inconsistent sum)
"""

import json
import random
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger

# Suppress matplotlib interactive mode
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.config.models import load_config


class BaselineTotalLSTM(nn.Module):
    """LSTM for predicting total PV production only (no tracker knowledge)."""

    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)  # Single output: total

    def forward(self, x):
        # x shape: (batch, sequence, features)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Last timestep
        return out.squeeze(-1)


class SeparateTrackerLSTM(nn.Module):
    """LSTM for predicting a single tracker (North OR South)."""

    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)  # Single output: north OR south

    def forward(self, x):
        # x shape: (batch, sequence, features)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out.squeeze(-1)


class SeparateTrackerExperiment:
    """Comparison experiment: Baseline (Total-only) vs. Separate Tracker Models."""

    def __init__(self):
        # Experiment parameters
        self.lookback = 48  # 24 hours at 15-min resolution
        self.horizon = 24   # 6 hours ahead
        self.test_fraction = 0.2
        self.seed = 42

        # Model hyperparameters (identical for fair comparison)
        self.hidden_size = 32
        self.num_layers = 2
        self.dropout = 0.2
        self.learning_rate = 0.001
        self.epochs = 20
        self.batch_size = 64

        # Paths
        self.output_dir = Path("outputs/experiment_separate_tracker")
        self.plots_dir = self.output_dir / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Data containers
        self.df: pd.DataFrame = None
        self.scaler = None
        self.feature_cols: List[str] = []

        # Training data
        self.X_train: np.ndarray = None
        self.X_test: np.ndarray = None
        self.y_train_total: np.ndarray = None
        self.y_test_total: np.ndarray = None
        self.y_train_north: np.ndarray = None
        self.y_test_north: np.ndarray = None
        self.y_train_south: np.ndarray = None
        self.y_test_south: np.ndarray = None

        # Models
        self.model_baseline: BaselineTotalLSTM = None
        self.model_north: SeparateTrackerLSTM = None
        self.model_south: SeparateTrackerLSTM = None

        # Predictions
        self.pred_baseline_total: np.ndarray = None
        self.pred_baseline_north_derived: np.ndarray = None
        self.pred_baseline_south_derived: np.ndarray = None
        self.pred_separate_north: np.ndarray = None
        self.pred_separate_south: np.ndarray = None
        self.pred_separate_total_sum: np.ndarray = None

        # Historical ratio (for baseline derivation)
        self.historical_ratio_south: float = None

        # Set random seeds
        self._set_seeds()

    def _set_seeds(self):
        """Set all random seeds for reproducibility."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Set all random seeds to {self.seed}")

    def load_and_preprocess(self):
        """Load data, create features, apply chronological split, and normalize."""
        logger.info("=" * 80)
        logger.info("LOADING AND PREPROCESSING DATA")
        logger.info("=" * 80)

        # Load configuration
        config = load_config(Path("config/config.yaml"))

        # Load PV data
        logger.info("Loading PV data...")
        pv_path = config.data_paths.pv
        pv_df = pd.read_csv(pv_path, sep=";")

        # Detect timestamp column
        timestamp_col = None
        for col in ["timestamp", "time", "datetime", "Timestamp"]:
            if col in pv_df.columns:
                timestamp_col = col
                break

        if timestamp_col is None:
            raise ValueError(f"Could not find timestamp column in {pv_df.columns}")

        pv_df["timestamp"] = pd.to_datetime(pv_df[timestamp_col], utc=True)
        pv_df = pv_df.set_index("timestamp").sort_index()

        # Map German columns
        column_mapping = {
            "Solarproduktion": "pv_total",
            "Solarproduktion Tracker 1": "pv_north",
            "Solarproduktion Tracker 2": "pv_south",
        }
        pv_df = pv_df.rename(columns=column_mapping)

        # Select only the columns we need
        pv_df = pv_df[["pv_total", "pv_north", "pv_south"]]

        logger.info(f"Loaded PV data: {len(pv_df)} rows")
        logger.info(f"  Date range: {pv_df.index.min()} to {pv_df.index.max()}")

        # Load irradiance data
        logger.info("Loading irradiance data...")
        irradiance_path = config.data_paths.irradiance
        irradiance_df = pd.read_csv(irradiance_path)

        # Detect timestamp column
        timestamp_col = None
        for col in ["dt_iso", "timestamp", "time", "datetime"]:
            if col in irradiance_df.columns:
                timestamp_col = col
                break

        if timestamp_col is None:
            raise ValueError(f"Could not find timestamp column in {irradiance_df.columns}")

        # Clean UTC timestamp format
        if irradiance_df[timestamp_col].dtype == object:
            irradiance_df[timestamp_col] = irradiance_df[timestamp_col].str.replace(
                r" \+\d{4} UTC$", "", regex=True
            )

        irradiance_df["timestamp"] = pd.to_datetime(irradiance_df[timestamp_col], utc=True)
        irradiance_df = irradiance_df.set_index("timestamp").sort_index()

        # Extract GHI, DNI, DHI
        irradiance_cols = {}
        if "ghi_cloudy_sky" in irradiance_df.columns:
            irradiance_cols["ghi"] = irradiance_df["ghi_cloudy_sky"]
        if "dni_cloudy_sky" in irradiance_df.columns:
            irradiance_cols["dni"] = irradiance_df["dni_cloudy_sky"]
        if "dhi_cloudy_sky" in irradiance_df.columns:
            irradiance_cols["dhi"] = irradiance_df["dhi_cloudy_sky"]

        irradiance_df = pd.DataFrame(irradiance_cols)
        logger.info(f"Loaded irradiance data: {len(irradiance_df)} rows")

        # Load weather data
        logger.info("Loading weather data...")
        weather_path = config.data_paths.weather
        weather_df = pd.read_csv(weather_path, skiprows=2)

        # Detect timestamp column
        timestamp_col = None
        for col in ["time", "timestamp", "datetime"]:
            if col in weather_df.columns:
                timestamp_col = col
                break

        if timestamp_col is None:
            raise ValueError(f"Could not find timestamp column in {weather_df.columns}")

        weather_df["timestamp"] = pd.to_datetime(weather_df[timestamp_col], utc=True)
        weather_df = weather_df.set_index("timestamp").sort_index()

        # Clean column names
        weather_df.columns = [col.split(" (")[0].strip() for col in weather_df.columns]

        # Resample weather from hourly to 15-min
        logger.info("Resampling weather data to 15-min intervals...")
        categorical_cols = ["weather_code"] if "weather_code" in weather_df.columns else []
        continuous_cols = [col for col in weather_df.columns if col not in categorical_cols]

        weather_resampled = weather_df[continuous_cols].resample("15min").interpolate(method="time")
        if categorical_cols:
            categorical_resampled = weather_df[categorical_cols].resample("15min").ffill()
            weather_resampled = pd.concat([weather_resampled, categorical_resampled], axis=1)

        logger.info(f"Resampled weather: {len(weather_resampled)} rows")

        # Merge all data
        logger.info("Merging all data sources...")
        common_index = pv_df.index.intersection(irradiance_df.index).intersection(weather_resampled.index)
        logger.info(f"Common timestamps: {len(common_index)}")

        df = pd.concat(
            [pv_df.loc[common_index], irradiance_df.loc[common_index], weather_resampled.loc[common_index]],
            axis=1
        )

        # Select weather features to use
        weather_features = ["temperature_2m", "relative_humidity_2m", "cloud_cover",
                           "wind_speed_10m", "pressure_msl"]
        weather_features = [f for f in weather_features if f in df.columns]

        # Drop the 'time' column if it exists
        if "time" in df.columns:
            df = df.drop(columns=["time"])

        # Add time features
        logger.info("Adding time features...")
        df["hour"] = df.index.hour
        df["dayofyear"] = df.index.dayofyear
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["day_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
        df["day_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

        # Drop intermediate time features
        df = df.drop(columns=["hour", "dayofyear"])

        # Drop any NaN
        df = df.dropna()

        logger.info(f"Final merged data: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")

        # Define feature columns (only important features)
        target_cols = ["pv_total", "pv_north", "pv_south"]
        important_features = [
            'ghi', 'dni', 'dhi', 'temperature_2m', 'cloud_cover',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
        ]

        # Filter to only include features that exist in the dataframe
        self.feature_cols = [col for col in important_features if col in df.columns]

        logger.info(f"Feature columns ({len(self.feature_cols)}): {self.feature_cols}")

        # Warn if any important features are missing
        missing_features = set(important_features) - set(self.feature_cols)
        if missing_features:
            logger.warning(f"Missing features from dataframe: {missing_features}")

        # Apply 6-hour forecast shift (shift features back 24 steps)
        logger.info(f"Applying 6-hour forecast shift (horizon={self.horizon} steps)...")
        df_shifted = df.copy()
        for col in self.feature_cols:
            df_shifted[col] = df[col].shift(self.horizon)

        # Drop NaN from shift
        df_shifted = df_shifted.dropna()
        logger.info(f"After shift: {len(df_shifted)} rows")

        self.df = df_shifted

        # Chronological split
        logger.info(f"Performing chronological split (test_fraction={self.test_fraction})...")
        split_idx = int(len(self.df) * (1 - self.test_fraction))
        df_train = self.df.iloc[:split_idx]
        df_test = self.df.iloc[split_idx:]

        logger.info(f"Train set: {len(df_train)} samples ({df_train.index.min()} to {df_train.index.max()})")
        logger.info(f"Test set: {len(df_test)} samples ({df_test.index.min()} to {df_test.index.max()})")

        # Berechne das Ratio basierend auf Gesamtenergie (wie im Dokument)
        total_energy_south = df_train["pv_south"].sum()
        total_energy_north = df_train["pv_north"].sum()
        total_energy_combined = total_energy_south + total_energy_north

        self.historical_ratio_south = total_energy_south / total_energy_combined

        logger.info(f"\n📊 Historical South Ratio (energy-based): {self.historical_ratio_south:.4f}")
        logger.info(f"   South total: {total_energy_south:.2f}, North total: {total_energy_north:.2f}")
        logger.info(f"   This should match ~65.5% from the document")

        # Normalize with single scaler
        logger.info("Normalizing data with MinMaxScaler...")
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler()

        # Fit scaler on training data only
        df_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(df_train),
            columns=df_train.columns,
            index=df_train.index
        )
        df_test_scaled = pd.DataFrame(
            self.scaler.transform(df_test),
            columns=df_test.columns,
            index=df_test.index
        )

        # Create supervised datasets
        logger.info(f"Creating supervised datasets (lookback={self.lookback})...")
        self.X_train, self.y_train_total, self.y_train_north, self.y_train_south = self._create_sequences(
            df_train_scaled[self.feature_cols].values,
            df_train_scaled["pv_total"].values,
            df_train_scaled["pv_north"].values,
            df_train_scaled["pv_south"].values
        )
        self.X_test, self.y_test_total, self.y_test_north, self.y_test_south = self._create_sequences(
            df_test_scaled[self.feature_cols].values,
            df_test_scaled["pv_total"].values,
            df_test_scaled["pv_north"].values,
            df_test_scaled["pv_south"].values
        )

        logger.info(f"Datasets created:")
        logger.info(f"  X_train: {self.X_train.shape}")
        logger.info(f"  X_test: {self.X_test.shape}")
        logger.info(f"  Targets: Total, North, South")

    def _create_sequences(self, X: np.ndarray, y_total: np.ndarray,
                         y_north: np.ndarray, y_south: np.ndarray) -> Tuple:
        """Create supervised learning sequences with lookback window."""
        X_seq = []
        y_total_seq = []
        y_north_seq = []
        y_south_seq = []

        for i in range(self.lookback, len(X)):
            X_seq.append(X[i - self.lookback:i])
            y_total_seq.append(y_total[i])
            y_north_seq.append(y_north[i])
            y_south_seq.append(y_south[i])

        return (np.array(X_seq), np.array(y_total_seq),
                np.array(y_north_seq), np.array(y_south_seq))

    def train_baseline_model(self):
        """Train baseline model that predicts total PV only (no tracker knowledge)."""
        logger.info("=" * 80)
        logger.info("TRAINING BASELINE MODEL (TOTAL-ONLY)")
        logger.info("=" * 80)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        input_dim = self.X_train.shape[2]
        self.model_baseline = BaselineTotalLSTM(
            input_size=input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(device)

        logger.info(f"Model architecture:")
        logger.info(f"  Input size: {input_dim}")
        logger.info(f"  Hidden size: {self.hidden_size}")
        logger.info(f"  Num layers: {self.num_layers}")
        logger.info(f"  Output: Total PV (no tracker information)")

        # Train the model
        self._train_model(
            self.model_baseline,
            self.X_train,
            self.y_train_total,
            device,
            "Baseline Total"
        )

        # Generate predictions
        logger.info("\nGenerating baseline predictions...")
        self.pred_baseline_total = self._predict(self.model_baseline, self.X_test, device)

        # Derive tracker predictions using historical ratio
        logger.info("\n📊 DERIVING tracker predictions from baseline total...")
        logger.info(f"   Using historical ratio: South = {self.historical_ratio_south:.4f}")
        logger.info(f"   (These are NOT predicted, but derived for evaluation!)")

        self.pred_baseline_south_derived = self.pred_baseline_total * self.historical_ratio_south
        self.pred_baseline_north_derived = self.pred_baseline_total * (1 - self.historical_ratio_south)

        logger.info(f"✓ Baseline predictions generated:")
        logger.info(f"  Total (predicted): {len(self.pred_baseline_total)} values")
        logger.info(f"  North (derived*): {len(self.pred_baseline_north_derived)} values")
        logger.info(f"  South (derived*): {len(self.pred_baseline_south_derived)} values")

    def train_separate_models(self):
        """Train two independent models for North and South trackers."""
        logger.info("=" * 80)
        logger.info("TRAINING SEPARATE TRACKER MODELS")
        logger.info("=" * 80)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        input_dim = self.X_train.shape[2]

        # North model
        logger.info("\n--- Training North Tracker Model ---")
        self.model_north = SeparateTrackerLSTM(
            input_size=input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(device)

        self._train_model(
            self.model_north,
            self.X_train,
            self.y_train_north,
            device,
            "North Tracker"
        )

        # South model
        logger.info("\n--- Training South Tracker Model ---")
        self.model_south = SeparateTrackerLSTM(
            input_size=input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(device)

        self._train_model(
            self.model_south,
            self.X_train,
            self.y_train_south,
            device,
            "South Tracker"
        )

        # Generate predictions
        logger.info("\nGenerating separate tracker predictions...")
        self.pred_separate_north = self._predict(self.model_north, self.X_test, device)
        self.pred_separate_south = self._predict(self.model_south, self.X_test, device)

        # Calculate sum (will likely be inconsistent with true total)
        self.pred_separate_total_sum = self.pred_separate_north + self.pred_separate_south

        logger.info(f"✓ Separate predictions generated:")
        logger.info(f"  North (predicted): {len(self.pred_separate_north)} values")
        logger.info(f"  South (predicted): {len(self.pred_separate_south)} values")
        logger.info(f"  Sum: {len(self.pred_separate_total_sum)} values")
        logger.info(f"  NOTE: Sum may not match true total (expected trade-off)")

    def _train_model(self, model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                     device, model_name: str):
        """Train a single model."""
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.FloatTensor(y_train).to(device)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # Training loop
        model.train()
        for epoch in range(self.epochs):
            # Mini-batch training
            indices = np.arange(len(X_train))
            np.random.shuffle(indices)

            epoch_loss = 0.0
            n_batches = 0

            for start_idx in range(0, len(X_train), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(X_train))
                batch_indices = indices[start_idx:end_idx]

                X_batch = X_train_t[batch_indices]
                y_batch = y_train_t[batch_indices]

                # Forward pass
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches

            if (epoch + 1) % 5 == 0:
                logger.info(f"{model_name} - Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.6f}")

    def _predict(self, model: nn.Module, X_test: np.ndarray, device) -> np.ndarray:
        """Generate predictions."""
        model.eval()
        X_test_t = torch.FloatTensor(X_test).to(device)

        with torch.no_grad():
            y_pred_t = model(X_test_t)
            y_pred = y_pred_t.cpu().numpy()

        return y_pred

    def evaluate(self):
        """Compute metrics and compare both approaches."""
        logger.info("=" * 80)
        logger.info("EVALUATION AND COMPARISON")
        logger.info("=" * 80)

        # Inverse transform all predictions and targets
        all_cols = list(self.df.columns)
        n_features = len(all_cols)

        idx_total = all_cols.index("pv_total")
        idx_north = all_cols.index("pv_north")
        idx_south = all_cols.index("pv_south")

        logger.info(f"Inverse transforming predictions...")

        # Helper function for inverse transform
        def inverse_transform_single(values, col_idx):
            dummy = np.zeros((len(values), n_features))
            dummy[:, col_idx] = values
            return self.scaler.inverse_transform(dummy)[:, col_idx]

        # Inverse transform ground truth
        y_true_total = inverse_transform_single(self.y_test_total, idx_total)
        y_true_north = inverse_transform_single(self.y_test_north, idx_north)
        y_true_south = inverse_transform_single(self.y_test_south, idx_south)

        # Inverse transform baseline predictions
        pred_baseline_total_inv = inverse_transform_single(self.pred_baseline_total, idx_total)
        pred_baseline_north_inv = inverse_transform_single(self.pred_baseline_north_derived, idx_north)
        pred_baseline_south_inv = inverse_transform_single(self.pred_baseline_south_derived, idx_south)

        # Inverse transform separate predictions
        pred_separate_north_inv = inverse_transform_single(self.pred_separate_north, idx_north)
        pred_separate_south_inv = inverse_transform_single(self.pred_separate_south, idx_south)
        pred_separate_total_inv = inverse_transform_single(self.pred_separate_total_sum, idx_total)

        # Compute metrics for baseline
        logger.info("\n--- Baseline Approach Metrics ---")
        metrics_baseline_total = self._compute_metrics(y_true_total, pred_baseline_total_inv, "Baseline Total")
        metrics_baseline_north = self._compute_metrics(y_true_north, pred_baseline_north_inv, "Baseline North (derived*)")
        metrics_baseline_south = self._compute_metrics(y_true_south, pred_baseline_south_inv, "Baseline South (derived*)")

        # Compute metrics for separate models
        logger.info("\n--- Separate Models Metrics ---")
        metrics_separate_north = self._compute_metrics(y_true_north, pred_separate_north_inv, "Separate North")
        metrics_separate_south = self._compute_metrics(y_true_south, pred_separate_south_inv, "Separate South")
        metrics_separate_total = self._compute_metrics(y_true_total, pred_separate_total_inv, "Separate Total (sum)")

        # Calculate sum inconsistency for separate approach
        sum_inconsistency = np.abs(pred_separate_total_inv - y_true_total)
        sum_inconsistency_mae = np.mean(sum_inconsistency)
        sum_inconsistency_max = np.max(sum_inconsistency)
        sum_inconsistency_pct = (sum_inconsistency_mae / np.mean(y_true_total)) * 100

        logger.info(f"\n📊 SUM INCONSISTENCY (Separate Approach):")
        logger.info(f"   MAE: {sum_inconsistency_mae:.2f} W ({sum_inconsistency_pct:.2f}%)")
        logger.info(f"   Max: {sum_inconsistency_max:.2f} W")

        # Calculate improvements
        logger.info("\n" + "=" * 80)
        logger.info("TRACKER ACCURACY COMPARISON")
        logger.info("=" * 80)

        def calc_improvement(baseline_val, separate_val):
            return ((baseline_val - separate_val) / baseline_val) * 100 if baseline_val != 0 else 0

        north_rmse_improvement = calc_improvement(metrics_baseline_north['rmse'], metrics_separate_north['rmse'])
        north_mae_improvement = calc_improvement(metrics_baseline_north['mae'], metrics_separate_north['mae'])
        north_r2_improvement = ((metrics_separate_north['r2'] - metrics_baseline_north['r2']) /
                                 abs(metrics_baseline_north['r2'])) * 100 if metrics_baseline_north['r2'] != 0 else 0

        south_rmse_improvement = calc_improvement(metrics_baseline_south['rmse'], metrics_separate_south['rmse'])
        south_mae_improvement = calc_improvement(metrics_baseline_south['mae'], metrics_separate_south['mae'])
        south_r2_improvement = ((metrics_separate_south['r2'] - metrics_baseline_south['r2']) /
                                 abs(metrics_baseline_south['r2'])) * 100 if metrics_baseline_south['r2'] != 0 else 0

        logger.info(f"\nNorth Tracker:")
        logger.info(f"  Baseline (derived*): RMSE = {metrics_baseline_north['rmse']:.2f} W, R² = {metrics_baseline_north['r2']:.4f}")
        logger.info(f"  Separate (predicted): RMSE = {metrics_separate_north['rmse']:.2f} W, R² = {metrics_separate_north['r2']:.4f}")
        logger.info(f"  → Improvement: {north_rmse_improvement:+.1f}% RMSE, {north_r2_improvement:+.1f}% R²")

        logger.info(f"\nSouth Tracker:")
        logger.info(f"  Baseline (derived*): RMSE = {metrics_baseline_south['rmse']:.2f} W, R² = {metrics_baseline_south['r2']:.4f}")
        logger.info(f"  Separate (predicted): RMSE = {metrics_separate_south['rmse']:.2f} W, R² = {metrics_separate_south['r2']:.4f}")
        logger.info(f"  → Improvement: {south_rmse_improvement:+.1f}% RMSE, {south_r2_improvement:+.1f}% R²")

        logger.info("\n" + "=" * 80)
        logger.info("TOTAL ACCURACY TRADE-OFF")
        logger.info("=" * 80)

        total_rmse_change = calc_improvement(metrics_baseline_total['rmse'], metrics_separate_total['rmse'])
        total_r2_change = ((metrics_separate_total['r2'] - metrics_baseline_total['r2']) /
                           abs(metrics_baseline_total['r2'])) * 100 if metrics_baseline_total['r2'] != 0 else 0

        logger.info(f"\nBaseline Total:  RMSE = {metrics_baseline_total['rmse']:.2f} W, R² = {metrics_baseline_total['r2']:.4f}")
        logger.info(f"Separate Sum:    RMSE = {metrics_separate_total['rmse']:.2f} W, R² = {metrics_separate_total['r2']:.4f}")
        logger.info(f"Inconsistency:   MAE = ±{sum_inconsistency_mae:.2f} W ({sum_inconsistency_pct:.1f}%)")
        logger.info(f"→ Total accuracy change: {total_rmse_change:+.1f}% RMSE, {total_r2_change:+.1f}% R²")

        # Save all results
        results = {
            "experiment_config": {
                "approach": "baseline_vs_separate_trackers",
                "seed": self.seed,
                "lookback": self.lookback,
                "horizon": self.horizon,
                "split": "chronological_80_20",
                "test_fraction": self.test_fraction,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "historical_ratio_south": float(self.historical_ratio_south)
            },
            "baseline": {
                "total": metrics_baseline_total,
                "north_derived": metrics_baseline_north,
                "south_derived": metrics_baseline_south
            },
            "separate": {
                "north": metrics_separate_north,
                "south": metrics_separate_south,
                "total_sum": metrics_separate_total,
                "sum_inconsistency": {
                    "mae": float(sum_inconsistency_mae),
                    "max": float(sum_inconsistency_max),
                    "percentage": float(sum_inconsistency_pct)
                }
            },
            "comparison": {
                "north_improvement": {
                    "rmse_pct": float(north_rmse_improvement),
                    "mae_pct": float(north_mae_improvement),
                    "r2_pct": float(north_r2_improvement)
                },
                "south_improvement": {
                    "rmse_pct": float(south_rmse_improvement),
                    "mae_pct": float(south_mae_improvement),
                    "r2_pct": float(south_r2_improvement)
                },
                "total_tradeoff": {
                    "rmse_pct": float(total_rmse_change),
                    "r2_pct": float(total_r2_change)
                }
            }
        }

        # Save JSON
        results_path = self.output_dir / "metrics.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\n✓ Saved results to {results_path}")

        # Generate visualizations
        logger.info("\nGenerating visualizations...")
        self._create_visualizations(
            y_true_total, y_true_north, y_true_south,
            pred_baseline_total_inv, pred_baseline_north_inv, pred_baseline_south_inv,
            pred_separate_total_inv, pred_separate_north_inv, pred_separate_south_inv
        )

        # Generate summary report
        self._generate_summary(results)

        # Print final conclusion
        logger.info("\n" + "=" * 80)
        logger.info("CONCLUSION")
        logger.info("=" * 80)

        avg_tracker_improvement = (north_rmse_improvement + south_rmse_improvement) / 2

        if avg_tracker_improvement > 0:
            logger.info(f"✓ Separate modeling IMPROVES tracker accuracy by {avg_tracker_improvement:.1f}% on average")
        else:
            logger.info(f"✗ Separate modeling WORSENS tracker accuracy by {abs(avg_tracker_improvement):.1f}% on average")

        if total_rmse_change < 0:
            logger.info(f"✗ Total accuracy degrades by {abs(total_rmse_change):.1f}% due to error accumulation")
        else:
            logger.info(f"✓ Total accuracy improves by {total_rmse_change:.1f}%")

        logger.info(f"\nTrade-off: {avg_tracker_improvement:+.1f}% individual tracker accuracy")
        logger.info(f"           vs. {total_rmse_change:+.1f}% total accuracy")
        logger.info(f"\nResearch question is {'POSITIVE' if avg_tracker_improvement > 0 else 'NEGATIVE'} to answer.")

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, name: str) -> Dict:
        """Compute regression metrics."""
        # RMSE
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        # MAE
        mae = np.mean(np.abs(y_true - y_pred))

        # R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # MAPE (skip zeros)
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else 0.0

        # NRMSE
        nrmse = (rmse / np.mean(y_true)) * 100 if np.mean(y_true) > 0 else 0.0

        # Mean Bias
        mean_bias = np.mean(y_pred - y_true)

        logger.info(f"\n{name}:")
        logger.info(f"  RMSE: {rmse:.2f} W")
        logger.info(f"  MAE: {mae:.2f} W")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  MAPE: {mape:.2f}%")
        logger.info(f"  NRMSE: {nrmse:.2f}%")
        logger.info(f"  Mean Bias: {mean_bias:.2f} W")

        return {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "mape": float(mape),
            "nrmse": float(nrmse),
            "mean_bias": float(mean_bias)
        }

    def _create_visualizations(self, y_true_total, y_true_north, y_true_south,
                              pred_baseline_total, pred_baseline_north, pred_baseline_south,
                              pred_separate_total, pred_separate_north, pred_separate_south):
        """Generate all comparison visualizations."""

        # 1. Scatter plots comparison (2x3 grid)
        logger.info("  Creating scatter_comparison.png...")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Row 1: Baseline
        axes[0, 0].scatter(y_true_total, pred_baseline_total, alpha=0.3, s=10, color='blue')
        axes[0, 0].plot([y_true_total.min(), y_true_total.max()],
                       [y_true_total.min(), y_true_total.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel("Actual Total (W)")
        axes[0, 0].set_ylabel("Predicted Total (W)")
        axes[0, 0].set_title("Baseline: Total (predicted)")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].scatter(y_true_north, pred_baseline_north, alpha=0.3, s=10, color='blue')
        axes[0, 1].plot([y_true_north.min(), y_true_north.max()],
                       [y_true_north.min(), y_true_north.max()], 'r--', lw=2)
        axes[0, 1].set_xlabel("Actual North (W)")
        axes[0, 1].set_ylabel("Predicted North (W)")
        axes[0, 1].set_title("Baseline: North (derived*)")
        axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].scatter(y_true_south, pred_baseline_south, alpha=0.3, s=10, color='blue')
        axes[0, 2].plot([y_true_south.min(), y_true_south.max()],
                       [y_true_south.min(), y_true_south.max()], 'r--', lw=2)
        axes[0, 2].set_xlabel("Actual South (W)")
        axes[0, 2].set_ylabel("Predicted South (W)")
        axes[0, 2].set_title("Baseline: South (derived*)")
        axes[0, 2].grid(True, alpha=0.3)

        # Row 2: Separate
        axes[1, 0].scatter(y_true_total, pred_separate_total, alpha=0.3, s=10, color='green')
        axes[1, 0].plot([y_true_total.min(), y_true_total.max()],
                       [y_true_total.min(), y_true_total.max()], 'r--', lw=2)
        axes[1, 0].set_xlabel("Actual Total (W)")
        axes[1, 0].set_ylabel("Predicted Total (W)")
        axes[1, 0].set_title("Separate: Total (sum)")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].scatter(y_true_north, pred_separate_north, alpha=0.3, s=10, color='green')
        axes[1, 1].plot([y_true_north.min(), y_true_north.max()],
                       [y_true_north.min(), y_true_north.max()], 'r--', lw=2)
        axes[1, 1].set_xlabel("Actual North (W)")
        axes[1, 1].set_ylabel("Predicted North (W)")
        axes[1, 1].set_title("Separate: North (predicted)")
        axes[1, 1].grid(True, alpha=0.3)

        axes[1, 2].scatter(y_true_south, pred_separate_south, alpha=0.3, s=10, color='green')
        axes[1, 2].plot([y_true_south.min(), y_true_south.max()],
                       [y_true_south.min(), y_true_south.max()], 'r--', lw=2)
        axes[1, 2].set_xlabel("Actual South (W)")
        axes[1, 2].set_ylabel("Predicted South (W)")
        axes[1, 2].set_title("Separate: South (predicted)")
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / "scatter_comparison.png", dpi=150)
        plt.close()

        # 2. Direct tracker comparison
        logger.info("  Creating tracker_direct_comparison.png...")
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        axes[0].scatter(y_true_north, pred_baseline_north, alpha=0.3, s=10,
                       color='blue', label='Baseline (derived*)')
        axes[0].scatter(y_true_north, pred_separate_north, alpha=0.3, s=10,
                       color='green', label='Separate (predicted)')
        axes[0].plot([y_true_north.min(), y_true_north.max()],
                    [y_true_north.min(), y_true_north.max()], 'r--', lw=2, label='Perfect')
        axes[0].set_xlabel("Actual North (W)")
        axes[0].set_ylabel("Predicted North (W)")
        axes[0].set_title("North Tracker: Baseline vs. Separate")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].scatter(y_true_south, pred_baseline_south, alpha=0.3, s=10,
                       color='blue', label='Baseline (derived*)')
        axes[1].scatter(y_true_south, pred_separate_south, alpha=0.3, s=10,
                       color='green', label='Separate (predicted)')
        axes[1].plot([y_true_south.min(), y_true_south.max()],
                    [y_true_south.min(), y_true_south.max()], 'r--', lw=2, label='Perfect')
        axes[1].set_xlabel("Actual South (W)")
        axes[1].set_ylabel("Predicted South (W)")
        axes[1].set_title("South Tracker: Baseline vs. Separate")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / "tracker_direct_comparison.png", dpi=150)
        plt.close()

        # 3. Residual distributions
        logger.info("  Creating residual_distributions.png...")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # North residuals
        res_baseline_north = pred_baseline_north - y_true_north
        res_separate_north = pred_separate_north - y_true_north

        axes[0, 0].hist(res_baseline_north, bins=50, alpha=0.6, color='blue',
                       edgecolor='black', label='Baseline (derived*)')
        axes[0, 0].hist(res_separate_north, bins=50, alpha=0.6, color='green',
                       edgecolor='black', label='Separate (predicted)')
        axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel("Residual (W)")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title(f"North Tracker Residuals\nBaseline: μ={np.mean(res_baseline_north):.1f}, σ={np.std(res_baseline_north):.1f}\nSeparate: μ={np.mean(res_separate_north):.1f}, σ={np.std(res_separate_north):.1f}")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # South residuals
        res_baseline_south = pred_baseline_south - y_true_south
        res_separate_south = pred_separate_south - y_true_south

        axes[0, 1].hist(res_baseline_south, bins=50, alpha=0.6, color='blue',
                       edgecolor='black', label='Baseline (derived*)')
        axes[0, 1].hist(res_separate_south, bins=50, alpha=0.6, color='green',
                       edgecolor='black', label='Separate (predicted)')
        axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel("Residual (W)")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title(f"South Tracker Residuals\nBaseline: μ={np.mean(res_baseline_south):.1f}, σ={np.std(res_baseline_south):.1f}\nSeparate: μ={np.mean(res_separate_south):.1f}, σ={np.std(res_separate_south):.1f}")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Total residuals
        res_baseline_total = pred_baseline_total - y_true_total
        res_separate_total = pred_separate_total - y_true_total

        axes[1, 0].hist(res_baseline_total, bins=50, alpha=0.6, color='blue',
                       edgecolor='black', label='Baseline')
        axes[1, 0].hist(res_separate_total, bins=50, alpha=0.6, color='green',
                       edgecolor='black', label='Separate (sum)')
        axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel("Residual (W)")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title(f"Total Residuals\nBaseline: μ={np.mean(res_baseline_total):.1f}, σ={np.std(res_baseline_total):.1f}\nSeparate: μ={np.mean(res_separate_total):.1f}, σ={np.std(res_separate_total):.1f}")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Sum inconsistency
        sum_inconsistency = pred_separate_total - y_true_total

        axes[1, 1].hist(sum_inconsistency, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[1, 1].axvline(np.mean(sum_inconsistency), color='darkred', linestyle='-', linewidth=2,
                          label=f'Mean: {np.mean(sum_inconsistency):.1f} W')
        axes[1, 1].set_xlabel("Sum Inconsistency (W)")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].set_title(f"Sum Inconsistency (Separate Only)\n|pred_north + pred_south - true_total|")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / "residual_distributions.png", dpi=150)
        plt.close()

        # 4. Time series example
        logger.info("  Creating time_series_example.png...")
        n_samples = min(500, len(y_true_total))

        fig, axes = plt.subplots(3, 1, figsize=(16, 12))

        x_axis = np.arange(n_samples)

        # North
        axes[0].plot(x_axis, y_true_north[:n_samples], label='True', color='black', linewidth=2)
        axes[0].plot(x_axis, pred_baseline_north[:n_samples], label='Baseline (derived*)',
                    color='blue', alpha=0.7, linewidth=1.5)
        axes[0].plot(x_axis, pred_separate_north[:n_samples], label='Separate (predicted)',
                    color='green', alpha=0.7, linewidth=1.5)
        axes[0].set_ylabel("Power (W)")
        axes[0].set_title("North Tracker: Time Series Comparison (first 500 samples)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # South
        axes[1].plot(x_axis, y_true_south[:n_samples], label='True', color='black', linewidth=2)
        axes[1].plot(x_axis, pred_baseline_south[:n_samples], label='Baseline (derived*)',
                    color='blue', alpha=0.7, linewidth=1.5)
        axes[1].plot(x_axis, pred_separate_south[:n_samples], label='Separate (predicted)',
                    color='green', alpha=0.7, linewidth=1.5)
        axes[1].set_ylabel("Power (W)")
        axes[1].set_title("South Tracker: Time Series Comparison")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Total
        axes[2].plot(x_axis, y_true_total[:n_samples], label='True', color='black', linewidth=2)
        axes[2].plot(x_axis, pred_baseline_total[:n_samples], label='Baseline',
                    color='blue', alpha=0.7, linewidth=1.5)
        axes[2].plot(x_axis, pred_separate_total[:n_samples], label='Separate (sum)',
                    color='green', alpha=0.7, linewidth=1.5)
        axes[2].set_xlabel("Time Step")
        axes[2].set_ylabel("Power (W)")
        axes[2].set_title("Total: Time Series Comparison")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / "time_series_example.png", dpi=150)
        plt.close()

        logger.info("✓ All visualizations saved!")

    def _generate_summary(self, results: Dict):
        """Generate summary markdown report."""
        baseline_total = results['baseline']['total']
        baseline_north = results['baseline']['north_derived']
        baseline_south = results['baseline']['south_derived']
        separate_north = results['separate']['north']
        separate_south = results['separate']['south']
        separate_total = results['separate']['total_sum']
        sum_inconsistency = results['separate']['sum_inconsistency']
        comparison = results['comparison']

        summary = f"""# Separate Tracker Modeling: Comparison Results

## Research Question

Are the individual tracker errors with separate modeling smaller than those
implicitly derived from the baseline, even if the sum becomes worse?

## Approach

### Baseline Approach
- Single LSTM predicts **total PV only** (no tracker knowledge during training)
- Tracker values derived using historical ratio for evaluation:
  - South ratio: {self.historical_ratio_south:.4f} (from training data)
  - North ratio: {1 - self.historical_ratio_south:.4f}
- **Note**: Tracker predictions are NOT predicted, but derived from total

### Separate Approach
- Two independent LSTMs predict North and South trackers
- Each model trained on its respective tracker target
- **Trade-off**: pred_north + pred_south ≠ true_total (errors accumulate)

### Model Architecture (identical for fair comparison)
- **Layers**: {self.num_layers}-layer LSTM
- **Hidden Size**: {self.hidden_size}
- **Dropout**: {self.dropout}
- **Learning Rate**: {self.learning_rate}
- **Epochs**: {self.epochs}
- **Batch Size**: {self.batch_size}
- **Lookback**: {self.lookback} steps (24 hours)
- **Forecast Horizon**: {self.horizon} steps (6 hours)

## Results

### Tracker Accuracy Comparison

| Metric | Baseline North* | Separate North | Improvement | Baseline South* | Separate South | Improvement |
|--------|----------------|----------------|-------------|----------------|----------------|-------------|
| RMSE (W) | {baseline_north['rmse']:.2f} | {separate_north['rmse']:.2f} | {comparison['north_improvement']['rmse_pct']:+.1f}% | {baseline_south['rmse']:.2f} | {separate_south['rmse']:.2f} | {comparison['south_improvement']['rmse_pct']:+.1f}% |
| MAE (W) | {baseline_north['mae']:.2f} | {separate_north['mae']:.2f} | {comparison['north_improvement']['mae_pct']:+.1f}% | {baseline_south['mae']:.2f} | {separate_south['mae']:.2f} | {comparison['south_improvement']['mae_pct']:+.1f}% |
| R² | {baseline_north['r2']:.4f} | {separate_north['r2']:.4f} | {comparison['north_improvement']['r2_pct']:+.1f}% | {baseline_south['r2']:.4f} | {separate_south['r2']:.4f} | {comparison['south_improvement']['r2_pct']:+.1f}% |
| MAPE (%) | {baseline_north['mape']:.2f}% | {separate_north['mape']:.2f}% | - | {baseline_south['mape']:.2f}% | {separate_south['mape']:.2f}% | - |

*Derived from baseline total using historical ratio (not directly predicted)

### Total Accuracy Trade-off

| Approach | RMSE (W) | MAE (W) | R² | Sum Consistency |
|----------|----------|---------|----|--------------------|
| Baseline | {baseline_total['rmse']:.2f} | {baseline_total['mae']:.2f} | {baseline_total['r2']:.4f} | Perfect (by design) |
| Separate | {separate_total['rmse']:.2f} | {separate_total['mae']:.2f} | {separate_total['r2']:.4f} | MAE: ±{sum_inconsistency['mae']:.2f} W ({sum_inconsistency['percentage']:.1f}%) |

**Total Accuracy Change**: {comparison['total_tradeoff']['rmse_pct']:+.1f}% RMSE, {comparison['total_tradeoff']['r2_pct']:+.1f}% R²

## Key Findings

### 1. Tracker-Level Accuracy
- North Tracker: {comparison['north_improvement']['rmse_pct']:+.1f}% RMSE improvement
- South Tracker: {comparison['south_improvement']['rmse_pct']:+.1f}% RMSE improvement
- Average: {(comparison['north_improvement']['rmse_pct'] + comparison['south_improvement']['rmse_pct']) / 2:+.1f}% improvement

### 2. Total Accuracy Trade-off
- Total RMSE: {comparison['total_tradeoff']['rmse_pct']:+.1f}%
- Sum inconsistency: ±{sum_inconsistency['mae']:.2f} W ({sum_inconsistency['percentage']:.1f}% of mean)
- Maximum inconsistency: {sum_inconsistency['max']:.2f} W

### 3. Research Question Answer
{'✓ **POSITIVE**: Separate modeling IMPROVES individual tracker accuracy despite worse total accuracy' if (comparison['north_improvement']['rmse_pct'] + comparison['south_improvement']['rmse_pct']) / 2 > 0 else '✗ **NEGATIVE**: Separate modeling DOES NOT improve individual tracker accuracy'}

## Interpretation

### Advantages of Separate Approach
1. Direct prediction of tracker-specific patterns
2. Individual tracker errors can be better optimized
3. Models learn tracker-specific weather sensitivities

### Disadvantages of Separate Approach
1. Sum inconsistency (±{sum_inconsistency['percentage']:.1f}% error)
2. {'Worse' if comparison['total_tradeoff']['rmse_pct'] < 0 else 'Better'} total accuracy ({comparison['total_tradeoff']['rmse_pct']:+.1f}%)
3. Two models to maintain instead of one

### Trade-off Summary
- **Gain**: {(comparison['north_improvement']['rmse_pct'] + comparison['south_improvement']['rmse_pct']) / 2:+.1f}% average tracker accuracy
- **Cost**: {comparison['total_tradeoff']['rmse_pct']:+.1f}% total accuracy, ±{sum_inconsistency['percentage']:.1f}% sum inconsistency

## Recommendation

{"**Use Separate Models** if tracker-level accuracy is critical and sum inconsistency is acceptable." if (comparison['north_improvement']['rmse_pct'] + comparison['south_improvement']['rmse_pct']) / 2 > 5 else "**Use Baseline (Total-only)** if total accuracy is priority and perfect sum constraint is required."}

## Visualizations

- `scatter_comparison.png`: Scatter plots for all predictions (2x3 grid)
- `tracker_direct_comparison.png`: Direct overlay of baseline vs. separate for trackers
- `residual_distributions.png`: Residual histograms and sum inconsistency
- `time_series_example.png`: Time series comparison (first 500 samples)

## Configuration

All experiment details saved in `metrics.json`.
"""

        summary_path = self.output_dir / "comparison_summary.md"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)

        logger.info(f"✓ Saved summary to {summary_path}")

    def run(self):
        """Execute complete comparison experiment."""
        logger.info("=" * 80)
        logger.info("SEPARATE TRACKER MODELING EXPERIMENT")
        logger.info("=" * 80)
        logger.info(f"Output directory: {self.output_dir}")

        self.load_and_preprocess()
        self.train_baseline_model()
        self.train_separate_models()
        self.evaluate()

        logger.info("\n" + "=" * 80)
        logger.info("EXPERIMENT COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info(f"Plots saved to: {self.plots_dir}")


def main():
    """CLI entry point."""
    from src.logging.logging_setup import setup_logging
    setup_logging(log_level="INFO")

    experiment = SeparateTrackerExperiment()
    experiment.run()


if __name__ == "__main__":
    main()
