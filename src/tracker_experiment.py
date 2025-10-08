"""
Tracker vs. Total LSTM Forecasting Experiment

Research Question:
Does forecasting the two PV trackers (North and South) separately and summing their
predictions yield a more accurate total forecast than directly predicting the total
PV production?

Experiment Design:
- Forecast Horizon: 6 hours = 24 steps (15 min resolution)
- Lookback Window: 24 hours = 96 steps
- Split: Chronological 80/20
- Models: LSTM_Total vs. (LSTM_North + LSTM_South)
- Loss: MAE
- Metrics: RMSE, MAE, MAPE, NRMSE, Mean Bias
- Statistical tests: Paired t-test and Wilcoxon test
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
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

# Suppress matplotlib interactive mode
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.config.models import load_config


class LSTMForecaster(nn.Module):
    """Simple 2-layer LSTM for time series forecasting."""

    def __init__(self, input_dim: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, sequence, features)
        lstm_out, _ = self.lstm(x)
        # Take the last time step
        out = self.fc(lstm_out[:, -1, :])
        return out.squeeze(-1)


class TrackerForecastExperiment:
    """Complete experiment comparing tracker-level vs. total forecasting."""

    def __init__(self):
        # Hardcoded experiment parameters
        self.lookback = 48  # 24 hours at 15-min resolution
        self.horizon = 24   # 6 hours ahead
        self.test_fraction = 0.2
        self.seed = 42

        # LSTM hyperparameters
        self.hidden_size = 32
        self.num_layers = 2
        self.dropout = 0.2
        self.learning_rate = 0.001
        self.epochs = 20
        self.batch_size = 64

        # Paths
        self.output_dir = Path("outputs/experiment_tracker_vs_total")
        self.plots_dir = self.output_dir / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Data containers
        self.df: pd.DataFrame = None
        self.scaler: MinMaxScaler = None
        self.feature_cols: List[str] = []

        # Training data
        self.X_train_total: np.ndarray = None
        self.y_train_total: np.ndarray = None
        self.X_test_total: np.ndarray = None
        self.y_test_total: np.ndarray = None

        self.X_train_north: np.ndarray = None
        self.y_train_north: np.ndarray = None
        self.X_test_north: np.ndarray = None
        self.y_test_north: np.ndarray = None

        self.X_train_south: np.ndarray = None
        self.y_train_south: np.ndarray = None
        self.X_test_south: np.ndarray = None
        self.y_test_south: np.ndarray = None

        # Models
        self.model_total: LSTMForecaster = None
        self.model_north: LSTMForecaster = None
        self.model_south: LSTMForecaster = None

        # Predictions
        self.y_pred_total: np.ndarray = None
        self.y_pred_north: np.ndarray = None
        self.y_pred_south: np.ndarray = None
        self.y_pred_sum: np.ndarray = None

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

        # Drop the 'time' column if it exists (it's a timestamp string from weather data)
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

        # Normalize with single scaler
        logger.info("Normalizing data with MinMaxScaler...")
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

        # Create supervised datasets for each target
        logger.info(f"Creating supervised datasets (lookback={self.lookback})...")

        # Total model
        self.X_train_total, self.y_train_total = self._create_sequences(
            df_train_scaled[self.feature_cols].values,
            df_train_scaled["pv_total"].values
        )
        self.X_test_total, self.y_test_total = self._create_sequences(
            df_test_scaled[self.feature_cols].values,
            df_test_scaled["pv_total"].values
        )

        # North model
        self.X_train_north, self.y_train_north = self._create_sequences(
            df_train_scaled[self.feature_cols].values,
            df_train_scaled["pv_north"].values
        )
        self.X_test_north, self.y_test_north = self._create_sequences(
            df_test_scaled[self.feature_cols].values,
            df_test_scaled["pv_north"].values
        )

        # South model
        self.X_train_south, self.y_train_south = self._create_sequences(
            df_train_scaled[self.feature_cols].values,
            df_train_scaled["pv_south"].values
        )
        self.X_test_south, self.y_test_south = self._create_sequences(
            df_test_scaled[self.feature_cols].values,
            df_test_scaled["pv_south"].values
        )

        logger.info(f"Sequence datasets created:")
        logger.info(f"  Total - Train: {self.X_train_total.shape}, Test: {self.X_test_total.shape}")
        logger.info(f"  North - Train: {self.X_train_north.shape}, Test: {self.X_test_north.shape}")
        logger.info(f"  South - Train: {self.X_train_south.shape}, Test: {self.X_test_south.shape}")

    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create supervised learning sequences with lookback window.

        Args:
            X: Feature array (n_samples, n_features)
            y: Target array (n_samples,)

        Returns:
            X_seq: (n_sequences, lookback, n_features)
            y_seq: (n_sequences,)
        """
        X_seq = []
        y_seq = []

        for i in range(self.lookback, len(X)):
            X_seq.append(X[i - self.lookback:i])
            y_seq.append(y[i])

        return np.array(X_seq), np.array(y_seq)

    def build_lstm(self, input_dim: int) -> LSTMForecaster:
        """Build LSTM model."""
        model = LSTMForecaster(
            input_dim=input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        return model

    def train_and_predict(self):
        """Train all three LSTM models and generate predictions."""
        logger.info("=" * 80)
        logger.info("TRAINING MODELS")
        logger.info("=" * 80)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        input_dim = self.X_train_total.shape[2]

        # Train LSTM_Total
        logger.info("\n" + "=" * 40)
        logger.info("Training LSTM_Total (pv_total)...")
        logger.info("=" * 40)
        self.model_total = self.build_lstm(input_dim).to(device)
        self._train_model(
            self.model_total,
            self.X_train_total,
            self.y_train_total,
            device
        )
        self.y_pred_total = self._predict(self.model_total, self.X_test_total, device)

        # Train LSTM_North
        logger.info("\n" + "=" * 40)
        logger.info("Training LSTM_North (pv_north)...")
        logger.info("=" * 40)
        self.model_north = self.build_lstm(input_dim).to(device)
        self._train_model(
            self.model_north,
            self.X_train_north,
            self.y_train_north,
            device
        )
        self.y_pred_north = self._predict(self.model_north, self.X_test_north, device)

        # Train LSTM_South
        logger.info("\n" + "=" * 40)
        logger.info("Training LSTM_South (pv_south)...")
        logger.info("=" * 40)
        self.model_south = self.build_lstm(input_dim).to(device)
        self._train_model(
            self.model_south,
            self.X_train_south,
            self.y_train_south,
            device
        )
        self.y_pred_south = self._predict(self.model_south, self.X_test_south, device)

        # Sum tracker predictions
        self.y_pred_sum = self.y_pred_north + self.y_pred_south
        logger.info("\nComputed summed prediction: y_pred_sum = y_pred_north + y_pred_south")

    def _train_model(self, model: LSTMForecaster, X_train: np.ndarray, y_train: np.ndarray, device):
        """Train a single LSTM model."""
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.FloatTensor(y_train).to(device)

        # Loss and optimizer
        criterion = nn.L1Loss()  # MAE
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

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.6f}")

    def _predict(self, model: LSTMForecaster, X_test: np.ndarray, device) -> np.ndarray:
        """Generate predictions."""
        model.eval()
        X_test_t = torch.FloatTensor(X_test).to(device)

        with torch.no_grad():
            y_pred_t = model(X_test_t)
            y_pred = y_pred_t.cpu().numpy()

        return y_pred

    def evaluate(self):
        """Compute metrics, run statistical tests, and generate visualizations."""
        logger.info("=" * 80)
        logger.info("EVALUATION")
        logger.info("=" * 80)

        # Inverse transform predictions and targets
        # The scaler was fit on the entire dataframe with all columns
        # Get the column order from the original dataframe
        all_cols = list(self.df.columns)
        n_features = len(all_cols)

        # Get indices of target columns
        idx_total = all_cols.index("pv_total")
        idx_north = all_cols.index("pv_north")
        idx_south = all_cols.index("pv_south")

        logger.info(f"Scaler expects {n_features} features, column order: {all_cols}")
        logger.info(f"Target indices - total: {idx_total}, north: {idx_north}, south: {idx_south}")

        # Inverse transform y_test_total
        dummy = np.zeros((len(self.y_test_total), n_features))
        dummy[:, idx_total] = self.y_test_total
        y_test_total_inv = self.scaler.inverse_transform(dummy)[:, idx_total]

        # Inverse transform y_pred_total
        dummy = np.zeros((len(self.y_pred_total), n_features))
        dummy[:, idx_total] = self.y_pred_total
        y_pred_total_inv = self.scaler.inverse_transform(dummy)[:, idx_total]

        # Inverse transform y_pred_north
        dummy = np.zeros((len(self.y_pred_north), n_features))
        dummy[:, idx_north] = self.y_pred_north
        y_pred_north_inv = self.scaler.inverse_transform(dummy)[:, idx_north]

        # Inverse transform y_pred_south
        dummy = np.zeros((len(self.y_pred_south), n_features))
        dummy[:, idx_south] = self.y_pred_south
        y_pred_south_inv = self.scaler.inverse_transform(dummy)[:, idx_south]

        # Compute summed prediction
        y_pred_sum_inv = y_pred_north_inv + y_pred_south_inv

        # Compute metrics
        logger.info("\nComputing metrics...")
        metrics_total = self._regression_metrics(y_test_total_inv, y_pred_total_inv)
        metrics_sum = self._regression_metrics(y_test_total_inv, y_pred_sum_inv)

        logger.info("\n" + "=" * 40)
        logger.info("LSTM_Total Metrics:")
        logger.info("=" * 40)
        for key, value in metrics_total.items():
            logger.info(f"  {key}: {value:.6f}")

        logger.info("\n" + "=" * 40)
        logger.info("LSTM_TrackerSum Metrics:")
        logger.info("=" * 40)
        for key, value in metrics_sum.items():
            logger.info(f"  {key}: {value:.6f}")

        # Statistical comparison
        logger.info("\n" + "=" * 40)
        logger.info("Statistical Comparison:")
        logger.info("=" * 40)
        stats_results = self._compare_models(y_test_total_inv, y_pred_total_inv, y_pred_sum_inv)

        for key, value in stats_results.items():
            logger.info(f"  {key}: {value}")

        # Save results
        results = {
            "experiment_config": {
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
                "batch_size": self.batch_size
            },
            "lstm_total": metrics_total,
            "lstm_tracker_sum": metrics_sum,
            "statistical_tests": stats_results
        }

        # Save JSON
        results_path = self.output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nSaved results to {results_path}")

        # Save CSV
        csv_data = {
            "Model": ["LSTM_Total", "LSTM_TrackerSum"],
            "RMSE": [metrics_total["rmse"], metrics_sum["rmse"]],
            "MAE": [metrics_total["mae"], metrics_sum["mae"]],
            "MAPE": [metrics_total["mape"], metrics_sum["mape"]],
            "NRMSE": [metrics_total["nrmse"], metrics_sum["nrmse"]],
            "MeanBias": [metrics_total["mean_bias"], metrics_sum["mean_bias"]],
            "R2": [metrics_total["r2"], metrics_sum["r2"]]
        }
        csv_df = pd.DataFrame(csv_data)
        csv_path = self.output_dir / "combined_results.csv"
        csv_df.to_csv(csv_path, index=False)
        logger.info(f"Saved combined results to {csv_path}")

        # Generate visualizations
        logger.info("\nGenerating visualizations...")
        self._create_visualizations(
            y_test_total_inv,
            y_pred_total_inv,
            y_pred_sum_inv
        )

        # Generate summary markdown
        logger.info("\nGenerating summary...")
        self._generate_summary(metrics_total, metrics_sum, stats_results)

    def _regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
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

        # NRMSE (normalized by mean)
        nrmse = (rmse / np.mean(y_true)) * 100 if np.mean(y_true) > 0 else 0.0

        # Mean Bias Error
        mean_bias = np.mean(y_pred - y_true)

        return {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "mape": float(mape),
            "nrmse": float(nrmse),
            "mean_bias": float(mean_bias)
        }

    def _compare_models(self, y_true: np.ndarray, y_pred_total: np.ndarray, y_pred_sum: np.ndarray) -> Dict:
        """Perform statistical comparison between models."""
        # Compute absolute errors
        errors_total = np.abs(y_true - y_pred_total)
        errors_sum = np.abs(y_true - y_pred_sum)

        # Paired t-test
        t_stat, t_pval = stats.ttest_rel(errors_total, errors_sum)

        # Wilcoxon signed-rank test
        w_stat, w_pval = stats.wilcoxon(errors_total, errors_sum)

        # Compute improvement percentage
        mae_total = np.mean(errors_total)
        mae_sum = np.mean(errors_sum)
        improvement_pct = ((mae_total - mae_sum) / mae_total) * 100

        return {
            "paired_t_test_statistic": float(t_stat),
            "paired_t_test_pvalue": float(t_pval),
            "wilcoxon_test_statistic": float(w_stat),
            "wilcoxon_test_pvalue": float(w_pval),
            "mae_improvement_percent": float(improvement_pct)
        }

    def _create_visualizations(self, y_true: np.ndarray, y_pred_total: np.ndarray, y_pred_sum: np.ndarray):
        """Generate all visualization plots."""
        # 1. Predictions vs Actual - Total
        logger.info("  Creating pred_vs_actual_total.png...")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_true, y_pred_total, alpha=0.5, s=10)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax.set_xlabel("Actual PV Power (kW)")
        ax.set_ylabel("Predicted PV Power (kW)")
        ax.set_title("LSTM_Total: Predictions vs Actual")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "pred_vs_actual_total.png", dpi=150)
        plt.close()

        # 2. Predictions vs Actual - Tracker Sum
        logger.info("  Creating pred_vs_actual_tracker_sum.png...")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_true, y_pred_sum, alpha=0.5, s=10, color='green')
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax.set_xlabel("Actual PV Power (kW)")
        ax.set_ylabel("Predicted PV Power (kW)")
        ax.set_title("LSTM_TrackerSum: Predictions vs Actual")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "pred_vs_actual_tracker_sum.png", dpi=150)
        plt.close()

        # 3. Residual distribution comparison
        logger.info("  Creating residual_distribution_comparison.png...")
        residuals_total = y_pred_total - y_true
        residuals_sum = y_pred_sum - y_true

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(residuals_total, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[0].set_xlabel("Residual (kW)")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title(f"LSTM_Total Residuals\nMean: {np.mean(residuals_total):.3f}, Std: {np.std(residuals_total):.3f}")
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(residuals_sum, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[1].set_xlabel("Residual (kW)")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title(f"LSTM_TrackerSum Residuals\nMean: {np.mean(residuals_sum):.3f}, Std: {np.std(residuals_sum):.3f}")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / "residual_distribution_comparison.png", dpi=150)
        plt.close()

        # 4. MAE by hour of day
        logger.info("  Creating mae_by_hour_comparison.png...")

        # Get test set timestamps (need to account for lookback offset)
        split_idx = int(len(self.df) * (1 - self.test_fraction))
        test_timestamps = self.df.iloc[split_idx + self.lookback:].index

        # Make sure we have the right number of timestamps
        if len(test_timestamps) != len(y_true):
            logger.warning(f"Timestamp mismatch: {len(test_timestamps)} timestamps vs {len(y_true)} predictions")
            # Truncate to match
            min_len = min(len(test_timestamps), len(y_true))
            test_timestamps = test_timestamps[:min_len]
            y_true = y_true[:min_len]
            y_pred_total = y_pred_total[:min_len]
            y_pred_sum = y_pred_sum[:min_len]

        hours = test_timestamps.hour
        errors_total = np.abs(y_true - y_pred_total)
        errors_sum = np.abs(y_true - y_pred_sum)

        mae_by_hour_total = []
        mae_by_hour_sum = []

        for h in range(24):
            mask = hours == h
            if mask.sum() > 0:
                mae_by_hour_total.append(errors_total[mask].mean())
                mae_by_hour_sum.append(errors_sum[mask].mean())
            else:
                mae_by_hour_total.append(0)
                mae_by_hour_sum.append(0)

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(24)
        width = 0.35

        ax.bar(x - width/2, mae_by_hour_total, width, label='LSTM_Total', alpha=0.8, color='blue')
        ax.bar(x + width/2, mae_by_hour_sum, width, label='LSTM_TrackerSum', alpha=0.8, color='green')

        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Mean Absolute Error (kW)")
        ax.set_title("MAE by Hour of Day: LSTM_Total vs LSTM_TrackerSum")
        ax.set_xticks(x)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.plots_dir / "mae_by_hour_comparison.png", dpi=150)
        plt.close()

        logger.info("All visualizations saved!")

    def _generate_summary(self, metrics_total: Dict, metrics_sum: Dict, stats: Dict):
        """Generate summary markdown file."""
        mae_improvement = stats["mae_improvement_percent"]

        summary = f"""# Tracker vs. Total LSTM Forecasting Experiment

## Research Question

Does forecasting the two PV trackers (North and South) separately and summing their
predictions yield a more accurate total forecast than directly predicting the total
PV production?

## Experiment Configuration

- **Forecast Horizon**: 6 hours (24 steps at 15-min resolution)
- **Lookback Window**: 24 hours (96 steps)
- **Train/Test Split**: Chronological 80/20
- **Model Architecture**: 2-layer LSTM with {self.hidden_size} hidden units
- **Dropout**: {self.dropout}
- **Loss Function**: MAE (L1)
- **Epochs**: {self.epochs}
- **Batch Size**: {self.batch_size}
- **Learning Rate**: {self.learning_rate}
- **Random Seed**: {self.seed}

## Results

### Performance Metrics

| Metric | LSTM_Total | LSTM_TrackerSum | Improvement |
|--------|------------|-----------------|-------------|
| RMSE | {metrics_total['rmse']:.4f} | {metrics_sum['rmse']:.4f} | {((metrics_total['rmse'] - metrics_sum['rmse']) / metrics_total['rmse'] * 100):.2f}% |
| MAE | {metrics_total['mae']:.4f} | {metrics_sum['mae']:.4f} | {mae_improvement:.2f}% |
| MAPE | {metrics_total['mape']:.2f}% | {metrics_sum['mape']:.2f}% | {((metrics_total['mape'] - metrics_sum['mape']) / metrics_total['mape'] * 100):.2f}% |
| NRMSE | {metrics_total['nrmse']:.2f}% | {metrics_sum['nrmse']:.2f}% | {((metrics_total['nrmse'] - metrics_sum['nrmse']) / metrics_total['nrmse'] * 100):.2f}% |
| R² | {metrics_total['r2']:.4f} | {metrics_sum['r2']:.4f} | - |
| Mean Bias | {metrics_total['mean_bias']:.4f} | {metrics_sum['mean_bias']:.4f} | - |

### Statistical Tests

- **Paired t-test p-value**: {stats['paired_t_test_pvalue']:.6f}
- **Wilcoxon test p-value**: {stats['wilcoxon_test_pvalue']:.6f}

## Interpretation

The separate per-tracker modeling {'**yielded {:.2f}% lower MAE**'.format(abs(mae_improvement)) if mae_improvement > 0 else '**yielded {:.2f}% higher MAE**'.format(abs(mae_improvement))} compared to the total model.

{'**Statistical significance**: The difference is statistically significant at alpha=0.05.' if stats['paired_t_test_pvalue'] < 0.05 else '**Not statistically significant**: The difference is not significant at alpha=0.05.'}

## Visualizations

- `pred_vs_actual_total.png`: Scatter plot of LSTM_Total predictions vs actual values
- `pred_vs_actual_tracker_sum.png`: Scatter plot of LSTM_TrackerSum predictions vs actual values
- `residual_distribution_comparison.png`: Histogram comparison of residuals for both models
- `mae_by_hour_comparison.png`: MAE breakdown by hour of day for both models

## Conclusion

This experiment provides {'strong' if stats['paired_t_test_pvalue'] < 0.05 and mae_improvement > 0 else 'limited'} evidence that {'separate tracker-level forecasting improves accuracy over direct total forecasting' if mae_improvement > 0 else 'direct total forecasting performs similarly to or better than separate tracker-level forecasting'}.
"""

        summary_path = self.output_dir / "summary.md"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)

        logger.info(f"Saved summary to {summary_path}")

    def run(self):
        """Execute complete experiment pipeline."""
        logger.info("=" * 80)
        logger.info("TRACKER VS. TOTAL LSTM FORECASTING EXPERIMENT")
        logger.info("=" * 80)
        logger.info(f"Output directory: {self.output_dir}")

        self.load_and_preprocess()
        self.train_and_predict()
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

    experiment = TrackerForecastExperiment()
    experiment.run()


if __name__ == "__main__":
    main()
