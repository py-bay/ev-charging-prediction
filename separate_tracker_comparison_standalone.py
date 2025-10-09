"""
Separate Tracker Modeling Comparison - Standalone Version

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
from sklearn.preprocessing import MinMaxScaler

# Suppress matplotlib interactive mode
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
# CONFIGURATION - Adjust these paths as needed
# ============================================================================
PV_DATA_PATH = Path("data/pv_data.csv")
IRRADIANCE_DATA_PATH = Path("data/irradiance.csv")
WEATHER_DATA_PATH = Path("data/weather.csv")
OUTPUT_DIR = Path("outputs/experiment_separate_tracker")


# ============================================================================
# LSTM MODEL CLASSES
# ============================================================================

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


# ============================================================================
# MAIN EXPERIMENT CLASS
# ============================================================================

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
        self.output_dir = OUTPUT_DIR
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
        print(f"Set all random seeds to {self.seed}")

    def load_and_preprocess(self):
        """Load data, create features, apply chronological split, and normalize."""
        print("=" * 80)
        print("LOADING AND PREPROCESSING DATA")
        print("=" * 80)

        # Load PV data
        print("Loading PV data...")
        pv_df = pd.read_csv(PV_DATA_PATH, sep=";")

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

        print(f"Loaded PV data: {len(pv_df)} rows")
        print(f"  Date range: {pv_df.index.min()} to {pv_df.index.max()}")

        # Load irradiance data
        print("Loading irradiance data...")
        irradiance_df = pd.read_csv(IRRADIANCE_DATA_PATH)

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
        print(f"Loaded irradiance data: {len(irradiance_df)} rows")

        # Load weather data
        print("Loading weather data...")
        weather_df = pd.read_csv(WEATHER_DATA_PATH, skiprows=2)

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
        print("Resampling weather data to 15-min intervals...")
        categorical_cols = ["weather_code"] if "weather_code" in weather_df.columns else []
        continuous_cols = [col for col in weather_df.columns if col not in categorical_cols]

        weather_resampled = weather_df[continuous_cols].resample("15min").interpolate(method="time")
        if categorical_cols:
            categorical_resampled = weather_df[categorical_cols].resample("15min").ffill()
            weather_resampled = pd.concat([weather_resampled, categorical_resampled], axis=1)

        print(f"Resampled weather: {len(weather_resampled)} rows")

        # Merge all data
        print("Merging all data sources...")
        common_index = pv_df.index.intersection(irradiance_df.index).intersection(weather_resampled.index)
        print(f"Common timestamps: {len(common_index)}")

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
        print("Adding time features...")
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

        print(f"Final merged data: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        # Define feature columns (only important features)
        important_features = [
            'ghi', 'dni', 'dhi', 'temperature_2m', 'cloud_cover',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
        ]

        # Filter to only include features that exist in the dataframe
        self.feature_cols = [col for col in important_features if col in df.columns]

        print(f"Feature columns ({len(self.feature_cols)}): {self.feature_cols}")

        # Warn if any important features are missing
        missing_features = set(important_features) - set(self.feature_cols)
        if missing_features:
            print(f"WARNING: Missing features from dataframe: {missing_features}")

        # Apply 6-hour forecast shift (shift features back 24 steps)
        print(f"Applying 6-hour forecast shift (horizon={self.horizon} steps)...")
        df_shifted = df.copy()
        for col in self.feature_cols:
            df_shifted[col] = df[col].shift(self.horizon)

        # Drop NaN from shift
        df_shifted = df_shifted.dropna()
        print(f"After shift: {len(df_shifted)} rows")

        self.df = df_shifted

        # Chronological split
        print(f"Performing chronological split (test_fraction={self.test_fraction})...")
        split_idx = int(len(self.df) * (1 - self.test_fraction))
        df_train = self.df.iloc[:split_idx]
        df_test = self.df.iloc[split_idx:]

        print(f"Train set: {len(df_train)} samples ({df_train.index.min()} to {df_train.index.max()})")
        print(f"Test set: {len(df_test)} samples ({df_test.index.min()} to {df_test.index.max()})")

        # Calculate energy-based ratio
        total_energy_south = df_train["pv_south"].sum()
        total_energy_north = df_train["pv_north"].sum()
        total_energy_combined = total_energy_south + total_energy_north

        self.historical_ratio_south = total_energy_south / total_energy_combined

        print(f"\n📊 Historical South Ratio (energy-based): {self.historical_ratio_south:.4f}")
        print(f"   South total: {total_energy_south:.2f}, North total: {total_energy_north:.2f}")

        # Normalize with single scaler
        print("Normalizing data with MinMaxScaler...")
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
        print(f"Creating supervised datasets (lookback={self.lookback})...")
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

        print(f"Datasets created:")
        print(f"  X_train: {self.X_train.shape}")
        print(f"  X_test: {self.X_test.shape}")
        print(f"  Targets: Total, North, South")

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
        print("=" * 80)
        print("TRAINING BASELINE MODEL (TOTAL-ONLY)")
        print("=" * 80)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        input_dim = self.X_train.shape[2]
        self.model_baseline = BaselineTotalLSTM(
            input_size=input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(device)

        print(f"Model architecture:")
        print(f"  Input size: {input_dim}")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Num layers: {self.num_layers}")
        print(f"  Output: Total PV (no tracker information)")

        # Train the model
        self._train_model(
            self.model_baseline,
            self.X_train,
            self.y_train_total,
            device,
            "Baseline Total"
        )

        # Generate predictions
        print("\nGenerating baseline predictions...")
        self.pred_baseline_total = self._predict(self.model_baseline, self.X_test, device)

        # Derive tracker predictions using historical ratio
        print("\n📊 DERIVING tracker predictions from baseline total...")
        print(f"   Using historical ratio: South = {self.historical_ratio_south:.4f}")

        self.pred_baseline_south_derived = self.pred_baseline_total * self.historical_ratio_south
        self.pred_baseline_north_derived = self.pred_baseline_total * (1 - self.historical_ratio_south)

        print(f"✓ Baseline predictions generated")

    def train_separate_models(self):
        """Train two independent models for North and South trackers."""
        print("=" * 80)
        print("TRAINING SEPARATE TRACKER MODELS")
        print("=" * 80)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        input_dim = self.X_train.shape[2]

        # North model
        print("\n--- Training North Tracker Model ---")
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
        print("\n--- Training South Tracker Model ---")
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
        print("\nGenerating separate tracker predictions...")
        self.pred_separate_north = self._predict(self.model_north, self.X_test, device)
        self.pred_separate_south = self._predict(self.model_south, self.X_test, device)

        # Calculate sum
        self.pred_separate_total_sum = self.pred_separate_north + self.pred_separate_south

        print(f"✓ Separate predictions generated")

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
                print(f"{model_name} - Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.6f}")

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
        print("=" * 80)
        print("EVALUATION AND COMPARISON")
        print("=" * 80)

        # Inverse transform all predictions and targets
        all_cols = list(self.df.columns)
        n_features = len(all_cols)

        idx_total = all_cols.index("pv_total")
        idx_north = all_cols.index("pv_north")
        idx_south = all_cols.index("pv_south")

        print(f"Inverse transforming predictions...")

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

        # Compute metrics
        print("\n--- Baseline Approach Metrics ---")
        metrics_baseline_total = self._compute_metrics(y_true_total, pred_baseline_total_inv, "Baseline Total")
        metrics_baseline_north = self._compute_metrics(y_true_north, pred_baseline_north_inv, "Baseline North (derived*)")
        metrics_baseline_south = self._compute_metrics(y_true_south, pred_baseline_south_inv, "Baseline South (derived*)")

        print("\n--- Separate Models Metrics ---")
        metrics_separate_north = self._compute_metrics(y_true_north, pred_separate_north_inv, "Separate North")
        metrics_separate_south = self._compute_metrics(y_true_south, pred_separate_south_inv, "Separate South")
        metrics_separate_total = self._compute_metrics(y_true_total, pred_separate_total_inv, "Separate Total (sum)")

        # Calculate improvements
        def calc_improvement(baseline_val, separate_val):
            return ((baseline_val - separate_val) / baseline_val) * 100 if baseline_val != 0 else 0

        north_rmse_improvement = calc_improvement(metrics_baseline_north['rmse'], metrics_separate_north['rmse'])
        south_rmse_improvement = calc_improvement(metrics_baseline_south['rmse'], metrics_separate_south['rmse'])

        print("\n" + "=" * 80)
        print("CONCLUSION")
        print("=" * 80)

        avg_tracker_improvement = (north_rmse_improvement + south_rmse_improvement) / 2

        if avg_tracker_improvement > 0:
            print(f"✓ Separate modeling IMPROVES tracker accuracy by {avg_tracker_improvement:.1f}% on average")
        else:
            print(f"✗ Separate modeling WORSENS tracker accuracy by {abs(avg_tracker_improvement):.1f}% on average")

        # Save results
        results = {
            "experiment_config": {
                "approach": "baseline_vs_separate_trackers",
                "seed": self.seed,
                "lookback": self.lookback,
                "horizon": self.horizon,
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
                "total_sum": metrics_separate_total
            }
        }

        results_path = self.output_dir / "metrics.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved results to {results_path}")

        # Generate visualizations
        self._create_visualizations(
            y_true_total, y_true_north, y_true_south,
            pred_baseline_total_inv, pred_baseline_north_inv, pred_baseline_south_inv,
            pred_separate_total_inv, pred_separate_north_inv, pred_separate_south_inv
        )

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

        print(f"\n{name}:")
        print(f"  RMSE: {rmse:.2f} W")
        print(f"  MAE: {mae:.2f} W")
        print(f"  R²: {r2:.4f}")

        return {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2)
        }

    def _create_visualizations(self, y_true_total, y_true_north, y_true_south,
                              pred_baseline_total, pred_baseline_north, pred_baseline_south,
                              pred_separate_total, pred_separate_north, pred_separate_south):
        """Generate comparison visualizations."""

        # Scatter plots comparison (2x3 grid)
        print("\n  Creating scatter_comparison.png...")
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

        print("✓ Visualizations saved!")

    def run(self):
        """Execute complete comparison experiment."""
        print("=" * 80)
        print("SEPARATE TRACKER MODELING EXPERIMENT")
        print("=" * 80)
        print(f"Output directory: {self.output_dir}")

        self.load_and_preprocess()
        self.train_baseline_model()
        self.train_separate_models()
        self.evaluate()

        print("\n" + "=" * 80)
        print("EXPERIMENT COMPLETE!")
        print("=" * 80)
        print(f"Results saved to: {self.output_dir}")


if __name__ == "__main__":
    experiment = SeparateTrackerExperiment()
    experiment.run()