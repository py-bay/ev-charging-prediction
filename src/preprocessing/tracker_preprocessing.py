"""Preprocessing pipeline for tracker-specific solar forecasting."""

from pathlib import Path
from typing import Tuple, List
import hashlib

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from src.config.models import Config


class TrackerDataPreprocessor:
    """Preprocessor for tracker-specific solar PV forecasting."""

    def __init__(self, config: Config):
        self.config = config
        self.df_combined: pd.DataFrame | None = None
        self.selected_features: List[str] = []

    def _detect_timestamp_column(self, df: pd.DataFrame) -> str:
        """Detect timestamp column name."""
        possible_names = [
            "timestamp", "time", "datetime", "date", "dt_iso",
            "Date", "Time", "DateTime", "Timestamp"
        ]
        for name in possible_names:
            if name in df.columns:
                logger.info(f"Detected timestamp column: '{name}'")
                return name

        for col in df.columns:
            try:
                sample = df[col].head(5).astype(str)
                pd.to_datetime(sample)
                logger.info(f"Detected timestamp column by parsing: '{col}'")
                return col
            except (ValueError, TypeError):
                continue

        raise ValueError(f"Could not find timestamp column. Available: {list(df.columns)}")

    def load_pv_data(self) -> pd.DataFrame:
        """Load PV data with tracker information."""
        logger.info("Loading PV data with trackers...")

        pv_path = self.config.data_paths.pv
        if not pv_path.exists():
            raise FileNotFoundError(f"PV data file not found: {pv_path}")

        pv_df = pd.read_csv(pv_path, sep=";")
        timestamp_col = self._detect_timestamp_column(pv_df)
        pv_df["timestamp"] = pd.to_datetime(pv_df[timestamp_col], utc=True)
        if timestamp_col != "timestamp":
            pv_df = pv_df.drop(columns=[timestamp_col])
        pv_df = pv_df.set_index("timestamp").sort_index()

        # Rename German columns
        # Note: User confirmed Tracker 2 is south, Tracker 1 is north
        column_mapping = {
            "Solarproduktion": "pv_total",
            "Solarproduktion Tracker 1": "tracker1_north",
            "Solarproduktion Tracker 2": "tracker2_south",
            "Solarproduktion Tracker 3": "tracker3",  # Will be ignored (always 0)
            # Fallback for shorter names
            "Tracker 1": "tracker1_north",
            "Tracker 2": "tracker2_south",
            "Tracker 3": "tracker3",
        }
        pv_df = pv_df.rename(columns=column_mapping)

        logger.info(f"Loaded PV data: {len(pv_df)} rows")
        logger.info(f"  Date range: {pv_df.index.min()} to {pv_df.index.max()}")
        logger.info(f"  Columns: {list(pv_df.columns)}")

        return pv_df

    def load_irradiance_data(self) -> pd.DataFrame:
        """Load solar irradiance data (GHI, DNI, DHI)."""
        logger.info("Loading irradiance data...")

        irradiance_path = self.config.data_paths.irradiance
        if not irradiance_path.exists():
            raise FileNotFoundError(f"Irradiance file not found: {irradiance_path}")

        irradiance_df = pd.read_csv(irradiance_path)
        timestamp_col = self._detect_timestamp_column(irradiance_df)

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
        logger.info(f"  Columns: {list(irradiance_df.columns)}")

        return irradiance_df

    def load_weather_data(self) -> pd.DataFrame:
        """Load weather data."""
        logger.info("Loading weather data...")

        weather_path = self.config.data_paths.weather
        if not weather_path.exists():
            raise FileNotFoundError(f"Weather file not found: {weather_path}")

        weather_df = pd.read_csv(weather_path, skiprows=2)
        timestamp_col = self._detect_timestamp_column(weather_df)
        weather_df["timestamp"] = pd.to_datetime(weather_df[timestamp_col], utc=True)
        if timestamp_col != "timestamp":
            weather_df = weather_df.drop(columns=[timestamp_col])
        weather_df = weather_df.set_index("timestamp").sort_index()

        # Clean column names
        weather_df.columns = [col.split(" (")[0].strip() for col in weather_df.columns]

        logger.info(f"Loaded weather data: {len(weather_df)} rows")
        logger.info(f"  Columns: {list(weather_df.columns)}")

        return weather_df

    def resample_weather(self, weather_df: pd.DataFrame, target_freq: str = "15T") -> pd.DataFrame:
        """Resample hourly weather to 15-minute intervals."""
        logger.info(f"Resampling weather data to {target_freq}...")

        categorical_cols = ["weather_code"] if "weather_code" in weather_df.columns else []
        continuous_cols = [col for col in weather_df.columns if col not in categorical_cols]

        # Resample continuous with interpolation
        weather_resampled = weather_df[continuous_cols].resample(target_freq).interpolate(method="time")

        # Resample categorical with forward fill
        if categorical_cols:
            categorical_resampled = weather_df[categorical_cols].resample(target_freq).ffill()
            weather_resampled = pd.concat([weather_resampled, categorical_resampled], axis=1)

        logger.info(f"Resampled weather: {len(weather_resampled)} rows")
        return weather_resampled

    def merge_all_data(
        self, pv_df: pd.DataFrame, irradiance_df: pd.DataFrame, weather_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge all data sources on timestamp."""
        logger.info("Merging all data sources...")

        # Align to common index
        common_index = pv_df.index.intersection(irradiance_df.index).intersection(weather_df.index)
        logger.info(f"Common timestamps: {len(common_index)}")

        df_combined = pd.concat(
            [pv_df.loc[common_index], irradiance_df.loc[common_index], weather_df.loc[common_index]],
            axis=1
        )

        logger.info(f"Combined data shape: {df_combined.shape}")
        return df_combined

    def apply_6hour_forecast_shift(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply 6-hour forecast logic: shift features back 6 hours.

        This simulates forecasting 6 hours ahead using past weather/irradiance data.
        """
        logger.info("Applying 6-hour forecast shift...")

        # Shift all feature columns (not targets) back by 6 hours (24 steps at 15-min intervals)
        forecast_horizon_steps = 24  # 6 hours * 4 steps/hour

        # Identify target columns (tracker power, total power)
        target_cols = ["pv_total", "tracker1_north", "tracker2_south"]
        feature_cols = [col for col in df.columns if col not in target_cols]

        df_shifted = df.copy()
        for col in feature_cols:
            df_shifted[col] = df[col].shift(forecast_horizon_steps)

        # Drop rows with NaN from shifting
        df_shifted = df_shifted.dropna()

        logger.info(f"After 6-hour shift: {len(df_shifted)} rows")
        return df_shifted

    def compute_correlation_matrix(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Compute correlation matrix between features and target."""
        logger.info(f"Computing correlation matrix for target: {target_col}")

        # Exclude other target columns
        target_cols = ["pv_total", "tracker1_north", "tracker2_south"]
        feature_cols = [col for col in df.columns if col not in target_cols]

        correlations = df[feature_cols + [target_col]].corr()[target_col].sort_values(ascending=False)
        logger.info(f"Top 10 correlated features with {target_col}:")
        for feat, corr in correlations.head(11).items():  # 11 because target is included
            if feat != target_col:
                logger.info(f"  {feat}: {corr:.4f}")

        return correlations

    def select_features_by_correlation(
        self, df: pd.DataFrame, target_col: str, threshold: float = 0.3, top_n: int = 15
    ) -> List[str]:
        """Select features based on correlation with target.

        Args:
            df: Combined dataframe
            target_col: Target column name
            threshold: Minimum absolute correlation
            top_n: Maximum number of features to select

        Returns:
            List of selected feature names
        """
        correlations = self.compute_correlation_matrix(df, target_col)

        # Get features with |correlation| > threshold
        selected = correlations[correlations.abs() > threshold].drop(target_col, errors='ignore')

        # If threshold is too strict and we have no features, take top_n regardless
        if len(selected) == 0:
            logger.warning(f"No features meet correlation threshold {threshold}. Selecting top {top_n} by absolute correlation instead.")
            selected = correlations.drop(target_col, errors='ignore').abs().nlargest(top_n)
            # Get original correlations for these features
            selected = correlations[selected.index]
        else:
            selected = selected.head(top_n)

        logger.info(f"Selected {len(selected)} features for {target_col}")
        return selected.index.tolist()

    def prepare_baseline_data(
        self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for baseline models: GHI → Total PV.

        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info("Preparing baseline data (GHI → Total PV)...")

        # Features: only GHI
        X = df[["ghi"]].values

        # Target: total PV power
        y = df["pv_total"].values

        # Random split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )

        logger.info(f"Baseline data prepared:")
        logger.info(f"  Train: {len(X_train)} samples")
        logger.info(f"  Test: {len(X_test)} samples")

        return X_train, X_test, y_train, y_test

    def prepare_tracker_data(
        self,
        df: pd.DataFrame,
        tracker_col: str,
        feature_cols: List[str],
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for tracker-specific models.

        Args:
            df: Combined dataframe
            tracker_col: Target tracker column
            feature_cols: List of feature column names
            test_size: Test split ratio
            random_state: Random seed

        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info(f"Preparing tracker data for {tracker_col}...")

        X = df[feature_cols].values
        y = df[tracker_col].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )

        logger.info(f"Tracker data prepared:")
        logger.info(f"  Features: {len(feature_cols)}")
        logger.info(f"  Train: {len(X_train)} samples")
        logger.info(f"  Test: {len(X_test)} samples")

        return X_train, X_test, y_train, y_test

    def run_full_pipeline(self) -> pd.DataFrame:
        """Run complete preprocessing pipeline.

        Returns:
            Fully preprocessed dataframe ready for model training
        """
        logger.info("=" * 80)
        logger.info("STARTING TRACKER PREPROCESSING PIPELINE")
        logger.info("=" * 80)

        # Load data
        pv_df = self.load_pv_data()
        irradiance_df = self.load_irradiance_data()
        weather_df = self.load_weather_data()

        # Resample weather to 15-min
        weather_df = self.resample_weather(weather_df)

        # Merge all
        df_combined = self.merge_all_data(pv_df, irradiance_df, weather_df)

        # Apply 6-hour forecast shift
        df_combined = self.apply_6hour_forecast_shift(df_combined)

        # Drop any remaining NaN
        df_combined = df_combined.dropna()

        logger.info(f"Final preprocessed data: {df_combined.shape}")
        logger.info("Preprocessing pipeline complete!")

        self.df_combined = df_combined
        return df_combined
