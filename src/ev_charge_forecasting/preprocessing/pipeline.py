"""Data preprocessing pipeline."""

import hashlib
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from ..config.models import Config


class DataPreprocessor:
    """
    Handles data loading, cleaning, resampling, feature engineering, and labeling.
    """

    def __init__(self, config: Config):
        """
        Initialize preprocessor with configuration.

        Args:
            config: Configuration object
        """
        self.config = config
        self.df_combined: pd.DataFrame | None = None
        self.feature_columns: list[str] = []

    def compute_data_hash(self) -> str:
        """
        Compute hash of input data files for integrity check.

        Returns:
            str: Hexadecimal hash string
        """
        hasher = hashlib.sha256()

        for path_name in ["pv", "irradiance", "weather"]:
            file_path = getattr(self.config.data_paths, path_name)
            if file_path.exists():
                with open(file_path, "rb") as f:
                    hasher.update(f.read())

        return hasher.hexdigest()

    def _detect_timestamp_column(self, df: pd.DataFrame) -> str:
        """
        Detect the timestamp column name in a dataframe.

        Args:
            df: Input dataframe

        Returns:
            Name of timestamp column

        Raises:
            ValueError: If no timestamp column found
        """
        # Common timestamp column names
        possible_names = [
            "timestamp", "time", "datetime", "date", "Date", "Time", "DateTime", "Timestamp",
            "date_time", "Date_Time", "TIME", "DATE", "DATETIME", "TIMESTAMP", "dt_iso",
        ]

        # Check if any of these columns exist
        for name in possible_names:
            if name in df.columns:
                logger.info(f"Detected timestamp column: '{name}'")
                return name

        # If not found, check for columns that look like dates
        for col in df.columns:
            try:
                # Try to parse first few values as datetime
                sample = df[col].head(5).astype(str)
                pd.to_datetime(sample)
                logger.info(f"Detected timestamp column by parsing: '{col}'")
                return col
            except (ValueError, TypeError):
                continue

        raise ValueError(
            f"Could not find timestamp column. Available columns: {list(df.columns)}. "
            "Please ensure your CSV has a timestamp/date/datetime column."
        )

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load PV, irradiance, and weather data from CSV files.

        Returns:
            Tuple of (pv_df, irradiance_df, weather_df)
        """
        logger.info("Loading data files...")

        # Load PV data (semicolon-separated, German columns)
        pv_path = self.config.data_paths.pv
        if not pv_path.exists():
            raise FileNotFoundError(f"PV data file not found: {pv_path}")
        pv_df = pd.read_csv(pv_path, sep=";")
        timestamp_col = self._detect_timestamp_column(pv_df)
        pv_df["timestamp"] = pd.to_datetime(pv_df[timestamp_col], utc=True)  # Make timezone-aware
        if timestamp_col != "timestamp":
            pv_df = pv_df.drop(columns=[timestamp_col])
        pv_df = pv_df.set_index("timestamp").sort_index()

        # Rename key German columns to English
        column_mapping = {
            "Solarproduktion": "pv_power",
            "Hausverbrauch": "household_consumption",
            "Ladezustand": "battery_soc",
            "Batterie (Laden)": "battery_charge",
            "Batterie (Entladen)": "battery_discharge",
            "Netzeinspeisung": "grid_export",
            "Netzbezug": "grid_import",
        }
        pv_df = pv_df.rename(columns=column_mapping)

        logger.info(f"Loaded PV data: {len(pv_df)} rows")
        logger.info(f"  Date range: {pv_df.index.min()} to {pv_df.index.max()}")
        logger.info(f"  Columns: {list(pv_df.columns)}")

        # Load irradiance data (comma-separated)
        irradiance_path = self.config.data_paths.irradiance
        if not irradiance_path.exists():
            raise FileNotFoundError(f"Irradiance data file not found: {irradiance_path}")
        irradiance_df = pd.read_csv(irradiance_path)
        timestamp_col = self._detect_timestamp_column(irradiance_df)

        # Parse the UTC timestamp format (e.g., "1979-01-02 00:00:00 +0000 UTC")
        # Remove the timezone info and parse
        if irradiance_df[timestamp_col].dtype == object:
            # Remove " +0000 UTC" or similar timezone suffixes
            irradiance_df[timestamp_col] = irradiance_df[timestamp_col].str.replace(r' \+\d{4} UTC$', '', regex=True)

        irradiance_df["timestamp"] = pd.to_datetime(irradiance_df[timestamp_col], utc=True)

        # Select relevant irradiance columns (GHI = Global Horizontal Irradiance)
        irradiance_columns = ["timestamp"]
        if "ghi_cloudy_sky" in irradiance_df.columns:
            irradiance_columns.append("ghi_cloudy_sky")
        if "ghi_clear_sky" in irradiance_df.columns:
            irradiance_columns.append("ghi_clear_sky")
        if "dni_cloudy_sky" in irradiance_df.columns:
            irradiance_columns.append("dni_cloudy_sky")

        irradiance_df = irradiance_df[irradiance_columns]
        irradiance_df = irradiance_df.set_index("timestamp").sort_index()

        # Rename to simpler names
        irradiance_df = irradiance_df.rename(columns={
            "ghi_cloudy_sky": "irradiance",
            "ghi_clear_sky": "irradiance_clear_sky",
            "dni_cloudy_sky": "dni"
        })

        logger.info(f"Loaded irradiance data: {len(irradiance_df)} rows")
        logger.info(f"  Date range: {irradiance_df.index.min()} to {irradiance_df.index.max()}")
        logger.info(f"  Columns: {list(irradiance_df.columns)}")

        # Load weather data (comma-separated, with header rows)
        weather_path = self.config.data_paths.weather
        if not weather_path.exists():
            raise FileNotFoundError(f"Weather data file not found: {weather_path}")

        # Skip the first 2 header rows
        weather_df = pd.read_csv(weather_path, skiprows=2)
        timestamp_col = self._detect_timestamp_column(weather_df)
        weather_df["timestamp"] = pd.to_datetime(weather_df[timestamp_col], utc=True)  # Make timezone-aware
        if timestamp_col != "timestamp":
            weather_df = weather_df.drop(columns=[timestamp_col])
        weather_df = weather_df.set_index("timestamp").sort_index()

        # Clean column names (remove units from parentheses)
        weather_df.columns = [col.split(" (")[0].strip() for col in weather_df.columns]

        logger.info(f"Loaded weather data: {len(weather_df)} rows")
        logger.info(f"  Date range: {weather_df.index.min()} to {weather_df.index.max()}")
        logger.info(f"  Columns: {list(weather_df.columns)}")

        # Find common date range
        max_start = max(pv_df.index.min(), irradiance_df.index.min(), weather_df.index.min())
        min_end = min(pv_df.index.max(), irradiance_df.index.max(), weather_df.index.max())

        logger.info(f"Common date range across all datasets: {max_start} to {min_end}")

        return pv_df, irradiance_df, weather_df

    def resample_weather_data(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample hourly weather data to 15-minute intervals.

        Args:
            weather_df: Hourly weather dataframe

        Returns:
            Resampled weather dataframe
        """
        logger.info(
            f"Resampling weather data to {self.config.preprocessing.resample_interval}..."
        )

        # Identify categorical columns (e.g., weather_code)
        categorical_cols = ["weather_code"] if "weather_code" in weather_df.columns else []
        continuous_cols = [col for col in weather_df.columns if col not in categorical_cols]

        # Resample continuous variables with time interpolation
        weather_resampled = weather_df[continuous_cols].resample(
            self.config.preprocessing.resample_interval
        )
        weather_resampled = weather_resampled.interpolate(
            method=self.config.preprocessing.weather_interp_method
        )

        # Resample categorical variables with forward fill
        if categorical_cols:
            categorical_resampled = weather_df[categorical_cols].resample(
                self.config.preprocessing.resample_interval
            )
            categorical_resampled = categorical_resampled.ffill()
            weather_resampled = pd.concat([weather_resampled, categorical_resampled], axis=1)

        logger.info(f"Resampled weather data: {len(weather_resampled)} rows")
        return weather_resampled

    def align_and_merge_data(
        self, pv_df: pd.DataFrame, irradiance_df: pd.DataFrame, weather_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Align timestamps and merge all datasets.

        Args:
            pv_df: PV production dataframe
            irradiance_df: Solar irradiance dataframe
            weather_df: Weather dataframe (already resampled)

        Returns:
            Merged dataframe
        """
        logger.info("Aligning and merging datasets...")

        # Merge PV and irradiance (both 15-min resolution)
        df_merged = pv_df.join(irradiance_df, how="inner", rsuffix="_irr")

        # Merge with resampled weather data
        df_merged = df_merged.join(weather_df, how="inner", rsuffix="_weather")

        # Drop rows with missing values
        initial_len = len(df_merged)
        df_merged = df_merged.dropna()
        dropped = initial_len - len(df_merged)
        if dropped > 0:
            logger.warning(f"Dropped {dropped} rows with missing values")

        logger.info(f"Merged dataset: {len(df_merged)} rows, {len(df_merged.columns)} columns")
        return df_merged

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add cyclical time features (hour of day, day of week) encoded with sine/cosine.

        Args:
            df: Input dataframe with datetime index

        Returns:
            Dataframe with added time features
        """
        if not self.config.features.cyclical_time_features:
            return df

        logger.info("Adding cyclical time features...")

        # Hour of day (0-23)
        df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)

        # Day of week (0-6)
        df["day_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df["day_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

        logger.info("Added time features: hour_sin, hour_cos, day_sin, day_cos")
        return df

    def generate_labels(
        self, df: pd.DataFrame, threshold: float, method: Literal["absolute", "relative"] = "absolute"
    ) -> pd.DataFrame:
        """
        Generate binary labels for optimal charging windows.

        Args:
            df: Input dataframe with PV power column
            threshold: Threshold value for labeling
            method: 'absolute' for kW surplus, 'relative' for proportion of max PV

        Returns:
            Dataframe with 'label' column added
        """
        logger.info(f"Generating labels with threshold={threshold}, method={method}...")

        # Find PV power column (should be 'pv_power' after renaming)
        if "pv_power" not in df.columns:
            raise ValueError(
                f"PV power column 'pv_power' not found. Available columns: {list(df.columns)}"
            )

        pv_col = "pv_power"

        if method == "absolute":
            # Check if we have actual household consumption data
            if "household_consumption" in df.columns:
                logger.info("Using actual household_consumption data from CSV")
                # Optimal if PV surplus > threshold
                # PV surplus = pv_power - household_consumption
                df["pv_surplus"] = df[pv_col] - df["household_consumption"]
                df["label"] = (df["pv_surplus"] > threshold).astype(int)

                logger.info(f"  Mean PV power: {df[pv_col].mean():.2f} W")
                logger.info(f"  Mean household consumption: {df['household_consumption'].mean():.2f} W")
                logger.info(f"  Mean PV surplus: {df['pv_surplus'].mean():.2f} W")
            else:
                # Use configured household consumption as fallback
                household_consumption = self.config.preprocessing.household_consumption_kw * 1000  # Convert to W
                logger.info(f"Using configured household_consumption: {household_consumption:.0f} W")
                df["pv_surplus"] = df[pv_col] - household_consumption
                df["label"] = (df["pv_surplus"] > threshold).astype(int)

        elif method == "relative":
            # Optimal if PV power is above threshold proportion of max
            max_pv = df[pv_col].max()
            logger.info(f"  Max PV power: {max_pv:.2f} W")
            df["label"] = (df[pv_col] / max_pv > threshold).astype(int)
        else:
            raise ValueError(f"Unknown labeling method: {method}")

        positive_ratio = df["label"].mean()
        logger.info(
            f"Generated labels: {df['label'].sum()} positive ({positive_ratio:.2%}), "
            f"{(~df['label'].astype(bool)).sum()} negative"
        )

        return df

    def prepare_features(
        self, df: pd.DataFrame, feature_set: Literal["weather", "irradiance", "combined"]
    ) -> pd.DataFrame:
        """
        Select and prepare feature columns based on feature set.

        Args:
            df: Input dataframe
            feature_set: Name of feature set to use

        Returns:
            Dataframe with selected features and label
        """
        logger.info(f"Preparing features for '{feature_set}' feature set...")

        # Get feature list from config
        feature_list = self.config.get_feature_set(feature_set)

        # Add time features if enabled
        time_features = []
        if self.config.features.cyclical_time_features:
            time_features = ["hour_sin", "hour_cos", "day_sin", "day_cos"]

        # Combine feature lists
        all_features = feature_list + time_features

        # Verify all features exist
        missing_features = [f for f in all_features if f not in df.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            all_features = [f for f in all_features if f in df.columns]

        # Store feature columns
        self.feature_columns = all_features

        # Select features + label
        if "label" not in df.columns:
            raise ValueError("Label column not found. Run generate_labels first.")

        df_selected = df[all_features + ["label"]].copy()
        logger.info(f"Selected {len(all_features)} features: {all_features}")

        return df_selected

    def split_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets.

        Args:
            df: Input dataframe with features and label

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Splitting data into train and test sets...")

        X = df.drop("label", axis=1)
        y = df["label"]

        split_ratio = self.config.preprocessing.train_test_split
        random_seed = self.config.preprocessing.random_seed

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=split_ratio, random_state=random_seed, stratify=y, shuffle=True
        )

        logger.info(
            f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples"
        )
        logger.info(
            f"Train positive ratio: {y_train.mean():.2%}, "
            f"Test positive ratio: {y_test.mean():.2%}"
        )

        return X_train, X_test, y_train, y_test

    def run_pipeline(
        self,
        feature_set: Literal["weather", "irradiance", "combined"],
        label_threshold: float = 1.0,
        label_method: Literal["absolute", "relative"] = "absolute",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Run the complete preprocessing pipeline.

        Args:
            feature_set: Feature set to use
            label_threshold: Threshold for label generation
            label_method: Labeling method ('absolute' or 'relative')

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("=" * 60)
        logger.info("Starting preprocessing pipeline")
        logger.info("=" * 60)

        # Compute data hash for integrity
        data_hash = self.compute_data_hash()
        logger.info(f"Data hash: {data_hash[:16]}...")

        # Load data
        pv_df, irradiance_df, weather_df = self.load_data()

        # Resample weather data to 15-min intervals
        weather_resampled = self.resample_weather_data(weather_df)

        # Align and merge datasets
        df_merged = self.align_and_merge_data(pv_df, irradiance_df, weather_resampled)

        # Add time features
        df_merged = self.add_time_features(df_merged)

        # Generate labels
        df_merged = self.generate_labels(df_merged, label_threshold, label_method)

        # Prepare features
        df_prepared = self.prepare_features(df_merged, feature_set)

        # Store combined dataframe for inspection
        self.df_combined = df_prepared

        # Split data
        X_train, X_test, y_train, y_test = self.split_data(df_prepared)

        logger.info("=" * 60)
        logger.info("Preprocessing pipeline completed")
        logger.info("=" * 60)

        return X_train, X_test, y_train, y_test

    def save_processed_data(self, output_dir: Path = Path("outputs/processed")):
        """
        Save processed data to CSV for inspection.

        Args:
            output_dir: Output directory
        """
        if self.df_combined is None:
            logger.warning("No processed data to save. Run pipeline first.")
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "processed_data.csv"

        self.df_combined.to_csv(output_path)
        logger.info(f"Saved processed data to {output_path}")
