"""Data processor for train/test splitting and feature engineering."""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Tuple
from loguru import logger
from sklearn.preprocessing import StandardScaler

from src.config.models import Config


class DataProcessor:
    """Processor for train/test splitting, feature engineering, and scaling."""

    def __init__(self, config: Config):
        """
        Initialize the data processor.

        Args:
            config: Configuration object with processing settings
        """
        self.config = config
        self.processing = config.processing
        self.scaler = StandardScaler()

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features from datetime index.

        Args:
            df: DataFrame with datetime as 'utc' column

        Returns:
            DataFrame with added temporal features
        """
        logger.info("Adding temporal features...")
        df = df.copy()

        # Parse UTC timestamp
        df["utc"] = pd.to_datetime(df["utc"])

        # Basic temporal features
        df["hour"] = df["utc"].dt.hour
        df["day_of_year"] = df["utc"].dt.dayofyear
        df["month"] = df["utc"].dt.month

        # Cyclical encoding for hour (0-23)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        # Cyclical encoding for day of year (1-365/366)
        df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

        logger.info(f"Added temporal features: {self.processing.temporal_features}")
        return df

    def split_train_test(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets based on temporal order.

        Args:
            df: DataFrame to split

        Returns:
            Tuple of (train_df, test_df)
        """
        split_ratio = self.processing.train_test_split
        split_idx = int(len(df) * split_ratio)

        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        logger.info(f"Train/test split: {split_ratio:.1%} / {1-split_ratio:.1%}")
        logger.info(f"Train set: {len(train_df)} rows")
        logger.info(f"Test set: {len(test_df)} rows")

        return train_df, test_df

    def scale_features(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale features using StandardScaler fitted on training data.

        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            target_col: Name of the target column (will not be scaled)

        Returns:
            Tuple of (scaled_train_df, scaled_test_df)
        """
        logger.info("Scaling features with StandardScaler...")

        # Identify feature columns (exclude utc and target)
        feature_cols = [col for col in train_df.columns if col not in ["utc", target_col]]

        # Fit scaler on training data only
        self.scaler.fit(train_df[feature_cols])

        # Transform both train and test
        train_scaled = train_df.copy()
        test_scaled = test_df.copy()

        train_scaled[feature_cols] = self.scaler.transform(train_df[feature_cols])
        test_scaled[feature_cols] = self.scaler.transform(test_df[feature_cols])

        logger.info(f"Scaled {len(feature_cols)} features")
        return train_scaled, test_scaled

    def process_dataset(
        self,
        input_file: Path,
        output_name: str
    ) -> None:
        """
        Process a single dataset: add features, split, scale, and save.

        Args:
            input_file: Path to input CSV file
            output_name: Name for output files (e.g., "total", "north", "south")
        """
        logger.info(f"Processing dataset: {input_file.name}")

        # Load data
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} rows")

        # Identify target column (first column after 'utc')
        target_col = [col for col in df.columns if col != "utc"][0]
        logger.info(f"Target column: {target_col}")

        # Add temporal features if configured
        if self.processing.add_temporal_features:
            df = self.add_temporal_features(df)

        # Split into train/test
        train_df, test_df = self.split_train_test(df)

        # Scale features
        train_scaled, test_scaled = self.scale_features(train_df, test_df, target_col)

        # Create output directory
        output_dir = self.processing.output_dir / output_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save train and test sets
        train_file = output_dir / "train.csv"
        test_file = output_dir / "test.csv"

        train_scaled.to_csv(train_file, index=False)
        test_scaled.to_csv(test_file, index=False)

        logger.info(f"Saved train set: {train_file} ({len(train_scaled)} rows)")
        logger.info(f"Saved test set: {test_file} ({len(test_scaled)} rows)")

        # Save scaler for future use
        scaler_file = output_dir / "scaler.pkl"
        with open(scaler_file, "wb") as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Saved scaler: {scaler_file}")

        # Save metadata
        metadata = {
            "input_file": str(input_file),
            "target_column": target_col,
            "n_features": len([col for col in train_scaled.columns if col not in ["utc", target_col]]),
            "train_samples": len(train_scaled),
            "test_samples": len(test_scaled),
            "train_period": f"{train_scaled['utc'].min()} to {train_scaled['utc'].max()}",
            "test_period": f"{test_scaled['utc'].min()} to {test_scaled['utc'].max()}",
            "split_ratio": self.processing.train_test_split,
            "temporal_features_added": self.processing.add_temporal_features,
        }

        metadata_file = output_dir / "metadata.txt"
        with open(metadata_file, "w") as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        logger.info(f"Saved metadata: {metadata_file}")

    def run(self) -> None:
        """
        Run the complete data processing pipeline for all datasets.

        Processes:
        1. total_production.csv -> data/processed/total/
        2. north_tracker.csv -> data/processed/north/
        3. south_tracker.csv -> data/processed/south/
        """
        logger.info("Starting data processing pipeline...")

        # Get preprocessed data directory
        preprocessed_dir = self.config.preprocessing.output_dir

        # Process each dataset
        datasets = [
            ("total_production.csv", "total"),
            ("north_tracker.csv", "north"),
            ("south_tracker.csv", "south"),
        ]

        for filename, output_name in datasets:
            input_file = preprocessed_dir / filename
            if not input_file.exists():
                logger.error(f"Input file not found: {input_file}")
                continue

            self.process_dataset(input_file, output_name)
            # Create new scaler for each dataset
            self.scaler = StandardScaler()

        logger.success("Data processing pipeline completed successfully!")
        logger.info(f"Output directory: {self.processing.output_dir}")
