"""Preprocessing pipeline for PV solar tracker data."""

import pandas as pd
from pathlib import Path
from typing import Tuple
from loguru import logger

from src.config.models import Config


class PreprocessingPipeline:
    """Pipeline for loading, merging, and preprocessing PV solar tracker data."""

    def __init__(self, config: Config):
        """
        Initialize the preprocessing pipeline.

        Args:
            config: Configuration object with data paths and preprocessing settings
        """
        self.config = config
        self.data_paths = config.data_paths
        self.preprocessing = config.preprocessing

    def load_pv_data(self) -> pd.DataFrame:
        """
        Load PV production data with 15min intervals.

        Returns:
            DataFrame with timestamp index and PV production columns
        """
        logger.info(f"Loading PV data from {self.data_paths.pv}")
        df = pd.read_csv(self.data_paths.pv, sep=";")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        # Convert timezone-naive to UTC (assuming local time is already in UTC)
        df.index = df.index.tz_localize("UTC")

        logger.info(f"PV data loaded: {len(df)} rows, {df.index.min()} to {df.index.max()}")
        return df

    def load_irradiance_data(self) -> pd.DataFrame:
        """
        Load irradiance data with 15min intervals.

        Returns:
            DataFrame with timestamp index and irradiance columns
        """
        logger.info(f"Loading irradiance data from {self.data_paths.irradiance}")
        df = pd.read_csv(self.data_paths.irradiance)

        # Parse datetime from dt_iso column
        df["timestamp"] = pd.to_datetime(df["dt_iso"], format="%Y-%m-%d %H:%M:%S %z UTC")
        df = df.set_index("timestamp")

        # Select only configured irradiance columns
        irradiance_cols = self.preprocessing.irradiance_cols
        df = df[irradiance_cols]

        logger.info(
            f"Irradiance data loaded: {len(df)} rows, {df.index.min()} to {df.index.max()}"
        )
        return df

    def load_weather_data(self) -> pd.DataFrame:
        """
        Load weather data with hourly intervals (needs interpolation to 15min).

        Returns:
            DataFrame with timestamp index and weather columns
        """
        logger.info(f"Loading weather data from {self.data_paths.weather}")
        # Skip the first 2 metadata rows
        df = pd.read_csv(self.data_paths.weather, skiprows=2)

        # Parse datetime and convert to UTC
        df["timestamp"] = pd.to_datetime(df["time"])
        # Assuming data is in local time (Europe/Berlin), convert to UTC
        # Handle DST transitions: prefer DST for ambiguous times, shift forward for nonexistent times
        df["timestamp"] = df["timestamp"].dt.tz_localize(
            "Europe/Berlin", ambiguous=True, nonexistent="shift_forward"
        ).dt.tz_convert("UTC")
        df = df.set_index("timestamp")

        # Drop the original time column
        df = df.drop(columns=["time"])

        # Remove any duplicate timestamps that might occur from DST transitions
        # Keep the first occurrence
        df = df[~df.index.duplicated(keep="first")]

        # Clean column names (remove units)
        df.columns = self.preprocessing.weather_cols

        logger.info(f"Weather data loaded: {len(df)} rows, {df.index.min()} to {df.index.max()}")
        return df

    def interpolate_weather_to_15min(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolate hourly weather data to 15min intervals.

        Args:
            weather_df: Weather DataFrame with hourly data

        Returns:
            DataFrame with 15min intervals
        """
        logger.info("Interpolating weather data to 15min intervals...")
        # Create a new index with 15min frequency
        start = weather_df.index.min()
        end = weather_df.index.max()
        new_index = pd.date_range(start=start, end=end, freq="15min")

        # Reindex and interpolate
        weather_15min = weather_df.reindex(weather_df.index.union(new_index))
        weather_15min = weather_15min.interpolate(method="linear")
        weather_15min = weather_15min.loc[new_index]

        logger.info(f"Weather data interpolated: {len(weather_15min)} rows")
        return weather_15min

    def find_common_timeframe(
        self, pv_df: pd.DataFrame, irradiance_df: pd.DataFrame, weather_df: pd.DataFrame
    ) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Find the common time range across all datasets.

        Args:
            pv_df: PV production DataFrame
            irradiance_df: Irradiance DataFrame
            weather_df: Weather DataFrame

        Returns:
            Tuple of (start_time, end_time)
        """
        start = max(pv_df.index.min(), irradiance_df.index.min(), weather_df.index.min())
        end = min(pv_df.index.max(), irradiance_df.index.max(), weather_df.index.max())

        logger.info(f"Common timeframe: {start} to {end}")
        return start, end

    def merge_datasets(
        self, pv_df: pd.DataFrame, irradiance_df: pd.DataFrame, weather_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge PV, irradiance, and weather data on timestamp.

        Args:
            pv_df: PV production DataFrame
            irradiance_df: Irradiance DataFrame
            weather_df: Weather DataFrame (already interpolated to 15min)

        Returns:
            Merged DataFrame
        """
        logger.info("Merging datasets...")
        # Find common timeframe
        start, end = self.find_common_timeframe(pv_df, irradiance_df, weather_df)

        # Filter all datasets to common timeframe
        pv_filtered = pv_df.loc[start:end]
        irradiance_filtered = irradiance_df.loc[start:end]
        weather_filtered = weather_df.loc[start:end]

        # Merge on timestamp index
        merged = pv_filtered.join(irradiance_filtered, how="inner")
        merged = merged.join(weather_filtered, how="inner")

        logger.info(
            f"Merged data: {len(merged)} rows, {merged.index.min()} to {merged.index.max()}"
        )
        return merged

    def create_total_production_dataset(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create dataset with total PV production.

        Args:
            merged_df: Merged DataFrame with all data

        Returns:
            DataFrame with total production and all features
        """
        pv_col = self.preprocessing.pv_cols.total
        irradiance_cols = self.preprocessing.irradiance_cols
        weather_cols = self.preprocessing.weather_cols

        result = merged_df[[pv_col] + irradiance_cols + weather_cols].copy()
        result = result.reset_index()
        # Rename the index column to 'utc'
        if result.columns[0] in ["timestamp", "index"]:
            result = result.rename(columns={result.columns[0]: "utc"})

        return result

    def create_tracker_dataset(
        self, merged_df: pd.DataFrame, tracker_name: str, tracker_col: str
    ) -> pd.DataFrame:
        """
        Create dataset for a specific tracker.

        Args:
            merged_df: Merged DataFrame with all data
            tracker_name: Name for the tracker (e.g., "north", "south")
            tracker_col: Column name in the DataFrame

        Returns:
            DataFrame with tracker production and all features
        """
        irradiance_cols = self.preprocessing.irradiance_cols
        weather_cols = self.preprocessing.weather_cols

        result = merged_df[[tracker_col] + irradiance_cols + weather_cols].copy()
        result = result.reset_index()
        # Rename the index column to 'utc' and tracker column
        rename_dict = {tracker_col: f"pv_production_{tracker_name}"}
        if result.columns[0] in ["timestamp", "index"]:
            rename_dict[result.columns[0]] = "utc"
        result = result.rename(columns=rename_dict)

        return result

    def run(self) -> None:
        """
        Run the complete preprocessing pipeline.

        Loads all data sources, merges them, and creates three output files:
        1. total_production.csv - Total PV production with all features
        2. north_tracker.csv - North tracker (Tracker 1) with all features
        3. south_tracker.csv - South tracker (Tracker 2) with all features
        """
        logger.info("Starting preprocessing pipeline...")

        # Load all data
        pv_df = self.load_pv_data()
        irradiance_df = self.load_irradiance_data()
        weather_df = self.load_weather_data()

        # Interpolate weather data
        weather_15min = self.interpolate_weather_to_15min(weather_df)

        # Merge all datasets
        merged_df = self.merge_datasets(pv_df, irradiance_df, weather_15min)

        # Create output directory
        output_path = self.preprocessing.output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_path}")

        # Create and save total production dataset
        logger.info("Creating total production dataset...")
        total_df = self.create_total_production_dataset(merged_df)
        output_file = output_path / "total_production.csv"
        total_df.to_csv(output_file, index=False)
        logger.info(f"Saved: {output_file} ({len(total_df)} rows, {len(total_df.columns)} columns)")

        # Create and save north tracker dataset
        logger.info("Creating north tracker dataset...")
        north_df = self.create_tracker_dataset(
            merged_df, "north", self.preprocessing.pv_cols.tracker_north
        )
        output_file = output_path / "north_tracker.csv"
        north_df.to_csv(output_file, index=False)
        logger.info(f"Saved: {output_file} ({len(north_df)} rows, {len(north_df.columns)} columns)")

        # Create and save south tracker dataset
        logger.info("Creating south tracker dataset...")
        south_df = self.create_tracker_dataset(
            merged_df, "south", self.preprocessing.pv_cols.tracker_south
        )
        output_file = output_path / "south_tracker.csv"
        south_df.to_csv(output_file, index=False)
        logger.info(f"Saved: {output_file} ({len(south_df)} rows, {len(south_df.columns)} columns)")

        logger.success("Preprocessing pipeline completed successfully!")
        logger.info(f"  Common timeframe: {merged_df.index.min()} to {merged_df.index.max()}")
        logger.info(f"  Total intervals: {len(merged_df)}")
        logger.info(f"  Output directory: {output_path}")
