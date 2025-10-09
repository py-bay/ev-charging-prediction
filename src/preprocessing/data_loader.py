"""Data loading and preprocessing pipeline for PV solar tracker prediction."""

import pandas as pd
from pathlib import Path
from typing import Tuple


def load_pv_data(file_path: str = "data/pv_data.csv") -> pd.DataFrame:
    """
    Load PV production data with 15min intervals.

    Args:
        file_path: Path to the PV data CSV file

    Returns:
        DataFrame with timestamp index and PV production columns
    """
    df = pd.read_csv(file_path, sep=";")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    # Convert timezone-naive to UTC (assuming local time is already in UTC)
    df.index = df.index.tz_localize("UTC")

    return df


def load_irradiance_data(file_path: str = "data/irradiance.csv") -> pd.DataFrame:
    """
    Load irradiance data with 15min intervals.

    Args:
        file_path: Path to the irradiance CSV file

    Returns:
        DataFrame with timestamp index and irradiance columns
    """
    df = pd.read_csv(file_path)

    # Parse datetime from dt_iso column
    df["timestamp"] = pd.to_datetime(df["dt_iso"], format="%Y-%m-%d %H:%M:%S %z UTC")
    df = df.set_index("timestamp")

    # Select only relevant irradiance columns
    irradiance_cols = [
        "ghi_cloudy_sky",
        "dni_cloudy_sky",
        "dhi_cloudy_sky",
        "ghi_clear_sky",
        "dni_clear_sky",
        "dhi_clear_sky",
    ]

    return df[irradiance_cols]


def load_weather_data(file_path: str = "data/weather.csv") -> pd.DataFrame:
    """
    Load weather data with hourly intervals (needs interpolation to 15min).

    Args:
        file_path: Path to the weather CSV file

    Returns:
        DataFrame with timestamp index and weather columns
    """
    # Skip the first 2 metadata rows
    df = pd.read_csv(file_path, skiprows=2)

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
    df.columns = [
        "temperature_2m",
        "cloud_cover",
        "wind_speed_10m",
        "precipitation",
        "weather_code",
        "relative_humidity_2m",
        "pressure_msl",
        "cloud_cover_low",
        "cloud_cover_mid",
        "cloud_cover_high",
        "wind_gusts_10m",
    ]

    return df


def interpolate_weather_to_15min(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate hourly weather data to 15min intervals.

    Args:
        weather_df: Weather DataFrame with hourly data

    Returns:
        DataFrame with 15min intervals
    """
    # Create a new index with 15min frequency
    start = weather_df.index.min()
    end = weather_df.index.max()
    new_index = pd.date_range(start=start, end=end, freq="15min")

    # Reindex and interpolate
    weather_15min = weather_df.reindex(weather_df.index.union(new_index))
    weather_15min = weather_15min.interpolate(method="linear")
    weather_15min = weather_15min.loc[new_index]

    return weather_15min


def find_common_timeframe(
    pv_df: pd.DataFrame, irradiance_df: pd.DataFrame, weather_df: pd.DataFrame
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

    return start, end


def merge_datasets(
    pv_df: pd.DataFrame, irradiance_df: pd.DataFrame, weather_df: pd.DataFrame
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
    # Find common timeframe
    start, end = find_common_timeframe(pv_df, irradiance_df, weather_df)

    # Filter all datasets to common timeframe
    pv_filtered = pv_df.loc[start:end]
    irradiance_filtered = irradiance_df.loc[start:end]
    weather_filtered = weather_df.loc[start:end]

    # Merge on timestamp index
    merged = pv_filtered.join(irradiance_filtered, how="inner")
    merged = merged.join(weather_filtered, how="inner")

    return merged


def create_total_production_dataset(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create dataset with total PV production.

    Args:
        merged_df: Merged DataFrame with all data

    Returns:
        DataFrame with total production and all features
    """
    # Select total production and all irradiance/weather columns
    pv_col = "Solarproduktion"

    # Get irradiance and weather columns (everything that's not PV-specific)
    irradiance_cols = [col for col in merged_df.columns if "sky" in col]
    weather_cols = [
        col
        for col in merged_df.columns
        if col not in irradiance_cols
        and col != pv_col
        and not col.startswith("Solarproduktion Tracker")
        and not col.startswith("Ladezustand")
        and not col.startswith("Batterie")
        and not col.startswith("Netzeinspeisung")
        and not col.startswith("Netzbezug")
        and not col.startswith("Hausverbrauch")
        and not col.startswith("Wallbox")
        and not col.startswith("Σ")
    ]

    result = merged_df[[pv_col] + irradiance_cols + weather_cols].copy()
    result = result.reset_index()
    # Rename the index column to 'utc'
    if result.columns[0] in ["timestamp", "index"]:
        result = result.rename(columns={result.columns[0]: "utc"})

    return result


def create_tracker_dataset(
    merged_df: pd.DataFrame, tracker_name: str, tracker_col: str
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
    # Get irradiance and weather columns
    irradiance_cols = [col for col in merged_df.columns if "sky" in col]
    weather_cols = [
        col
        for col in merged_df.columns
        if col not in irradiance_cols
        and not col.startswith("Solarproduktion")
        and not col.startswith("Ladezustand")
        and not col.startswith("Batterie")
        and not col.startswith("Netzeinspeisung")
        and not col.startswith("Netzbezug")
        and not col.startswith("Hausverbrauch")
        and not col.startswith("Wallbox")
        and not col.startswith("Σ")
    ]

    result = merged_df[[tracker_col] + irradiance_cols + weather_cols].copy()
    result = result.reset_index()
    # Rename the index column to 'utc' and tracker column
    rename_dict = {tracker_col: f"pv_production_{tracker_name}"}
    if result.columns[0] in ["timestamp", "index"]:
        rename_dict[result.columns[0]] = "utc"
    result = result.rename(columns=rename_dict)

    return result


def run_preprocessing_pipeline(
    output_dir: str = "data/preprocessed",
) -> None:
    """
    Run the complete preprocessing pipeline.

    Args:
        output_dir: Directory to save preprocessed data
    """
    print("Loading PV data...")
    pv_df = load_pv_data()
    print(f"PV data: {len(pv_df)} rows, {pv_df.index.min()} to {pv_df.index.max()}")

    print("\nLoading irradiance data...")
    irradiance_df = load_irradiance_data()
    print(f"Irradiance data: {len(irradiance_df)} rows, {irradiance_df.index.min()} to {irradiance_df.index.max()}")

    print("\nLoading weather data...")
    weather_df = load_weather_data()
    print(f"Weather data: {len(weather_df)} rows, {weather_df.index.min()} to {weather_df.index.max()}")

    print("\nInterpolating weather data to 15min intervals...")
    weather_15min = interpolate_weather_to_15min(weather_df)
    print(f"Interpolated weather data: {len(weather_15min)} rows")

    print("\nMerging datasets...")
    merged_df = merge_datasets(pv_df, irradiance_df, weather_15min)
    print(f"Merged data: {len(merged_df)} rows, {merged_df.index.min()} to {merged_df.index.max()}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\nCreating total production dataset...")
    total_df = create_total_production_dataset(merged_df)
    output_file = output_path / "total_production.csv"
    total_df.to_csv(output_file, index=False)
    print(f"Saved: {output_file} ({len(total_df)} rows, {len(total_df.columns)} columns)")

    print("\nCreating north tracker dataset (Tracker 1)...")
    north_df = create_tracker_dataset(merged_df, "north", "Solarproduktion Tracker 1")
    output_file = output_path / "north_tracker.csv"
    north_df.to_csv(output_file, index=False)
    print(f"Saved: {output_file} ({len(north_df)} rows, {len(north_df.columns)} columns)")

    print("\nCreating south tracker dataset (Tracker 2)...")
    south_df = create_tracker_dataset(merged_df, "south", "Solarproduktion Tracker 2")
    output_file = output_path / "south_tracker.csv"
    south_df.to_csv(output_file, index=False)
    print(f"Saved: {output_file} ({len(south_df)} rows, {len(south_df.columns)} columns)")

    print("\n[SUCCESS] Preprocessing complete!")
    print(f"  Common timeframe: {merged_df.index.min()} to {merged_df.index.max()}")
    print(f"  Total intervals: {len(merged_df)}")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    run_preprocessing_pipeline()
