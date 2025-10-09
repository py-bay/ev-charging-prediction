"""Pydantic models for configuration validation."""

from pathlib import Path
from typing import Any, List

import yaml
from pydantic import BaseModel, Field, field_validator


class DataPaths(BaseModel):
    """Data file paths configuration."""

    pv: Path
    irradiance: Path
    weather: Path

    @field_validator("pv", "irradiance", "weather", mode="before")
    @classmethod
    def convert_to_path(cls, v: Any) -> Path:
        """Convert string to Path object."""
        if isinstance(v, str):
            return Path(v)
        return v


class PVColumns(BaseModel):
    """PV data column names."""

    total: str
    tracker_north: str
    tracker_south: str


class PreprocessingConfig(BaseModel):
    """Preprocessing configuration."""

    output_dir: Path = Field(default=Path("data/preprocessed"))
    irradiance_cols: List[str]
    weather_cols: List[str]
    pv_cols: PVColumns

    @field_validator("output_dir", mode="before")
    @classmethod
    def convert_to_path(cls, v: Any) -> Path:
        """Convert string to Path object."""
        if isinstance(v, str):
            return Path(v)
        return v


class ProcessingConfig(BaseModel):
    """Data processing configuration."""

    output_dir: Path = Field(default=Path("data/processed"))
    train_test_split: float = Field(default=0.8, ge=0.0, le=1.0)
    add_temporal_features: bool = Field(default=True)
    temporal_features: List[str] = Field(default_factory=list)

    @field_validator("output_dir", mode="before")
    @classmethod
    def convert_to_path(cls, v: Any) -> Path:
        """Convert string to Path object."""
        if isinstance(v, str):
            return Path(v)
        return v


class ModelConfig(BaseModel):
    """LSTM model configuration."""

    hidden_size: int = Field(default=64)
    num_layers: int = Field(default=2)
    dropout: float = Field(default=0.2, ge=0.0, le=1.0)
    lookback_window: int = Field(default=12)
    batch_size: int = Field(default=32)
    epochs: int = Field(default=50)
    learning_rate: float = Field(default=0.001)
    patience: int = Field(default=10)
    output_dir: Path = Field(default=Path("outputs/models"))
    checkpoint_dir: Path = Field(default=Path("outputs/checkpoints"))

    @field_validator("output_dir", "checkpoint_dir", mode="before")
    @classmethod
    def convert_to_path(cls, v: Any) -> Path:
        """Convert string to Path object."""
        if isinstance(v, str):
            return Path(v)
        return v


class OutputPaths(BaseModel):
    """Output paths configuration."""

    logs: Path
    models: Path
    predictions: Path
    results: Path
    plots: Path

    @field_validator("logs", "models", "predictions", "results", "plots", mode="before")
    @classmethod
    def convert_to_path(cls, v: Any) -> Path:
        """Convert string to Path object."""
        if isinstance(v, str):
            return Path(v)
        return v


class Config(BaseModel):
    """Main configuration model."""

    data_paths: DataPaths
    preprocessing: PreprocessingConfig
    processing: ProcessingConfig
    model: ModelConfig
    output_paths: OutputPaths


def load_config(config_path: Path = Path("config.yaml")) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config YAML file

    Returns:
        Config: Validated configuration object
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return Config(**config_dict)
