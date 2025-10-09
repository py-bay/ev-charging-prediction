"""Pydantic models for configuration validation."""

from pathlib import Path
from typing import Any, Dict, List

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


class OutputPaths(BaseModel):
    """Output paths configuration."""

    logs: Path

    @field_validator("logs", mode="before")
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
