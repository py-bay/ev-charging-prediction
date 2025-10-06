"""Pydantic models for configuration validation."""

from pathlib import Path
from typing import Any, Dict, List, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class DataPaths(BaseModel):
    """Data file paths configuration."""

    pv: Path = Field(default=Path("data/pv_data.csv"))
    irradiance: Path = Field(default=Path("data/solar_irradiance.csv"))
    weather: Path = Field(default=Path("data/weather.csv"))

    @field_validator("pv", "irradiance", "weather", mode="before")
    @classmethod
    def convert_to_path(cls, v: Any) -> Path:
        """Convert string to Path object."""
        if isinstance(v, str):
            return Path(v)
        return v


class PreprocessingConfig(BaseModel):
    """Preprocessing configuration."""

    resample_interval: str = Field(default="15T")
    weather_interp_method: str = Field(default="time")
    label_thresholds: List[float] = Field(default=[0.5, 0.8, 1.0])
    household_consumption_kw: float = Field(
        default=0.5, description="Average household consumption in kW"
    )
    train_test_split: float = Field(default=0.8, ge=0.0, le=1.0)
    random_seed: int = Field(default=42)


class FeatureSetsConfig(BaseModel):
    """Feature sets configuration."""

    weather: List[str] = Field(
        default=[
            "temperature_2m",
            "relative_humidity_2m",
            "dewpoint_2m",
            "precipitation",
            "cloud_cover",
            "wind_speed_10m",
            "wind_gusts_10m",
            "pressure_msl",
            "weather_code",
        ]
    )
    irradiance: List[str] = Field(default=["irradiance"])
    combined: List[str] = Field(
        default=[
            "irradiance",
            "temperature_2m",
            "relative_humidity_2m",
            "cloud_cover",
            "precipitation",
            "wind_speed_10m",
            "pressure_msl",
        ]
    )


class FeaturesConfig(BaseModel):
    """Features configuration."""

    sets: FeatureSetsConfig = Field(default_factory=FeatureSetsConfig)
    cyclical_time_features: bool = Field(
        default=True, description="Add sine/cosine encoded time features"
    )


class RandomForestConfig(BaseModel):
    """RandomForest hyperparameters."""

    n_estimators: int = Field(default=100, ge=1)
    max_depth: int | None = Field(default=None)
    min_samples_split: int = Field(default=2, ge=2)
    min_samples_leaf: int = Field(default=1, ge=1)
    random_state: int = Field(default=42)
    n_jobs: int = Field(default=-1)


class LSTMConfig(BaseModel):
    """LSTM hyperparameters."""

    hidden_size: int = Field(default=64, ge=1)
    num_layers: int = Field(default=2, ge=1)
    dropout: float = Field(default=0.2, ge=0.0, le=1.0)
    sequence_length: int = Field(default=12, ge=1, description="Number of timesteps to look back")
    batch_size: int = Field(default=32, ge=1)
    epochs: int = Field(default=50, ge=1)
    learning_rate: float = Field(default=0.001, gt=0.0)
    weight_decay: float = Field(default=1e-5, ge=0.0)
    patience: int = Field(default=10, ge=1, description="Early stopping patience")


class ModelsConfig(BaseModel):
    """Models configuration."""

    enabled: List[Literal["randomforest", "lstm"]] = Field(default=["randomforest", "lstm"])
    randomforest: RandomForestConfig = Field(default_factory=RandomForestConfig)
    lstm: LSTMConfig = Field(default_factory=LSTMConfig)


class OutputPaths(BaseModel):
    """Output paths configuration."""

    models: Path = Field(default=Path("outputs/models"))
    results: Path = Field(default=Path("outputs/results"))
    plots: Path = Field(default=Path("outputs/plots"))
    logs: Path = Field(default=Path("logs"))

    @field_validator("models", "results", "plots", "logs", mode="before")
    @classmethod
    def convert_to_path(cls, v: Any) -> Path:
        """Convert string to Path object."""
        if isinstance(v, str):
            return Path(v)
        return v


class Config(BaseModel):
    """Main configuration model."""

    data_paths: DataPaths = Field(default_factory=DataPaths)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    output_paths: OutputPaths = Field(default_factory=OutputPaths)

    def get_feature_set(self, feature_set_name: str) -> List[str]:
        """Get feature list by name."""
        feature_sets = {
            "weather": self.features.sets.weather,
            "irradiance": self.features.sets.irradiance,
            "combined": self.features.sets.combined,
        }
        if feature_set_name not in feature_sets:
            raise ValueError(
                f"Unknown feature set: {feature_set_name}. "
                f"Available: {list(feature_sets.keys())}"
            )
        return feature_sets[feature_set_name]


def load_config(config_path: str | Path = "config/config.yaml") -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config YAML file

    Returns:
        Config: Validated configuration object
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return Config(**config_dict)
