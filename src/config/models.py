"""Pydantic models for configuration validation."""

from pathlib import Path
from typing import Any, List, Literal

import yaml
from pydantic import BaseModel


class DataPaths(BaseModel):
    """Data file paths configuration."""

    pv: Path
    irradiance: Path
    weather: Path

    @classmethod
    def convert_to_path(cls, v: Any) -> Path:
        """Convert string to Path object."""
        if isinstance(v, str):
            return Path(v)
        return v


class PreprocessingConfig(BaseModel):
    """Preprocessing configuration."""

    resample_interval: str
    weather_interp_method: str
    label_thresholds: List[float]
    household_consumption_kw: float
    train_test_split: float
    random_seed: int


class FeatureSetsConfig(BaseModel):
    """Feature sets configuration."""

    weather: List[str]
    irradiance: List[str]
    combined: List[str]


class FeaturesConfig(BaseModel):
    """Features configuration."""

    sets: FeatureSetsConfig
    cyclical_time_features: bool


class RandomForestConfig(BaseModel):
    """RandomForest hyperparameters."""

    n_estimators: int
    max_depth: int | None
    min_samples_split: int
    min_samples_leaf: int
    random_state: int
    n_jobs: int


class LSTMConfig(BaseModel):
    """LSTM hyperparameters."""

    hidden_size: int
    num_layers: int
    dropout: float
    sequence_length: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    patience: int


class ModelsConfig(BaseModel):
    """Models configuration."""

    enabled: List[Literal["randomforest", "lstm"]]
    randomforest: RandomForestConfig
    lstm: LSTMConfig


class OutputPaths(BaseModel):
    """Output paths configuration."""

    models: Path
    results: Path
    plots: Path
    logs: Path

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
    features: FeaturesConfig
    models: ModelsConfig
    output_paths: OutputPaths

    def get_feature_set(self, feature_set_name: str) -> List[str]:
        """Get feature list by name."""
        feature_sets = {
            "weather": self.features.sets.weather,
            "irradiance": self.features.sets.irradiance,
            "combined": self.features.sets.combined,
        }
        if feature_set_name not in feature_sets:
            raise ValueError(
                f"Unknown feature set: {feature_set_name}. Available: {list(feature_sets.keys())}"
            )
        return feature_sets[feature_set_name]


def load_config(config_path: Path = "config/config.yaml") -> Config:
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
