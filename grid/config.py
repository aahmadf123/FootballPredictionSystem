"""Configuration management for Grid Football Prediction System."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class GarbageTimeThresholds(BaseModel):
    """Garbage time threshold configuration."""
    q1: int
    q2: int
    q3: int
    q4: int


class FeatureConfig(BaseModel):
    """Feature engineering configuration."""
    windows: List[int]
    opponent_adjust: bool
    enable_player_level: bool
    enable_special_teams: bool
    enable_garbage_time_filter: bool
    garbage_time_thresholds: Dict[str, GarbageTimeThresholds]
    leverage_smoothing_alpha: float


class PBPTransformerConfig(BaseModel):
    """Play-by-play transformer configuration."""
    enabled: bool
    d_model: int
    layers: int
    heads: int
    seq_len: int


class DRLConfig(BaseModel):
    """Deep reinforcement learning configuration."""
    enabled: bool
    max_adjust_pct: float
    algo: str


class ModelConfig(BaseModel):
    """Model configuration."""
    use_text: bool
    use_gnn: bool
    pbp_transformer: PBPTransformerConfig
    calibrator: str
    drl: DRLConfig


class ConformalConfig(BaseModel):
    """Conformal prediction configuration."""
    levels: List[float]


class UncertaintyConfig(BaseModel):
    """Uncertainty quantification configuration."""
    deep_ensembles: int
    conformal: ConformalConfig


class ServingConfig(BaseModel):
    """Serving configuration."""
    onnx: bool
    batch_cache: bool


class PrivacyConfig(BaseModel):
    """Privacy configuration."""
    telemetry_opt_in: bool


class DataConfig(BaseModel):
    """Data configuration."""
    update_intervals: Dict[str, str]


class AppConfig(BaseModel):
    """Application configuration."""
    port: int
    data_dir: str
    leagues: List[str]
    model_version_pin: Optional[str]


class GridConfig(BaseModel):
    """Main configuration model."""
    app: AppConfig
    data: DataConfig
    features: FeatureConfig
    models: ModelConfig
    uncertainty: UncertaintyConfig
    serving: ServingConfig
    privacy: PrivacyConfig


class Settings(BaseSettings):
    """Environment settings."""
    cfbd_api_key: Optional[str] = None
    weather_api_key: Optional[str] = None
    news_feeds: Optional[str] = None
    
    class Config:
        env_file = ".env"


def load_config(config_path: str = "config.yaml") -> GridConfig:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, "r") as f:
        config_data = yaml.safe_load(f)
    
    return GridConfig(**config_data)


def get_settings() -> Settings:
    """Get environment settings."""
    return Settings()


def get_data_dir(config: GridConfig) -> Path:
    """Get data directory path."""
    data_dir = Path(config.app.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def ensure_data_structure(data_dir: Path) -> None:
    """Ensure data directory structure exists."""
    subdirs = [
        "raw",
        "bronze", 
        "silver",
        "gold/feature_store",
        "snapshots/packs"
    ]
    
    for subdir in subdirs:
        (data_dir / subdir).mkdir(parents=True, exist_ok=True)
