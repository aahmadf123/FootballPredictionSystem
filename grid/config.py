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
    # Core APIs
    cfbd_api_key: Optional[str] = None
    espn_api_key: Optional[str] = None
    
    # Weather APIs
    weather_api_key: Optional[str] = None
    weatherapi_key: Optional[str] = None
    
    # Player & Recruiting
    recruiting_247_api_key: Optional[str] = None
    recruiting_rivals_api_key: Optional[str] = None
    pfr_api_key: Optional[str] = None
    
    # Injury & Transactions
    espn_injury_api_key: Optional[str] = None
    fantasypros_api_key: Optional[str] = None
    
    # Market Data
    odds_api_key: Optional[str] = None
    fanduel_api_key: Optional[str] = None
    draftkings_api_key: Optional[str] = None
    
    # News & Media
    news_feeds: Optional[str] = None
    news_api_key: Optional[str] = None
    twitter_bearer_token: Optional[str] = None
    
    # Officials & Advanced Analytics
    fo_api_key: Optional[str] = None
    pff_api_key: Optional[str] = None
    sis_api_key: Optional[str] = None
    
    # Venue & Travel
    google_maps_api_key: Optional[str] = None
    timezone_api_key: Optional[str] = None
    
    # Database & Storage
    database_url: Optional[str] = "sqlite:///./data/grid.db"
    redis_url: Optional[str] = "redis://localhost:6379"
    
    # Development & Monitoring
    log_level: str = "INFO"
    dev_mode: bool = False
    telemetry_enabled: bool = False
    
    # Performance
    global_rate_limit: int = 10
    cache_ttl: int = 3600
    parallel_fetching: bool = True
    
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
