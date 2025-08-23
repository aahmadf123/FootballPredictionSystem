"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    timestamp: datetime
    version: str
    uptime_seconds: float


class UpdateRequest(BaseModel):
    """Request to update data sources."""
    sources: List[str] = Field(..., description="List of data sources to update")
    force: bool = Field(False, description="Force update even if recently updated")


class UpdateResponse(BaseModel):
    """Response from data update operation."""
    success: bool
    updated_sources: List[str]
    failed_sources: List[str]
    timestamp: datetime
    message: str


class TeamInfo(BaseModel):
    """Team information."""
    team_id: str
    name: str
    league: str
    conference: Optional[str] = None
    division: Optional[str] = None
    season: int


class TeamsResponse(BaseModel):
    """Response for teams endpoint."""
    teams: List[TeamInfo]
    count: int
    league: Optional[str] = None
    season: Optional[int] = None


class GameInfo(BaseModel):
    """Game information."""
    game_id: str
    league: str
    season: int
    week: int
    home_team: str
    away_team: str
    kickoff_utc: datetime
    venue: Optional[str] = None
    status: str
    home_score: Optional[int] = None
    away_score: Optional[int] = None


class GamesResponse(BaseModel):
    """Response for games endpoint."""
    games: List[GameInfo]
    count: int
    league: Optional[str] = None
    season: Optional[int] = None
    week: Optional[int] = None


class PredictionDriver(BaseModel):
    """Individual prediction driver/factor."""
    name: str
    contribution: float
    description: Optional[str] = None


class ConformalInterval(BaseModel):
    """Conformal prediction interval."""
    level: float = Field(..., description="Confidence level (e.g., 0.8, 0.9)")
    lower: float
    upper: float


class MarginPrediction(BaseModel):
    """Margin prediction with uncertainty."""
    mu: float = Field(..., description="Mean predicted margin")
    sigma: float = Field(..., description="Standard deviation")


class GamePrediction(BaseModel):
    """Game prediction response."""
    game_id: str
    model_version: str
    feature_version_id: str
    p_home: float = Field(..., ge=0.0, le=1.0, description="Home team win probability")
    p_away: float = Field(..., ge=0.0, le=1.0, description="Away team win probability")
    margin: MarginPrediction
    total_points: Optional[MarginPrediction] = None
    conformal_intervals: Dict[str, ConformalInterval]
    drivers: List[PredictionDriver]
    generated_at: datetime
    uncertainty_type: str = "epistemic+aleatoric"


class ExplanationFactor(BaseModel):
    """SHAP explanation factor."""
    feature: str
    value: float
    shap_value: float
    description: Optional[str] = None


class GameExplanation(BaseModel):
    """Game explanation response."""
    game_id: str
    model_version: str
    shap_factors: List[ExplanationFactor]
    text_rationale: str
    base_prediction: float
    final_prediction: float
    explanation_quality: float = Field(..., ge=0.0, le=1.0)


class CounterfactualDelta(BaseModel):
    """Counterfactual scenario delta."""
    QB_out: Optional[bool] = None
    wind: Optional[float] = None
    neutral_field: Optional[bool] = None
    kicker_change: Optional[bool] = None
    key_injury: Optional[str] = None
    weather_change: Optional[str] = None


class CounterfactualRequest(BaseModel):
    """Counterfactual analysis request."""
    game_id: str
    deltas: CounterfactualDelta


class CounterfactualResponse(BaseModel):
    """Counterfactual analysis response."""
    original_prediction: GamePrediction
    counterfactual_prediction: GamePrediction
    deltas_applied: Dict[str, Any]
    impact_summary: Dict[str, float]


class PlayerProjection(BaseModel):
    """Player projection data."""
    player_id: str
    name: str
    position: str
    team_id: str
    season: int
    projected_stats: Dict[str, float]
    confidence_intervals: Dict[str, ConformalInterval]
    development_stage: str
    injury_risk: Optional[float] = None


class PlayerProjectionsResponse(BaseModel):
    """Player projections response."""
    player: PlayerProjection
    historical_performance: List[Dict[str, Any]]
    peer_comparisons: List[Dict[str, Any]]


class SpecialTeamsSummary(BaseModel):
    """Special teams summary."""
    team_id: str
    season: int
    fg_percentage: float
    fg_attempts: int
    fg_made: int
    avg_fg_distance: float
    punt_net_avg: float
    punt_return_avg: float
    kick_return_avg: float
    st_epa_total: float
    ranking: Dict[str, int]


class TalentMetrics(BaseModel):
    """Team talent metrics."""
    team_id: str
    season: int
    composite_talent: float
    class_rank: int
    portal_net_impact: float
    talent_percentile: float
    talent_trajectory: str


class MomentumEvent(BaseModel):
    """Momentum event in game."""
    play_id: int
    quarter: int
    time_remaining: str
    event_type: str
    wpa_change: float
    leverage: float
    description: str


class MomentumAnalysis(BaseModel):
    """Game momentum analysis."""
    game_id: str
    momentum_events: List[MomentumEvent]
    momentum_curve: List[Dict[str, float]]
    final_momentum_impact: float


class HistoricalAnalogue(BaseModel):
    """Historical game analogue."""
    game_id: str
    season: int
    week: int
    teams: str
    similarity_score: float
    outcome: str
    key_similarities: List[str]
    final_score: str


class HistoricalAnaloguesResponse(BaseModel):
    """Historical analogues response."""
    query_game_id: str
    analogues: List[HistoricalAnalogue]
    similarity_methodology: str


class TeamComparison(BaseModel):
    """Team vs team comparison."""
    team_a: str
    team_b: str
    season: int
    comparison_metrics: Dict[str, Dict[str, float]]
    head_to_head_record: Dict[str, Any]
    strength_advantages: Dict[str, List[str]]
    predicted_outcome: Dict[str, float]


class CalibrationBucket(BaseModel):
    """Calibration bucket data."""
    bucket: int
    predicted_prob: float
    actual_rate: float
    count: int
    brier_contribution: float


class CalibrationMetrics(BaseModel):
    """Weekly calibration metrics."""
    week: int
    season: int
    buckets: List[CalibrationBucket]
    overall_brier: float
    overall_logloss: float
    ece: float
    accuracy: float
    coverage_80: Optional[float] = None
    coverage_90: Optional[float] = None


class CalibrationResponse(BaseModel):
    """Calibration analysis response."""
    current_week: CalibrationMetrics
    historical_trend: List[CalibrationMetrics]
    model_health: str
    recommendations: List[str]


class ExportFormat(BaseModel):
    """Export format specification."""
    format: str = Field(..., regex="^(csv|xlsx|json|parquet)$")
    tables: List[str]
    date_range: Optional[Dict[str, str]] = None
    filters: Optional[Dict[str, Any]] = None


class ExportResponse(BaseModel):
    """Export response."""
    download_url: str
    expires_at: datetime
    format: str
    size_bytes: int
    record_count: int


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    message: str
    timestamp: datetime
    request_id: Optional[str] = None


class PaginatedResponse(BaseModel):
    """Base paginated response."""
    page: int = 1
    per_page: int = 50
    total: int
    pages: int
    has_next: bool
    has_prev: bool


class ModelVersion(BaseModel):
    """Model version information."""
    model_version: str
    feature_version_id: str
    created_at: datetime
    performance_metrics: Dict[str, float]
    status: str
    description: Optional[str] = None


class SystemStatus(BaseModel):
    """System status information."""
    api_status: str
    data_freshness: Dict[str, datetime]
    model_versions: List[ModelVersion]
    last_predictions_run: datetime
    active_jobs: List[str]
    system_load: Dict[str, float]
