"""Data schemas for silver tables."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field, validator


class Team(BaseModel):
    """Team information schema."""
    team_id: str
    league: str = Field(..., regex="^(NFL|NCAA)$")
    conf: Optional[str] = None
    division: Optional[str] = None
    venue_id: str
    tz: str
    altitude: int = Field(ge=0)
    surface: str
    coach_id: str
    season: int = Field(ge=1900, le=2100)


class Coach(BaseModel):
    """Coach information schema."""
    coach_id: str
    name: str
    role: str
    tenure: int = Field(ge=0)
    agg_4th: float = Field(ge=0.0, le=1.0)
    motion_rate: float = Field(ge=0.0, le=1.0)
    pa_rate: float = Field(ge=0.0, le=1.0)
    blitz_rate: float = Field(ge=0.0, le=1.0)
    season: int = Field(ge=1900, le=2100)


class Player(BaseModel):
    """Player information schema."""
    player_id: str
    team_id: str
    pos: str
    depth: int = Field(ge=1)
    height: int = Field(ge=60, le=90)  # inches
    weight: int = Field(ge=150, le=400)  # pounds
    age: float = Field(ge=16.0, le=40.0)
    snaps: int = Field(ge=0)
    season: int = Field(ge=1900, le=2100)


class Game(BaseModel):
    """Game information schema."""
    game_id: str
    league: str = Field(..., regex="^(NFL|NCAA)$")
    season: int = Field(ge=1900, le=2100)
    week: int = Field(ge=0, le=25)
    home_id: str
    away_id: str
    kickoff_utc: datetime
    venue_id: str
    ref_crew_id: str
    weather_id: Optional[str] = None
    line_implied: Optional[float] = None
    total_implied: Optional[float] = None
    home_score: Optional[int] = Field(None, ge=0)
    away_score: Optional[int] = Field(None, ge=0)
    status: str = Field(default="scheduled")


class PlayByPlay(BaseModel):
    """Play-by-play data schema."""
    game_id: str
    play_id: int = Field(ge=1)
    sec_left: int = Field(ge=0, le=3600)
    quarter: int = Field(ge=1, le=5)
    down: Optional[int] = Field(None, ge=1, le=4)
    dist: Optional[int] = Field(None, ge=0)
    yardline_100: int = Field(ge=0, le=100)
    offense_id: str
    defense_id: str
    play_type: str
    yards: int = Field(ge=-50, le=110)
    ep_before: float
    ep_after: float
    epa: float
    wpa: float


class Injury(BaseModel):
    """Injury report schema."""
    game_id: str
    player_id: str
    status: str
    designation: str
    note: Optional[str] = None
    impact: int = Field(ge=0, le=10)
    as_of: datetime


class SpecialTeams(BaseModel):
    """Special teams statistics schema."""
    game_id: str
    team_id: str
    fg_att: int = Field(ge=0)
    fg_made: int = Field(ge=0)
    fg_dist_bins: Dict[str, Any]
    punt_net_avg: float
    pr_ypa: float = Field(ge=0.0)
    kr_ypa: float = Field(ge=0.0)
    blocks: int = Field(ge=0)

    @validator('fg_made')
    def fg_made_not_exceed_attempts(cls, v, values):
        if 'fg_att' in values and v > values['fg_att']:
            raise ValueError('fg_made cannot exceed fg_att')
        return v


class Recruiting(BaseModel):
    """Recruiting data schema."""
    team_id: str
    season: int = Field(ge=1900, le=2100)
    composite_talent: float = Field(ge=0.0, le=1.0)
    class_rank: int = Field(ge=1, le=150)


class Transfer(BaseModel):
    """Transfer portal data schema."""
    team_id: str
    season: int = Field(ge=1900, le=2100)
    player_id: str
    direction: str = Field(..., regex="^(in|out)$")
    prev_team: Optional[str] = None
    proj_role: Optional[str] = None


class RefCrew(BaseModel):
    """Referee crew data schema."""
    ref_crew_id: str
    season: int = Field(ge=1900, le=2100)
    penalties_per_game: float = Field(ge=0.0)
    std: float = Field(ge=0.0)
    pace_adj: float


class News(BaseModel):
    """News article schema."""
    news_id: str
    published_at: datetime
    url: str
    team_ids: List[str]
    player_ids: List[str]
    coach_ids: List[str]
    summary: str
    impact: Dict[str, Any]
    sentiment: float = Field(ge=-1.0, le=1.0)


class Situation(BaseModel):
    """Game situation schema."""
    game_id: str
    play_id: int = Field(ge=1)
    is_garbage_time: bool
    leverage: float = Field(ge=0.0)


class Calibration(BaseModel):
    """Model calibration tracking schema."""
    pred_id: str
    game_id: str
    p_hat: float = Field(ge=0.0, le=1.0)
    y: int = Field(ge=0, le=1)
    bucket: int = Field(ge=0, le=9)
    model_version: str
    created_at: datetime


def validate_dataframe(df: pd.DataFrame, schema_class: BaseModel) -> bool:
    """Validate a pandas DataFrame against a Pydantic schema."""
    try:
        for _, row in df.iterrows():
            schema_class(**row.to_dict())
        return True
    except Exception as e:
        print(f"Validation error: {e}")
        return False


def get_schema_fields(schema_class: BaseModel) -> Dict[str, str]:
    """Get field names and types from a Pydantic schema."""
    return {name: str(field.type_) for name, field in schema_class.__fields__.items()}


def create_empty_dataframe(schema_class: BaseModel) -> pd.DataFrame:
    """Create an empty DataFrame with correct schema columns."""
    fields = get_schema_fields(schema_class)
    return pd.DataFrame(columns=list(fields.keys()))
