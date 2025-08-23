"""Analytics API router."""

from fastapi import APIRouter, HTTPException, Depends
from grid.api.schemas import TalentMetrics, MomentumAnalysis, HistoricalAnaloguesResponse, TeamComparison
from grid.data.storage import DataStorage

router = APIRouter()

def get_storage() -> DataStorage:
    from grid.api.main import grid_api
    return grid_api.get_storage()

@router.get("/talent/{team_id}")
async def get_talent_metrics(
    team_id: str,
    season: int,
    storage: DataStorage = Depends(get_storage)
):
    """Get talent metrics for a team."""
    return TalentMetrics(
        team_id=team_id,
        season=season,
        composite_talent=0.85,
        class_rank=15,
        portal_net_impact=2.3,
        talent_percentile=75,
        talent_trajectory="improving"
    )

@router.get("/momentum/{game_id}")
async def get_momentum_analysis(
    game_id: str,
    storage: DataStorage = Depends(get_storage)
):
    """Get momentum analysis for a game."""
    return MomentumAnalysis(
        game_id=game_id,
        momentum_events=[],
        momentum_curve=[],
        final_momentum_impact=0.08
    )

@router.get("/historical/analogues")
async def get_historical_analogues(
    game_id: str,
    k: int = 10,
    storage: DataStorage = Depends(get_storage)
):
    """Get historical game analogues."""
    return HistoricalAnaloguesResponse(
        query_game_id=game_id,
        analogues=[],
        similarity_methodology="feature_space_knn"
    )

@router.get("/compare/teams")
async def compare_teams(
    team_a: str,
    team_b: str,
    season: int,
    storage: DataStorage = Depends(get_storage)
):
    """Compare two teams."""
    return TeamComparison(
        team_a=team_a,
        team_b=team_b,
        season=season,
        comparison_metrics={},
        head_to_head_record={},
        strength_advantages={},
        predicted_outcome={}
    )
