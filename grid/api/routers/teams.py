"""Teams API router."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Query
import pandas as pd

from grid.api.schemas import TeamsResponse, TeamInfo
from grid.data.storage import DataStorage


router = APIRouter()


def get_storage() -> DataStorage:
    from grid.api.main import grid_api
    return grid_api.get_storage()


@router.get("/", response_model=TeamsResponse)
async def get_teams(
    league: Optional[str] = Query(None, regex="^(NFL|NCAA)$"),
    season: Optional[int] = Query(None, ge=1900, le=2100),
    storage: DataStorage = Depends(get_storage)
):
    """Get list of teams."""
    try:
        teams_df = storage.load_data("teams")
        
        if league:
            teams_df = teams_df[teams_df["league"] == league]
        
        if season:
            teams_df = teams_df[teams_df["season"] == season]
        
        teams = []
        for _, team in teams_df.iterrows():
            teams.append(TeamInfo(
                team_id=team["team_id"],
                name=team["team_id"],  # Would map to actual names
                league=team["league"],
                conference=team.get("conf"),
                division=team.get("division"),
                season=team["season"]
            ))
        
        return TeamsResponse(
            teams=teams,
            count=len(teams),
            league=league,
            season=season
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{team_id}")
async def get_team(
    team_id: str,
    season: Optional[int] = Query(None),
    storage: DataStorage = Depends(get_storage)
):
    """Get specific team information."""
    try:
        teams_df = storage.load_data("teams")
        team_data = teams_df[teams_df["team_id"] == team_id]
        
        if season:
            team_data = team_data[team_data["season"] == season]
        
        if team_data.empty:
            raise HTTPException(status_code=404, detail="Team not found")
        
        team = team_data.iloc[0]
        return TeamInfo(
            team_id=team["team_id"],
            name=team["team_id"],
            league=team["league"],
            conference=team.get("conf"),
            division=team.get("division"),
            season=team["season"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
