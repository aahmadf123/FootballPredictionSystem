"""Games API router."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Query
import pandas as pd

from grid.api.schemas import GamesResponse, GameInfo
from grid.data.storage import DataStorage


router = APIRouter()


def get_storage() -> DataStorage:
    from grid.api.main import grid_api
    return grid_api.get_storage()


@router.get("/", response_model=GamesResponse)
async def get_games(
    league: Optional[str] = Query(None, regex="^(NFL|NCAA)$"),
    season: Optional[int] = Query(None, ge=1900, le=2100),
    week: Optional[int] = Query(None, ge=0, le=25),
    storage: DataStorage = Depends(get_storage)
):
    """Get list of games."""
    try:
        games_df = storage.load_data("games")
        
        if league:
            games_df = games_df[games_df["league"] == league]
        
        if season:
            games_df = games_df[games_df["season"] == season]
            
        if week is not None:
            games_df = games_df[games_df["week"] == week]
        
        games = []
        for _, game in games_df.iterrows():
            games.append(GameInfo(
                game_id=game["game_id"],
                league=game["league"],
                season=game["season"],
                week=game["week"],
                home_team=game["home_id"],
                away_team=game["away_id"],
                kickoff_utc=game["kickoff_utc"],
                venue=game.get("venue_id"),
                status=game["status"],
                home_score=game.get("home_score"),
                away_score=game.get("away_score")
            ))
        
        return GamesResponse(
            games=games,
            count=len(games),
            league=league,
            season=season,
            week=week
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
