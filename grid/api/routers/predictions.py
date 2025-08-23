"""Predictions API router."""

from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Depends, Query
import pandas as pd
from loguru import logger

from grid.api.schemas import (
    GamePrediction, MarginPrediction, ConformalInterval, 
    PredictionDriver, CounterfactualRequest, CounterfactualResponse
)
from grid.data.storage import DataStorage
from grid.config import GridConfig
from grid.models.elo import EloEnsemble


router = APIRouter()


# Placeholder dependency functions - would be imported from main
def get_storage() -> DataStorage:
    from grid.api.main import grid_api
    return grid_api.get_storage()

def get_config() -> GridConfig:
    from grid.api.main import grid_api
    return grid_api.get_config()

def get_elo_ensemble() -> EloEnsemble:
    from grid.api.main import grid_api
    return grid_api.elo_ensemble


@router.get("/game/{game_id}", response_model=GamePrediction)
async def predict_game(
    game_id: str,
    storage: DataStorage = Depends(get_storage),
    config: GridConfig = Depends(get_config),
    elo_ensemble: EloEnsemble = Depends(get_elo_ensemble)
):
    """Get prediction for a specific game."""
    try:
        # Load game data
        games_df = storage.load_data("games")
        game_data = games_df[games_df["game_id"] == game_id]
        
        if game_data.empty:
            raise HTTPException(status_code=404, detail="Game not found")
        
        game = game_data.iloc[0]
        
        # Get Elo prediction
        elo_pred = elo_ensemble.models[game["league"]].predict_game(
            game["home_id"], game["away_id"]
        )
        
        # Create prediction response
        prediction = GamePrediction(
            game_id=game_id,
            model_version="v2024.1",
            feature_version_id="fv_2024_latest",
            p_home=elo_pred["home_win_prob"],
            p_away=elo_pred["away_win_prob"],
            margin=MarginPrediction(
                mu=elo_pred["point_spread"],
                sigma=7.2  # Typical NFL margin std dev
            ),
            total_points=MarginPrediction(
                mu=45.0,  # Placeholder
                sigma=6.5
            ),
            conformal_intervals={
                "win_prob_80": ConformalInterval(
                    level=0.8,
                    lower=max(0.0, elo_pred["home_win_prob"] - 0.08),
                    upper=min(1.0, elo_pred["home_win_prob"] + 0.08)
                ),
                "win_prob_90": ConformalInterval(
                    level=0.9,
                    lower=max(0.0, elo_pred["home_win_prob"] - 0.12),
                    upper=min(1.0, elo_pred["home_win_prob"] + 0.12)
                )
            },
            drivers=[
                PredictionDriver(
                    name="team_strength_diff",
                    contribution=elo_pred["rating_diff"] / 400,
                    description="Difference in team strength ratings"
                ),
                PredictionDriver(
                    name="home_field_advantage",
                    contribution=0.12,
                    description="Home field advantage factor"
                ),
                PredictionDriver(
                    name="recent_form",
                    contribution=0.05,
                    description="Recent team performance"
                )
            ],
            generated_at=datetime.now()
        )
        
        return prediction
        
    except Exception as e:
        logger.error(f"Prediction error for game {game_id}: {e}")
        raise HTTPException(status_code=500, detail="Prediction generation failed")


@router.get("/week/{league}/{season}/{week}")
async def predict_week(
    league: str,
    season: int,
    week: int,
    storage: DataStorage = Depends(get_storage),
    elo_ensemble: EloEnsemble = Depends(get_elo_ensemble)
):
    """Get predictions for all games in a specific week."""
    try:
        # Load games for the week
        games_df = storage.load_data("games")
        week_games = games_df[
            (games_df["league"] == league) & 
            (games_df["season"] == season) & 
            (games_df["week"] == week)
        ]
        
        if week_games.empty:
            raise HTTPException(status_code=404, detail="No games found for specified week")
        
        predictions = []
        
        for _, game in week_games.iterrows():
            try:
                # Get prediction for each game
                prediction = await predict_game(
                    game["game_id"], storage, elo_ensemble
                )
                predictions.append(prediction)
            except Exception as e:
                logger.warning(f"Failed to predict game {game['game_id']}: {e}")
                continue
        
        return {
            "league": league,
            "season": season,
            "week": week,
            "predictions": predictions,
            "count": len(predictions)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Week prediction error: {e}")
        raise HTTPException(status_code=500, detail="Week predictions failed")


@router.post("/counterfactual", response_model=CounterfactualResponse)
async def counterfactual_analysis(
    request: CounterfactualRequest,
    storage: DataStorage = Depends(get_storage),
    elo_ensemble: EloEnsemble = Depends(get_elo_ensemble)
):
    """Perform counterfactual analysis on a game prediction."""
    try:
        # Get original prediction
        original_pred = await predict_game(request.game_id, storage, elo_ensemble)
        
        # Apply counterfactual deltas
        modified_pred = _apply_counterfactual_deltas(original_pred, request.deltas)
        
        # Calculate impact summary
        impact_summary = {
            "win_prob_change": modified_pred.p_home - original_pred.p_home,
            "margin_change": modified_pred.margin.mu - original_pred.margin.mu,
            "total_factors_changed": len([d for d in request.deltas.dict().values() if d is not None])
        }
        
        return CounterfactualResponse(
            original_prediction=original_pred,
            counterfactual_prediction=modified_pred,
            deltas_applied=request.deltas.dict(exclude_none=True),
            impact_summary=impact_summary
        )
        
    except Exception as e:
        logger.error(f"Counterfactual analysis error: {e}")
        raise HTTPException(status_code=500, detail="Counterfactual analysis failed")


@router.get("/team/{team_id}/season/{season}")
async def predict_team_season(
    team_id: str,
    season: int,
    storage: DataStorage = Depends(get_storage),
    elo_ensemble: EloEnsemble = Depends(get_elo_ensemble)
):
    """Get season predictions for a specific team."""
    try:
        # Load team's games for the season
        games_df = storage.load_data("games")
        team_games = games_df[
            ((games_df["home_id"] == team_id) | (games_df["away_id"] == team_id)) &
            (games_df["season"] == season)
        ]
        
        if team_games.empty:
            raise HTTPException(status_code=404, detail="No games found for team/season")
        
        predictions = []
        wins = 0
        total_games = 0
        
        for _, game in team_games.iterrows():
            try:
                pred = await predict_game(game["game_id"], storage, elo_ensemble)
                predictions.append(pred)
                
                # Calculate expected wins
                if game["home_id"] == team_id:
                    wins += pred.p_home
                else:
                    wins += pred.p_away
                total_games += 1
                
            except Exception as e:
                logger.warning(f"Failed to predict game {game['game_id']}: {e}")
                continue
        
        return {
            "team_id": team_id,
            "season": season,
            "predictions": predictions,
            "season_projection": {
                "expected_wins": round(wins, 1),
                "expected_losses": round(total_games - wins, 1),
                "games_remaining": len([p for p in predictions if p.generated_at > datetime.now()]),
                "playoff_probability": min(1.0, max(0.0, (wins - 8) / 4))  # Simplified
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Team season prediction error: {e}")
        raise HTTPException(status_code=500, detail="Team season predictions failed")


@router.get("/live/{game_id}")
async def get_live_prediction(
    game_id: str,
    storage: DataStorage = Depends(get_storage),
    elo_ensemble: EloEnsemble = Depends(get_elo_ensemble)
):
    """Get live in-game prediction (if available)."""
    try:
        # This would integrate with live play-by-play data
        # For now, return enhanced prediction with live context
        
        base_prediction = await predict_game(game_id, storage, elo_ensemble)
        
        # Add live context (placeholder)
        live_context = {
            "quarter": 3,
            "time_remaining": "7:23",
            "score": {"home": 21, "away": 17},
            "possession": "home",
            "field_position": 35,
            "live_win_probability": 0.68,
            "momentum": "home",
            "key_events": [
                "TD pass - 3:45 Q3",
                "Fumble recovery - 9:12 Q3"
            ]
        }
        
        return {
            "game_id": game_id,
            "base_prediction": base_prediction,
            "live_context": live_context,
            "updated_at": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Live prediction error: {e}")
        raise HTTPException(status_code=500, detail="Live prediction failed")


def _apply_counterfactual_deltas(prediction: GamePrediction, deltas) -> GamePrediction:
    """Apply counterfactual deltas to a prediction."""
    # This is a simplified implementation
    # In production, would re-run models with modified features
    
    modified_pred = prediction.copy(deep=True)
    
    # QB out effect
    if deltas.QB_out:
        modified_pred.p_home *= 0.85 if deltas.QB_out else 1.0
        modified_pred.p_away = 1 - modified_pred.p_home
        modified_pred.margin.mu -= 3.5
    
    # Wind effect
    if deltas.wind is not None:
        wind_effect = -deltas.wind * 0.02  # -2% per mph
        modified_pred.p_home += wind_effect
        modified_pred.p_away = 1 - modified_pred.p_home
    
    # Neutral field
    if deltas.neutral_field:
        # Remove home field advantage
        modified_pred.p_home -= 0.06
        modified_pred.p_away = 1 - modified_pred.p_home
        modified_pred.margin.mu -= 2.5
    
    # Kicker change
    if deltas.kicker_change:
        modified_pred.margin.sigma += 0.5  # Increased uncertainty
    
    # Ensure probabilities are valid
    modified_pred.p_home = max(0.01, min(0.99, modified_pred.p_home))
    modified_pred.p_away = 1 - modified_pred.p_home
    
    return modified_pred
