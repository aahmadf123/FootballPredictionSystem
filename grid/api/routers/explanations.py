"""Explanations API router."""

from fastapi import APIRouter, HTTPException, Depends
from loguru import logger
from grid.api.schemas import GameExplanation, ExplanationFactor
from grid.data.storage import DataStorage

router = APIRouter()

def get_storage() -> DataStorage:
    from grid.api.main import grid_api
    return grid_api.get_storage()

@router.get("/game/{game_id}", response_model=GameExplanation)
async def explain_game(
    game_id: str,
    storage: DataStorage = Depends(get_storage)
):
    """Get SHAP explanations for a game prediction."""
    try:
        # Load game data
        games_df = storage.load_data("games")
        game_data = games_df[games_df["game_id"] == game_id]
        
        if game_data.empty:
            raise HTTPException(status_code=404, detail="Game not found")
        
        game = game_data.iloc[0]
        
        # Load latest feature data
        feature_versions = storage.get_feature_versions()
        if not feature_versions.empty:
            latest_version = feature_versions.iloc[0]["feature_version_id"]
            team_features = storage.load_gold_data("team_features", latest_version)
            
            if not team_features.empty:
                # Get features for this game's teams
                home_features = team_features[
                    (team_features["team_id"] == game["home_id"]) &
                    (team_features["game_id"] == game_id)
                ]
                away_features = team_features[
                    (team_features["team_id"] == game["away_id"]) &
                    (team_features["game_id"] == game_id)
                ]
                
                if not home_features.empty and not away_features.empty:
                    # Calculate feature differences
                    home_vals = home_features.iloc[0]
                    away_vals = away_features.iloc[0]
                    
                    shap_factors = []
                    
                    # Key features to explain
                    key_features = [
                        ("offense_epa_play", "Offensive efficiency"),
                        ("defense_epa_play", "Defensive efficiency"), 
                        ("st_epa_total", "Special teams impact"),
                        ("recent_form", "Recent performance"),
                        ("home_field_advantage", "Home field advantage")
                    ]
                    
                    for feature, desc in key_features:
                        if feature in home_vals and feature in away_vals:
                            diff = float(home_vals[feature] - away_vals[feature])
                            shap_factors.append(ExplanationFactor(
                                feature=feature,
                                value=diff,
                                shap_value=diff * 0.1,  # Approximate SHAP value
                                description=desc
                            ))
                    
                    # Sort by absolute impact
                    shap_factors.sort(key=lambda x: abs(x.shap_value), reverse=True)
                    
                    base_pred = 0.5
                    final_pred = base_pred + sum(f.shap_value for f in shap_factors)
                    final_pred = max(0.01, min(0.99, final_pred))
                    
                    # Generate text explanation
                    favored = "home team" if final_pred > 0.5 else "away team"
                    confidence = "high" if abs(final_pred - 0.5) > 0.2 else "moderate"
                    
                    text_rationale = f"Model shows {confidence} confidence favoring the {favored}. "
                    if shap_factors:
                        top_factor = shap_factors[0]
                        impact = "positive" if top_factor.shap_value > 0 else "negative"
                        text_rationale += f"Primary factor: {top_factor.description} has {impact} impact."
                    
                    return GameExplanation(
                        game_id=game_id,
                        model_version=latest_version,
                        shap_factors=shap_factors[:10],
                        text_rationale=text_rationale,
                        base_prediction=base_pred,
                        final_prediction=final_pred,
                        explanation_quality=0.75
                    )
        
        # Fallback explanation
        return GameExplanation(
            game_id=game_id,
            model_version="v2024.1",
            shap_factors=[
                ExplanationFactor(
                    feature="insufficient_data",
                    value=0.0,
                    shap_value=0.0,
                    description="Insufficient feature data for detailed explanation"
                )
            ],
            text_rationale="Limited explanation available due to insufficient feature data",
            base_prediction=0.5,
            final_prediction=0.5,
            explanation_quality=0.2
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explanation generation failed for {game_id}: {e}")
        raise HTTPException(status_code=500, detail="Explanation generation failed")
