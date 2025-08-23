"""Players API router."""

from fastapi import APIRouter, HTTPException, Depends
from grid.api.schemas import PlayerProjectionsResponse, PlayerProjection, ConformalInterval
from grid.data.storage import DataStorage

router = APIRouter()

def get_storage() -> DataStorage:
    from grid.api.main import grid_api
    return grid_api.get_storage()

@router.get("/{player_id}/projections", response_model=PlayerProjectionsResponse)
async def get_player_projections(
    player_id: str,
    storage: DataStorage = Depends(get_storage)
):
    """Get projections for a specific player."""
    try:
        # Load player data
        players_df = storage.load_data("players")
        player_data = players_df[players_df["player_id"] == player_id]
        
        if player_data.empty:
            raise HTTPException(status_code=404, detail="Player not found")
        
        player = player_data.iloc[0]
        
        # Load player features if available
        feature_versions = storage.get_feature_versions()
        projected_stats = {}
        confidence_intervals = {}
        historical_performance = []
        
        if not feature_versions.empty:
            latest_version = feature_versions.iloc[0]["feature_version_id"]
            player_features = storage.load_gold_data("player_features", latest_version)
            
            if not player_features.empty:
                player_stats = player_features[player_features["player_id"] == player_id]
                
                if not player_stats.empty:
                    stats = player_stats.iloc[0]
                    
                    # Generate position-specific projections
                    if player["pos"] == "QB":
                        projected_stats = {
                            "dropback_epa": float(stats.get("qb_dropback_epa_play", 0.0)),
                            "success_rate": float(stats.get("qb_success_rate", 0.6)),
                            "pressure_rate": float(stats.get("qb_pressure_rate", 0.25))
                        }
                        confidence_intervals = {
                            "dropback_epa": ConformalInterval(
                                level=0.8,
                                lower=projected_stats["dropback_epa"] - 0.05,
                                upper=projected_stats["dropback_epa"] + 0.05
                            )
                        }
                    elif player["pos"] == "RB":
                        projected_stats = {
                            "rushing_epa": float(stats.get("rb_rush_epa_carry", 0.0)),
                            "yards_per_carry": float(stats.get("rb_yards_carry", 4.0)),
                            "breakaway_rate": float(stats.get("rb_breakaway_rate", 0.1))
                        }
                        confidence_intervals = {
                            "rushing_epa": ConformalInterval(
                                level=0.8,
                                lower=projected_stats["rushing_epa"] - 0.03,
                                upper=projected_stats["rushing_epa"] + 0.03
                            )
                        }
                    elif player["pos"] in ["WR", "TE"]:
                        projected_stats = {
                            "target_share": float(stats.get("recv_target_share", 0.15)),
                            "epa_per_target": float(stats.get("recv_epa_target", 0.1)),
                            "air_yards": float(stats.get("recv_air_yards", 12.0))
                        }
                        confidence_intervals = {
                            "target_share": ConformalInterval(
                                level=0.8,
                                lower=projected_stats["target_share"] - 0.05,
                                upper=projected_stats["target_share"] + 0.05
                            )
                        }
        
        # Determine development stage
        age = float(player.get("age", 25))
        if age < 24:
            development_stage = "developing"
        elif age < 30:
            development_stage = "prime"
        else:
            development_stage = "veteran"
        
        # Calculate injury risk (simplified)
        injury_risk = min(0.3, max(0.05, (age - 20) * 0.01))
        
        # Get peer comparisons (simplified)
        same_position = players_df[
            (players_df["pos"] == player["pos"]) &
            (players_df["team_id"] == player["team_id"]) &
            (players_df["player_id"] != player_id)
        ]
        
        peer_comparisons = []
        for _, peer in same_position.head(3).iterrows():
            peer_comparisons.append({
                "player_id": peer["player_id"],
                "name": peer["player_id"],  # Would map to actual name
                "similarity_score": 0.8,
                "comparison_metrics": {"age": peer["age"], "experience": peer.get("experience", 0)}
            })
        
        return PlayerProjectionsResponse(
            player=PlayerProjection(
                player_id=player_id,
                name=player_id,  # Would map to actual name
                position=player["pos"],
                team_id=player["team_id"],
                season=int(player["season"]),
                projected_stats=projected_stats,
                confidence_intervals=confidence_intervals,
                development_stage=development_stage,
                injury_risk=injury_risk
            ),
            historical_performance=historical_performance,
            peer_comparisons=peer_comparisons
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get player projections: {str(e)}")
