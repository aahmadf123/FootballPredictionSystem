"""Special Teams API router."""

from fastapi import APIRouter, HTTPException, Depends
from grid.api.schemas import SpecialTeamsSummary
from grid.data.storage import DataStorage

router = APIRouter()

def get_storage() -> DataStorage:
    from grid.api.main import grid_api
    return grid_api.get_storage()

@router.get("/summary")
async def get_special_teams_summary(
    team_id: str,
    season: int,
    storage: DataStorage = Depends(get_storage)
):
    """Get special teams summary for a team."""
    try:
        # Load special teams data
        st_df = storage.load_data("special_teams")
        
        if st_df.empty:
            raise HTTPException(status_code=404, detail="No special teams data found")
        
        # Filter for team and season
        team_st_data = st_df[
            (st_df["team_id"] == team_id) & 
            (st_df.get("season", season) == season)
        ]
        
        if team_st_data.empty:
            raise HTTPException(status_code=404, detail="No data found for this team/season")
        
        # Aggregate season stats
        total_fg_att = team_st_data["fg_att"].sum()
        total_fg_made = team_st_data["fg_made"].sum()
        fg_percentage = total_fg_made / total_fg_att if total_fg_att > 0 else 0.0
        
        # Calculate averages
        punt_net_avg = team_st_data["punt_net_avg"].mean()
        punt_return_avg = team_st_data["pr_ypa"].mean()
        kick_return_avg = team_st_data["kr_ypa"].mean()
        total_blocks = team_st_data["blocks"].sum()
        
        # Load ST features if available
        st_epa_total = 0.0
        avg_fg_distance = 40.0
        
        feature_versions = storage.get_feature_versions()
        if not feature_versions.empty:
            latest_version = feature_versions.iloc[0]["feature_version_id"]
            st_features = storage.load_gold_data("st_features", latest_version)
            
            if not st_features.empty:
                team_st_features = st_features[st_features["team_id"] == team_id]
                if not team_st_features.empty:
                    st_epa_total = team_st_features["st_epa_total"].sum()
                    avg_fg_distance = team_st_features.get("fg_avg_distance", 40.0).mean()
        
        # Calculate rankings (simplified)
        all_teams_st = st_df[st_df.get("season", season) == season]
        team_fg_pcts = all_teams_st.groupby("team_id").apply(
            lambda x: x["fg_made"].sum() / x["fg_att"].sum() if x["fg_att"].sum() > 0 else 0
        ).sort_values(ascending=False)
        
        fg_rank = list(team_fg_pcts.index).index(team_id) + 1 if team_id in team_fg_pcts.index else 50
        
        team_punt_avgs = all_teams_st.groupby("team_id")["punt_net_avg"].mean().sort_values(ascending=False)
        punt_rank = list(team_punt_avgs.index).index(team_id) + 1 if team_id in team_punt_avgs.index else 50
        
        return SpecialTeamsSummary(
            team_id=team_id,
            season=season,
            fg_percentage=float(fg_percentage),
            fg_attempts=int(total_fg_att),
            fg_made=int(total_fg_made),
            avg_fg_distance=float(avg_fg_distance),
            punt_net_avg=float(punt_net_avg),
            punt_return_avg=float(punt_return_avg),
            kick_return_avg=float(kick_return_avg),
            st_epa_total=float(st_epa_total),
            ranking={
                "fg_percentage": fg_rank,
                "punt_net": punt_rank,
                "blocks": min(32, max(1, 33 - total_blocks))
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get special teams summary: {str(e)}")
