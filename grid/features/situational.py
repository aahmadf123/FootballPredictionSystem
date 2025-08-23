"""Situational feature engineering including garbage time detection."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from grid.config import GridConfig
from loguru import logger


class GarbageTimeDetector:
    """Detect garbage time plays based on configurable thresholds."""
    
    def __init__(self, config: GridConfig):
        self.config = config
        self.thresholds = config.features.garbage_time_thresholds
    
    def detect_garbage_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add garbage time flags to play-by-play data."""
        df = df.copy()
        
        # Initialize garbage time column
        df["is_garbage_time"] = False
        
        for _, row in df.iterrows():
            league = row.get("league", "NFL").lower()
            quarter = row.get("quarter", 1)
            score_diff = abs(row.get("score_differential", 0))
            
            # Get thresholds for league
            thresholds = self.thresholds.get(league, self.thresholds.get("nfl"))
            
            # Determine threshold based on quarter
            if quarter == 1:
                threshold = thresholds.q1
            elif quarter == 2:
                threshold = thresholds.q2
            elif quarter == 3:
                threshold = thresholds.q3
            elif quarter >= 4:
                threshold = thresholds.q4
            else:
                threshold = 999  # No garbage time for other situations
            
            # Mark as garbage time if score differential exceeds threshold
            if score_diff >= threshold:
                df.loc[df.index == row.name, "is_garbage_time"] = True
        
        logger.info(f"Marked {df['is_garbage_time'].sum()} plays as garbage time out of {len(df)} total plays")
        return df
    
    def get_non_garbage_stats(self, df: pd.DataFrame, group_cols: List[str], 
                             value_cols: List[str]) -> pd.DataFrame:
        """Calculate statistics excluding garbage time plays."""
        non_garbage = df[~df["is_garbage_time"]]
        
        if non_garbage.empty:
            logger.warning("No non-garbage time plays found")
            return pd.DataFrame()
        
        stats = non_garbage.groupby(group_cols)[value_cols].agg([
            "mean", "sum", "count", "std"
        ]).round(3)
        
        # Flatten column names
        stats.columns = ["_".join(col).strip() for col in stats.columns]
        stats = stats.reset_index()
        
        return stats


class LeverageCalculator:
    """Calculate leverage index for plays."""
    
    def __init__(self, config: GridConfig):
        self.smoothing_alpha = config.features.leverage_smoothing_alpha
    
    def calculate_leverage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate leverage index for each play."""
        df = df.copy()
        
        # Basic leverage calculation based on win probability sensitivity
        df["leverage"] = 0.0
        
        for _, row in df.iterrows():
            quarter = row.get("quarter", 1)
            sec_left = row.get("sec_left", 900)
            score_diff = abs(row.get("score_differential", 0))
            
            # Base leverage increases as game progresses
            time_factor = 1.0 - (sec_left / (15 * 60))  # 0 to 1 as time decreases
            
            # Leverage higher for close games
            score_factor = np.exp(-score_diff / 7.0)  # Exponential decay with score diff
            
            # Quarter effects
            quarter_multiplier = {1: 0.3, 2: 0.6, 3: 0.8, 4: 1.0}.get(quarter, 1.0)
            
            leverage = time_factor * score_factor * quarter_multiplier
            
            # Apply smoothing
            if _ > 0:
                prev_leverage = df.iloc[_ - 1]["leverage"]
                leverage = self.smoothing_alpha * leverage + (1 - self.smoothing_alpha) * prev_leverage
            
            df.loc[df.index == row.name, "leverage"] = min(leverage, 10.0)  # Cap at 10
        
        return df


class SituationalFeatures:
    """Generate situational features for games and plays."""
    
    def __init__(self, config: GridConfig):
        self.config = config
        self.garbage_detector = GarbageTimeDetector(config)
        self.leverage_calc = LeverageCalculator(config)
    
    def process_game_situations(self, pbp_df: pd.DataFrame, 
                               games_df: pd.DataFrame) -> pd.DataFrame:
        """Process situational features for all plays in games."""
        if pbp_df.empty:
            return pd.DataFrame()
        
        # Add score differential
        pbp_df = self._add_score_differential(pbp_df)
        
        # Detect garbage time
        pbp_df = self.garbage_detector.detect_garbage_time(pbp_df)
        
        # Calculate leverage
        pbp_df = self.leverage_calc.calculate_leverage(pbp_df)
        
        # Create situations output
        situations = pbp_df[["game_id", "play_id", "is_garbage_time", "leverage"]].copy()
        
        return situations
    
    def _add_score_differential(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add score differential to play data."""
        df = df.copy()
        
        # This would typically come from cumulative scoring in the game
        # For now, use a simplified approach
        df["score_differential"] = 0
        
        # Group by game and calculate running score differential
        for game_id in df["game_id"].unique():
            game_plays = df[df["game_id"] == game_id].copy()
            
            # Sort by play order (quarter, then play_id)
            game_plays = game_plays.sort_values(["quarter", "play_id"])
            
            # Calculate running score (simplified)
            home_score = 0
            away_score = 0
            
            for idx, row in game_plays.iterrows():
                # This is a simplified scoring model
                # In production, would track actual scoring plays
                if row.get("play_type") == "touchdown":
                    if row.get("offense_id"):  # Assuming offense scored
                        if self._is_home_team(row["offense_id"], row["game_id"]):
                            home_score += 7
                        else:
                            away_score += 7
                elif row.get("play_type") == "field_goal":
                    if row.get("offense_id"):
                        if self._is_home_team(row["offense_id"], row["game_id"]):
                            home_score += 3
                        else:
                            away_score += 3
                
                # Set score differential (positive if home leading)
                score_diff = home_score - away_score
                df.loc[df.index == idx, "score_differential"] = score_diff
        
        return df
    
    def _is_home_team(self, team_id: str, game_id: str) -> bool:
        """Determine if team is home team for this game."""
        # Parse game_id to determine home team
        # Format: LEAGUE_YEAR_WEEK_AWAY_HOME
        try:
            parts = game_id.split("_")
            if len(parts) >= 5:
                home_team = parts[-1]
                return team_id == home_team
        except:
            pass
        return False
    
    def get_situational_stats(self, situations_df: pd.DataFrame, 
                             pbp_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate situational statistics for teams."""
        # Merge situations with play data
        merged = pbp_df.merge(situations_df, on=["game_id", "play_id"], how="left")
        
        # Calculate stats by team and situation
        stats_list = []
        
        for team_id in merged["offense_id"].unique():
            if pd.isna(team_id):
                continue
                
            team_plays = merged[merged["offense_id"] == team_id]
            
            # Overall stats
            overall_stats = self._calculate_team_situational_stats(team_plays, "overall")
            overall_stats["team_id"] = team_id
            stats_list.append(overall_stats)
            
            # High leverage stats
            high_leverage = team_plays[team_plays["leverage"] > 2.0]
            if not high_leverage.empty:
                hl_stats = self._calculate_team_situational_stats(high_leverage, "high_leverage")
                hl_stats["team_id"] = team_id
                stats_list.append(hl_stats)
            
            # Non-garbage time stats
            non_garbage = team_plays[~team_plays["is_garbage_time"]]
            if not non_garbage.empty:
                ng_stats = self._calculate_team_situational_stats(non_garbage, "non_garbage")
                ng_stats["team_id"] = team_id
                stats_list.append(ng_stats)
        
        return pd.DataFrame(stats_list)
    
    def _calculate_team_situational_stats(self, df: pd.DataFrame, 
                                        situation_type: str) -> Dict:
        """Calculate statistics for a specific situation."""
        if df.empty:
            return {"situation_type": situation_type}
        
        stats = {
            "situation_type": situation_type,
            "plays": len(df),
            "avg_epa": df["epa"].mean() if "epa" in df.columns else 0.0,
            "success_rate": (df["epa"] > 0).mean() if "epa" in df.columns else 0.0,
            "explosive_rate": (df["yards"] >= 20).mean() if "yards" in df.columns else 0.0,
            "avg_leverage": df["leverage"].mean() if "leverage" in df.columns else 0.0,
            "total_yards": df["yards"].sum() if "yards" in df.columns else 0,
        }
        
        return stats
    
    def create_momentum_features(self, pbp_df: pd.DataFrame) -> pd.DataFrame:
        """Create momentum-based features."""
        momentum_features = []
        
        for game_id in pbp_df["game_id"].unique():
            game_plays = pbp_df[pbp_df["game_id"] == game_id].copy()
            game_plays = game_plays.sort_values(["quarter", "play_id"])
            
            # EWMA for win probability changes
            if "wpa" in game_plays.columns:
                game_plays["momentum_ewma"] = game_plays["wpa"].ewm(span=10).mean()
            else:
                game_plays["momentum_ewma"] = 0.0
            
            # Consecutive successful plays
            if "epa" in game_plays.columns:
                game_plays["success"] = game_plays["epa"] > 0
                game_plays["consec_success"] = game_plays.groupby(
                    (game_plays["success"] != game_plays["success"].shift()).cumsum()
                )["success"].cumsum()
                game_plays["consec_success"] = game_plays["consec_success"] * game_plays["success"]
            else:
                game_plays["consec_success"] = 0
            
            # Cap momentum contribution
            game_plays["momentum_capped"] = np.clip(game_plays["momentum_ewma"], -2.0, 2.0)
            
            momentum_features.append(game_plays[["game_id", "play_id", "momentum_ewma", 
                                               "consec_success", "momentum_capped"]])
        
        if momentum_features:
            return pd.concat(momentum_features, ignore_index=True)
        return pd.DataFrame()
