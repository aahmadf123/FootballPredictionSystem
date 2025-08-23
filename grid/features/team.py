"""Team and unit-level feature engineering."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import Ridge

from grid.config import GridConfig
from loguru import logger


class TeamFeatureBuilder:
    """Build team-level features with rolling windows and opponent adjustments."""
    
    def __init__(self, config: GridConfig):
        self.config = config
        self.windows = config.features.windows
        self.opponent_adjust = config.features.opponent_adjust
    
    def build_team_features(self, pbp_df: pd.DataFrame, games_df: pd.DataFrame,
                           situations_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Build comprehensive team features."""
        logger.info("Building team features...")
        
        # Base efficiency features
        efficiency_features = self._build_efficiency_features(pbp_df, situations_df)
        
        # Rolling window features
        rolling_features = self._build_rolling_features(efficiency_features, games_df)
        
        # Opponent adjustments
        if self.opponent_adjust:
            adjusted_features = self._apply_opponent_adjustments(rolling_features, games_df)
        else:
            adjusted_features = rolling_features
        
        # Advanced team features
        advanced_features = self._build_advanced_features(pbp_df, games_df)
        
        # Merge all features
        final_features = self._merge_team_features([adjusted_features, advanced_features])
        
        logger.info(f"Built team features for {len(final_features)} team-game combinations")
        return final_features
    
    def _build_efficiency_features(self, pbp_df: pd.DataFrame,
                                 situations_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Build base efficiency features from play-by-play data."""
        if pbp_df.empty:
            return pd.DataFrame()
        
        # Filter out garbage time if situations provided
        if situations_df is not None:
            pbp_with_situations = pbp_df.merge(
                situations_df[["game_id", "play_id", "is_garbage_time"]], 
                on=["game_id", "play_id"], 
                how="left"
            )
            core_pbp = pbp_with_situations[~pbp_with_situations["is_garbage_time"].fillna(False)]
        else:
            core_pbp = pbp_df
        
        # Calculate features by team and game
        team_game_features = []
        
        # Process offense
        offense_features = self._calculate_unit_features(
            core_pbp, "offense_id", "offense"
        )
        team_game_features.append(offense_features)
        
        # Process defense
        defense_features = self._calculate_unit_features(
            core_pbp, "defense_id", "defense"
        )
        team_game_features.append(defense_features)
        
        # Merge offense and defense features
        if team_game_features:
            return pd.concat(team_game_features, ignore_index=True)
        return pd.DataFrame()
    
    def _calculate_unit_features(self, pbp_df: pd.DataFrame, 
                                team_col: str, unit_type: str) -> pd.DataFrame:
        """Calculate features for offense or defense."""
        features_list = []
        
        for (team_id, game_id), group in pbp_df.groupby([team_col, "game_id"]):
            if pd.isna(team_id):
                continue
            
            features = {
                "team_id": team_id,
                "game_id": game_id,
                "unit": unit_type
            }
            
            # Basic efficiency metrics
            features[f"{unit_type}_plays"] = len(group)
            
            if "epa" in group.columns:
                features[f"{unit_type}_epa_play"] = group["epa"].mean()
                features[f"{unit_type}_success_rate"] = (group["epa"] > 0).mean()
                features[f"{unit_type}_total_epa"] = group["epa"].sum()
            
            if "yards" in group.columns:
                features[f"{unit_type}_yards_play"] = group["yards"].mean()
                features[f"{unit_type}_explosive_rate"] = (group["yards"] >= 20).mean()
                features[f"{unit_type}_total_yards"] = group["yards"].sum()
            
            # Down and distance features
            if "down" in group.columns:
                # Early down pass rate (1st and 2nd down)
                early_downs = group[group["down"].isin([1, 2])]
                if not early_downs.empty and "play_type" in group.columns:
                    pass_plays = early_downs["play_type"].str.contains("pass", case=False, na=False)
                    features[f"{unit_type}_early_down_pass_rate"] = pass_plays.mean()
                
                # Third down conversion rate
                third_downs = group[group["down"] == 3]
                if not third_downs.empty and "epa" in group.columns:
                    features[f"{unit_type}_third_down_conv"] = (third_downs["epa"] > 0).mean()
            
            # Red zone efficiency (simplified - within 20 yards)
            if "yardline_100" in group.columns:
                red_zone = group[group["yardline_100"] <= 20]
                if not red_zone.empty:
                    if "play_type" in group.columns:
                        td_plays = red_zone["play_type"].str.contains("touchdown", case=False, na=False)
                        features[f"{unit_type}_red_zone_td_rate"] = td_plays.mean()
            
            # Pace (seconds per snap)
            if "sec_left" in group.columns and len(group) > 1:
                # Calculate time between plays
                time_diffs = group["sec_left"].diff().abs()
                valid_diffs = time_diffs[(time_diffs > 0) & (time_diffs < 60)]  # Reasonable range
                if not valid_diffs.empty:
                    features[f"{unit_type}_pace_sec_snap"] = valid_diffs.mean()
            
            # Play type distribution
            if "play_type" in group.columns:
                play_types = group["play_type"].value_counts(normalize=True)
                for play_type, rate in play_types.items():
                    features[f"{unit_type}_{play_type.lower()}_rate"] = rate
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def _build_rolling_features(self, efficiency_df: pd.DataFrame, 
                               games_df: pd.DataFrame) -> pd.DataFrame:
        """Build rolling window features."""
        if efficiency_df.empty:
            return pd.DataFrame()
        
        # Merge with games to get temporal ordering
        merged = efficiency_df.merge(
            games_df[["game_id", "season", "week"]], 
            on="game_id", 
            how="left"
        )
        
        rolling_features = []
        
        for team_id in merged["team_id"].unique():
            if pd.isna(team_id):
                continue
            
            team_data = merged[merged["team_id"] == team_id].copy()
            team_data = team_data.sort_values(["season", "week"])
            
            # Calculate rolling features for each window
            for window in self.windows:
                team_rolling = self._calculate_rolling_window(team_data, window)
                rolling_features.append(team_rolling)
        
        if rolling_features:
            return pd.concat(rolling_features, ignore_index=True)
        return pd.DataFrame()
    
    def _calculate_rolling_window(self, team_df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Calculate rolling statistics for a specific window."""
        numeric_cols = team_df.select_dtypes(include=[np.number]).columns
        exclude_cols = ["team_id", "game_id", "season", "week"]
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Calculate rolling means
        rolling_means = team_df[feature_cols].rolling(window=window, min_periods=1).mean()
        
        # Rename columns to include window size
        rolling_means.columns = [f"{col}_rolling_{window}" for col in rolling_means.columns]
        
        # Combine with identifiers
        result = team_df[["team_id", "game_id", "season", "week", "unit"]].copy()
        result = pd.concat([result, rolling_means], axis=1)
        
        return result
    
    def _apply_opponent_adjustments(self, features_df: pd.DataFrame, 
                                   games_df: pd.DataFrame) -> pd.DataFrame:
        """Apply opponent adjustments using ridge regression (SRS-style)."""
        logger.info("Applying opponent adjustments...")
        
        # Create opponent schedule matrix
        schedule_matrix = self._create_schedule_matrix(games_df)
        
        if schedule_matrix is None:
            logger.warning("Could not create schedule matrix, skipping opponent adjustments")
            return features_df
        
        adjusted_features = features_df.copy()
        
        # Apply adjustments to key metrics
        adjustment_metrics = [col for col in features_df.columns if 
                            any(keyword in col for keyword in ["_epa_", "_yards_", "_success_"])]
        
        for metric in adjustment_metrics:
            if metric in features_df.columns:
                adjusted_values = self._calculate_srs_adjustment(
                    features_df, metric, schedule_matrix
                )
                adjusted_features[f"{metric}_adj"] = adjusted_values
        
        return adjusted_features
    
    def _create_schedule_matrix(self, games_df: pd.DataFrame) -> Optional[np.ndarray]:
        """Create schedule matrix for SRS calculations."""
        if games_df.empty:
            return None
        
        teams = sorted(set(games_df["home_id"].tolist() + games_df["away_id"].tolist()))
        team_to_idx = {team: idx for idx, team in enumerate(teams)}
        
        n_teams = len(teams)
        schedule_matrix = np.zeros((n_teams, n_teams))
        
        for _, game in games_df.iterrows():
            home_idx = team_to_idx.get(game["home_id"])
            away_idx = team_to_idx.get(game["away_id"])
            
            if home_idx is not None and away_idx is not None:
                schedule_matrix[home_idx, away_idx] += 1
                schedule_matrix[away_idx, home_idx] += 1
        
        return schedule_matrix
    
    def _calculate_srs_adjustment(self, features_df: pd.DataFrame, 
                                 metric: str, schedule_matrix: np.ndarray) -> pd.Series:
        """Calculate SRS-style opponent adjustments."""
        try:
            # Prepare data for ridge regression
            team_values = features_df.groupby("team_id")[metric].mean()
            
            # Simple opponent adjustment using team averages
            league_avg = team_values.mean()
            adjustments = team_values - league_avg
            
            # Map back to original dataframe
            adjusted_series = features_df["team_id"].map(
                lambda x: team_values.get(x, league_avg) + adjustments.get(x, 0)
            )
            
            return adjusted_series
            
        except Exception as e:
            logger.warning(f"Failed to calculate opponent adjustment for {metric}: {e}")
            return features_df[metric]
    
    def _build_advanced_features(self, pbp_df: pd.DataFrame, 
                                games_df: pd.DataFrame) -> pd.DataFrame:
        """Build advanced team features."""
        advanced_features = []
        
        for (team_id, game_id), group in pbp_df.groupby(["offense_id", "game_id"]):
            if pd.isna(team_id):
                continue
            
            features = {
                "team_id": team_id,
                "game_id": game_id,
                "unit": "advanced"
            }
            
            # Success rate by down and distance
            features.update(self._calculate_down_distance_features(group))
            
            # Havoc rate (for defense)
            features.update(self._calculate_havoc_features(group))
            
            # Pressure rates
            features.update(self._calculate_pressure_features(group))
            
            advanced_features.append(features)
        
        return pd.DataFrame(advanced_features)
    
    def _calculate_down_distance_features(self, group: pd.DataFrame) -> Dict:
        """Calculate success rates by down and distance."""
        features = {}
        
        if "down" not in group.columns or "dist" not in group.columns or "epa" not in group.columns:
            return features
        
        # Success rate definitions from spec
        # 1st down: gain >= 50% of yards to go
        # 2nd down: gain >= 70% of yards to go  
        # 3rd/4th down: gain >= 100% of yards to go
        
        for down in [1, 2, 3, 4]:
            down_plays = group[group["down"] == down]
            if not down_plays.empty:
                if down == 1:
                    success_threshold = 0.5
                elif down == 2:
                    success_threshold = 0.7
                else:
                    success_threshold = 1.0
                
                # Calculate success based on yards gained vs distance
                if "yards" in down_plays.columns:
                    success_plays = down_plays["yards"] >= (down_plays["dist"] * success_threshold)
                    features[f"success_rate_down_{down}"] = success_plays.mean()
        
        return features
    
    def _calculate_havoc_features(self, group: pd.DataFrame) -> Dict:
        """Calculate havoc rate features."""
        features = {}
        
        if "play_type" not in group.columns:
            return features
        
        # Havoc rate = (TFL + FF + INT + PD) / defensive plays
        # Simplified implementation
        total_plays = len(group)
        if total_plays > 0:
            # Count negative plays as proxy for TFL
            negative_plays = (group.get("yards", 0) < 0).sum()
            
            # Count turnovers as proxy for FF + INT
            turnover_plays = group["play_type"].str.contains(
                "fumble|interception", case=False, na=False
            ).sum()
            
            havoc_plays = negative_plays + turnover_plays
            features["havoc_rate"] = havoc_plays / total_plays
        
        return features
    
    def _calculate_pressure_features(self, group: pd.DataFrame) -> Dict:
        """Calculate pressure-related features."""
        features = {}
        
        if "play_type" not in group.columns:
            return features
        
        # Pass plays only
        pass_plays = group[group["play_type"].str.contains("pass", case=False, na=False)]
        
        if not pass_plays.empty:
            # Simplified pressure rate (would need more detailed data in production)
            sacks = pass_plays["play_type"].str.contains("sack", case=False, na=False).sum()
            features["pressure_rate"] = sacks / len(pass_plays)
            
            # Time to throw proxy
            if "yards" in pass_plays.columns:
                short_passes = (pass_plays["yards"] < 10).mean()
                features["quick_pass_rate"] = short_passes
        
        return features
    
    def _merge_team_features(self, feature_list: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge multiple team feature dataframes."""
        if not feature_list:
            return pd.DataFrame()
        
        # Start with first dataframe
        merged = feature_list[0].copy()
        
        # Merge subsequent dataframes
        for df in feature_list[1:]:
            if not df.empty:
                merged = merged.merge(
                    df, 
                    on=["team_id", "game_id"], 
                    how="outer",
                    suffixes=("", "_dup")
                )
        
        return merged
