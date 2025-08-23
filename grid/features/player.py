"""Player-level feature engineering including EPA, matchups, and development curves."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats

from grid.config import GridConfig
from loguru import logger


class PlayerFeatureBuilder:
    """Build player-level features for EPA, matchups, and development."""
    
    def __init__(self, config: GridConfig):
        self.config = config
        self.min_snaps_threshold = 50  # Minimum snaps for reliable stats
        self.dev_alpha_pos = 0.7  # Shrinkage parameter for development curves
    
    def build_player_features(self, pbp_df: pd.DataFrame, players_df: pd.DataFrame,
                             games_df: pd.DataFrame) -> pd.DataFrame:
        """Build comprehensive player features."""
        logger.info("Building player features...")
        
        # EPA features by involvement
        epa_features = self._build_player_epa_features(pbp_df, players_df)
        
        # Matchup features
        matchup_features = self._build_matchup_features(pbp_df, players_df)
        
        # Development/aging curves
        development_features = self._build_development_features(players_df, games_df)
        
        # Merge all features
        player_features = self._merge_player_features([
            epa_features, matchup_features, development_features
        ])
        
        logger.info(f"Built player features for {len(player_features)} player-game combinations")
        return player_features
    
    def _build_player_epa_features(self, pbp_df: pd.DataFrame, 
                                  players_df: pd.DataFrame) -> pd.DataFrame:
        """Build EPA features by player involvement."""
        player_features = []
        
        # QB features
        qb_features = self._calculate_qb_features(pbp_df, players_df)
        player_features.append(qb_features)
        
        # RB features
        rb_features = self._calculate_rb_features(pbp_df, players_df)
        player_features.append(rb_features)
        
        # WR/TE features
        receiver_features = self._calculate_receiver_features(pbp_df, players_df)
        player_features.append(receiver_features)
        
        # OL features (proxy through team pressure)
        ol_features = self._calculate_ol_features(pbp_df, players_df)
        player_features.append(ol_features)
        
        # Defensive features
        defense_features = self._calculate_defense_features(pbp_df, players_df)
        player_features.append(defense_features)
        
        # Combine all features
        if player_features:
            return pd.concat([df for df in player_features if not df.empty], ignore_index=True)
        return pd.DataFrame()
    
    def _calculate_qb_features(self, pbp_df: pd.DataFrame, 
                              players_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate QB-specific features."""
        # Get QBs
        qbs = players_df[players_df["pos"] == "QB"]
        
        features_list = []
        
        for _, qb in qbs.iterrows():
            # Get dropback plays for this QB
            qb_plays = pbp_df[
                (pbp_df["play_type"].str.contains("pass", case=False, na=False)) |
                (pbp_df["play_type"].str.contains("sack", case=False, na=False))
            ]
            
            # Filter by team (simplified - would need player tracking data)
            team_plays = qb_plays[qb_plays["offense_id"] == qb["team_id"]]
            
            if len(team_plays) < self.min_snaps_threshold:
                continue
            
            features = {
                "player_id": qb["player_id"],
                "team_id": qb["team_id"],
                "position": "QB",
                "season": qb["season"]
            }
            
            # Dropback EPA/play
            features["qb_dropback_epa_play"] = team_plays["epa"].mean()
            features["qb_dropbacks"] = len(team_plays)
            features["qb_success_rate"] = (team_plays["epa"] > 0).mean()
            
            # Completion percentage (proxy)
            if "yards" in team_plays.columns:
                completions = team_plays[team_plays["yards"] > 0]
                features["qb_completion_pct"] = len(completions) / len(team_plays)
                
                # Air yards and YAC (simplified)
                features["qb_avg_air_yards"] = team_plays["yards"].mean()
                features["qb_deep_ball_rate"] = (team_plays["yards"] >= 20).mean()
            
            # Pressure handling (sacks as proxy)
            sacks = team_plays[team_plays["play_type"].str.contains("sack", case=False, na=False)]
            features["qb_sack_rate"] = len(sacks) / len(team_plays)
            
            # Red zone efficiency
            red_zone_plays = team_plays[
                (team_plays.get("yardline_100", 100) <= 20) & 
                (team_plays.get("yardline_100", 100) > 0)
            ]
            if not red_zone_plays.empty:
                features["qb_red_zone_epa"] = red_zone_plays["epa"].mean()
                td_passes = red_zone_plays[
                    red_zone_plays["play_type"].str.contains("touchdown", case=False, na=False)
                ]
                features["qb_red_zone_td_rate"] = len(td_passes) / len(red_zone_plays)
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def _calculate_rb_features(self, pbp_df: pd.DataFrame, 
                              players_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RB-specific features."""
        rbs = players_df[players_df["pos"] == "RB"]
        
        features_list = []
        
        for _, rb in rbs.iterrows():
            # Get rush and receiving plays (simplified)
            team_plays = pbp_df[pbp_df["offense_id"] == rb["team_id"]]
            
            rush_plays = team_plays[
                team_plays["play_type"].str.contains("rush|run", case=False, na=False)
            ]
            
            receiving_plays = team_plays[
                team_plays["play_type"].str.contains("pass", case=False, na=False)
            ]
            
            total_touches = len(rush_plays) + len(receiving_plays) * 0.3  # Estimate
            
            if total_touches < self.min_snaps_threshold:
                continue
            
            features = {
                "player_id": rb["player_id"],
                "team_id": rb["team_id"],
                "position": "RB",
                "season": rb["season"]
            }
            
            # Rushing features
            if not rush_plays.empty:
                features["rb_rush_epa_carry"] = rush_plays["epa"].mean()
                features["rb_carries"] = len(rush_plays)
                features["rb_yards_carry"] = rush_plays["yards"].mean()
                features["rb_stuff_rate"] = (rush_plays["yards"] <= 0).mean()
                features["rb_breakaway_rate"] = (rush_plays["yards"] >= 20).mean()
            
            # Receiving features (estimated)
            recv_estimate = len(receiving_plays) * 0.2  # Estimate RB targets
            if recv_estimate > 10:
                features["rb_target_share"] = recv_estimate / len(receiving_plays)
                features["rb_receiving_epa"] = receiving_plays["epa"].mean() * 0.2
            
            # Goal line carries
            goal_line_plays = rush_plays[
                (rush_plays.get("yardline_100", 100) <= 5) & 
                (rush_plays.get("yardline_100", 100) > 0)
            ]
            if not goal_line_plays.empty:
                features["rb_goal_line_td_rate"] = (
                    goal_line_plays["play_type"].str.contains("touchdown", case=False, na=False)
                ).mean()
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def _calculate_receiver_features(self, pbp_df: pd.DataFrame, 
                                   players_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate WR/TE features."""
        receivers = players_df[players_df["pos"].isin(["WR", "TE"])]
        
        features_list = []
        
        for _, receiver in receivers.iterrows():
            team_plays = pbp_df[pbp_df["offense_id"] == receiver["team_id"]]
            pass_plays = team_plays[
                team_plays["play_type"].str.contains("pass", case=False, na=False)
            ]
            
            # Estimate target share (simplified)
            estimated_targets = len(pass_plays) * (0.15 if receiver["pos"] == "WR" else 0.08)
            
            if estimated_targets < 20:  # Minimum targets threshold
                continue
            
            features = {
                "player_id": receiver["player_id"],
                "team_id": receiver["team_id"],
                "position": receiver["pos"],
                "season": receiver["season"]
            }
            
            # Target and reception features (estimated)
            features["recv_target_share"] = estimated_targets / len(pass_plays)
            features["recv_targets_game"] = estimated_targets / 16  # Estimate per game
            
            # EPA features (estimated based on team passing)
            if not pass_plays.empty:
                features["recv_epa_target"] = pass_plays["epa"].mean() * features["recv_target_share"]
                features["recv_air_yards"] = pass_plays["yards"].mean()
                features["recv_deep_target_rate"] = (pass_plays["yards"] >= 20).mean()
            
            # Red zone features
            red_zone_passes = pass_plays[
                (pass_plays.get("yardline_100", 100) <= 20) & 
                (pass_plays.get("yardline_100", 100) > 0)
            ]
            if not red_zone_passes.empty:
                features["recv_red_zone_target_share"] = (
                    len(red_zone_passes) * features["recv_target_share"] / len(red_zone_passes)
                )
            
            # Route distribution (estimated based on position)
            if receiver["pos"] == "WR":
                features["recv_slot_rate"] = 0.4 if receiver["depth"] >= 2 else 0.2
            else:  # TE
                features["recv_slot_rate"] = 0.6
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def _calculate_ol_features(self, pbp_df: pd.DataFrame, 
                              players_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate OL features (proxy through team pressure rates)."""
        ol_players = players_df[players_df["pos"].isin(["C", "G", "T", "OL"])]
        
        features_list = []
        
        for _, ol_player in ol_players.iterrows():
            team_plays = pbp_df[pbp_df["offense_id"] == ol_player["team_id"]]
            pass_plays = team_plays[
                team_plays["play_type"].str.contains("pass", case=False, na=False)
            ]
            
            if len(pass_plays) < 100:  # Minimum team pass plays
                continue
            
            features = {
                "player_id": ol_player["player_id"],
                "team_id": ol_player["team_id"],
                "position": ol_player["pos"],
                "season": ol_player["season"]
            }
            
            # Pressure rate allowed (team level, attributed to OL)
            sacks = pass_plays[pass_plays["play_type"].str.contains("sack", case=False, na=False)]
            team_pressure_rate = len(sacks) / len(pass_plays)
            
            # Expected pressure rate based on opponent strength (simplified)
            expected_pressure_rate = 0.07
            
            features["ol_pressure_rate_allowed"] = team_pressure_rate
            features["ol_pressure_over_expected"] = team_pressure_rate - expected_pressure_rate
            
            # Run blocking (simplified)
            rush_plays = team_plays[
                team_plays["play_type"].str.contains("rush|run", case=False, na=False)
            ]
            if not rush_plays.empty:
                features["ol_run_block_epa"] = rush_plays["epa"].mean()
                features["ol_stuff_rate_allowed"] = (rush_plays["yards"] <= 0).mean()
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def _calculate_defense_features(self, pbp_df: pd.DataFrame, 
                                   players_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate defensive player features."""
        defensive_players = players_df[
            players_df["pos"].isin(["DE", "DT", "LB", "CB", "S", "DB", "DL"])
        ]
        
        features_list = []
        
        for _, def_player in defensive_players.iterrows():
            team_plays = pbp_df[pbp_df["defense_id"] == def_player["team_id"]]
            
            if len(team_plays) < self.min_snaps_threshold:
                continue
            
            features = {
                "player_id": def_player["player_id"],
                "team_id": def_player["team_id"],
                "position": def_player["pos"],
                "season": def_player["season"]
            }
            
            # Position-specific features
            if def_player["pos"] in ["DE", "DT", "DL"]:
                # Pass rush features
                pass_plays = team_plays[
                    team_plays["play_type"].str.contains("pass", case=False, na=False)
                ]
                if not pass_plays.empty:
                    sacks = pass_plays[
                        pass_plays["play_type"].str.contains("sack", case=False, na=False)
                    ]
                    features["def_pressure_rate"] = len(sacks) / len(pass_plays)
                
                # Run defense
                rush_plays = team_plays[
                    team_plays["play_type"].str.contains("rush|run", case=False, na=False)
                ]
                if not rush_plays.empty:
                    features["def_run_stop_rate"] = (rush_plays["yards"] <= 2).mean()
            
            elif def_player["pos"] in ["CB", "S", "DB"]:
                # Coverage features (simplified)
                pass_plays = team_plays[
                    team_plays["play_type"].str.contains("pass", case=False, na=False)
                ]
                if not pass_plays.empty:
                    features["def_pass_epa_against"] = -pass_plays["epa"].mean()  # Negative EPA is good for defense
                    interceptions = pass_plays[
                        pass_plays["play_type"].str.contains("interception", case=False, na=False)
                    ]
                    features["def_int_rate"] = len(interceptions) / len(pass_plays)
            
            elif def_player["pos"] == "LB":
                # Linebacker features (run and coverage)
                features["def_tackles_per_play"] = 0.08  # Estimated based on position
                features["def_coverage_epa"] = -team_plays["epa"].mean() * 0.3  # Partial attribution
            
            # General defensive features
            features["def_epa_play"] = -team_plays["epa"].mean()
            features["def_success_rate_against"] = (team_plays["epa"] <= 0).mean()
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def _build_matchup_features(self, pbp_df: pd.DataFrame, 
                               players_df: pd.DataFrame) -> pd.DataFrame:
        """Build matchup-specific features."""
        matchup_features = []
        
        # WR vs CB matchups
        wr_cb_matchups = self._calculate_wr_cb_matchups(pbp_df, players_df)
        matchup_features.append(wr_cb_matchups)
        
        # OL vs DL matchups
        ol_dl_matchups = self._calculate_ol_dl_matchups(pbp_df, players_df)
        matchup_features.append(ol_dl_matchups)
        
        if matchup_features:
            return pd.concat([df for df in matchup_features if not df.empty], ignore_index=True)
        return pd.DataFrame()
    
    def _calculate_wr_cb_matchups(self, pbp_df: pd.DataFrame, 
                                 players_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate WR vs CB matchup features."""
        # This would require detailed player tracking data
        # Simplified implementation based on team-level data
        
        wrs = players_df[players_df["pos"] == "WR"]
        cbs = players_df[players_df["pos"] == "CB"]
        
        matchup_features = []
        
        # Create team-level matchup proxies
        for _, wr in wrs.iterrows():
            wr_team = wr["team_id"]
            
            # Find games where this WR's team played
            team_games = pbp_df[pbp_df["offense_id"] == wr_team]["game_id"].unique()
            
            for game_id in team_games:
                # Get opposing team
                game_pbp = pbp_df[pbp_df["game_id"] == game_id]
                opposing_team = game_pbp[game_pbp["defense_id"] != wr_team]["defense_id"].iloc[0]
                
                # Find CBs on opposing team
                opposing_cbs = cbs[cbs["team_id"] == opposing_team]
                
                for _, cb in opposing_cbs.iterrows():
                    # Calculate matchup features (simplified)
                    features = {
                        "wr_player_id": wr["player_id"],
                        "cb_player_id": cb["player_id"],
                        "game_id": game_id,
                        "wr_team": wr_team,
                        "cb_team": opposing_team
                    }
                    
                    # Height/speed archetype similarity (using height as proxy)
                    height_diff = abs(wr["height"] - cb["height"])
                    features["archetype_similarity"] = max(0, 1 - height_diff / 12)  # Normalize
                    
                    # Expected separation (simplified)
                    wr_speed_proxy = max(0, (72 - wr["height"]) / 12)  # Shorter = faster proxy
                    cb_speed_proxy = max(0, (72 - cb["height"]) / 12)
                    features["expected_separation"] = max(0, wr_speed_proxy - cb_speed_proxy)
                    
                    matchup_features.append(features)
        
        return pd.DataFrame(matchup_features)
    
    def _calculate_ol_dl_matchups(self, pbp_df: pd.DataFrame, 
                                 players_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate OL vs DL matchup features."""
        ol_players = players_df[players_df["pos"].isin(["C", "G", "T", "OL"])]
        dl_players = players_df[players_df["pos"].isin(["DE", "DT", "DL"])]
        
        matchup_features = []
        
        for _, ol_player in ol_players.iterrows():
            ol_team = ol_player["team_id"]
            team_games = pbp_df[pbp_df["offense_id"] == ol_team]["game_id"].unique()
            
            for game_id in team_games:
                game_pbp = pbp_df[pbp_df["game_id"] == game_id]
                opposing_team = game_pbp[game_pbp["defense_id"] != ol_team]["defense_id"].iloc[0]
                
                opposing_dl = dl_players[dl_players["team_id"] == opposing_team]
                
                for _, dl_player in opposing_dl.iterrows():
                    features = {
                        "ol_player_id": ol_player["player_id"],
                        "dl_player_id": dl_player["player_id"],
                        "game_id": game_id,
                        "ol_team": ol_team,
                        "dl_team": opposing_team
                    }
                    
                    # Size matchup
                    weight_advantage = ol_player["weight"] - dl_player["weight"]
                    features["weight_advantage"] = weight_advantage
                    
                    # Win rate proxy (simplified)
                    expected_win_rate = 0.5 + (weight_advantage / 100) * 0.1
                    features["expected_win_rate"] = np.clip(expected_win_rate, 0.1, 0.9)
                    
                    matchup_features.append(features)
        
        return pd.DataFrame(matchup_features)
    
    def _build_development_features(self, players_df: pd.DataFrame, 
                                   games_df: pd.DataFrame) -> pd.DataFrame:
        """Build aging and development curve features."""
        development_features = []
        
        # Group by position for position-specific curves
        for position in players_df["pos"].unique():
            if pd.isna(position):
                continue
            
            pos_players = players_df[players_df["pos"] == position]
            pos_features = self._calculate_position_development(pos_players, position)
            development_features.append(pos_features)
        
        if development_features:
            return pd.concat([df for df in development_features if not df.empty], ignore_index=True)
        return pd.DataFrame()
    
    def _calculate_position_development(self, players_df: pd.DataFrame, 
                                      position: str) -> pd.DataFrame:
        """Calculate development curves for a specific position."""
        if players_df.empty:
            return pd.DataFrame()
        
        features_list = []
        
        # Calculate team mean performance by position (for shrinkage)
        team_means = players_df.groupby("team_id")["age"].mean()
        
        for _, player in players_df.iterrows():
            features = {
                "player_id": player["player_id"],
                "team_id": player["team_id"],
                "position": position,
                "season": player["season"]
            }
            
            # Age curve features
            age = player["age"]
            features["age"] = age
            features["age_squared"] = age ** 2
            
            # Position-specific peak ages and curves
            peak_age = self._get_position_peak_age(position)
            features["years_from_peak"] = age - peak_age
            features["in_prime"] = 1 if abs(age - peak_age) <= 2 else 0
            
            # Development stage
            if age <= 23:
                features["development_stage"] = "developing"
                features["development_factor"] = 1.1  # Potential for growth
            elif age <= 29:
                features["development_stage"] = "prime"
                features["development_factor"] = 1.0
            else:
                features["development_stage"] = "declining"
                features["development_factor"] = 0.95  # Expected decline
            
            # Experience proxy (age - typical college grad age)
            features["experience_years"] = max(0, age - 22)
            
            # Hierarchical shrinkage to team mean
            team_mean_age = team_means.get(player["team_id"], 26.0)
            shrunk_age_effect = (
                self.dev_alpha_pos * (age - 26) + 
                (1 - self.dev_alpha_pos) * (team_mean_age - 26)
            )
            features["age_effect_shrunk"] = shrunk_age_effect
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def _get_position_peak_age(self, position: str) -> float:
        """Get typical peak age for position."""
        peak_ages = {
            "QB": 30.0,
            "RB": 26.0,
            "WR": 27.0,
            "TE": 28.0,
            "C": 29.0,
            "G": 28.5,
            "T": 29.0,
            "OL": 28.5,
            "DE": 27.5,
            "DT": 28.0,
            "DL": 28.0,
            "LB": 27.0,
            "CB": 26.5,
            "S": 28.0,
            "DB": 27.0
        }
        return peak_ages.get(position, 27.0)
    
    def _merge_player_features(self, feature_list: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge multiple player feature dataframes."""
        if not feature_list:
            return pd.DataFrame()
        
        # Start with first non-empty dataframe
        merged = None
        for df in feature_list:
            if not df.empty:
                if merged is None:
                    merged = df.copy()
                else:
                    merged = merged.merge(
                        df, 
                        on=["player_id"], 
                        how="outer",
                        suffixes=("", "_dup")
                    )
        
        return merged if merged is not None else pd.DataFrame()
