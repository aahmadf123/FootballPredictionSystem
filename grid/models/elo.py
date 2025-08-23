"""Elo rating system for baseline predictions."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from grid.config import GridConfig
from loguru import logger


class EloRating:
    """Elo rating system for football teams."""
    
    def __init__(self, config: GridConfig, league: str = "NFL"):
        self.config = config
        self.league = league
        
        # Elo parameters (tuned per league)
        if league == "NFL":
            self.k_factor = 20.0
            self.home_advantage = 55.0
            self.initial_rating = 1500.0
            self.regression_factor = 0.75  # Regress toward mean between seasons
        else:  # NCAA
            self.k_factor = 25.0
            self.home_advantage = 65.0
            self.initial_rating = 1500.0
            self.regression_factor = 0.70
        
        # Rating storage
        self.ratings: Dict[str, float] = {}
        self.rating_history: List[Dict] = []
        
        # Recency decay
        self.decay_factor = 0.95
        self.max_days_decay = 365
    
    def initialize_ratings(self, teams: List[str], season: int) -> None:
        """Initialize ratings for teams."""
        logger.info(f"Initializing Elo ratings for {len(teams)} teams in {self.league} {season}")
        
        for team in teams:
            self.ratings[team] = self.initial_rating
        
        # Log initial ratings
        for team, rating in self.ratings.items():
            self.rating_history.append({
                "team_id": team,
                "rating": rating,
                "season": season,
                "week": 0,
                "timestamp": datetime.now(),
                "reason": "initialization"
            })
    
    def apply_season_regression(self, season: int) -> None:
        """Apply regression toward mean at season start."""
        logger.info(f"Applying season regression for {season}")
        
        mean_rating = np.mean(list(self.ratings.values()))
        
        for team in self.ratings:
            old_rating = self.ratings[team]
            self.ratings[team] = (
                self.regression_factor * old_rating + 
                (1 - self.regression_factor) * mean_rating
            )
            
            # Log regression
            self.rating_history.append({
                "team_id": team,
                "rating": self.ratings[team],
                "season": season,
                "week": 0,
                "timestamp": datetime.now(),
                "reason": f"season_regression_from_{old_rating:.1f}"
            })
    
    def expected_score(self, team1_rating: float, team2_rating: float, 
                      is_team1_home: bool = False) -> float:
        """Calculate expected score for team1 vs team2."""
        rating_diff = team1_rating - team2_rating
        
        # Add home field advantage
        if is_team1_home:
            rating_diff += self.home_advantage
        
        # Elo expected score formula
        expected = 1 / (1 + 10 ** (-rating_diff / 400))
        return expected
    
    def update_ratings(self, home_team: str, away_team: str, 
                      home_score: int, away_score: int,
                      game_date: datetime, season: int, week: int) -> Tuple[float, float]:
        """Update ratings after a game."""
        # Get current ratings
        home_rating = self.ratings.get(home_team, self.initial_rating)
        away_rating = self.ratings.get(away_team, self.initial_rating)
        
        # Calculate expected scores
        home_expected = self.expected_score(home_rating, away_rating, is_team1_home=True)
        away_expected = 1 - home_expected
        
        # Actual scores (1 for win, 0.5 for tie, 0 for loss)
        if home_score > away_score:
            home_actual = 1.0
            away_actual = 0.0
        elif away_score > home_score:
            home_actual = 0.0
            away_actual = 1.0
        else:
            home_actual = 0.5
            away_actual = 0.5
        
        # Apply margin of victory multiplier
        mov_multiplier = self._calculate_mov_multiplier(abs(home_score - away_score))
        
        # Apply recency decay
        decay_multiplier = self._calculate_decay_multiplier(game_date)
        
        # Update ratings
        effective_k = self.k_factor * mov_multiplier * decay_multiplier
        
        home_rating_change = effective_k * (home_actual - home_expected)
        away_rating_change = effective_k * (away_actual - away_expected)
        
        new_home_rating = home_rating + home_rating_change
        new_away_rating = away_rating + away_rating_change
        
        # Store new ratings
        self.ratings[home_team] = new_home_rating
        self.ratings[away_team] = new_away_rating
        
        # Log rating changes
        self.rating_history.extend([
            {
                "team_id": home_team,
                "rating": new_home_rating,
                "season": season,
                "week": week,
                "timestamp": game_date,
                "reason": f"game_vs_{away_team}_change_{home_rating_change:+.1f}",
                "game_id": f"{season}_{week}_{away_team}_{home_team}"
            },
            {
                "team_id": away_team,
                "rating": new_away_rating,
                "season": season,
                "week": week,
                "timestamp": game_date,
                "reason": f"game_vs_{home_team}_change_{away_rating_change:+.1f}",
                "game_id": f"{season}_{week}_{away_team}_{home_team}"
            }
        ])
        
        return new_home_rating, new_away_rating
    
    def _calculate_mov_multiplier(self, margin: int) -> float:
        """Calculate margin of victory multiplier."""
        # Logarithmic scaling for margin of victory
        if margin <= 3:
            return 1.0
        elif margin <= 7:
            return 1.1
        elif margin <= 14:
            return 1.2
        elif margin <= 21:
            return 1.3
        else:
            return 1.4
    
    def _calculate_decay_multiplier(self, game_date: datetime) -> float:
        """Calculate recency decay multiplier."""
        days_old = (datetime.now() - game_date).days
        
        if days_old <= 0:
            return 1.0
        
        # Apply exponential decay
        decay_power = min(days_old / self.max_days_decay, 1.0)
        return self.decay_factor ** decay_power
    
    def predict_game(self, home_team: str, away_team: str) -> Dict[str, float]:
        """Predict outcome of a game."""
        home_rating = self.ratings.get(home_team, self.initial_rating)
        away_rating = self.ratings.get(away_team, self.initial_rating)
        
        home_win_prob = self.expected_score(home_rating, away_rating, is_team1_home=True)
        away_win_prob = 1 - home_win_prob
        
        # Estimate point spread (rating difference / 25 is common)
        rating_diff = home_rating - away_rating + self.home_advantage
        point_spread = rating_diff / 25.0
        
        return {
            "home_win_prob": home_win_prob,
            "away_win_prob": away_win_prob,
            "home_rating": home_rating,
            "away_rating": away_rating,
            "rating_diff": rating_diff,
            "point_spread": point_spread
        }
    
    def get_current_ratings(self) -> pd.DataFrame:
        """Get current ratings as DataFrame."""
        ratings_data = []
        for team, rating in self.ratings.items():
            ratings_data.append({
                "team_id": team,
                "rating": rating,
                "league": self.league
            })
        
        df = pd.DataFrame(ratings_data)
        df = df.sort_values("rating", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        
        return df
    
    def get_rating_history(self) -> pd.DataFrame:
        """Get rating history as DataFrame."""
        return pd.DataFrame(self.rating_history)
    
    def simulate_season(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Simulate a full season and return predictions."""
        predictions = []
        
        # Sort games by date
        games_sorted = games_df.sort_values("kickoff_utc")
        
        for _, game in games_sorted.iterrows():
            # Make prediction
            pred = self.predict_game(game["home_id"], game["away_id"])
            
            prediction_record = {
                "game_id": game["game_id"],
                "season": game["season"],
                "week": game["week"],
                "home_team": game["home_id"],
                "away_team": game["away_id"],
                "home_win_prob": pred["home_win_prob"],
                "away_win_prob": pred["away_win_prob"],
                "point_spread": pred["point_spread"],
                "home_rating_pre": pred["home_rating"],
                "away_rating_pre": pred["away_rating"],
                "model": "elo"
            }
            predictions.append(prediction_record)
            
            # Update ratings if game is completed
            if pd.notna(game["home_score"]) and pd.notna(game["away_score"]):
                self.update_ratings(
                    game["home_id"], game["away_id"],
                    game["home_score"], game["away_score"],
                    game["kickoff_utc"], game["season"], game["week"]
                )
        
        return pd.DataFrame(predictions)


class EloEnsemble:
    """Ensemble of Elo models for different contexts."""
    
    def __init__(self, config: GridConfig):
        self.config = config
        self.models = {}
        
        # Create Elo models for different leagues
        for league in config.app.leagues:
            self.models[league] = EloRating(config, league)
    
    def initialize_all_ratings(self, teams_df: pd.DataFrame) -> None:
        """Initialize ratings for all leagues."""
        for league in self.models:
            league_teams = teams_df[teams_df["league"] == league]["team_id"].unique()
            if len(league_teams) > 0:
                # Get latest season
                latest_season = teams_df[teams_df["league"] == league]["season"].max()
                self.models[league].initialize_ratings(league_teams, latest_season)
    
    def predict_games(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Predict multiple games across leagues."""
        all_predictions = []
        
        for league in self.models:
            league_games = games_df[games_df["league"] == league]
            if not league_games.empty:
                predictions = self.models[league].simulate_season(league_games)
                all_predictions.append(predictions)
        
        if all_predictions:
            return pd.concat(all_predictions, ignore_index=True)
        return pd.DataFrame()
    
    def get_all_ratings(self) -> pd.DataFrame:
        """Get current ratings for all leagues."""
        all_ratings = []
        
        for league, model in self.models.items():
            ratings = model.get_current_ratings()
            all_ratings.append(ratings)
        
        if all_ratings:
            return pd.concat(all_ratings, ignore_index=True)
        return pd.DataFrame()
    
    def export_ratings(self, filepath: str) -> None:
        """Export all ratings to file."""
        ratings_df = self.get_all_ratings()
        ratings_df.to_csv(filepath, index=False)
        logger.info(f"Exported ratings to {filepath}")
    
    def load_ratings(self, filepath: str) -> None:
        """Load ratings from file."""
        try:
            ratings_df = pd.read_csv(filepath)
            
            for _, row in ratings_df.iterrows():
                league = row["league"]
                team = row["team_id"]
                rating = row["rating"]
                
                if league in self.models:
                    self.models[league].ratings[team] = rating
            
            logger.info(f"Loaded ratings from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load ratings: {e}")


def calculate_elo_accuracy(predictions_df: pd.DataFrame, results_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate Elo model accuracy metrics."""
    merged = predictions_df.merge(
        results_df[["game_id", "home_score", "away_score"]], 
        on="game_id", 
        how="inner"
    )
    
    if merged.empty:
        return {}
    
    # Calculate actual outcomes
    merged["home_win"] = merged["home_score"] > merged["away_score"]
    merged["predicted_home_win"] = merged["home_win_prob"] > 0.5
    
    # Accuracy
    accuracy = (merged["home_win"] == merged["predicted_home_win"]).mean()
    
    # Log loss
    log_loss = -np.mean(
        merged["home_win"] * np.log(np.clip(merged["home_win_prob"], 1e-15, 1-1e-15)) +
        (1 - merged["home_win"]) * np.log(np.clip(1 - merged["home_win_prob"], 1e-15, 1-1e-15))
    )
    
    # Brier score
    brier_score = np.mean((merged["home_win_prob"] - merged["home_win"]) ** 2)
    
    return {
        "accuracy": accuracy,
        "log_loss": log_loss,
        "brier_score": brier_score,
        "n_games": len(merged)
    }
