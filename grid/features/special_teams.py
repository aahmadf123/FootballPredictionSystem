"""Special teams feature engineering including FG model, returns, and ST-EPA."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

from grid.config import GridConfig
from loguru import logger


class FieldGoalModel:
    """Field goal make probability model."""
    
    def __init__(self):
        self.model = LogisticRegression()
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Feature coefficients (will be learned)
        self.coefficients = {
            "distance": -0.05,  # Negative - harder from farther
            "wind": -0.02,      # Negative - wind hurts accuracy
            "temp": 0.001,      # Slight positive - better in warm weather
            "precip": -0.15,    # Negative - rain/snow hurts
            "hash_mark": -0.05, # Negative - harder from hash marks
            "altitude": 0.0001, # Slight positive - thinner air helps
            "dome": 0.08        # Positive - controlled environment
        }
    
    def fit(self, fg_data: pd.DataFrame) -> None:
        """Fit the field goal model."""
        if fg_data.empty:
            logger.warning("No field goal data provided for fitting")
            return
        
        # Prepare features
        X = self._prepare_features(fg_data)
        y = fg_data["made"].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit logistic regression
        self.model.fit(X_scaled, y)
        
        # Fit calibrator
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        self.calibrator.fit(probabilities, y)
        
        self.is_fitted = True
        logger.info(f"Fitted FG model on {len(fg_data)} attempts")
    
    def predict_probability(self, distance: float, wind: float = 0, 
                          temp: float = 70, precip: float = 0,
                          hash_mark: bool = False, altitude: float = 0,
                          dome: bool = False) -> float:
        """Predict field goal make probability."""
        if not self.is_fitted:
            # Use rule-based model if not fitted
            return self._rule_based_probability(distance, wind, temp, precip, hash_mark, dome)
        
        # Create feature vector
        features = np.array([[
            distance, wind, temp, precip, 
            int(hash_mark), altitude, int(dome)
        ]])
        
        # Scale and predict
        features_scaled = self.scaler.transform(features)
        raw_prob = self.model.predict_proba(features_scaled)[0, 1]
        
        # Apply calibration
        calibrated_prob = self.calibrator.predict([raw_prob])[0]
        
        return float(np.clip(calibrated_prob, 0.01, 0.99))
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for model fitting."""
        features = []
        
        for _, row in df.iterrows():
            feature_row = [
                row.get("distance", 40),
                row.get("wind", 0),
                row.get("temp", 70),
                row.get("precip", 0),
                int(row.get("hash_mark", False)),
                row.get("altitude", 0),
                int(row.get("dome", False))
            ]
            features.append(feature_row)
        
        return np.array(features)
    
    def _rule_based_probability(self, distance: float, wind: float, 
                               temp: float, precip: float,
                               hash_mark: bool, dome: bool) -> float:
        """Rule-based probability when model not fitted."""
        # Base probability by distance
        if distance <= 30:
            base_prob = 0.95
        elif distance <= 40:
            base_prob = 0.90
        elif distance <= 50:
            base_prob = 0.80
        elif distance <= 60:
            base_prob = 0.60
        else:
            base_prob = 0.30
        
        # Apply adjustments
        wind_adj = wind * self.coefficients["wind"]
        temp_adj = (temp - 70) * self.coefficients["temp"]
        precip_adj = precip * self.coefficients["precip"]
        hash_adj = self.coefficients["hash_mark"] if hash_mark else 0
        dome_adj = self.coefficients["dome"] if dome else 0
        
        # Logistic transformation
        logit = np.log(base_prob / (1 - base_prob))
        adjusted_logit = logit + wind_adj + temp_adj + precip_adj + hash_adj + dome_adj
        
        adjusted_prob = 1 / (1 + np.exp(-adjusted_logit))
        return float(np.clip(adjusted_prob, 0.01, 0.99))
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from fitted model."""
        if not self.is_fitted:
            return self.coefficients
        
        feature_names = ["distance", "wind", "temp", "precip", "hash_mark", "altitude", "dome"]
        return dict(zip(feature_names, self.model.coef_[0]))


class ReturnEfficiencyModel:
    """Model for punt and kick return efficiency."""
    
    def __init__(self):
        self.punt_model = LogisticRegression()
        self.kick_model = LogisticRegression()
        self.is_fitted = False
    
    def calculate_return_efficiency(self, team_returns: pd.DataFrame, 
                                   coverage_strength: float = 0.5) -> Dict[str, float]:
        """Calculate return efficiency metrics."""
        efficiency = {}
        
        # Punt returns
        punt_returns = team_returns[team_returns["return_type"] == "punt"]
        if not punt_returns.empty:
            actual_ypa = punt_returns["yards"].mean()
            expected_ypa = self._get_expected_return_yards("punt", coverage_strength)
            efficiency["punt_return_efficiency"] = actual_ypa - expected_ypa
            efficiency["punt_return_ypa"] = actual_ypa
        
        # Kick returns
        kick_returns = team_returns[team_returns["return_type"] == "kick"]
        if not kick_returns.empty:
            actual_ypa = kick_returns["yards"].mean()
            expected_ypa = self._get_expected_return_yards("kick", coverage_strength)
            efficiency["kick_return_efficiency"] = actual_ypa - expected_ypa
            efficiency["kick_return_ypa"] = actual_ypa
        
        return efficiency
    
    def _get_expected_return_yards(self, return_type: str, coverage_strength: float) -> float:
        """Get expected return yards based on coverage strength."""
        # League averages
        if return_type == "punt":
            base_expected = 8.5  # Average punt return
        else:  # kick
            base_expected = 22.0  # Average kick return
        
        # Adjust for coverage strength (0 = worst, 1 = best)
        coverage_adjustment = (0.5 - coverage_strength) * 5  # +/- 2.5 yards
        
        return base_expected + coverage_adjustment


class SpecialTeamsEPA:
    """Calculate Special Teams EPA components."""
    
    def __init__(self, fg_model: FieldGoalModel):
        self.fg_model = fg_model
        
        # EPA values (would be calibrated from historical data)
        self.fg_epa_values = {
            "make": 3.0,      # Points from making FG
            "miss_short": -2.5,  # Field position loss
            "miss_long": -1.0,   # Less penalty for long attempts
            "block": -3.0     # Significant negative value
        }
        
        self.punt_epa_values = {
            "net_yard": 0.04,    # EPA per net yard
            "touchback": -0.5,   # Touchback penalty
            "block": -4.0,       # Blocked punt penalty
            "return_td": -6.0    # Return TD
        }
    
    def calculate_fg_epa(self, attempts: pd.DataFrame, weather_data: pd.DataFrame = None) -> pd.DataFrame:
        """Calculate EPA for field goal attempts."""
        fg_epa_list = []
        
        for _, attempt in attempts.iterrows():
            # Get weather conditions
            weather = self._get_weather_for_attempt(attempt, weather_data)
            
            # Predict make probability
            make_prob = self.fg_model.predict_probability(
                distance=attempt["distance"],
                wind=weather.get("wind", 0),
                temp=weather.get("temp", 70),
                precip=weather.get("precip", 0),
                hash_mark=attempt.get("hash_mark", False),
                altitude=weather.get("altitude", 0),
                dome=weather.get("dome", False)
            )
            
            # Calculate EPA components
            make_epa = make_prob * self.fg_epa_values["make"]
            
            # Miss EPA depends on distance
            if attempt["distance"] >= 50:
                miss_epa = (1 - make_prob) * self.fg_epa_values["miss_long"]
            else:
                miss_epa = (1 - make_prob) * self.fg_epa_values["miss_short"]
            
            # Block probability (simplified)
            block_prob = 0.02 + max(0, (attempt["distance"] - 45) * 0.001)
            block_epa = block_prob * self.fg_epa_values["block"]
            
            total_epa = make_epa + miss_epa + block_epa
            
            fg_epa_list.append({
                "game_id": attempt["game_id"],
                "team_id": attempt["team_id"],
                "distance": attempt["distance"],
                "make_probability": make_prob,
                "expected_epa": total_epa,
                "make_epa": make_epa,
                "miss_epa": miss_epa,
                "block_epa": block_epa
            })
        
        return pd.DataFrame(fg_epa_list)
    
    def calculate_punt_epa(self, punts: pd.DataFrame) -> pd.DataFrame:
        """Calculate EPA for punts."""
        punt_epa_list = []
        
        for _, punt in punts.iterrows():
            net_yards = punt.get("net_yards", punt.get("gross_yards", 40))
            
            # Base EPA from field position change
            base_epa = net_yards * self.punt_epa_values["net_yard"]
            
            # Touchback penalty
            if punt.get("touchback", False):
                base_epa += self.punt_epa_values["touchback"]
            
            # Return considerations
            return_yards = punt.get("return_yards", 0)
            return_epa = -return_yards * 0.04  # Negative for punting team
            
            # Blocked punt
            if punt.get("blocked", False):
                base_epa += self.punt_epa_values["block"]
            
            total_epa = base_epa + return_epa
            
            punt_epa_list.append({
                "game_id": punt["game_id"],
                "team_id": punt["team_id"],
                "net_yards": net_yards,
                "return_yards": return_yards,
                "expected_epa": total_epa,
                "field_position_epa": base_epa,
                "return_epa": return_epa
            })
        
        return pd.DataFrame(punt_epa_list)
    
    def calculate_return_epa(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate EPA for returns."""
        return_epa_list = []
        
        for _, return_play in returns.iterrows():
            return_yards = return_play.get("yards", 0)
            
            # EPA from return yards
            return_epa = return_yards * 0.04
            
            # Touchdown bonus
            if return_play.get("touchdown", False):
                return_epa += 6.0
            
            # Fumble penalty
            if return_play.get("fumble", False):
                return_epa -= 4.0
            
            return_epa_list.append({
                "game_id": return_play["game_id"],
                "team_id": return_play["team_id"],
                "return_type": return_play.get("return_type", "kick"),
                "yards": return_yards,
                "expected_epa": return_epa
            })
        
        return pd.DataFrame(return_epa_list)
    
    def _get_weather_for_attempt(self, attempt: pd.Series, 
                                weather_data: Optional[pd.DataFrame]) -> Dict[str, float]:
        """Get weather conditions for field goal attempt."""
        if weather_data is None or weather_data.empty:
            return {"wind": 0, "temp": 70, "precip": 0, "altitude": 0, "dome": False}
        
        # Match by game_id
        game_weather = weather_data[weather_data["game_id"] == attempt.get("game_id")]
        
        if game_weather.empty:
            return {"wind": 0, "temp": 70, "precip": 0, "altitude": 0, "dome": False}
        
        weather_row = game_weather.iloc[0]
        return {
            "wind": weather_row.get("wind_speed", 0),
            "temp": weather_row.get("temperature", 70),
            "precip": weather_row.get("precipitation", 0),
            "altitude": weather_row.get("altitude", 0),
            "dome": weather_row.get("dome", False)
        }


class SpecialTeamsFeatureBuilder:
    """Build comprehensive special teams features."""
    
    def __init__(self, config: GridConfig):
        self.config = config
        self.fg_model = FieldGoalModel()
        self.return_model = ReturnEfficiencyModel()
        self.st_epa = SpecialTeamsEPA(self.fg_model)
    
    def build_special_teams_features(self, st_data: pd.DataFrame, 
                                   weather_data: Optional[pd.DataFrame] = None,
                                   games_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Build comprehensive special teams features."""
        logger.info("Building special teams features...")
        
        st_features = []
        
        # Process by team and game
        for (team_id, game_id), group in st_data.groupby(["team_id", "game_id"]):
            features = {
                "team_id": team_id,
                "game_id": game_id,
                "unit": "special_teams"
            }
            
            # Field goal features
            fg_features = self._build_fg_features(group, weather_data)
            features.update(fg_features)
            
            # Punt features
            punt_features = self._build_punt_features(group)
            features.update(punt_features)
            
            # Return features
            return_features = self._build_return_features(group)
            features.update(return_features)
            
            # Overall ST-EPA
            st_epa_total = self._calculate_total_st_epa(group, weather_data)
            features.update(st_epa_total)
            
            st_features.append(features)
        
        result_df = pd.DataFrame(st_features)
        logger.info(f"Built special teams features for {len(result_df)} team-game combinations")
        return result_df
    
    def _build_fg_features(self, team_game_data: pd.DataFrame, 
                          weather_data: Optional[pd.DataFrame]) -> Dict[str, float]:
        """Build field goal features."""
        features = {}
        
        # Extract FG attempts from special teams data
        fg_attempts = team_game_data.get("fg_att", [0]).iloc[0] if not team_game_data.empty else 0
        fg_made = team_game_data.get("fg_made", [0]).iloc[0] if not team_game_data.empty else 0
        
        features["fg_attempts"] = fg_attempts
        features["fg_made"] = fg_made
        
        if fg_attempts > 0:
            features["fg_percentage"] = fg_made / fg_attempts
            
            # Get distance bins if available
            if "fg_dist_bins" in team_game_data.columns:
                dist_bins = team_game_data["fg_dist_bins"].iloc[0]
                if isinstance(dist_bins, dict):
                    # Calculate average distance (weighted)
                    total_attempts = sum(dist_bins.values())
                    if total_attempts > 0:
                        weighted_distance = sum(
                            float(dist.split("-")[0]) * count 
                            for dist, count in dist_bins.items()
                        ) / total_attempts
                        features["fg_avg_distance"] = weighted_distance
                        
                        # Expected make percentage based on distance
                        expected_makes = sum(
                            count * self.fg_model.predict_probability(
                                distance=float(dist.split("-")[0])
                            )
                            for dist, count in dist_bins.items()
                        )
                        features["fg_expected_makes"] = expected_makes
                        features["fg_makes_over_expected"] = fg_made - expected_makes
        
        return features
    
    def _build_punt_features(self, team_game_data: pd.DataFrame) -> Dict[str, float]:
        """Build punt features."""
        features = {}
        
        if "punt_net_avg" in team_game_data.columns:
            features["punt_net_avg"] = team_game_data["punt_net_avg"].iloc[0]
            
            # Expected punt net yards (league average)
            expected_net = 40.0
            features["punt_net_over_expected"] = features["punt_net_avg"] - expected_net
        
        return features
    
    def _build_return_features(self, team_game_data: pd.DataFrame) -> Dict[str, float]:
        """Build return features."""
        features = {}
        
        # Punt returns
        if "pr_ypa" in team_game_data.columns:
            features["punt_return_ypa"] = team_game_data["pr_ypa"].iloc[0]
            
            # Expected based on coverage (simplified)
            expected_pr_ypa = 8.5
            features["punt_return_over_expected"] = features["punt_return_ypa"] - expected_pr_ypa
        
        # Kick returns
        if "kr_ypa" in team_game_data.columns:
            features["kick_return_ypa"] = team_game_data["kr_ypa"].iloc[0]
            
            expected_kr_ypa = 22.0
            features["kick_return_over_expected"] = features["kick_return_ypa"] - expected_kr_ypa
        
        return features
    
    def _calculate_total_st_epa(self, team_game_data: pd.DataFrame,
                               weather_data: Optional[pd.DataFrame]) -> Dict[str, float]:
        """Calculate total special teams EPA."""
        total_epa = 0.0
        
        # FG EPA
        fg_attempts = team_game_data.get("fg_att", [0]).iloc[0] if not team_game_data.empty else 0
        fg_made = team_game_data.get("fg_made", [0]).iloc[0] if not team_game_data.empty else 0
        
        if fg_attempts > 0:
            # Simplified FG EPA calculation
            avg_distance = 42  # Estimate
            make_prob = self.fg_model.predict_probability(avg_distance)
            expected_points = make_prob * 3
            actual_points = fg_made * 3
            fg_epa = actual_points - expected_points
            total_epa += fg_epa
        
        # Punt EPA (simplified)
        if "punt_net_avg" in team_game_data.columns:
            punt_net = team_game_data["punt_net_avg"].iloc[0]
            expected_net = 40.0
            punt_epa = (punt_net - expected_net) * 0.04  # EPA per yard
            total_epa += punt_epa
        
        # Return EPA
        if "pr_ypa" in team_game_data.columns:
            pr_ypa = team_game_data["pr_ypa"].iloc[0]
            expected_pr = 8.5
            pr_epa = (pr_ypa - expected_pr) * 0.04
            total_epa += pr_epa
        
        if "kr_ypa" in team_game_data.columns:
            kr_ypa = team_game_data["kr_ypa"].iloc[0]
            expected_kr = 22.0
            kr_epa = (kr_ypa - expected_kr) * 0.04
            total_epa += kr_epa
        
        # Blocks
        if "blocks" in team_game_data.columns:
            blocks = team_game_data["blocks"].iloc[0]
            block_epa = blocks * 4.0  # High value for blocks
            total_epa += block_epa
        
        return {
            "st_epa_total": total_epa,
            "st_epa_per_play": total_epa / 10  # Estimate plays per game
        }
    
    def fit_models(self, historical_data: Dict[str, pd.DataFrame]) -> None:
        """Fit special teams models on historical data."""
        # Fit field goal model
        if "field_goals" in historical_data:
            self.fg_model.fit(historical_data["field_goals"])
        
        logger.info("Fitted special teams models")
    
    def get_model_performance(self) -> Dict[str, float]:
        """Get model performance metrics."""
        performance = {}
        
        if self.fg_model.is_fitted:
            performance["fg_model_fitted"] = True
            performance.update(self.fg_model.get_feature_importance())
        else:
            performance["fg_model_fitted"] = False
        
        return performance
