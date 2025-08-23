"""Meta-ensemble and calibration framework for football predictions."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize
from loguru import logger

from grid.config import GridConfig
from grid.models.elo import EloEnsemble
from grid.models.tabular import GBDTModel, DeepTabularTrainer


class MetaEnsemble:
    """Meta-ensemble with constrained weights for combining multiple models."""
    
    def __init__(self, config: GridConfig):
        self.config = config
        self.models = {}
        self.weights = {}
        self.calibrator = None
        self.is_fitted = False
        
        # Model components
        self.elo_ensemble = None
        self.gbdt_model = None
        self.dl_model = None
        
        # Ensemble constraints
        self.weight_bounds = (0.0, 1.0)
        self.sum_constraint = 1.0
    
    def add_model(self, name: str, model: Any, weight: float = None) -> None:
        """Add a model to the ensemble."""
        self.models[name] = model
        if weight is not None:
            self.weights[name] = weight
        
        logger.info(f"Added model {name} to ensemble")
    
    def fit_ensemble_weights(self, predictions_dict: Dict[str, np.ndarray], 
                           targets: np.ndarray, method: str = "ridge") -> Dict[str, float]:
        """Fit ensemble weights using constrained optimization."""
        if not predictions_dict:
            raise ValueError("No model predictions provided")
        
        # Stack predictions
        model_names = list(predictions_dict.keys())
        X = np.column_stack([predictions_dict[name] for name in model_names])
        
        if method == "ridge":
            # Ridge regression with non-negativity constraints
            weights = self._fit_constrained_ridge(X, targets)
        elif method == "minimize":
            # Direct optimization of log loss
            weights = self._fit_minimize_logloss(X, targets)
        else:
            # Equal weights as fallback
            weights = np.ones(len(model_names)) / len(model_names)
        
        # Normalize to sum to 1
        weights = weights / np.sum(weights)
        
        # Store weights
        for i, name in enumerate(model_names):
            self.weights[name] = weights[i]
        
        logger.info(f"Fitted ensemble weights: {dict(zip(model_names, weights))}")
        return dict(zip(model_names, weights))
    
    def _fit_constrained_ridge(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit Ridge regression with non-negative weights."""
        from sklearn.linear_model import Ridge
        from scipy.optimize import nnls
        
        # Use non-negative least squares
        weights, _ = nnls(X, y)
        
        return weights
    
    def _fit_minimize_logloss(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit weights by minimizing log loss directly."""
        n_models = X.shape[1]
        
        def objective(weights):
            # Ensure weights sum to 1
            weights = weights / np.sum(weights)
            ensemble_pred = X @ weights
            # Clip to avoid log(0)
            ensemble_pred = np.clip(ensemble_pred, 1e-15, 1 - 1e-15)
            return log_loss(y, ensemble_pred)
        
        # Constraints: non-negative weights that sum to 1
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0) for _ in range(n_models)]
        
        # Initial guess: equal weights
        x0 = np.ones(n_models) / n_models
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            return result.x
        else:
            logger.warning("Optimization failed, using equal weights")
            return np.ones(n_models) / n_models
    
    def predict_ensemble(self, predictions_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Make ensemble predictions using fitted weights."""
        if not self.weights:
            raise ValueError("Ensemble weights not fitted")
        
        ensemble_pred = np.zeros(len(list(predictions_dict.values())[0]))
        
        for name, pred in predictions_dict.items():
            if name in self.weights:
                ensemble_pred += self.weights[name] * pred
        
        return ensemble_pred
    
    def cross_validate_ensemble(self, predictions_dict: Dict[str, np.ndarray], 
                               targets: np.ndarray, cv_splits: int = 5) -> Dict[str, float]:
        """Cross-validate ensemble performance."""
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(targets):
            # Split data
            train_preds = {name: pred[train_idx] for name, pred in predictions_dict.items()}
            val_preds = {name: pred[val_idx] for name, pred in predictions_dict.items()}
            
            train_targets = targets[train_idx]
            val_targets = targets[val_idx]
            
            # Fit weights on training data
            weights = self.fit_ensemble_weights(train_preds, train_targets)
            
            # Predict on validation data
            val_ensemble = self.predict_ensemble(val_preds)
            
            # Calculate metrics
            val_logloss = log_loss(val_targets, val_ensemble)
            val_brier = brier_score_loss(val_targets, val_ensemble)
            
            cv_scores.append({
                'logloss': val_logloss,
                'brier': val_brier,
                'accuracy': np.mean((val_ensemble > 0.5) == val_targets)
            })
        
        # Average scores
        avg_scores = {
            'cv_logloss': np.mean([s['logloss'] for s in cv_scores]),
            'cv_brier': np.mean([s['brier'] for s in cv_scores]),
            'cv_accuracy': np.mean([s['accuracy'] for s in cv_scores]),
            'cv_std_logloss': np.std([s['logloss'] for s in cv_scores])
        }
        
        return avg_scores


class BayesianCalibrator:
    """Bayesian calibration for probability predictions."""
    
    def __init__(self):
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.is_fitted = False
    
    def fit(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        """Fit calibration mapping."""
        self.calibrator.fit(predictions, targets)
        self.is_fitted = True
        
        logger.info("Fitted Bayesian calibrator")
    
    def predict(self, predictions: np.ndarray) -> np.ndarray:
        """Apply calibration to predictions."""
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted")
        
        return self.calibrator.predict(predictions)
    
    def evaluate_calibration(self, predictions: np.ndarray, targets: np.ndarray, 
                           n_bins: int = 10) -> Dict[str, float]:
        """Evaluate calibration quality."""
        calibrated_preds = self.predict(predictions)
        
        # Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = (calibrated_preds > bin_lower) & (calibrated_preds <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = targets[in_bin].mean()
                avg_confidence_in_bin = calibrated_preds[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        # Brier Score
        brier_score = brier_score_loss(targets, calibrated_preds)
        
        # Reliability (weighted squared differences)
        reliability = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (calibrated_preds > bin_lower) & (calibrated_preds <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = targets[in_bin].mean()
                avg_confidence_in_bin = calibrated_preds[in_bin].mean()
                reliability += prop_in_bin * (avg_confidence_in_bin - accuracy_in_bin) ** 2
        
        return {
            'ece': ece,
            'brier_score': brier_score,
            'reliability': reliability,
            'calibrated_accuracy': np.mean((calibrated_preds > 0.5) == targets)
        }


class EnsembleFramework:
    """Complete ensemble framework with multiple models and calibration."""
    
    def __init__(self, config: GridConfig):
        self.config = config
        self.meta_ensemble = MetaEnsemble(config)
        self.calibrator = BayesianCalibrator()
        
        # Initialize component models
        self.elo_ensemble = EloEnsemble(config)
        self.gbdt_model = GBDTModel(config)
        self.dl_trainer = DeepTabularTrainer(config)
        
        self.is_fitted = False
    
    def fit(self, train_data: Dict[str, pd.DataFrame], 
           val_data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, float]:
        """Fit the complete ensemble framework."""
        logger.info("Training ensemble framework...")
        
        # Train individual models
        model_predictions = {}
        
        # Elo predictions
        if "games" in train_data and "teams" in train_data:
            self.elo_ensemble.initialize_all_ratings(train_data["teams"])
            elo_preds = self.elo_ensemble.predict_games(train_data["games"])
            if not elo_preds.empty:
                model_predictions["elo"] = elo_preds["home_win_prob"].values
        
        # GBDT predictions
        if "features" in train_data:
            gbdt_metrics = self.gbdt_model.fit(train_data["features"], val_data.get("features") if val_data else None)
            gbdt_preds = self.gbdt_model.predict_proba(train_data["features"])
            model_predictions["gbdt"] = gbdt_preds[:, 1]
        
        # Deep learning predictions
        if "features" in train_data:
            dl_history = self.dl_trainer.train(train_data["features"], val_data.get("features") if val_data else None)
            dl_preds = self.dl_trainer.predict(train_data["features"])
            model_predictions["deep_learning"] = dl_preds["win_prob"]
        
        # Extract targets (assuming home wins)
        if "games" in train_data:
            games_df = train_data["games"]
            completed_games = games_df.dropna(subset=["home_score", "away_score"])
            if not completed_games.empty:
                targets = (completed_games["home_score"] > completed_games["away_score"]).astype(int).values
                
                # Align predictions with completed games
                aligned_preds = {}
                for name, preds in model_predictions.items():
                    if len(preds) >= len(targets):
                        aligned_preds[name] = preds[:len(targets)]
                
                if aligned_preds:
                    # Fit ensemble weights
                    ensemble_weights = self.meta_ensemble.fit_ensemble_weights(aligned_preds, targets)
                    
                    # Get ensemble predictions
                    ensemble_preds = self.meta_ensemble.predict_ensemble(aligned_preds)
                    
                    # Fit calibrator
                    self.calibrator.fit(ensemble_preds, targets)
                    
                    # Evaluate
                    calibrated_preds = self.calibrator.predict(ensemble_preds)
                    
                    metrics = {
                        'ensemble_logloss': log_loss(targets, ensemble_preds),
                        'calibrated_logloss': log_loss(targets, calibrated_preds),
                        'ensemble_brier': brier_score_loss(targets, ensemble_preds),
                        'calibrated_brier': brier_score_loss(targets, calibrated_preds)
                    }
                    
                    # Add calibration metrics
                    cal_metrics = self.calibrator.evaluate_calibration(ensemble_preds, targets)
                    metrics.update(cal_metrics)
                    
                    self.is_fitted = True
                    
                    logger.info(f"Ensemble training completed. Final LogLoss: {metrics['calibrated_logloss']:.4f}")
                    return metrics
        
        raise ValueError("Insufficient data to train ensemble")
    
    def predict(self, data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """Make ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted")
        
        model_predictions = {}
        
        # Get predictions from each model
        if "games" in data:
            elo_preds = self.elo_ensemble.predict_games(data["games"])
            if not elo_preds.empty:
                model_predictions["elo"] = elo_preds["home_win_prob"].values
        
        if "features" in data:
            gbdt_preds = self.gbdt_model.predict_proba(data["features"])
            model_predictions["gbdt"] = gbdt_preds[:, 1]
            
            dl_preds = self.dl_trainer.predict(data["features"])
            model_predictions["deep_learning"] = dl_preds["win_prob"]
        
        # Ensemble predictions
        ensemble_preds = self.meta_ensemble.predict_ensemble(model_predictions)
        
        # Calibrated predictions
        calibrated_preds = self.calibrator.predict(ensemble_preds)
        
        return {
            'ensemble_raw': ensemble_preds,
            'ensemble_calibrated': calibrated_preds,
            'individual_models': model_predictions
        }
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get current ensemble weights."""
        return self.meta_ensemble.weights.copy()
    
    def evaluate_on_holdout(self, holdout_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Evaluate ensemble on holdout data."""
        predictions = self.predict(holdout_data)
        
        # Extract targets
        if "games" in holdout_data:
            games_df = holdout_data["games"]
            completed_games = games_df.dropna(subset=["home_score", "away_score"])
            
            if not completed_games.empty:
                targets = (completed_games["home_score"] > completed_games["away_score"]).astype(int).values
                
                calibrated_preds = predictions['ensemble_calibrated'][:len(targets)]
                
                metrics = {
                    'holdout_logloss': log_loss(targets, calibrated_preds),
                    'holdout_brier': brier_score_loss(targets, calibrated_preds),
                    'holdout_accuracy': np.mean((calibrated_preds > 0.5) == targets)
                }
                
                # Add calibration metrics
                cal_metrics = self.calibrator.evaluate_calibration(calibrated_preds, targets)
                metrics.update({f'holdout_{k}': v for k, v in cal_metrics.items()})
                
                return metrics
        
        return {}
