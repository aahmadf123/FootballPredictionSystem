"""Conformal prediction for uncertainty quantification."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
from loguru import logger


class BaseConformalPredictor(ABC):
    """Base class for conformal prediction methods."""
    
    def __init__(self, alpha: float = 0.1):
        """
        Initialize conformal predictor.
        
        Args:
            alpha: Miscoverage rate (e.g., 0.1 for 90% coverage)
        """
        self.alpha = alpha
        self.quantile_level = 1 - alpha
        self.calibration_scores = None
        self.quantile = None
        self.is_fitted = False
    
    @abstractmethod
    def compute_nonconformity_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute nonconformity scores."""
        pass
    
    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Fit conformal predictor on calibration data."""
        # Compute nonconformity scores
        scores = self.compute_nonconformity_score(y_true, y_pred)
        
        # Store calibration scores
        self.calibration_scores = scores
        
        # Compute quantile
        n = len(scores)
        quantile_index = int(np.ceil((n + 1) * self.quantile_level))
        
        if quantile_index >= n:
            self.quantile = np.max(scores)
        else:
            self.quantile = np.sort(scores)[quantile_index - 1]
        
        self.is_fitted = True
        logger.info(f"Fitted conformal predictor with quantile {self.quantile:.4f} for {self.quantile_level:.1%} coverage")
    
    @abstractmethod
    def predict_interval(self, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict conformity intervals."""
        pass
    
    def evaluate_coverage(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate empirical coverage on test data."""
        if not self.is_fitted:
            raise ValueError("Conformal predictor not fitted")
        
        lower, upper = self.predict_interval(y_pred)
        
        # Calculate coverage
        in_interval = (y_true >= lower) & (y_true <= upper)
        empirical_coverage = np.mean(in_interval)
        
        # Calculate average interval width
        avg_width = np.mean(upper - lower)
        
        # Calculate conditional coverage by prediction value
        coverage_by_pred = []
        pred_bins = np.linspace(np.min(y_pred), np.max(y_pred), 10)
        
        for i in range(len(pred_bins) - 1):
            bin_mask = (y_pred >= pred_bins[i]) & (y_pred < pred_bins[i + 1])
            if np.sum(bin_mask) > 0:
                bin_coverage = np.mean(in_interval[bin_mask])
                coverage_by_pred.append(bin_coverage)
        
        return {
            'empirical_coverage': empirical_coverage,
            'target_coverage': self.quantile_level,
            'coverage_gap': abs(empirical_coverage - self.quantile_level),
            'average_width': avg_width,
            'conditional_coverage_std': np.std(coverage_by_pred) if coverage_by_pred else 0.0
        }


class RegressionConformalPredictor(BaseConformalPredictor):
    """Conformal predictor for regression problems (e.g., margin prediction)."""
    
    def compute_nonconformity_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute absolute residuals as nonconformity scores."""
        return np.abs(y_true - y_pred)
    
    def predict_interval(self, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict confidence intervals for regression."""
        if not self.is_fitted:
            raise ValueError("Conformal predictor not fitted")
        
        lower = y_pred - self.quantile
        upper = y_pred + self.quantile
        
        return lower, upper


class ClassificationConformalPredictor(BaseConformalPredictor):
    """Conformal predictor for classification problems (e.g., win probability)."""
    
    def compute_nonconformity_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute 1 - predicted probability for true class as nonconformity score."""
        # For binary classification, nonconformity = 1 - p_predicted_class
        scores = np.zeros(len(y_true))
        
        for i in range(len(y_true)):
            if y_true[i] == 1:
                scores[i] = 1 - y_pred[i]  # True class is 1, so score is 1 - p(class=1)
            else:
                scores[i] = y_pred[i]      # True class is 0, so score is p(class=1)
        
        return scores
    
    def predict_interval(self, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict confidence intervals for probabilities."""
        if not self.is_fitted:
            raise ValueError("Conformal predictor not fitted")
        
        # For classification, we create intervals around probabilities
        lower = np.maximum(0.0, y_pred - self.quantile)
        upper = np.minimum(1.0, y_pred + self.quantile)
        
        return lower, upper


class AdaptiveConformalPredictor(BaseConformalPredictor):
    """Adaptive conformal predictor that adjusts for changing distributions."""
    
    def __init__(self, alpha: float = 0.1, gamma: float = 0.005):
        """
        Initialize adaptive conformal predictor.
        
        Args:
            alpha: Target miscoverage rate
            gamma: Learning rate for adaptation
        """
        super().__init__(alpha)
        self.gamma = gamma
        self.alpha_t = alpha  # Current adaptive alpha
        self.coverage_history = []
    
    def compute_nonconformity_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute absolute residuals as nonconformity scores."""
        return np.abs(y_true - y_pred)
    
    def update_alpha(self, coverage_error: float) -> None:
        """Update alpha based on recent coverage error."""
        # Gradient step to adjust alpha
        self.alpha_t = self.alpha_t + self.gamma * coverage_error
        self.alpha_t = np.clip(self.alpha_t, 0.01, 0.5)  # Keep alpha in reasonable range
        
        self.quantile_level = 1 - self.alpha_t
    
    def predict_interval_adaptive(self, y_pred: np.ndarray, 
                                recent_coverage: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Predict intervals with adaptive alpha."""
        if not self.is_fitted:
            raise ValueError("Conformal predictor not fitted")
        
        # Update alpha if recent coverage provided
        if recent_coverage is not None:
            coverage_error = self.alpha - (1 - recent_coverage)  # Error in miscoverage
            self.update_alpha(coverage_error)
            self.coverage_history.append(recent_coverage)
        
        # Recompute quantile with updated alpha
        if self.calibration_scores is not None:
            n = len(self.calibration_scores)
            quantile_index = int(np.ceil((n + 1) * self.quantile_level))
            
            if quantile_index >= n:
                self.quantile = np.max(self.calibration_scores)
            else:
                self.quantile = np.sort(self.calibration_scores)[quantile_index - 1]
        
        lower = y_pred - self.quantile
        upper = y_pred + self.quantile
        
        return lower, upper
    
    def predict_interval(self, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Standard predict interval interface."""
        return self.predict_interval_adaptive(y_pred)


class SplitConformalPredictor:
    """Split conformal prediction for guaranteed coverage."""
    
    def __init__(self, model, alpha: float = 0.1, random_state: int = 42):
        """
        Initialize split conformal predictor.
        
        Args:
            model: ML model with fit and predict methods
            alpha: Miscoverage rate
            random_state: Random seed for data splitting
        """
        self.model = model
        self.alpha = alpha
        self.random_state = random_state
        self.conformal_predictor = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
           problem_type: str = "regression") -> None:
        """
        Fit split conformal predictor.
        
        Args:
            X: Features
            y: Targets
            problem_type: "regression" or "classification"
        """
        # Split data
        X_train, X_cal, y_train, y_cal = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        
        # Fit model on training data
        self.model.fit(X_train, y_train)
        
        # Get predictions on calibration data
        if hasattr(self.model, 'predict_proba') and problem_type == "classification":
            y_cal_pred = self.model.predict_proba(X_cal)[:, 1]
        else:
            y_cal_pred = self.model.predict(X_cal)
        
        # Fit conformal predictor
        if problem_type == "regression":
            self.conformal_predictor = RegressionConformalPredictor(self.alpha)
        else:
            self.conformal_predictor = ClassificationConformalPredictor(self.alpha)
        
        self.conformal_predictor.fit(y_cal, y_cal_pred)
        self.is_fitted = True
        
        logger.info(f"Fitted split conformal predictor for {problem_type}")
    
    def predict_with_intervals(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with conformal intervals.
        
        Returns:
            predictions, lower_bounds, upper_bounds
        """
        if not self.is_fitted:
            raise ValueError("Split conformal predictor not fitted")
        
        # Get model predictions
        if hasattr(self.model, 'predict_proba'):
            predictions = self.model.predict_proba(X)[:, 1]
        else:
            predictions = self.model.predict(X)
        
        # Get conformal intervals
        lower, upper = self.conformal_predictor.predict_interval(predictions)
        
        return predictions, lower, upper
    
    def evaluate_coverage(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate coverage on test data."""
        predictions, lower, upper = self.predict_with_intervals(X)
        return self.conformal_predictor.evaluate_coverage(y, predictions)


class MultiOutputConformalPredictor:
    """Conformal predictor for multiple outputs (e.g., win prob + margin)."""
    
    def __init__(self, alphas: Dict[str, float]):
        """
        Initialize multi-output conformal predictor.
        
        Args:
            alphas: Dictionary mapping output names to miscoverage rates
        """
        self.alphas = alphas
        self.predictors = {}
        self.is_fitted = False
    
    def fit(self, predictions: Dict[str, np.ndarray], 
           targets: Dict[str, np.ndarray]) -> None:
        """Fit conformal predictors for each output."""
        for output_name in self.alphas:
            if output_name not in predictions or output_name not in targets:
                continue
            
            alpha = self.alphas[output_name]
            
            # Determine problem type based on output values
            if output_name in ["win_prob", "probability"]:
                predictor = ClassificationConformalPredictor(alpha)
            else:
                predictor = RegressionConformalPredictor(alpha)
            
            predictor.fit(targets[output_name], predictions[output_name])
            self.predictors[output_name] = predictor
        
        self.is_fitted = True
        logger.info(f"Fitted multi-output conformal predictor for {list(self.predictors.keys())}")
    
    def predict_intervals(self, predictions: Dict[str, np.ndarray]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Predict intervals for all outputs."""
        if not self.is_fitted:
            raise ValueError("Multi-output conformal predictor not fitted")
        
        intervals = {}
        
        for output_name, predictor in self.predictors.items():
            if output_name in predictions:
                lower, upper = predictor.predict_interval(predictions[output_name])
                intervals[output_name] = (lower, upper)
        
        return intervals
    
    def evaluate_all_coverage(self, predictions: Dict[str, np.ndarray], 
                            targets: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Evaluate coverage for all outputs."""
        coverage_results = {}
        
        for output_name, predictor in self.predictors.items():
            if output_name in predictions and output_name in targets:
                coverage = predictor.evaluate_coverage(targets[output_name], predictions[output_name])
                coverage_results[output_name] = coverage
        
        return coverage_results


def create_football_conformal_framework(alpha_win_prob: float = 0.1, 
                                      alpha_margin: float = 0.1) -> MultiOutputConformalPredictor:
    """Create conformal prediction framework for football predictions."""
    alphas = {
        "win_prob": alpha_win_prob,
        "margin": alpha_margin,
        "total_points": alpha_margin
    }
    
    return MultiOutputConformalPredictor(alphas)
