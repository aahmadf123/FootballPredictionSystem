"""SHAP-based explanations for football predictions."""

import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.base import BaseEstimator
import torch
import torch.nn as nn
from loguru import logger

from grid.config import GridConfig
from grid.models.ensemble import EnsembleFramework
from grid.models.tabular import GBDTModel, DeepTabularTrainer


class FootballExplainer:
    """Comprehensive explainer for football prediction models."""
    
    def __init__(self, config: GridConfig):
        self.config = config
        self.explainers = {}
        self.feature_names = []
        self.feature_descriptions = {}
        self.background_data = None
        
        # Initialize feature descriptions
        self._setup_feature_descriptions()
    
    def _setup_feature_descriptions(self) -> None:
        """Setup human-readable feature descriptions."""
        self.feature_descriptions = {
            # Team strength features
            "elo_rating_diff": "Difference in team Elo ratings",
            "home_field_advantage": "Home field advantage factor",
            "recent_form_diff": "Difference in recent team performance",
            
            # Offensive features
            "offense_epa_play": "Offensive expected points added per play",
            "offense_success_rate": "Offensive success rate",
            "offense_explosive_rate": "Offensive explosive play rate",
            "offense_red_zone_td_rate": "Red zone touchdown conversion rate",
            "offense_third_down_conv": "Third down conversion rate",
            
            # Defensive features
            "defense_epa_play": "Defensive expected points added per play",
            "defense_success_rate": "Defensive success rate",
            "defense_pressure_rate": "Pass rush pressure rate",
            "defense_havoc_rate": "Defensive havoc rate",
            
            # Special teams features
            "st_epa_total": "Special teams expected points added",
            "fg_percentage": "Field goal success percentage",
            "punt_net_avg": "Net punting average",
            "return_efficiency": "Return game efficiency",
            
            # Player features
            "qb_epa_dropback": "Quarterback EPA per dropback",
            "qb_pressure_rate": "Quarterback pressure rate faced",
            "rb_epa_carry": "Running back EPA per carry",
            "wr_target_share": "Wide receiver target share",
            
            # Situational features
            "leverage_avg": "Average leverage of game situations",
            "garbage_time_filtered": "Stats with garbage time filtered",
            "momentum_factor": "Team momentum factor",
            
            # Contextual features
            "injury_impact": "Key player injury impact",
            "weather_factor": "Weather impact on game",
            "travel_factor": "Travel/rest advantage",
            "referee_bias": "Referee crew tendencies",
            
            # Advanced features
            "talent_advantage": "Recruiting talent advantage",
            "portal_impact": "Transfer portal net impact",
            "coaching_advantage": "Coaching efficiency advantage"
        }
    
    def fit_explainers(self, models: Dict[str, Any], X_background: pd.DataFrame, 
                      feature_names: List[str]) -> None:
        """Fit SHAP explainers for different model types."""
        self.feature_names = feature_names
        self.background_data = X_background.sample(min(100, len(X_background)))
        
        logger.info(f"Fitting explainers for {len(models)} models...")
        
        for model_name, model in models.items():
            try:
                if isinstance(model, GBDTModel):
                    # Tree explainer for GBDT models
                    self.explainers[model_name] = shap.TreeExplainer(model.model)
                    
                elif hasattr(model, 'predict_proba'):
                    # Kernel explainer for sklearn-like models
                    self.explainers[model_name] = shap.KernelExplainer(
                        lambda x: model.predict_proba(pd.DataFrame(x, columns=feature_names))[:, 1],
                        self.background_data.values
                    )
                    
                elif isinstance(model, (nn.Module, DeepTabularTrainer)):
                    # Deep explainer for neural networks
                    self.explainers[model_name] = self._create_deep_explainer(model)
                    
                else:
                    # Fallback to kernel explainer
                    predict_func = self._create_predict_function(model)
                    self.explainers[model_name] = shap.KernelExplainer(
                        predict_func,
                        self.background_data.values
                    )
                
                logger.info(f"Fitted {model_name} explainer")
                
            except Exception as e:
                logger.warning(f"Failed to fit explainer for {model_name}: {e}")
    
    def _create_deep_explainer(self, model: Union[nn.Module, DeepTabularTrainer]) -> shap.Explainer:
        """Create deep explainer for neural network models."""
        if isinstance(model, DeepTabularTrainer):
            def predict_func(x):
                df = pd.DataFrame(x, columns=self.feature_names)
                predictions = model.predict(df)
                return predictions["win_prob"]
        else:
            def predict_func(x):
                with torch.no_grad():
                    x_tensor = torch.FloatTensor(x)
                    outputs = model(x_tensor)
                    if isinstance(outputs, dict):
                        return outputs["win_prob"].numpy()
                    return outputs.numpy()
        
        return shap.KernelExplainer(predict_func, self.background_data.values)
    
    def _create_predict_function(self, model: Any) -> callable:
        """Create prediction function for arbitrary models."""
        def predict_func(x):
            if hasattr(model, 'predict_proba'):
                df = pd.DataFrame(x, columns=self.feature_names)
                return model.predict_proba(df)[:, 1]
            elif hasattr(model, 'predict'):
                df = pd.DataFrame(x, columns=self.feature_names)
                predictions = model.predict(df)
                if isinstance(predictions, dict):
                    return predictions.get("win_prob", predictions.get("prediction", 0.5))
                return predictions
            else:
                return np.full(len(x), 0.5)  # Fallback
        
        return predict_func
    
    def explain_prediction(self, X: pd.DataFrame, model_name: str = None, 
                          top_k: int = 10) -> Dict[str, Any]:
        """Generate SHAP explanations for predictions."""
        if not self.explainers:
            raise ValueError("Explainers not fitted. Call fit_explainers first.")
        
        if model_name and model_name not in self.explainers:
            raise ValueError(f"Model {model_name} not found in fitted explainers")
        
        # Use first explainer if no specific model requested
        if not model_name:
            model_name = list(self.explainers.keys())[0]
        
        explainer = self.explainers[model_name]
        
        try:
            # Calculate SHAP values
            if isinstance(explainer, shap.TreeExplainer):
                shap_values = explainer.shap_values(X.values)
                # For binary classification, take positive class
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            else:
                shap_values = explainer.shap_values(X.values, nsamples=100)
            
            # Handle single prediction
            if len(X) == 1:
                shap_values = shap_values[0]
                feature_values = X.iloc[0].values
            else:
                # Average for multiple predictions
                shap_values = np.mean(shap_values, axis=0)
                feature_values = X.mean().values
            
            # Create explanation dictionary
            explanations = []
            for i, feature_name in enumerate(self.feature_names):
                explanations.append({
                    'feature': feature_name,
                    'feature_value': float(feature_values[i]),
                    'shap_value': float(shap_values[i]),
                    'abs_shap_value': abs(float(shap_values[i])),
                    'description': self.feature_descriptions.get(feature_name, feature_name),
                    'impact': 'positive' if shap_values[i] > 0 else 'negative'
                })
            
            # Sort by absolute SHAP value
            explanations.sort(key=lambda x: x['abs_shap_value'], reverse=True)
            
            # Calculate base value and prediction
            base_value = float(explainer.expected_value)
            if isinstance(explainer.expected_value, (list, np.ndarray)):
                base_value = float(explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0])
            
            prediction = base_value + sum(exp['shap_value'] for exp in explanations)
            
            return {
                'model_name': model_name,
                'base_prediction': base_value,
                'final_prediction': prediction,
                'top_features': explanations[:top_k],
                'all_features': explanations,
                'explanation_quality': self._calculate_explanation_quality(shap_values)
            }
            
        except Exception as e:
            logger.error(f"SHAP explanation failed for {model_name}: {e}")
            return self._create_fallback_explanation(X, model_name)
    
    def _calculate_explanation_quality(self, shap_values: np.ndarray) -> float:
        """Calculate quality score for explanation."""
        # Quality based on the concentration of important features
        abs_values = np.abs(shap_values)
        if len(abs_values) == 0:
            return 0.0
        
        # Normalize
        abs_values = abs_values / np.sum(abs_values)
        
        # Calculate entropy (lower entropy = higher quality)
        entropy = -np.sum(abs_values * np.log(abs_values + 1e-10))
        max_entropy = np.log(len(abs_values))
        
        # Convert to 0-1 scale where 1 = highest quality
        quality = 1.0 - (entropy / max_entropy)
        
        return float(quality)
    
    def _create_fallback_explanation(self, X: pd.DataFrame, model_name: str) -> Dict[str, Any]:
        """Create fallback explanation when SHAP fails."""
        logger.warning(f"Creating fallback explanation for {model_name}")
        
        # Simple feature importance based on correlation
        feature_importances = []
        
        for i, feature_name in enumerate(self.feature_names):
            if i < len(X.columns):
                value = float(X.iloc[0, i]) if len(X) > 0 else 0.0
                # Simple heuristic importance
                importance = abs(value - X[feature_name].mean()) / (X[feature_name].std() + 1e-6)
                
                feature_importances.append({
                    'feature': feature_name,
                    'feature_value': value,
                    'shap_value': importance * 0.1,  # Scale down
                    'abs_shap_value': abs(importance * 0.1),
                    'description': self.feature_descriptions.get(feature_name, feature_name),
                    'impact': 'positive' if importance > 0 else 'negative'
                })
        
        feature_importances.sort(key=lambda x: x['abs_shap_value'], reverse=True)
        
        return {
            'model_name': model_name,
            'base_prediction': 0.5,
            'final_prediction': 0.5,
            'top_features': feature_importances[:10],
            'all_features': feature_importances,
            'explanation_quality': 0.3  # Lower quality for fallback
        }
    
    def generate_text_explanation(self, explanation: Dict[str, Any], 
                                 game_context: Optional[Dict[str, str]] = None) -> str:
        """Generate human-readable text explanation."""
        model_name = explanation['model_name']
        prediction = explanation['final_prediction']
        top_features = explanation['top_features'][:5]  # Top 5 features
        
        # Start with prediction
        confidence = "highly confident" if abs(prediction - 0.5) > 0.3 else "moderately confident"
        favored_team = "home team" if prediction > 0.5 else "away team"
        
        text_parts = [
            f"The {model_name} model predicts the {favored_team} will win with {confidence} ({prediction:.1%} probability)."
        ]
        
        # Add game context if available
        if game_context:
            context_str = f"In this matchup between {game_context.get('away_team', 'Away')} at {game_context.get('home_team', 'Home')}"
            if 'week' in game_context:
                context_str += f" in Week {game_context['week']}"
            text_parts.append(context_str + ":")
        
        # Explain key factors
        text_parts.append("Key factors driving this prediction:")
        
        for i, feature in enumerate(top_features, 1):
            impact_verb = "favors" if feature['impact'] == 'positive' else "hurts"
            team_ref = "home team" if feature['impact'] == 'positive' else "away team"
            
            feature_text = f"{i}. {feature['description']} {impact_verb} the {team_ref}"
            
            # Add specific value context
            if feature['feature_value'] != 0:
                feature_text += f" (value: {feature['feature_value']:.2f})"
            
            text_parts.append(feature_text)
        
        # Add confidence qualifier
        quality = explanation['explanation_quality']
        if quality > 0.8:
            text_parts.append("This explanation is based on high-quality feature analysis.")
        elif quality > 0.6:
            text_parts.append("This explanation has moderate confidence in feature importance.")
        else:
            text_parts.append("This explanation should be interpreted with caution due to model complexity.")
        
        return " ".join(text_parts)
    
    def explain_ensemble(self, X: pd.DataFrame, ensemble_framework: EnsembleFramework,
                        top_k: int = 10) -> Dict[str, Any]:
        """Explain ensemble predictions by combining individual model explanations."""
        if not hasattr(ensemble_framework, 'meta_ensemble'):
            raise ValueError("Ensemble framework not properly initialized")
        
        # Get individual model explanations
        individual_explanations = {}
        model_weights = ensemble_framework.get_model_weights()
        
        for model_name in model_weights.keys():
            if model_name in self.explainers:
                try:
                    explanation = self.explain_prediction(X, model_name, top_k)
                    individual_explanations[model_name] = explanation
                except Exception as e:
                    logger.warning(f"Failed to explain {model_name}: {e}")
        
        if not individual_explanations:
            raise ValueError("No individual explanations could be generated")
        
        # Combine explanations weighted by model importance
        combined_features = {}
        
        for model_name, explanation in individual_explanations.items():
            weight = model_weights.get(model_name, 0.0)
            
            for feature in explanation['all_features']:
                feature_name = feature['feature']
                
                if feature_name not in combined_features:
                    combined_features[feature_name] = {
                        'feature': feature_name,
                        'feature_value': feature['feature_value'],
                        'weighted_shap_value': 0.0,
                        'description': feature['description'],
                        'model_contributions': {}
                    }
                
                combined_features[feature_name]['weighted_shap_value'] += weight * feature['shap_value']
                combined_features[feature_name]['model_contributions'][model_name] = {
                    'shap_value': feature['shap_value'],
                    'weight': weight
                }
        
        # Convert to list and sort
        ensemble_features = []
        for feature_data in combined_features.values():
            feature_data['abs_shap_value'] = abs(feature_data['weighted_shap_value'])
            feature_data['impact'] = 'positive' if feature_data['weighted_shap_value'] > 0 else 'negative'
            ensemble_features.append(feature_data)
        
        ensemble_features.sort(key=lambda x: x['abs_shap_value'], reverse=True)
        
        # Calculate ensemble prediction
        ensemble_base = sum(
            model_weights.get(name, 0.0) * exp['base_prediction']
            for name, exp in individual_explanations.items()
        )
        
        ensemble_prediction = ensemble_base + sum(
            feature['weighted_shap_value'] for feature in ensemble_features
        )
        
        return {
            'model_name': 'ensemble',
            'base_prediction': ensemble_base,
            'final_prediction': ensemble_prediction,
            'top_features': ensemble_features[:top_k],
            'all_features': ensemble_features,
            'individual_explanations': individual_explanations,
            'model_weights': model_weights,
            'explanation_quality': np.mean([
                exp['explanation_quality'] for exp in individual_explanations.values()
            ])
        }
    
    def create_feature_importance_summary(self, explanations: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create summary of feature importance across multiple explanations."""
        feature_stats = {}
        
        for explanation in explanations:
            for feature in explanation['all_features']:
                feature_name = feature['feature']
                
                if feature_name not in feature_stats:
                    feature_stats[feature_name] = {
                        'count': 0,
                        'avg_shap_value': 0.0,
                        'max_abs_shap': 0.0,
                        'description': feature['description']
                    }
                
                stats = feature_stats[feature_name]
                stats['count'] += 1
                stats['avg_shap_value'] += feature['shap_value']
                stats['max_abs_shap'] = max(stats['max_abs_shap'], feature['abs_shap_value'])
        
        # Calculate averages
        for stats in feature_stats.values():
            if stats['count'] > 0:
                stats['avg_shap_value'] /= stats['count']
        
        # Convert to DataFrame
        summary_df = pd.DataFrame([
            {
                'feature': name,
                'description': stats['description'],
                'frequency': stats['count'],
                'avg_importance': stats['avg_shap_value'],
                'max_importance': stats['max_abs_shap']
            }
            for name, stats in feature_stats.items()
        ])
        
        return summary_df.sort_values('max_importance', ascending=False)


class CounterfactualAnalyzer:
    """Analyze counterfactual scenarios for predictions."""
    
    def __init__(self, explainer: FootballExplainer):
        self.explainer = explainer
    
    def analyze_counterfactual(self, X_original: pd.DataFrame, 
                              modifications: Dict[str, float],
                              model_name: str = None) -> Dict[str, Any]:
        """Analyze impact of modifying specific features."""
        # Create modified input
        X_modified = X_original.copy()
        
        for feature, new_value in modifications.items():
            if feature in X_modified.columns:
                X_modified[feature] = new_value
        
        # Get explanations for both scenarios
        original_explanation = self.explainer.explain_prediction(X_original, model_name)
        modified_explanation = self.explainer.explain_prediction(X_modified, model_name)
        
        # Calculate impact
        prediction_change = modified_explanation['final_prediction'] - original_explanation['final_prediction']
        
        # Identify which features changed most
        feature_changes = []
        
        original_features = {f['feature']: f['shap_value'] for f in original_explanation['all_features']}
        modified_features = {f['feature']: f['shap_value'] for f in modified_explanation['all_features']}
        
        for feature in original_features:
            if feature in modified_features:
                change = modified_features[feature] - original_features[feature]
                if abs(change) > 0.001:  # Meaningful change threshold
                    feature_changes.append({
                        'feature': feature,
                        'original_shap': original_features[feature],
                        'modified_shap': modified_features[feature],
                        'shap_change': change,
                        'description': self.explainer.feature_descriptions.get(feature, feature)
                    })
        
        feature_changes.sort(key=lambda x: abs(x['shap_change']), reverse=True)
        
        return {
            'modifications_applied': modifications,
            'original_prediction': original_explanation['final_prediction'],
            'modified_prediction': modified_explanation['final_prediction'],
            'prediction_change': prediction_change,
            'feature_changes': feature_changes,
            'original_explanation': original_explanation,
            'modified_explanation': modified_explanation
        }
