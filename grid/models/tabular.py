"""Tabular models including GBDT and Deep Learning for football prediction."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
import xgboost as xgb
import lightgbm as lgb
import optuna
from loguru import logger

from grid.config import GridConfig


class GBDTModel:
    """Gradient Boosting Decision Tree model using XGBoost and LightGBM."""
    
    def __init__(self, config: GridConfig, model_type: str = "xgboost"):
        self.config = config
        self.model_type = model_type
        self.model = None
        self.feature_columns = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Default hyperparameters
        self.hyperparams = self._get_default_hyperparams()
    
    def _get_default_hyperparams(self) -> Dict[str, Any]:
        """Get default hyperparameters for GBDT models."""
        if self.model_type == "xgboost":
            return {
                "n_estimators": 500,
                "max_depth": 8,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "random_state": 42,
                "eval_metric": "logloss",
                "early_stopping_rounds": 50
            }
        else:  # lightgbm
            return {
                "n_estimators": 500,
                "max_depth": 8,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "random_state": 42,
                "metric": "binary_logloss",
                "early_stopping_rounds": 50,
                "verbosity": -1
            }
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = "home_win") -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for training."""
        # Separate features and target
        if target_col in df.columns:
            y = df[target_col].values
            X_df = df.drop(columns=[target_col])
        else:
            y = None
            X_df = df.copy()
        
        # Handle categorical variables
        categorical_columns = X_df.select_dtypes(include=["object", "category"]).columns
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X_df[col] = self.label_encoders[col].fit_transform(X_df[col].astype(str))
            else:
                # Handle unseen categories
                X_df[col] = X_df[col].astype(str)
                known_categories = set(self.label_encoders[col].classes_)
                X_df[col] = X_df[col].apply(
                    lambda x: x if x in known_categories else "unknown"
                )
                X_df[col] = self.label_encoders[col].transform(X_df[col])
        
        # Handle missing values
        X_df = X_df.fillna(0)
        
        # Store feature columns if first time
        if self.feature_columns is None:
            self.feature_columns = X_df.columns.tolist()
        
        # Ensure same feature order
        X_df = X_df.reindex(columns=self.feature_columns, fill_value=0)
        
        X = X_df.values
        
        return X, y
    
    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None,
            target_col: str = "home_win") -> Dict[str, float]:
        """Fit the GBDT model."""
        logger.info(f"Training {self.model_type} model...")
        
        # Prepare training data
        X_train, y_train = self.prepare_features(train_df, target_col)
        
        # Prepare validation data
        if val_df is not None:
            X_val, y_val = self.prepare_features(val_df, target_col)
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
        
        # Initialize model
        if self.model_type == "xgboost":
            self.model = xgb.XGBClassifier(**self.hyperparams)
            
            # Fit with early stopping
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:  # lightgbm
            self.model = lgb.LGBMClassifier(**self.hyperparams)
            
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
        
        self.is_fitted = True
        
        # Calculate validation metrics
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        
        metrics = {
            "val_logloss": log_loss(y_val, y_pred_proba),
            "val_brier": brier_score_loss(y_val, y_pred_proba),
            "val_accuracy": accuracy_score(y_val, y_pred_proba > 0.5)
        }
        
        logger.info(f"Model trained. Validation LogLoss: {metrics['val_logloss']:.4f}")
        return metrics
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X, _ = self.prepare_features(df)
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        
        if self.model_type == "xgboost":
            importance = self.model.feature_importances_
        else:
            importance = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            "feature": self.feature_columns,
            "importance": importance
        }).sort_values("importance", ascending=False)
        
        return importance_df
    
    def optimize_hyperparameters(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                                target_col: str = "home_win", n_trials: int = 100) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        logger.info(f"Optimizing {self.model_type} hyperparameters...")
        
        def objective(trial):
            # Suggest hyperparameters
            params = self._suggest_hyperparameters(trial)
            
            # Update model parameters
            old_params = self.hyperparams.copy()
            self.hyperparams.update(params)
            
            try:
                # Train and evaluate
                metrics = self.fit(train_df, val_df, target_col)
                score = metrics["val_logloss"]
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                score = float("inf")
            finally:
                # Restore old parameters
                self.hyperparams = old_params
            
            return score
        
        # Run optimization
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        
        # Update with best parameters
        best_params = study.best_params
        self.hyperparams.update(best_params)
        
        logger.info(f"Best hyperparameters: {best_params}")
        logger.info(f"Best validation score: {study.best_value:.4f}")
        
        return best_params
    
    def _suggest_hyperparameters(self, trial) -> Dict[str, Any]:
        """Suggest hyperparameters for optimization."""
        if self.model_type == "xgboost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0)
            }
        else:  # lightgbm
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
                "num_leaves": trial.suggest_int("num_leaves", 10, 100)
            }


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network block for tabular deep learning."""
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Skip connection adjustment
        if input_dim != hidden_dim:
            self.skip_projection = nn.Linear(input_dim, hidden_dim)
        else:
            self.skip_projection = nn.Identity()
    
    def forward(self, x):
        # Store input for skip connection
        skip = self.skip_projection(x)
        
        # First linear transformation
        h = torch.relu(self.linear1(x))
        h = self.dropout(h)
        
        # Second linear transformation
        h = self.linear2(h)
        
        # Gating mechanism
        gate = torch.sigmoid(self.gate(h))
        h = gate * h
        
        # Skip connection and layer norm
        output = self.layer_norm(h + skip)
        
        return output


class DeepTabularModel(nn.Module):
    """Deep Learning model for tabular data with multi-task learning."""
    
    def __init__(self, input_dim: int, embedding_dims: Dict[str, int],
                 hidden_dim: int = 256, num_blocks: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dims = embedding_dims
        self.hidden_dim = hidden_dim
        
        # Embedding layers for categorical features
        self.embeddings = nn.ModuleDict()
        embedding_output_dim = 0
        
        for feature, num_categories in embedding_dims.items():
            embed_dim = min(50, (num_categories + 1) // 2)
            self.embeddings[feature] = nn.Embedding(num_categories, embed_dim)
            embedding_output_dim += embed_dim
        
        # Input projection
        total_input_dim = input_dim + embedding_output_dim
        self.input_projection = nn.Linear(total_input_dim, hidden_dim)
        
        # GRN blocks
        self.grn_blocks = nn.ModuleList([
            GatedResidualNetwork(hidden_dim, hidden_dim, dropout)
            for _ in range(num_blocks)
        ])
        
        # Multi-task heads
        self.win_prob_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.margin_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        self.total_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        self.aleatoric_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()  # Ensure positive variance
        )
    
    def forward(self, x_numeric, x_categorical=None):
        # Handle embeddings
        if x_categorical is not None and self.embeddings:
            embedded_features = []
            for feature, embedding in self.embeddings.items():
                if feature in x_categorical:
                    embedded = embedding(x_categorical[feature])
                    embedded_features.append(embedded)
            
            if embedded_features:
                embedded_concat = torch.cat(embedded_features, dim=1)
                x = torch.cat([x_numeric, embedded_concat], dim=1)
            else:
                x = x_numeric
        else:
            x = x_numeric
        
        # Input projection
        x = self.input_projection(x)
        
        # GRN blocks
        for grn in self.grn_blocks:
            x = grn(x)
        
        # Multi-task outputs
        win_prob = self.win_prob_head(x)
        margin = self.margin_head(x)
        total = self.total_head(x)
        aleatoric_var = self.aleatoric_head(x)
        
        return {
            "win_prob": win_prob.squeeze(),
            "margin": margin.squeeze(),
            "total": total.squeeze(),
            "aleatoric_var": aleatoric_var.squeeze()
        }


class DeepTabularTrainer:
    """Trainer for deep tabular models."""
    
    def __init__(self, config: GridConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.categorical_features = []
        
        # Loss weights
        self.loss_weights = {
            "win_prob": 1.0,
            "margin": 0.5,
            "total": 0.5,
            "aleatoric": 0.2
        }
    
    def prepare_data(self, df: pd.DataFrame, 
                    target_cols: Dict[str, str] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Prepare data for training."""
        if target_cols is None:
            target_cols = {
                "win_prob": "home_win",
                "margin": "score_margin", 
                "total": "total_points"
            }
        
        # Separate features and targets
        feature_df = df.copy()
        targets = {}
        
        for task, col in target_cols.items():
            if col in df.columns:
                targets[task] = torch.FloatTensor(df[col].values).to(self.device)
                feature_df = feature_df.drop(columns=[col])
        
        # Handle categorical features
        categorical_columns = feature_df.select_dtypes(include=["object", "category"]).columns
        categorical_data = {}
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                feature_df[col] = self.label_encoders[col].fit_transform(feature_df[col].astype(str))
            else:
                feature_df[col] = feature_df[col].astype(str)
                known_categories = set(self.label_encoders[col].classes_)
                feature_df[col] = feature_df[col].apply(
                    lambda x: x if x in known_categories else "unknown"
                )
                feature_df[col] = self.label_encoders[col].transform(feature_df[col])
            
            categorical_data[col] = torch.LongTensor(feature_df[col].values).to(self.device)
            self.categorical_features.append(col)
        
        # Remove categorical columns from numeric features
        numeric_df = feature_df.drop(columns=categorical_columns)
        
        # Handle missing values and scale
        numeric_df = numeric_df.fillna(0)
        
        if self.feature_columns is None:
            self.feature_columns = numeric_df.columns.tolist()
            numeric_scaled = self.scaler.fit_transform(numeric_df.values)
        else:
            numeric_df = numeric_df.reindex(columns=self.feature_columns, fill_value=0)
            numeric_scaled = self.scaler.transform(numeric_df.values)
        
        numeric_tensor = torch.FloatTensor(numeric_scaled).to(self.device)
        
        return numeric_tensor, categorical_data, targets
    
    def create_model(self, input_dim: int, embedding_dims: Dict[str, int]) -> DeepTabularModel:
        """Create the deep tabular model."""
        model = DeepTabularModel(
            input_dim=input_dim,
            embedding_dims=embedding_dims,
            hidden_dim=256,
            num_blocks=4,
            dropout=0.1
        ).to(self.device)
        
        return model
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute multi-task loss."""
        total_loss = 0.0
        
        # Win probability loss (BCE)
        if "win_prob" in outputs and "win_prob" in targets:
            win_loss = nn.BCELoss()(outputs["win_prob"], targets["win_prob"])
            total_loss += self.loss_weights["win_prob"] * win_loss
        
        # Margin loss (Huber)
        if "margin" in outputs and "margin" in targets:
            margin_loss = nn.HuberLoss()(outputs["margin"], targets["margin"])
            total_loss += self.loss_weights["margin"] * margin_loss
        
        # Total points loss (MSE)
        if "total" in outputs and "total" in targets:
            total_loss_component = nn.MSELoss()(outputs["total"], targets["total"])
            total_loss += self.loss_weights["total"] * total_loss_component
        
        # Aleatoric uncertainty loss (negative log-likelihood)
        if "aleatoric_var" in outputs and "win_prob" in targets:
            # Simplified aleatoric loss
            pred_var = outputs["aleatoric_var"]
            pred_mean = outputs["win_prob"]
            true_values = targets["win_prob"]
            
            # Negative log-likelihood of Gaussian
            nll_loss = 0.5 * (
                torch.log(2 * np.pi * pred_var) + 
                (true_values - pred_mean) ** 2 / pred_var
            ).mean()
            total_loss += self.loss_weights["aleatoric"] * nll_loss
        
        return total_loss
    
    def train(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None,
             epochs: int = 100, batch_size: int = 256, lr: float = 0.001) -> Dict[str, List[float]]:
        """Train the deep tabular model."""
        logger.info("Training deep tabular model...")
        
        # Prepare data
        X_train_num, X_train_cat, y_train = self.prepare_data(train_df)
        
        if val_df is not None:
            X_val_num, X_val_cat, y_val = self.prepare_data(val_df)
        else:
            # Split training data
            split_idx = int(0.8 * len(X_train_num))
            X_val_num = X_train_num[split_idx:]
            X_train_num = X_train_num[:split_idx]
            
            X_val_cat = {k: v[split_idx:] for k, v in X_train_cat.items()}
            X_train_cat = {k: v[:split_idx] for k, v in X_train_cat.items()}
            
            y_val = {k: v[split_idx:] for k, v in y_train.items()}
            y_train = {k: v[:split_idx] for k, v in y_train.items()}
        
        # Create model
        embedding_dims = {
            feature: len(self.label_encoders[feature].classes_)
            for feature in self.categorical_features
        }
        
        self.model = self.create_model(X_train_num.shape[1], embedding_dims)
        
        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            # Simple batch processing (would use DataLoader in production)
            for i in range(0, len(X_train_num), batch_size):
                end_idx = min(i + batch_size, len(X_train_num))
                
                batch_num = X_train_num[i:end_idx]
                batch_cat = {k: v[i:end_idx] for k, v in X_train_cat.items()}
                batch_targets = {k: v[i:end_idx] for k, v in y_train.items()}
                
                optimizer.zero_grad()
                
                outputs = self.model(batch_num, batch_cat)
                loss = self.compute_loss(outputs, batch_targets)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= (len(X_train_num) / batch_size)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_num, X_val_cat)
                val_loss = self.compute_loss(val_outputs, y_val).item()
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                torch.save(self.model.state_dict(), "best_model.pth")
            else:
                patience_counter += 1
                
            if patience_counter >= 20:  # Early stopping patience
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Load best model
        self.model.load_state_dict(torch.load("best_model.pth"))
        
        logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        return history
    
    def predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Make predictions with the trained model."""
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        X_num, X_cat, _ = self.prepare_data(df)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_num, X_cat)
        
        # Convert to numpy
        predictions = {}
        for key, tensor in outputs.items():
            predictions[key] = tensor.cpu().numpy()
        
        return predictions
