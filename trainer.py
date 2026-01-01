"""
Model Training Module

This module handles training ML models based on ModelDesignResponse,
validating them on validation data, and persisting trained models to disk.
"""

import os
import json
import pickle
import numpy as np
import math
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)


# Model pool mappings
CLASSIFICATION_MODELS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "linear_svm": LinearSVC
}

REGRESSION_MODELS = {
    "linear_regression": LinearRegression,
    "ridge_regression": Ridge,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor
}


class ModelTrainer:
    """
    Trains ML models from ModelDesignResponse, validates them,
    and saves to disk with metadata.
    """
    
    def __init__(self, dataset_registry, models_dir: str = "./trained_models"):
        """
        Initialize ModelTrainer.
        
        Args:
            dataset_registry: Registry containing training/validation data
            models_dir: Directory to save trained models
        """
        self.dataset_registry = dataset_registry
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def train_models(
        self,
        model_design_response: Dict,
        x_train_id: str,
        y_train_id: str,
        x_val_id: str,
        y_val_id: str,
        task_type: str
    ) -> List[Dict[str, Any]]:
        """
        Train all models from ModelDesignResponse.
        
        Args:
            model_design_response: Dict with 'models' and 'primary_metric'
            x_train_id: Training features ID in registry
            y_train_id: Training target ID in registry
            x_val_id: Validation features ID in registry
            y_val_id: Validation target ID in registry
            task_type: 'binary', 'multiclass', or 'regression'
            
        Returns:
            List of dicts with model info: {name, path, metrics, timestamp}
        """
        # Retrieve data from registry
        X_train = self.dataset_registry.get(x_train_id)
        y_train = self.dataset_registry.get(y_train_id)
        X_val = self.dataset_registry.get(x_val_id)
        y_val = self.dataset_registry.get(y_val_id)
        
        models_list = model_design_response["models"]
        primary_metric = model_design_response.get("primary_metric", "accuracy")
        
        trained_models = []
        
        print(f"\n{'='*80}")
        print(f"TRAINING {len(models_list)} MODEL(S)")
        print(f"{'='*80}\n")
        
        for idx, model_spec in enumerate(models_list, 1):
            model_name = model_spec["name"]
            print(f"[{idx}/{len(models_list)}] Training {model_name}...")
            
            try:
                # Instantiate model
                model = self._instantiate_model(model_spec, task_type)
                
                # Train
                model = self._train_single_model(model, X_train, y_train)
                
                # Validate
                metrics = self._validate_model(
                    model, X_val, y_val, task_type, primary_metric
                )
                
                # Save to disk
                model_info = self._save_model(model, model_name, model_spec, metrics)
                
                trained_models.append(model_info)
                
                print(f"  ✓ {model_name} trained successfully")
                print(f"    {primary_metric}: {metrics.get(primary_metric, 'N/A'):.4f}")
                
            except Exception as e:
                print(f"  ✗ {model_name} training failed: {str(e)}")
                continue
        
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETE: {len(trained_models)}/{len(models_list)} models successful")
        print(f"{'='*80}\n")
        
        return trained_models
    
    def _instantiate_model(self, model_spec: Dict, task_type: str):
        """
        Create sklearn model instance from ModelSpec.
        
        Args:
            model_spec: Dict with 'name' and 'params' (List[{name, value}])
            task_type: Task type to select correct model pool
            
        Returns:
            Instantiated sklearn model
        """
        model_name = model_spec["name"]
        params_list = model_spec.get("params", [])
        
        # Convert List[ModelParam] to dict
        params = {}
        for param in params_list:
            name = param["name"]
            value = param["value"]
            
            # Type conversion
            if value is None:
                params[name] = None
            elif isinstance(value, bool):
                params[name] = value
            elif isinstance(value, (int, float)):
                # Try to convert to int if it's a whole number
                if isinstance(value, float) and value.is_integer():
                    params[name] = int(value)
                else:
                    params[name] = value
            else:
                params[name] = value  # Keep as string
        
        # Select model class
        if task_type in ["binary", "multiclass"]:
            model_pool = CLASSIFICATION_MODELS
        else:
            model_pool = REGRESSION_MODELS
        
        if model_name not in model_pool:
            raise ValueError(
                f"Model '{model_name}' not found in {task_type} model pool. "
                f"Available: {list(model_pool.keys())}"
            )
        
        ModelClass = model_pool[model_name]
        
        # Instantiate with parameters
        try:
            model = ModelClass(**params)
        except TypeError as e:
            print(f"  Warning: Invalid parameters for {model_name}: {e}")
            print(f"  Falling back to default parameters")
            model = ModelClass()
        
        return model
    
    def _train_single_model(self, model, X_train, y_train):
        """
        Fit a single model on training data.
        
        Args:
            model: sklearn model instance
            X_train: Training features
            y_train: Training target
            
        Returns:
            Fitted model
        """
        model.fit(X_train, y_train)
        return model
    
    def _validate_model(
        self,
        model,
        X_val,
        y_val,
        task_type: str,
        primary_metric: str
    ) -> Dict[str, float]:
        """
        Compute validation metrics for trained model.
        
        Args:
            model: Trained sklearn model
            X_val: Validation features
            y_val: Validation target
            task_type: Task type
            primary_metric: Primary metric name
            
        Returns:
            Dict of metric name -> value
        """
        y_pred = model.predict(X_val)
        
        metrics = {}
        
        if task_type in ["binary", "multiclass"]:
            # Classification metrics
            metrics["accuracy"] = accuracy_score(y_val, y_pred)
            
            # Handle binary vs multiclass
            average = "binary" if task_type == "binary" else "weighted"
            
            metrics["precision"] = precision_score(
                y_val, y_pred, average=average, zero_division=0
            )
            metrics["recall"] = recall_score(
                y_val, y_pred, average=average, zero_division=0
            )
            metrics["f1"] = f1_score(
                y_val, y_pred, average=average, zero_division=0
            )
            
            # ROC AUC for binary classification
            if task_type == "binary" and hasattr(model, "predict_proba"):
                try:
                    y_proba = model.predict_proba(X_val)[:, 1]
                    metrics["roc_auc"] = roc_auc_score(y_val, y_proba)
                except:
                    metrics["roc_auc"] = None
            
        else:
            # Regression metrics
            metrics["mse"] = mean_squared_error(y_val, y_pred)
            metrics["rmse"] = math.sqrt(metrics["mse"])
            metrics["mae"] = mean_absolute_error(y_val, y_pred)
            metrics["r2"] = r2_score(y_val, y_pred)
            
            # RMSLE (Root Mean Squared Log Error)
            # Only compute if all predictions and targets are non-negative
            try:

                if np.all(y_val >= 0) and np.all(y_pred >= 0):
                    metrics["rmsle"] = np.sqrt(mean_squared_error(
                        np.log1p(y_val), np.log1p(y_pred)
                    ))
                else:
                    metrics["rmsle"] = None  # Cannot compute for negative values
            except Exception:
                metrics["rmsle"] = None
        
        return metrics
    
    def _save_model(
        self,
        model,
        model_name: str,
        model_spec: Dict,
        metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Save trained model and metadata to disk.
        
        Args:
            model: Trained sklearn model
            model_name: Model name
            model_spec: Original ModelSpec dict
            metrics: Validation metrics
            
        Returns:
            Dict with model info: {name, path, metadata_path, metrics, timestamp}
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Model filename
        model_filename = f"{model_name}_{timestamp}.pkl"
        model_path = self.models_dir / model_filename
        
        # Metadata filename
        metadata_filename = f"{model_name}_{timestamp}_metadata.json"
        metadata_path = self.models_dir / metadata_filename
        
        # Save model
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata = {
            "model_name": model_name,
            "timestamp": timestamp,
            "params": model_spec.get("params", []),
            "rationale": model_spec.get("rationale", ""),
            "metrics": {k: float(v) if v is not None else None for k, v in metrics.items()},
            "model_path": str(model_path)
        }
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        return {
            "name": model_name,
            "path": str(model_path),
            "metadata_path": str(metadata_path),
            "metrics": metrics,
            "timestamp": timestamp
        }
