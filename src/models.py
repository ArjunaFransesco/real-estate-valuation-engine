"""
Machine Learning Models, Stacking Ensemble, and Model Persistence Module
Trains Ridge, Random Forest, Gradient Boosting, XGBoost, LightGBM, and Stacking Ensemble.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor
)
import xgboost as xgb
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

import joblib


def get_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns a dictionary of candidate regression models configured for real estate valuation.
    """
    models = {
        "Ridge_Regression": Ridge(alpha=10.0, random_state=random_state),
        "Random_Forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            n_jobs=-1,
            random_state=random_state
        ),
        "Gradient_Boosting": GradientBoostingRegressor(
            n_estimators=180,
            learning_rate=0.07,
            max_depth=5,
            random_state=random_state
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=220,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            n_jobs=-1,
            random_state=random_state
        )
    }
    
    if HAS_LIGHTGBM:
        models["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=220,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            n_jobs=-1,
            random_state=random_state,
            verbose=-1
        )
        
    return models


def build_stacking_ensemble(random_state: int = 42) -> StackingRegressor:
    """
    Builds a multi-model Stacking Ensemble with cross-validated out-of-fold predictions.
    """
    estimators = [
        ("rf", RandomForestRegressor(n_estimators=120, max_depth=10, n_jobs=-1, random_state=random_state)),
        ("gb", GradientBoostingRegressor(n_estimators=150, learning_rate=0.07, max_depth=5, random_state=random_state)),
        ("xgb", xgb.XGBRegressor(n_estimators=180, learning_rate=0.05, max_depth=6, n_jobs=-1, random_state=random_state))
    ]
    
    if HAS_LIGHTGBM:
        estimators.append(
            ("lgbm", lgb.LGBMRegressor(n_estimators=180, learning_rate=0.05, num_leaves=31, n_jobs=-1, random_state=random_state, verbose=-1))
        )
        
    final_estimator = Ridge(alpha=5.0)
    
    stacking_reg = StackingRegressor(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=5,
        n_jobs=-1,
        passthrough=True
    )
    
    return stacking_reg


def save_artifacts(
    pipeline: Any,
    model: Any,
    metrics: Dict[str, Any],
    feature_names: list,
    output_dir: str
) -> None:
    """Saves serialized model pipeline and evaluation metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save preprocessing pipeline and stacking model
    joblib.dump(pipeline, os.path.join(output_dir, "feature_pipeline.joblib"))
    joblib.dump(model, os.path.join(output_dir, "stacking_model.joblib"))
    
    # Save feature metadata & metrics JSON
    with open(os.path.join(output_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    with open(os.path.join(output_dir, "feature_metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"feature_names": feature_names, "total_features": len(feature_names)}, f, indent=2)
        
    print(f"[+] Artifacts saved to {output_dir}")
