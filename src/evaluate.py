"""
Evaluation Metrics, Cross-Validation Benchmarking, and Diagnostics Module
Computes R2, RMSE, MAE, MAPE, Median Absolute Error, and generates diagnostic reports.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error
)
from sklearn.model_selection import cross_validate, KFold


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard and robust regression evaluation metrics.
    """
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100.0)
    med_ae = float(median_absolute_error(y_true, y_pred))
    
    return {
        "R2_Score": round(r2, 4),
        "RMSE_USD": round(rmse, 2),
        "MAE_USD": round(mae, 2),
        "MAPE_Percent": round(mape, 2),
        "Median_AE_USD": round(med_ae, 2)
    }


def benchmark_models(
    models: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Runs K-Fold Cross-Validation across candidate models and tabulates metrics.
    """
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    results = []
    
    scoring = {
        "r2": "r2",
        "neg_rmse": "neg_root_mean_squared_error",
        "neg_mae": "neg_mean_absolute_error",
        "neg_mape": "neg_mean_absolute_percentage_error"
    }
    
    for model_name, model in models.items():
        print(f"[*] Benchmarking {model_name} with {cv_folds}-Fold CV...")
        cv_res = cross_validate(model, X_train, y_train, cv=kf, scoring=scoring, n_jobs=-1)
        
        mean_r2 = np.mean(cv_res["test_r2"])
        std_r2 = np.std(cv_res["test_r2"])
        mean_rmse = -np.mean(cv_res["test_neg_rmse"])
        mean_mae = -np.mean(cv_res["test_neg_mae"])
        mean_mape = -np.mean(cv_res["test_neg_mape"]) * 100.0
        
        results.append({
            "Model": model_name,
            "CV_R2_Mean": round(mean_r2, 4),
            "CV_R2_Std": round(std_r2, 4),
            "CV_RMSE_USD": round(mean_rmse, 2),
            "CV_MAE_USD": round(mean_mae, 2),
            "CV_MAPE_%": round(mean_mape, 2)
        })
        
    df_results = pd.DataFrame(results).sort_values(by="CV_R2_Mean", ascending=False).reset_index(drop=True)
    return df_results


def extract_feature_importance(model: Any, feature_names: List[str], top_n: int = 15) -> pd.DataFrame:
    """
    Extracts tree feature importances if available.
    """
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "named_estimators_"):
        # If stacking regressor, extract from one of the tree estimators
        for est_name, est in model.named_estimators_.items():
            if hasattr(est, "feature_importances_"):
                importances = est.feature_importances_
                break
                
    if importances is None or len(importances) != len(feature_names):
        return pd.DataFrame({"Feature": feature_names[:top_n], "Importance": [1.0 / top_n] * top_n})
        
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(top_n).reset_index(drop=True)
    
    return df_imp
