"""
End-to-End Automated Real Estate Valuation Machine Learning Pipeline
Executes data generation, feature engineering, model training, stacking, and artifact persistence.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import load_dataset, split_features_target
from src.feature_engineering import build_preprocessor
from src.models import get_candidate_models, build_stacking_ensemble, save_artifacts
from src.evaluate import calculate_regression_metrics, benchmark_models, extract_feature_importance


def run_pipeline(
    data_path: str = None,
    artifacts_dir: str = None,
    random_state: int = 42
):
    print("=" * 80)
    print("       AUTOMATED REAL ESTATE VALUATION & SPATIAL PRICE PREDICTION       ")
    print("=" * 80)
    
    # 1. Paths configuration
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if data_path is None:
        data_path = os.path.join(base_dir, "data", "properties_dataset.csv")
    if artifacts_dir is None:
        artifacts_dir = os.path.join(base_dir, "artifacts")
        
    # 2. Data Loading
    print(f"\n[1/5] Loading / Generating Dataset...")
    df = load_dataset(data_path)
    print(f"      Total records: {len(df):,} | Features: {df.shape[1]}")
    print(f"      Median Price: ${df['price'].median():,.2f} | Mean: ${df['price'].mean():,.2f}")
    
    # 3. Train-Test Splitting
    print(f"\n[2/5] Splitting Features and Target (80% Train / 20% Test)...")
    X, y = split_features_target(df, target_col="price")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state
    )
    print(f"      Train samples: {len(X_train):,} | Test samples: {len(X_test):,}")
    
    # 4. Feature Engineering & Preprocessing
    print(f"\n[3/5] Fitting Spatial Clustering & Feature Transformer...")
    preprocessor = build_preprocessor(n_clusters=8)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    # Extract feature column names
    ohe_cats = preprocessor.named_steps["column_transforms"].named_transformers_["cat"].get_feature_names_out()
    num_cols = preprocessor.named_steps["column_transforms"].transformers[0][2]
    all_feature_names = list(num_cols) + list(ohe_cats)
    print(f"      Total transformed features: {len(all_feature_names)}")
    
    # 5. Candidate Benchmarking
    print(f"\n[4/5] Running 5-Fold Cross-Validation across candidate models...")
    candidate_models = get_candidate_models(random_state=random_state)
    benchmark_df = benchmark_models(
        candidate_models,
        X_train_trans,
        y_train.values,
        cv_folds=5,
        random_state=random_state
    )
    print("\n--- 5-Fold Cross-Validation Benchmark Results ---")
    print(benchmark_df.to_string(index=False))
    
    # 6. Training Final Stacking Ensemble
    print(f"\n[5/5] Training Meta-Stacking Ensemble Regressor...")
    stacking_model = build_stacking_ensemble(random_state=random_state)
    stacking_model.fit(X_train_trans, y_train.values)
    
    # Test evaluation
    y_test_pred = stacking_model.predict(X_test_trans)
    test_metrics = calculate_regression_metrics(y_test.values, y_test_pred)
    
    print("\n" + "=" * 55)
    print("       FINAL TEST SET PERFORMANCE (STACKING ENSEMBLE)       ")
    print("=" * 55)
    for k, v in test_metrics.items():
        print(f"  * {k:<20}: {v}")
    print("=" * 55)
    
    # Save artifacts
    all_metrics = {
        "test_performance": test_metrics,
        "cv_benchmark": benchmark_df.to_dict(orient="records"),
        "dataset_summary": {
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features_count": len(all_feature_names)
        }
    }
    
    save_artifacts(
        pipeline=preprocessor,
        model=stacking_model,
        metrics=all_metrics,
        feature_names=all_feature_names,
        output_dir=artifacts_dir
    )
    
    print("\n[+] Real Estate Valuation Pipeline execution completed successfully!\n")
    return test_metrics


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    run_pipeline()
