"""
Healthcare Treatment & Medical Expense Predictor
================================================
Production Training Script
Trains a Scikit-learn Pipeline with ColumnTransformer and RandomForestRegressor.
Saves the serialized model pipeline, metrics, and feature importances.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def load_data(filepath="data/insurance.csv"):
    """Load and validate the insurance dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"Loaded dataset from {filepath} with shape {df.shape}")
    return df


def build_preprocessor(numeric_features, categorical_features):
    """
    Constructs a ColumnTransformer to apply:
    - StandardScaler to continuous numerical features
    - OneHotEncoder to categorical features (dropping first level)
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop"
    )
    return preprocessor


def train_models():
    """Execute end-to-end training and evaluation pipeline."""
    # Ensure artifacts directory exists
    os.makedirs("artifacts", exist_ok=True)
    
    # 1. Load data
    df = load_data("data/insurance.csv")
    
    X = df.drop("charges", axis=1)
    y = df["charges"]
    
    numeric_features = ["age", "bmi", "children"]
    categorical_features = ["sex", "smoker", "region"]
    
    # 2. Train-Test Split (80/20) with fixed random seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    
    # 3. Construct Preprocessor
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    
    # 4. Baseline Model: Linear Regression
    baseline_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])
    baseline_pipe.fit(X_train, y_train)
    y_pred_baseline = baseline_pipe.predict(X_test)
    
    baseline_r2 = r2_score(y_test, y_pred_baseline)
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_baseline)))
    baseline_mae = float(mean_absolute_error(y_test, y_pred_baseline))
    
    # 5. Primary Model: Tuned RandomForestRegressor
    rf_regressor = RandomForestRegressor(
        n_estimators=250,
        max_depth=5,
        min_samples_leaf=4,
        min_samples_split=6,
        max_features=None,
        random_state=42,
        n_jobs=-1
    )
    
    model_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", rf_regressor)
    ])
    
    # 5-Fold Cross Validation on Training Set
    cv_scores = cross_val_score(model_pipe, X_train, y_train, cv=5, scoring="r2")
    cv_r2_mean = float(np.mean(cv_scores))
    cv_r2_std = float(np.std(cv_scores))
    
    # Fit on full training set
    model_pipe.fit(X_train, y_train)
    y_pred_rf = model_pipe.predict(X_test)
    
    rf_r2 = float(r2_score(y_test, y_pred_rf))
    rf_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_rf)))
    rf_mae = float(mean_absolute_error(y_test, y_pred_rf))
    
    mae_reduction_pct = float(((baseline_mae - rf_mae) / baseline_mae) * 100)
    
    # 6. Extract Feature Importances
    preprocessor_fitted = model_pipe.named_steps["preprocessor"]
    cat_feature_names = preprocessor_fitted.named_transformers_["cat"].get_feature_names_out(categorical_features)
    all_feature_names = numeric_features + list(cat_feature_names)
    
    importances = model_pipe.named_steps["regressor"].feature_importances_
    feat_imp_dict = {
        name: float(round(imp, 5))
        for name, imp in sorted(zip(all_feature_names, importances), key=lambda x: x[1], reverse=True)
    }
    
    # 7. Print Results Table
    print("\n" + "=" * 65)
    print("           MODEL PERFORMANCE BENCHMARK RESULTS")
    print("=" * 65)
    print(f"{'Metric':<25} | {'Baseline (Linear Reg)':<20} | {'Random Forest':<15}")
    print("-" * 65)
    print(f"{'R² Score (Test)':<25} | {baseline_r2:<20.4f} | {rf_r2:<15.4f}")
    print(f"{'5-Fold CV R² (Train)':<25} | {'N/A':<20} | {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
    print(f"{'RMSE ($)':<25} | ${baseline_rmse:<19,.2f} | ${rf_rmse:<14,.2f}")
    print(f"{'MAE ($)':<25} | ${baseline_mae:<19,.2f} | ${rf_mae:<14,.2f}")
    print(f"{'MAE Improvement':<25} | {'Baseline':<20} | {mae_reduction_pct:>+.1f}%")
    print("=" * 65)
    
    print("\nFeature Importances:")
    for feat, imp in feat_imp_dict.items():
        print(f"  - {feat:<20}: {imp * 100:.2f}%")
    
    # 8. Save Model Artifacts
    joblib.dump(model_pipe, "artifacts/model_pipeline.joblib")
    joblib.dump(baseline_pipe, "artifacts/baseline_pipeline.joblib")
    
    metrics = {
        "model_name": "RandomForestRegressor",
        "hyperparameters": {
            "n_estimators": 250,
            "max_depth": 5,
            "min_samples_leaf": 4,
            "min_samples_split": 6,
            "max_features": None,
            "random_state": 42
        },
        "test_metrics": {
            "r2_score": round(rf_r2, 4),
            "rmse": round(rf_rmse, 2),
            "mae": round(rf_mae, 2)
        },
        "baseline_metrics": {
            "r2_score": round(baseline_r2, 4),
            "rmse": round(baseline_rmse, 2),
            "mae": round(baseline_mae, 2)
        },
        "comparison": {
            "mae_reduction_percentage": round(mae_reduction_pct, 2),
            "rmse_reduction_dollars": round(baseline_rmse - rf_rmse, 2)
        },
        "cross_validation": {
            "folds": 5,
            "mean_r2": round(cv_r2_mean, 4),
            "std_r2": round(cv_r2_std, 4)
        },
        "features": all_feature_names
    }
    
    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    with open("artifacts/feature_importance.json", "w") as f:
        json.dump(feat_imp_dict, f, indent=4)
        
    print("\n[SUCCESS] Artifacts successfully saved to 'artifacts/' folder.")
    return model_pipe, metrics


if __name__ == "__main__":
    train_models()
