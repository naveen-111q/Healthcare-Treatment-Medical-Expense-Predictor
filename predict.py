"""
Healthcare Treatment & Medical Expense Predictor
================================================
Inference Module & CLI
Provides programmatic prediction functions and CLI inference.
"""

import os
import argparse
import joblib
import pandas as pd
import numpy as np


def load_model(model_path="artifacts/model_pipeline.joblib"):
    """Load the trained Scikit-learn model pipeline."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model artifact not found at '{model_path}'. "
            "Please run 'python train.py' first to train and save the pipeline."
        )
    return joblib.load(model_path)


def predict_single(
    age: int,
    sex: str,
    bmi: float,
    children: int,
    smoker: str,
    region: str,
    model=None
) -> dict:
    """
    Predict medical expense for a single patient record.
    
    Parameters:
        age (int): Patient age (18 - 100)
        sex (str): 'male' or 'female'
        bmi (float): Body Mass Index (10.0 - 60.0)
        children (int): Number of dependents (0 - 10)
        smoker (str): 'yes' or 'no'
        region (str): 'southwest', 'southeast', 'northwest', or 'northeast'
        model: Pre-loaded model pipeline (optional)
        
    Returns:
        dict: Detailed prediction results including annual cost, monthly cost,
              risk tier, and lifestyle recommendations.
    """
    if model is None:
        model = load_model()
        
    # Input normalization
    sex = sex.lower().strip()
    smoker = smoker.lower().strip()
    region = region.lower().strip()
    
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": float(bmi),
        "children": int(children),
        "smoker": smoker,
        "region": region
    }])
    
    pred_annual = float(model.predict(input_df)[0])
    pred_monthly = pred_annual / 12.0
    
    # Classify Risk Tier
    if pred_annual < 5000:
        risk_tier = "Low"
        risk_color = "green"
    elif pred_annual < 12000:
        risk_tier = "Moderate"
        risk_color = "blue"
    elif pred_annual < 25000:
        risk_tier = "Elevated"
        risk_color = "orange"
    else:
        risk_tier = "High"
        risk_color = "red"
        
    # Simulate Smoking Cessation Savings
    annual_savings_quit_smoking = 0.0
    if smoker == "yes":
        non_smoker_df = input_df.copy()
        non_smoker_df["smoker"] = "no"
        non_smoker_cost = float(model.predict(non_smoker_df)[0])
        annual_savings_quit_smoking = max(0.0, pred_annual - non_smoker_cost)
        
    # Simulate BMI Optimization Savings (to normal range ~24.5)
    annual_savings_bmi_reduction = 0.0
    if bmi > 25.0:
        opt_bmi_df = input_df.copy()
        opt_bmi_df["bmi"] = 24.5
        opt_bmi_cost = float(model.predict(opt_bmi_df)[0])
        annual_savings_bmi_reduction = max(0.0, pred_annual - opt_bmi_cost)
        
    return {
        "predicted_annual_charges": round(pred_annual, 2),
        "predicted_monthly_charges": round(pred_monthly, 2),
        "risk_tier": risk_tier,
        "risk_color": risk_color,
        "potential_annual_savings_quit_smoking": round(annual_savings_quit_smoking, 2),
        "potential_annual_savings_normal_bmi": round(annual_savings_bmi_reduction, 2),
        "input_profile": {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region
        }
    }


def predict_batch(csv_filepath: str, output_filepath: str = None, model=None) -> pd.DataFrame:
    """
    Run batch inference on a CSV file containing patient records.
    """
    if model is None:
        model = load_model()
        
    df = pd.read_csv(csv_filepath)
    required_cols = {"age", "sex", "bmi", "children", "smoker", "region"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
        
    predictions = model.predict(df[list(required_cols)])
    df["predicted_annual_charges"] = np.round(predictions, 2)
    df["predicted_monthly_charges"] = np.round(predictions / 12.0, 2)
    
    if output_filepath:
        df.to_csv(output_filepath, index=False)
        print(f"Batch predictions saved to {output_filepath}")
        
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict medical treatment and insurance expenses.")
    parser.add_argument("--age", type=int, default=35, help="Age of patient")
    parser.add_argument("--sex", type=str, default="male", choices=["male", "female"], help="Biological sex")
    parser.add_argument("--bmi", type=float, default=28.5, help="Body Mass Index (BMI)")
    parser.add_argument("--children", type=int, default=1, help="Number of dependent children")
    parser.add_argument("--smoker", type=str, default="no", choices=["yes", "no"], help="Smoking status")
    parser.add_argument("--region", type=str, default="northeast", choices=["northeast", "northwest", "southeast", "southwest"], help="Residential region")
    
    args = parser.parse_args()
    
    result = predict_single(
        age=args.age,
        sex=args.sex,
        bmi=args.bmi,
        children=args.children,
        smoker=args.smoker,
        region=args.region
    )
    
    print("\n" + "=" * 55)
    print("      HEALTHCARE EXPENSE PREDICTION REPORT")
    print("=" * 55)
    print(f" Patient Profile : {args.age}yo {args.sex}, BMI={args.bmi}, {args.children} child(ren), Smoker={args.smoker}, {args.region}")
    print(f" Risk Category   : {result['risk_tier']}")
    print(f" Est. Annual Cost: ${result['predicted_annual_charges']:,.2f}")
    print(f" Est. Monthly    : ${result['predicted_monthly_charges']:,.2f}")
    if args.smoker == "yes":
        print(f" Smoking Cessation Savings: -${result['potential_annual_savings_quit_smoking']:,.2f} / year")
    if args.bmi > 25.0:
        print(f" BMI Normalization Savings: -${result['potential_annual_savings_normal_bmi']:,.2f} / year")
    print("=" * 55)
