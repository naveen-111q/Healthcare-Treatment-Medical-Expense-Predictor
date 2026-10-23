"""
Healthcare Treatment & Medical Expense Predictor
================================================
Automated Test Suite
Verifies data loading, preprocessing, model inference, and output formats.
"""

import os
import unittest
import pandas as pd
import numpy as np
from predict import load_model, predict_single, predict_batch


class TestMedicalExpensePredictor(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.data_path = "data/insurance.csv"
        cls.model_path = "artifacts/model_pipeline.joblib"
        cls.model = load_model(cls.model_path)

    def test_01_dataset_integrity(self):
        """Verify the raw dataset exists, has no null values, and has correct columns."""
        self.assertTrue(os.path.exists(self.data_path), "Dataset file is missing")
        df = pd.read_csv(self.data_path)
        self.assertEqual(len(df), 1338, "Expected 1338 records in insurance.csv")
        self.assertEqual(df.isnull().sum().sum(), 0, "Dataset contains unexpected null values")
        
        expected_columns = {"age", "sex", "bmi", "children", "smoker", "region", "charges"}
        self.assertEqual(set(df.columns), expected_columns)

    def test_02_model_artifacts_exist(self):
        """Verify that serialized model and metrics files exist."""
        self.assertTrue(os.path.exists(self.model_path), "Model pipeline joblib is missing")
        self.assertTrue(os.path.exists("artifacts/metrics.json"), "metrics.json is missing")
        self.assertTrue(os.path.exists("artifacts/feature_importance.json"), "feature_importance.json is missing")

    def test_03_prediction_validity_and_bounds(self):
        """Verify that predictions return valid positive monetary values within sensible ranges."""
        res = predict_single(age=30, sex="female", bmi=25.0, children=1, smoker="no", region="northwest", model=self.model)
        self.assertIn("predicted_annual_charges", res)
        self.assertIn("risk_tier", res)
        self.assertGreater(res["predicted_annual_charges"], 1000.0)
        self.assertLess(res["predicted_annual_charges"], 60000.0)
        self.assertAlmostEqual(res["predicted_monthly_charges"], res["predicted_annual_charges"] / 12.0, places=1)

    def test_04_smoker_multiplier_impact(self):
        """Verify the critical domain rule: smokers incur substantially higher predicted charges than non-smokers."""
        non_smoker = predict_single(age=40, sex="male", bmi=30.0, children=2, smoker="no", region="southeast", model=self.model)
        smoker = predict_single(age=40, sex="male", bmi=30.0, children=2, smoker="yes", region="southeast", model=self.model)
        
        # Smokers typically pay 2x to 4x higher expenses
        self.assertGreater(smoker["predicted_annual_charges"], non_smoker["predicted_annual_charges"] * 2.0)
        self.assertGreater(smoker["potential_annual_savings_quit_smoking"], 10000.0)

    def test_05_batch_prediction(self):
        """Verify batch prediction works on DataFrame input."""
        sample_df = pd.DataFrame([
            {"age": 20, "sex": "male", "bmi": 22.0, "children": 0, "smoker": "no", "region": "northeast"},
            {"age": 55, "sex": "female", "bmi": 35.0, "children": 3, "smoker": "yes", "region": "southwest"}
        ])
        temp_csv = "artifacts/temp_test.csv"
        sample_df.to_csv(temp_csv, index=False)
        
        try:
            res_df = predict_batch(temp_csv, model=self.model)
            self.assertEqual(len(res_df), 2)
            self.assertIn("predicted_annual_charges", res_df.columns)
            self.assertIn("predicted_monthly_charges", res_df.columns)
            self.assertGreater(res_df.iloc[1]["predicted_annual_charges"], res_df.iloc[0]["predicted_annual_charges"])
        finally:
            if os.path.exists(temp_csv):
                os.remove(temp_csv)


if __name__ == "__main__":
    unittest.main()
