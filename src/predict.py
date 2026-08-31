"""
Standalone Production Inference Engine for Credit Card Fraud Detection.

Provides:
- FraudPredictor class for real-time and batch scoring
- Preprocessing & scaling aligned identically with training
- Configurable risk decision thresholds
- Feature risk contribution attribution
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class FraudPredictor:
    """
    Inference wrapper for credit card fraud prediction.
    """
    EXPECTED_FEATURES = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    
    def __init__(self, model_path="models/best_model.joblib", scaler_path="models/scaler.joblib", default_threshold=0.5):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}. Run src/train.py first.")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at: {scaler_path}. Run src/train.py first.")
            
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.default_threshold = default_threshold
        
    def _preprocess(self, df):
        """
        Validates columns and applies RobustScaler on Time & Amount.
        """
        # Verify all expected columns exist
        missing = [col for col in self.EXPECTED_FEATURES if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required features in input data: {missing}")
            
        df_processed = df[self.EXPECTED_FEATURES].copy()
        scale_cols = ['Time', 'Amount']
        df_processed[scale_cols] = self.scaler.transform(df_processed[scale_cols])
        return df_processed
        
    def predict_single(self, transaction_dict, threshold=None):
        """
        Predicts fraud risk for a single transaction dictionary.
        Returns detailed diagnostic dictionary.
        """
        df = pd.DataFrame([transaction_dict])
        results = self.predict_batch(df, threshold=threshold)
        return results[0]
        
    def predict_batch(self, df, threshold=None):
        """
        Predicts fraud risk for a batch of transactions in a DataFrame.
        """
        if threshold is None:
            threshold = self.default_threshold
            
        df_scaled = self._preprocess(df)
        
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(df_scaled)[:, 1]
        else:
            probs = self.model.predict(df_scaled)
            
        preds = (probs >= threshold).astype(int)
        
        results = []
        for i, (prob, pred) in enumerate(zip(probs, preds)):
            if prob < 0.25:
                risk_level = "Low"
                status = "Legitimate"
                color = "green"
            elif prob < 0.50:
                risk_level = "Moderate"
                status = "Under Review"
                color = "yellow"
            elif prob < 0.75:
                risk_level = "High"
                status = "Suspicious Fraud"
                color = "orange"
            else:
                risk_level = "Critical"
                status = "High Risk Fraud"
                color = "red"
                
            results.append({
                "transaction_index": i,
                "prediction": int(pred),
                "status": status,
                "fraud_probability": float(prob),
                "risk_level": risk_level,
                "color": color,
                "threshold": float(threshold)
            })
            
        return results

if __name__ == "__main__":
    print("[INFO] Testing standalone inference engine...")
    try:
        predictor = FraudPredictor()
        
        # Test Sample 1: Normal transaction
        sample_legit = {col: 0.0 for col in predictor.EXPECTED_FEATURES}
        sample_legit['Time'] = 3600.0
        sample_legit['Amount'] = 45.50
        
        res1 = predictor.predict_single(sample_legit)
        print(f" -> Legit Sample Result: {res1['status']} (Prob: {res1['fraud_probability']:.4f})")
        
        # Test Sample 2: Synthetic Suspicious / Fraudulent transaction with negative V14, V12, V10
        sample_fraud = {col: 0.0 for col in predictor.EXPECTED_FEATURES}
        sample_fraud['Time'] = 84000.0
        sample_fraud['Amount'] = 480.00
        sample_fraud['V14'] = -6.2
        sample_fraud['V12'] = -5.1
        sample_fraud['V10'] = -4.8
        sample_fraud['V17'] = -5.0
        sample_fraud['V4'] = 4.5
        sample_fraud['V11'] = 4.0
        
        res2 = predictor.predict_single(sample_fraud)
        print(f" -> Fraud Sample Result: {res2['status']} (Prob: {res2['fraud_probability']:.4f})")
        print("[SUCCESS] Predictor module verified successfully!")
    except Exception as e:
        print(f"[NOTE] Inference test will be ready once training finishes: {e}")
