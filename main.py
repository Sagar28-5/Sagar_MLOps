"""
Main CLI Entrypoint for Credit Card Fraud Detection System.

Usage:
    python main.py --train        # Runs data preprocessing, model training, and evaluation
    python main.py --app          # Launches the Streamlit interactive dashboard
    python main.py --predict      # Runs inference on sample test cases
    python main.py --generate-data# Generates/verifies benchmark dataset
"""

import os
import sys
import argparse

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(
        description="💳 Credit Card Fraud Detection System - MLOps Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--train", action="store_true", help="Run complete data preprocessing, training, and evaluation pipeline")
    parser.add_argument("--app", action="store_true", help="Launch Streamlit web application")
    parser.add_argument("--predict", action="store_true", help="Test standalone inference engine")
    parser.add_argument("--generate-data", action="store_true", help="Generate or verify benchmark dataset")
    
    args = parser.parse_args()
    
    print("=" * 65)
    print(" [PROJECT] CREDIT CARD FRAUD DETECTION SYSTEM (MLOps Pipeline)")
    print("=" * 65)
    
    if args.generate_data:
        from data.download_or_generate_data import ensure_dataset
        ensure_dataset()
    elif args.train:
        from src.train import run_training_pipeline
        run_training_pipeline()
    elif args.predict:
        from src.predict import FraudPredictor
        predictor = FraudPredictor()
        
        sample_legit = {col: 0.0 for col in predictor.EXPECTED_FEATURES}
        sample_legit['Time'] = 3600.0
        sample_legit['Amount'] = 45.50
        res1 = predictor.predict_single(sample_legit)
        print(f"\n[SAMPLE 1 - Normal Transaction]")
        print(f" -> Result: {res1['status']} | Fraud Probability: {res1['fraud_probability']:.4f} | Risk: {res1['risk_level']}")
        
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
        print(f"\n[SAMPLE 2 - Suspicious Fraud Transaction]")
        print(f" -> Result: {res2['status']} | Fraud Probability: {res2['fraud_probability']:.4f} | Risk: {res2['risk_level']}")
    elif args.app:
        print("\n[INFO] Launching Streamlit web application on http://localhost:8501 ...")
        os.system("streamlit run app/app.py")
    else:
        print("\n[STATUS] Ready for execution. Available commands:")
        print("  1. Train ML Pipeline:    python main.py --train")
        print("  2. Launch Web App:       python main.py --app")
        print("  3. Test Single Scoring:  python main.py --predict")
        print("  4. Verify/Create Data:   python main.py --generate-data")
        print("\nFor more details, see README.md.")
    print("=" * 65)

if __name__ == "__main__":
    main()