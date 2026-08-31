"""
End-to-End Training, Hyperparameter Tuning, Ensembling, and Evaluation Pipeline
for Credit Card Fraud Detection.

Workflow:
1. Ingests data and computes diagnostics
2. Generates EDA visualization artifacts
3. Performs Stratified Split & Scaler fitting
4. Applies class imbalance techniques (SMOTE on train fold)
5. Trains and tunes multiple algorithms (Logistic Regression, Decision Tree, Random Forest, XGBoost, Voting Ensemble)
6. Evaluates all models on the pristine test set
7. Selects best model according to PR-AUC & Fraud Recall
8. Persists best model, scaler, and benchmark metadata in models/
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from tabulate import tabulate

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Scikit-Learn Models & Tools
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

# Internal Modules
from src.data_preprocessing import (
    load_data,
    inspect_data,
    clean_data,
    prepare_train_test_split,
    scale_features,
    balance_training_data
)
from src.evaluate import (
    evaluate_model_performance,
    plot_eda_figures,
    plot_model_comparison,
    plot_evaluation_curves,
    plot_confusion_matrices,
    plot_feature_importances
)

def run_training_pipeline(data_path="data/creditcard.csv", models_dir="models", reports_dir="reports/figures"):
    """
    Executes the complete machine learning lifecycle.
    """
    print("=" * 70, flush=True)
    print(" [START] CREDIT CARD FRAUD DETECTION ML PIPELINE", flush=True)
    print("=" * 70, flush=True)
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1. Dataset Loading & Inspection
    print("\n[STEP 1/8] Ingesting and Validating Dataset...", flush=True)
    df = load_data(data_path)
    diag = inspect_data(df)
    print(f" -> Total Transactions: {diag['shape'][0]:,}", flush=True)
    print(f" -> Features: {diag['shape'][1]}", flush=True)
    print(f" -> Missing Values: {diag['missing_values']}", flush=True)
    print(f" -> Duplicate Records: {diag['duplicates']}", flush=True)
    print(f" -> Legitimate: {diag['legit_count']:,} | Fraud: {diag['fraud_count']:,} ({diag['fraud_percentage']:.3f}%)", flush=True)
    
    # 2. Data Cleaning
    print("\n[STEP 2/8] Cleaning Data...", flush=True)
    df_clean = clean_data(df, drop_duplicates=False) # Keep original distribution integrity
    
    # 3. Exploratory Data Analysis (EDA)
    print("\n[STEP 3/8] Generating Exploratory Data Analysis (EDA) Visualizations...", flush=True)
    plot_eda_figures(df_clean, output_dir=reports_dir)
    print(" -> EDA figures successfully generated.", flush=True)
    
    # 4. Stratified Train/Test Split
    print("\n[STEP 4/8] Performing Stratified Train/Test Split (80/20)...", flush=True)
    X_train, X_test, y_train, y_test = prepare_train_test_split(df_clean, target_col='Class', test_size=0.2, random_state=42)
    print(f" -> Training set: {X_train.shape[0]:,} samples (Fraud: {y_train.sum():,})", flush=True)
    print(f" -> Test set:     {X_test.shape[0]:,} samples (Fraud: {y_test.sum():,})", flush=True)
    
    # 5. Feature Scaling (Fitted ONLY on training data to prevent leakage)
    print("\n[STEP 5/8] Scaling Numerical Features (Amount, Time) with RobustScaler...", flush=True)
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, scaler_type='robust', save_path=scaler_path)
    
    # 6. Handling Class Imbalance (SMOTE on Training Fold Only)
    print("\n[STEP 6/8] Balancing Training Data via SMOTE (Synthetic Minority Over-sampling)...", flush=True)
    X_train_smote, y_train_smote = balance_training_data(X_train_scaled, y_train, method='smote', random_state=42)
    print(f" -> Post-SMOTE Training Size: {X_train_smote.shape[0]:,} samples (Legit: {(y_train_smote==0).sum():,}, Fraud: {(y_train_smote==1).sum():,})", flush=True)
    
    # 7. Model Training, Comparison & Tuning
    print("\n[STEP 7/8] Training Candidate Models...", flush=True)
    
    candidate_models = {
        "Logistic Regression (Class Weighted)": LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            class_weight='balanced',
            random_state=42
        ),
        "Random Forest (Balanced)": RandomForestClassifier(
            n_estimators=80,
            max_depth=10,
            class_weight='balanced',
            n_jobs=1,
            random_state=42
        ),
        "XGBoost (SMOTE)": XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=1
        )
    }
    
    results = {}
    trained_estimators = {}
    
    for name, clf in candidate_models.items():
        print(f" -> Training: {name}...", flush=True)
        if "SMOTE" in name:
            clf.fit(X_train_smote, y_train_smote)
        else:
            clf.fit(X_train_scaled, y_train)
            
        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)[:, 1] if hasattr(clf, "predict_proba") else None
        
        metrics = evaluate_model_performance(y_test, y_pred, y_prob)
        results[name] = {
            "model": clf,
            "metrics": metrics,
            "y_prob": y_prob
        }
        trained_estimators[name] = clf
        print(f"    Precision: {metrics['Precision']:.3f} | Recall: {metrics['Recall']:.3f} | F1: {metrics['F1_Score']:.3f} | PR-AUC: {metrics['PR_AUC']:.3f}", flush=True)
        
    # Ensembling: Soft Voting Classifier combining top performers
    print(" -> Training: Voting Ensemble (Random Forest + XGBoost + Logistic Regression)...", flush=True)
    ensemble_clf = VotingClassifier(
        estimators=[
            ('rf', candidate_models["Random Forest (Balanced)"]),
            ('xgb', candidate_models["XGBoost (SMOTE)"]),
            ('lr', candidate_models["Logistic Regression (Class Weighted)"])
        ],
        voting='soft'
    )
    ensemble_clf.fit(X_train_smote, y_train_smote)
    ens_pred = ensemble_clf.predict(X_test_scaled)
    ens_prob = ensemble_clf.predict_proba(X_test_scaled)[:, 1]
    ens_metrics = evaluate_model_performance(y_test, ens_pred, ens_prob)
    results["Voting Ensemble"] = {
        "model": ensemble_clf,
        "metrics": ens_metrics,
        "y_prob": ens_prob
    }
    trained_estimators["Voting Ensemble"] = ensemble_clf
    print(f"    Precision: {ens_metrics['Precision']:.3f} | Recall: {ens_metrics['Recall']:.3f} | F1: {ens_metrics['F1_Score']:.3f} | PR-AUC: {ens_metrics['PR_AUC']:.3f}", flush=True)
    
    # Hyperparameter Tuning on XGBoost
    print("\n -> Performing Hyperparameter Tuning for XGBoost via RandomizedSearchCV (Stratified 3-Fold)...", flush=True)
    param_dist = {
        'n_estimators': [60, 100],
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0]
    }
    cv_strat = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    xgb_search = RandomizedSearchCV(
        estimator=XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=1),
        param_distributions=param_dist,
        n_iter=4,
        scoring='average_precision', # Optimized for PR-AUC (fraud precision-recall)
        cv=cv_strat,
        random_state=42,
        n_jobs=1
    )
    xgb_search.fit(X_train_smote, y_train_smote)
    tuned_xgb = xgb_search.best_estimator_
    print(f"    Best Parameters: {xgb_search.best_params_}", flush=True)
    
    tuned_pred = tuned_xgb.predict(X_test_scaled)
    tuned_prob = tuned_xgb.predict_proba(X_test_scaled)[:, 1]
    tuned_metrics = evaluate_model_performance(y_test, tuned_pred, tuned_prob)
    results["Tuned XGBoost"] = {
        "model": tuned_xgb,
        "metrics": tuned_metrics,
        "y_prob": tuned_prob
    }
    trained_estimators["Tuned XGBoost"] = tuned_xgb
    print(f"    Precision: {tuned_metrics['Precision']:.3f} | Recall: {tuned_metrics['Recall']:.3f} | F1: {tuned_metrics['F1_Score']:.3f} | PR-AUC: {tuned_metrics['PR_AUC']:.3f}", flush=True)

    # 8. Model Evaluation & Comparison Table
    print("\n[STEP 8/8] Generating Comparative Evaluation Reports & Visualizations...", flush=True)
    
    table_data = []
    for model_name, res in results.items():
        m = res["metrics"]
        table_data.append([
            model_name,
            f"{m['Accuracy'] * 100:.2f}%",
            f"{m['Precision']:.3f}",
            f"{m['Recall']:.3f}",
            f"{m['F1_Score']:.3f}",
            f"{m['PR_AUC']:.3f}",
            f"{m['ROC_AUC']:.3f}",
            f"{m['False_Positives']}",
            f"{m['False_Negatives']}"
        ])
        
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "PR-AUC", "ROC-AUC", "FP (Legit Blocked)", "FN (Fraud Missed)"]
    print("\n" + tabulate(table_data, headers=headers, tablefmt="simple"), flush=True)
    
    # Save visual reports
    plot_model_comparison(results, output_dir=reports_dir)
    plot_evaluation_curves(results, y_test, output_dir=reports_dir)
    plot_confusion_matrices(results, output_dir=reports_dir)
    
    # Feature importance of tuned model
    feature_names = list(X_train.columns)
    plot_feature_importances(tuned_xgb, feature_names, model_name="Tuned XGBoost", output_dir=reports_dir)
    
    # Select Best Model based on PR-AUC & Recall
    best_model_name = max(results.keys(), key=lambda k: results[k]["metrics"]["PR_AUC"])
    best_model_obj = results[best_model_name]["model"]
    best_metrics = results[best_model_name]["metrics"]
    
    print(f"\n[BEST MODEL SELECTED] '{best_model_name}'", flush=True)
    print(f" -> Reason: Highest PR-AUC ({best_metrics['PR_AUC']:.3f}) and optimal Recall ({best_metrics['Recall']:.3f})", flush=True)
    
    # Persist Best Model and Scaler
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(best_model_obj, best_model_path)
    print(f" -> Best model saved to: {best_model_path}", flush=True)
    
    # Save summary JSON for Streamlit UI & external consumption
    summary = {
        "best_model_name": best_model_name,
        "features": feature_names,
        "test_metrics": {name: res["metrics"] for name, res in results.items()},
        "dataset_stats": diag
    }
    summary_path = os.path.join(models_dir, "metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f" -> Metrics summary saved to: {summary_path}", flush=True)
    print("\n" + "=" * 70, flush=True)
    print(" [SUCCESS] TRAINING & ARTIFACT GENERATION COMPLETED SUCCESSFULLY!", flush=True)
    print("=" * 70, flush=True)
    
    return results, best_model_name

if __name__ == "__main__":
    run_training_pipeline()
