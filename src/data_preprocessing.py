"""
Data Preprocessing and Imbalance Handling Module for Credit Card Fraud Detection.

Handles:
- Data Ingestion & Data Validation
- Data Cleaning & Duplicate Removal
- Stratified Train-Test Split (Preventing Data Leakage)
- Feature Scaling (RobustScaler for skewed Amount/Time features)
- Class Imbalance Handling (SMOTE, Random Under-Sampling, Class Weights)
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def load_data(file_path):
    """
    Loads credit card dataset from CSV file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please run data/download_or_generate_data.py first.")
    df = pd.read_csv(file_path)
    return df

def inspect_data(df):
    """
    Computes summary diagnostics on raw dataset.
    """
    diagnostics = {
        "shape": df.shape,
        "missing_values": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "columns": list(df.columns),
        "legit_count": int((df['Class'] == 0).sum()),
        "fraud_count": int((df['Class'] == 1).sum()),
        "fraud_percentage": float((df['Class'] == 1).mean() * 100)
    }
    return diagnostics

def clean_data(df, drop_duplicates=True):
    """
    Performs data cleaning: removes duplicate records if requested and validates nulls.
    """
    df_cleaned = df.copy()
    if drop_duplicates and df_cleaned.duplicated().sum() > 0:
        df_cleaned = df_cleaned.drop_duplicates().reset_index(drop=True)
    
    # Fill or drop missing values if any exist
    if df_cleaned.isnull().sum().sum() > 0:
        df_cleaned = df_cleaned.dropna().reset_index(drop=True)
        
    return df_cleaned

def prepare_train_test_split(df, target_col='Class', test_size=0.2, random_state=42):
    """
    Performs Stratified Train-Test Split to ensure fraud ratio is preserved in both folds.
    Returns X_train, X_test, y_train, y_test.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test, scaler_type='robust', save_path=None):
    """
    Scales 'Time' and 'Amount' features using RobustScaler (or StandardScaler).
    CRITICAL: Scaler is fitted ONLY on training data to prevent data leakage,
    then applied to test data.
    """
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    scale_cols = ['Time', 'Amount']
    
    if scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
        
    scaler.fit(X_train[scale_cols])
    
    X_train_scaled[scale_cols] = scaler.transform(X_train[scale_cols])
    X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(scaler, save_path)
        print(f"[INFO] Scaler successfully saved to: {save_path}")
        
    return X_train_scaled, X_test_scaled, scaler

def balance_training_data(X_train, y_train, method='smote', random_state=42):
    """
    Handles class imbalance.
    CRITICAL: Applied ONLY to training fold. Test fold remains untouched.
    Methods:
    - 'smote': Synthetic Minority Over-sampling Technique
    - 'undersample': Random Under-Sampling
    - 'none': No resampling (rely on class weights in models)
    """
    if method == 'smote':
        # Default sampling_strategy=0.1 or 0.2 to balance enough without overwhelming with synthetic samples
        smote = SMOTE(sampling_strategy='auto', random_state=random_state)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        return X_res, y_res
    elif method == 'undersample':
        rus = RandomUnderSampler(sampling_strategy='auto', random_state=random_state)
        X_res, y_res = rus.fit_resample(X_train, y_train)
        return X_res, y_res
    elif method == 'none':
        return X_train, y_train
    else:
        raise ValueError(f"Unknown imbalance method: {method}. Choose from 'smote', 'undersample', 'none'.")
