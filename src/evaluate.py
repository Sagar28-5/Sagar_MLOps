"""
Comprehensive Model Evaluation and Visualization Module for Credit Card Fraud Detection.

Computes:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC (Area Under Receiver Operating Characteristic)
- PR-AUC (Average Precision / Area Under Precision-Recall Curve)
- Confusion Matrix (TN, FP, FN, TP)
- Business cost-benefit trade-offs (False Positives vs False Negatives)

Generates and saves publication-ready figures under reports/figures/.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    classification_report
)

# Configure plot aesthetic
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def evaluate_model_performance(y_true, y_pred, y_prob=None, threshold=0.5):
    """
    Computes comprehensive evaluation metrics for binary classification.
    """
    if y_prob is not None and threshold != 0.5:
        y_pred = (y_prob >= threshold).astype(int)
        
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    roc_auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.0
    pr_auc = average_precision_score(y_true, y_prob) if y_prob is not None else 0.0
    
    # Business trade-off metrics
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0 # False Positive Rate: Legit flagged as Fraud
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0 # False Negative Rate: Fraud missed

    metrics = {
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1_Score": float(f1),
        "ROC_AUC": float(roc_auc),
        "PR_AUC": float(pr_auc),
        "True_Negatives": int(tn),
        "False_Positives": int(fp),
        "False_Negatives": int(fn),
        "True_Positives": int(tp),
        "False_Positive_Rate": float(fpr),
        "False_Negative_Rate": float(fnr),
        "Threshold": float(threshold)
    }
    return metrics

def plot_eda_figures(df, output_dir="reports/figures"):
    """
    Generates and saves core Exploratory Data Analysis figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Class Distribution Plot
    plt.figure(figsize=(8, 5))
    class_counts = df['Class'].value_counts()
    colors = ['#2b5c8f', '#e63946']
    bars = plt.bar(['Legitimate (0)', 'Fraudulent (1)'], class_counts.values, color=colors, width=0.5, edgecolor='black', alpha=0.9)
    plt.title("Credit Card Transaction Class Distribution (Severe Imbalance)", fontsize=13, fontweight='bold', pad=12)
    plt.ylabel("Transaction Count", fontsize=11)
    
    for bar in bars:
        height = bar.get_height()
        pct = (height / len(df)) * 100
        plt.annotate(f'{height:,}\n({pct:.2f}%)',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 5), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
                     
    plt.yscale('log')
    plt.ylabel("Transaction Count (Log Scale)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=300)
    plt.close()
    
    # 2. Transaction Amount & Time Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Amount distribution
    sns.histplot(df[df['Class'] == 0]['Amount'], ax=axes[0], color='#2b5c8f', label='Legitimate', bins=40, kde=True, stat='density')
    sns.histplot(df[df['Class'] == 1]['Amount'], ax=axes[0], color='#e63946', label='Fraud', bins=40, kde=True, stat='density')
    axes[0].set_title("Transaction Amount Density (Fraud vs Legit)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Amount ($)", fontsize=10)
    axes[0].set_xlim(0, 1500)
    axes[0].legend()
    
    # Time distribution (hours)
    time_hours = df['Time'] / 3600
    sns.histplot(time_hours[df['Class'] == 0], ax=axes[1], color='#2b5c8f', label='Legitimate', bins=48, kde=True, stat='density')
    sns.histplot(time_hours[df['Class'] == 1], ax=axes[1], color='#e63946', label='Fraud', bins=48, kde=True, stat='density')
    axes[1].set_title("Transaction Time Distribution across 48 Hours", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Time (Hours)", fontsize=10)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "amount_and_time_distribution.png"), dpi=300)
    plt.close()
    
    # 3. Correlation Heatmap of top correlated features with Class
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    top_corr_features = corr['Class'].abs().sort_values(ascending=False).head(12).index
    sns.heatmap(df[top_corr_features].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar=True, square=True)
    plt.title("Top Feature Correlations with Fraud Class", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=300)
    plt.close()
    
    print(f"[SUCCESS] EDA figures saved under: {output_dir}")

def plot_model_comparison(results_dict, output_dir="reports/figures"):
    """
    Plots benchmark bar chart comparing models across Precision, Recall, F1, PR-AUC, and ROC-AUC.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_to_plot = ["Precision", "Recall", "F1_Score", "PR_AUC", "ROC_AUC"]
    model_names = list(results_dict.keys())
    
    data = []
    for model_name, res in results_dict.items():
        for m in metrics_to_plot:
            data.append({
                "Model": model_name,
                "Metric": m.replace("_", " "),
                "Score": res["metrics"][m]
            })
            
    df_plot = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 6))
    palette = sns.color_palette("Set2", len(metrics_to_plot))
    ax = sns.barplot(data=df_plot, x="Model", y="Score", hue="Metric", palette=palette)
    plt.title("Model Performance Comparison (Target: Fraud Class)", fontsize=14, fontweight='bold', pad=14)
    plt.ylim(0.0, 1.08)
    plt.ylabel("Score", fontsize=12)
    plt.xlabel("Machine Learning Model", fontsize=12)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Add values on top of bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0.05:
            ax.annotate(f"{height:.2f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=8, rotation=90,
                        xytext=(0, 3), textcoords='offset points')
                        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=300)
    plt.close()

def plot_evaluation_curves(results_dict, y_test, output_dir="reports/figures"):
    """
    Generates ROC curves and Precision-Recall curves for all evaluated models.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left: ROC Curves
    for model_name, res in results_dict.items():
        if "y_prob" in res and res["y_prob"] is not None:
            fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
            auc = res["metrics"]["ROC_AUC"]
            axes[0].plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {auc:.3f})")
            
    axes[0].plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Guess')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    axes[0].set_ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=11)
    axes[0].set_title('Receiver Operating Characteristic (ROC) Curves', fontsize=13, fontweight='bold')
    axes[0].legend(loc="lower right", fontsize=9)
    
    # Right: Precision-Recall Curves
    baseline_pr = y_test.mean()
    for model_name, res in results_dict.items():
        if "y_prob" in res and res["y_prob"] is not None:
            precision, recall, _ = precision_recall_curve(y_test, res["y_prob"])
            pr_auc = res["metrics"]["PR_AUC"]
            axes[1].plot(recall, precision, lw=2, label=f"{model_name} (PR-AUC = {pr_auc:.3f})")
            
    axes[1].axhline(y=baseline_pr, color='navy', linestyle='--', label=f'Baseline ({baseline_pr:.3f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall (Fraud Detection Rate)', fontsize=11)
    axes[1].set_ylabel('Precision (True Fraud Ratio in Flags)', fontsize=11)
    axes[1].set_title('Precision-Recall Curves (Critical for Rare Fraud)', fontsize=13, fontweight='bold')
    axes[1].legend(loc="lower left", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "evaluation_curves.png"), dpi=300)
    plt.close()

def plot_confusion_matrices(results_dict, output_dir="reports/figures"):
    """
    Plots side-by-side confusion matrices for all models.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_models = len(results_dict)
    
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (model_name, res) in enumerate(results_dict.items()):
        m = res["metrics"]
        cm = np.array([[m["True_Negatives"], m["False_Positives"]],
                       [m["False_Negatives"], m["True_Positives"]]])
        
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx], cbar=False,
                    xticklabels=['Pred Legit', 'Pred Fraud'],
                    yticklabels=['Actual Legit', 'Actual Fraud'])
        axes[idx].set_title(f"{model_name}\nRecall: {m['Recall']:.2f} | Prec: {m['Precision']:.2f}", fontsize=11, fontweight='bold')
        
    # Hide any unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrices.png"), dpi=300)
    plt.close()

def plot_feature_importances(model, feature_names, model_name="Random Forest", output_dir="reports/figures", top_n=15):
    """
    Plots feature importances for tree-based models.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return
        
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_imp, x="Importance", y="Feature", palette="viridis")
    plt.title(f"Top {top_n} Most Discriminative Features ({model_name})", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Relative Importance Score", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=300)
    plt.close()
