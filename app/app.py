"""
Production-Grade Streamlit Web Application for Credit Card Fraud Detection System.
Provides Real-Time Scoring, Batch CSV Screening, Model Benchmarks, EDA Insights, and Business Cost Simulation.
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Configure page settings
st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    
    /* Card design */
    .metric-card {
        background: linear-gradient(135deg, #1e2530 0%, #151a21 100%);
        border: 1px solid #2a3441;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        margin-bottom: 15px;
    }
    
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #8b9bb4;
        margin-bottom: 5px;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    /* Status Badges */
    .badge-safe {
        background-color: rgba(46, 204, 113, 0.15);
        color: #2ecc71;
        border: 1px solid #2ecc71;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .badge-warning {
        background-color: rgba(243, 156, 18, 0.15);
        color: #f39c12;
        border: 1px solid #f39c12;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .badge-danger {
        background-color: rgba(231, 76, 60, 0.15);
        color: #e74c3c;
        border: 1px solid #e74c3c;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }

    /* Streamlit overrides */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load metadata and model safely
@st.cache_resource
def load_model_resources():
    import joblib
    model_path = "models/best_model.joblib"
    scaler_path = "models/scaler.joblib"
    summary_path = "models/metrics_summary.json"
    
    model = joblib.load(model_path) if os.path.exists(model_path) else None
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
            
    return model, scaler, summary

model, scaler, summary = load_model_resources()

# Sidebar Navigation & System Status
st.sidebar.image("https://img.icons8.com/fluency/96/bank-card-back-side.png", width=70)
st.sidebar.title("Fraud Guard AI")
st.sidebar.caption("Enterprise ML Fraud Prevention Engine")
st.sidebar.divider()

if model is not None:
    best_name = summary.get("best_model_name", "Tuned XGBoost")
    st.sidebar.success(f"● Active Model: **{best_name}**")
else:
    st.sidebar.warning("● Model not yet loaded. Training in progress.")

st.sidebar.markdown("### ⚙️ Detection Settings")
global_threshold = st.sidebar.slider(
    "Decision Threshold (Sensitivity)",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05,
    help="Lower threshold catches more fraud (higher Recall) but increases False Positives (legit cards blocked)."
)

st.sidebar.divider()
st.sidebar.info(
    "💡 **Education & Security Disclaimer**:\n"
    "This system processes anonymized PCA transaction vectors for educational and benchmarking demonstrations. "
    "Predictions are probabilistic."
)

# Header Section
st.title("💳 Credit Card Fraud Detection System")
st.markdown(
    "An end-to-end Machine Learning pipeline engineered to detect rare, fraudulent financial transactions "
    "with high precision, recall, and cost-effective risk thresholding."
)

# Main Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Real-Time Transaction Predictor",
    "📁 Batch Transaction Screener",
    "📊 Model Performance & Benchmarks",
    "🔍 Exploratory Data Insights (EDA)",
    "💼 Business Cost-Benefit Simulator"
])

# ==========================================
# TAB 1: REAL-TIME TRANSACTION PREDICTOR
# ==========================================
with tab1:
    st.subheader("Real-Time Transaction Risk Assessment")
    st.markdown("Enter transaction parameters or load pre-configured test scenarios to evaluate fraud probability.")
    
    # Preset quick-load buttons
    col_p1, col_p2, col_p3 = st.columns(3)
    
    # Default values dictionary
    default_vals = {f"V{i}": 0.0 for i in range(1, 29)}
    default_vals["Amount"] = 85.00
    default_vals["Time"] = 3600.0
    
    if "form_data" not in st.session_state:
        st.session_state.form_data = default_vals.copy()
        
    with col_p1:
        if st.button("🟢 Load Normal Purchase Scenario", use_container_width=True):
            st.session_state.form_data = {f"V{i}": float(np.random.normal(0, 0.5)) for i in range(1, 29)}
            st.session_state.form_data["Amount"] = 42.50
            st.session_state.form_data["Time"] = 12450.0
            st.rerun()
            
    with col_p2:
        if st.button("🔴 Load Known Fraud Scenario", use_container_width=True):
            st.session_state.form_data = {f"V{i}": float(np.random.normal(0, 0.5)) for i in range(1, 29)}
            # Severe anomalies in key discriminant features
            st.session_state.form_data["V14"] = -6.85
            st.session_state.form_data["V12"] = -5.40
            st.session_state.form_data["V10"] = -4.95
            st.session_state.form_data["V17"] = -5.10
            st.session_state.form_data["V4"] = 4.20
            st.session_state.form_data["V11"] = 3.80
            st.session_state.form_data["V2"] = 3.10
            st.session_state.form_data["Amount"] = 490.00
            st.session_state.form_data["Time"] = 86200.0
            st.rerun()
            
    with col_p3:
        if st.button("🔄 Reset to Neutral", use_container_width=True):
            st.session_state.form_data = default_vals.copy()
            st.rerun()
            
    st.markdown("---")
    
    # Input Form
    with st.form("single_predict_form"):
        st.markdown("##### 1. Primary Transaction Metadata")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            input_amount = st.number_input(
                "Transaction Amount ($)",
                min_value=0.01,
                max_value=100000.0,
                value=float(st.session_state.form_data.get("Amount", 85.0)),
                step=1.0
            )
        with col_m2:
            input_time = st.number_input(
                "Elapsed Time (Seconds)",
                min_value=0.0,
                max_value=172800.0,
                value=float(st.session_state.form_data.get("Time", 3600.0)),
                step=100.0,
                help="Seconds elapsed since the initial reference timestamp (0 to 172,800s = 48 hours)."
            )
            
        st.markdown("##### 2. Key Fraud Discriminant Features (Top Correlated PCA Vectors)")
        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        with col_k1:
            v14 = st.number_input("V14 (High Negative Impact)", value=float(st.session_state.form_data.get("V14", 0.0)), step=0.1)
            v12 = st.number_input("V12 (High Negative Impact)", value=float(st.session_state.form_data.get("V12", 0.0)), step=0.1)
        with col_k2:
            v10 = st.number_input("V10 (High Negative Impact)", value=float(st.session_state.form_data.get("V10", 0.0)), step=0.1)
            v17 = st.number_input("V17 (High Negative Impact)", value=float(st.session_state.form_data.get("V17", 0.0)), step=0.1)
        with col_k3:
            v4 = st.number_input("V4 (High Positive Impact)", value=float(st.session_state.form_data.get("V4", 0.0)), step=0.1)
            v11 = st.number_input("V11 (High Positive Impact)", value=float(st.session_state.form_data.get("V11", 0.0)), step=0.1)
        with col_k4:
            v2 = st.number_input("V2 (Positive Correlation)", value=float(st.session_state.form_data.get("V2", 0.0)), step=0.1)
            v7 = st.number_input("V7 (Latent Projection)", value=float(st.session_state.form_data.get("V7", 0.0)), step=0.1)
            
        with st.expander("➕ Expand Remaining Anonymized PCA Features (V1, V3, V5, V6, V8, V9, V13, V15–V28)"):
            pca_cols = st.columns(4)
            remaining_v = [f"V{i}" for i in range(1, 29) if f"V{i}" not in ['V14', 'V12', 'V10', 'V17', 'V4', 'V11', 'V2', 'V7']]
            extra_vals = {}
            for idx, v_name in enumerate(remaining_v):
                col_target = pca_cols[idx % 4]
                with col_target:
                    extra_vals[v_name] = st.number_input(
                        v_name,
                        value=float(st.session_state.form_data.get(v_name, 0.0)),
                        step=0.1,
                        key=f"input_{v_name}"
                    )
                    
        submit_btn = st.form_submit_button("⚡ Analyze Transaction Risk", use_container_width=True, type="primary")
        
    if submit_btn:
        if model is None or scaler is None:
            st.error("⚠️ Model or Scaler artifacts not found. Please run `python src/train.py` first.")
        else:
            # Assemble feature vector
            feature_dict = {"Time": input_time}
            for i in range(1, 29):
                v_name = f"V{i}"
                if v_name == "V14": feature_dict[v_name] = v14
                elif v_name == "V12": feature_dict[v_name] = v12
                elif v_name == "V10": feature_dict[v_name] = v10
                elif v_name == "V17": feature_dict[v_name] = v17
                elif v_name == "V4": feature_dict[v_name] = v4
                elif v_name == "V11": feature_dict[v_name] = v11
                elif v_name == "V2": feature_dict[v_name] = v2
                elif v_name == "V7": feature_dict[v_name] = v7
                else: feature_dict[v_name] = extra_vals.get(v_name, 0.0)
            feature_dict["Amount"] = input_amount
            
            # Predict
            df_in = pd.DataFrame([feature_dict])
            df_scaled = df_in.copy()
            df_scaled[['Time', 'Amount']] = scaler.transform(df_scaled[['Time', 'Amount']])
            
            prob = float(model.predict_proba(df_scaled)[0, 1])
            is_fraud = prob >= global_threshold
            
            st.markdown("### 📋 Risk Assessment Results")
            res_col1, res_col2 = st.columns([1.2, 1])
            
            with res_col1:
                if is_fraud:
                    st.markdown(f"""
                    <div style="background-color: rgba(231, 76, 60, 0.15); border: 2px solid #e74c3c; border-radius: 12px; padding: 20px;">
                        <h3 style="color: #e74c3c; margin: 0;">🚨 HIGH RISK: TRANSACTION FLAGGED AS FRAUD</h3>
                        <p style="font-size: 1.1rem; margin-top: 10px;">
                            The machine learning model estimated a <b>{prob*100:.2f}%</b> probability of fraudulent behavior, 
                            exceeding the active threshold of <b>{global_threshold*100:.1f}%</b>.
                        </p>
                        <hr style="border-color: #e74c3c; opacity: 0.3;"/>
                        <b>Recommended Action:</b> Temporarily hold funds, trigger Step-Up Multi-Factor Authentication (MFA), or request cardholder confirmation.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color: rgba(46, 204, 113, 0.15); border: 2px solid #2ecc71; border-radius: 12px; padding: 20px;">
                        <h3 style="color: #2ecc71; margin: 0;">✅ TRANSACTION APPROVED: LEGITIMATE</h3>
                        <p style="font-size: 1.1rem; margin-top: 10px;">
                            Fraud Probability is <b>{prob*100:.2f}%</b> (Well within safe operating boundaries below threshold of <b>{global_threshold*100:.1f}%</b>).
                        </p>
                        <hr style="border-color: #2ecc71; opacity: 0.3;"/>
                        <b>Recommended Action:</b> Allow transaction to proceed smoothly without friction.
                    </div>
                    """, unsafe_allow_html=True)
                    
            with res_col2:
                # Plotly Gauge Chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={'suffix': "%", 'valueformat': ".1f"},
                    title={'text': "Fraud Probability Meter", 'font': {'size': 18}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': "#e74c3c" if is_fraud else "#2ecc71"},
                        'steps': [
                            {'range': [0, 30], 'color': "rgba(46, 204, 113, 0.2)"},
                            {'range': [30, 70], 'color': "rgba(243, 156, 18, 0.2)"},
                            {'range': [70, 100], 'color': "rgba(231, 76, 60, 0.2)"}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': global_threshold * 100
                        }
                    }
                ))
                fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_gauge, use_container_width=True)

# ==========================================
# TAB 2: BATCH TRANSACTION SCREENER
# ==========================================
with tab2:
    st.subheader("Batch Transaction Screening & Auditing")
    st.markdown("Upload a batch CSV file or test with a randomized transaction stream.")
    
    col_b1, col_b2 = st.columns([2, 1])
    
    with col_b1:
        uploaded_file = st.file_uploader("Upload CSV Batch for Screening", type=["csv"])
        
    with col_b2:
        st.markdown("<br>", unsafe_allow_html=True)
        demo_batch_btn = st.button("📂 Load 50 Sample Synthetic Transactions", use_container_width=True)
        
    batch_df = None
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
    elif demo_batch_btn:
        if os.path.exists("data/creditcard.csv"):
            full_df = pd.read_csv("data/creditcard.csv")
            # Grab a mixed sample of legit and fraud for demonstration
            sample_legit = full_df[full_df['Class'] == 0].sample(45, random_state=42)
            sample_fraud = full_df[full_df['Class'] == 1].sample(min(5, (full_df['Class'] == 1).sum()), random_state=42)
            batch_df = pd.concat([sample_legit, sample_fraud]).sample(frac=1.0, random_state=42).reset_index(drop=True)
        else:
            st.warning("Please generate dataset first.")
            
    if batch_df is not None:
        st.markdown(f"**Loaded Batch:** {len(batch_df)} transactions")
        
        if model is not None and scaler is not None:
            batch_features = batch_df[[col for col in batch_df.columns if col != 'Class']].copy()
            batch_scaled = batch_features.copy()
            batch_scaled[['Time', 'Amount']] = scaler.transform(batch_scaled[['Time', 'Amount']])
            
            probs = model.predict_proba(batch_scaled)[:, 1]
            flags = (probs >= global_threshold).astype(int)
            
            batch_results = batch_df.copy()
            batch_results["Fraud_Probability"] = np.round(probs, 4)
            batch_results["Status"] = ["🚨 FLAGGED FRAUD" if f == 1 else "✅ APPROVED" for f in flags]
            
            # Summary Metrics Cards
            total_n = len(batch_results)
            fraud_n = int(flags.sum())
            legit_n = total_n - fraud_n
            fraud_vol = batch_results[flags == 1]["Amount"].sum()
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Scanned", f"{total_n}")
            m2.metric("Flagged Fraud", f"{fraud_n}", delta=f"{fraud_n/total_n*100:.1f}% flag rate", delta_color="inverse")
            m3.metric("Approved Legit", f"{legit_n}")
            m4.metric("At-Risk Volume Intercepted", f"${fraud_vol:,.2f}")
            
            st.markdown("#### Detailed Batch Audit Table")
            st.dataframe(
                batch_results[['Time', 'Amount', 'Status', 'Fraud_Probability', 'V14', 'V12', 'V10', 'V4']],
                use_container_width=True
            )
            
            # Download Flagged Transactions
            csv_export = batch_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Complete Audited Report (CSV)",
                data=csv_export,
                file_name="fraud_detection_audit_report.csv",
                mime="text/csv"
            )

# ==========================================
# TAB 3: MODEL PERFORMANCE & BENCHMARKS
# ==========================================
with tab3:
    st.subheader("Model Performance Comparison & Benchmarks")
    st.markdown(
        "Because credit card fraud is a severe class imbalance scenario (~0.17% positive class), "
        "traditional Accuracy (99.8%) is misleading. Models must be evaluated based on **PR-AUC, Fraud Recall, and Precision**."
    )
    
    if summary and "test_metrics" in summary:
        test_metrics = summary["test_metrics"]
        
        # Build comparison DataFrame
        comp_data = []
        for model_name, m in test_metrics.items():
            comp_data.append({
                "Model": model_name,
                "Accuracy": f"{m['Accuracy']*100:.2f}%",
                "Precision": round(m['Precision'], 3),
                "Recall (Detection Rate)": round(m['Recall'], 3),
                "F1 Score": round(m['F1_Score'], 3),
                "PR-AUC (Avg Prec)": round(m['PR_AUC'], 3),
                "ROC-AUC": round(m['ROC_AUC'], 3),
                "False Positives (Blocked Legit)": m['False_Positives'],
                "False Negatives (Missed Fraud)": m['False_Negatives']
            })
            
        df_comp = pd.DataFrame(comp_data)
        st.dataframe(df_comp, use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### 📈 Benchmark Visualizations")
        
        vcol1, vcol2 = st.columns(2)
        with vcol1:
            if os.path.exists("reports/figures/model_comparison.png"):
                st.image("reports/figures/model_comparison.png", caption="Multi-Model Metric Comparison", use_container_width=True)
            elif os.path.exists("reports/figures/evaluation_curves.png"):
                st.image("reports/figures/evaluation_curves.png", caption="ROC and Precision-Recall Curves", use_container_width=True)
                
        with vcol2:
            if os.path.exists("reports/figures/confusion_matrices.png"):
                st.image("reports/figures/confusion_matrices.png", caption="Side-by-Side Confusion Matrices", use_container_width=True)
            elif os.path.exists("reports/figures/feature_importance.png"):
                st.image("reports/figures/feature_importance.png", caption="Top Discriminative Features", use_container_width=True)
    else:
        st.info("Training results will appear once `python src/train.py` completes execution.")

# ==========================================
# TAB 4: EXPLORATORY DATA INSIGHTS
# ==========================================
with tab4:
    st.subheader("Exploratory Data Analysis (EDA) & Feature Insights")
    st.markdown("In-depth statistical breakdown of transaction amounts, time-of-day dynamics, and feature correlations.")
    
    if os.path.exists("reports/figures/class_distribution.png"):
        ecol1, ecol2 = st.columns(2)
        with ecol1:
            st.image("reports/figures/class_distribution.png", caption="Class Distribution (Severe Imbalance)", use_container_width=True)
        with ecol2:
            if os.path.exists("reports/figures/amount_and_time_distribution.png"):
                st.image("reports/figures/amount_and_time_distribution.png", caption="Amount & Time Distributions", use_container_width=True)
                
    if os.path.exists("reports/figures/correlation_matrix.png"):
        st.markdown("---")
        st.image("reports/figures/correlation_matrix.png", caption="Top Correlated Features Heatmap with Fraud Target", use_container_width=True)

# ==========================================
# TAB 5: BUSINESS COST-BENEFIT SIMULATOR
# ==========================================
with tab5:
    st.subheader("Business Impact & Cost-Benefit Threshold Optimizer")
    st.markdown(
        "In production banking, setting the fraud threshold is a financial optimization problem balancing "
        "**Direct Fraud Loss (False Negatives)** against **Customer Friction & Investigation Costs (False Positives)**."
    )
    
    c_col1, c_col2 = st.columns(2)
    with c_col1:
        avg_fraud_loss = st.number_input("Average Loss per Missed Fraud Incident ($)", value=350.0, step=25.0)
        investigation_cost = st.number_input("Cost per False Positive Investigation / User Friction ($)", value=20.0, step=5.0)
        
    with c_col2:
        monthly_volume = st.number_input("Monthly Card Transaction Volume", value=100000, step=10000)
        est_fraud_rate = st.slider("Estimated Baseline Fraud Rate (%)", min_value=0.05, max_value=1.00, value=0.20, step=0.05) / 100.0

    # Simulate cost curve across thresholds [0.1 to 0.9]
    thresholds = np.linspace(0.05, 0.95, 19)
    sim_costs = []
    
    for t in thresholds:
        # Heuristic recall & precision response curve
        est_recall = 1.0 / (1.0 + np.exp(6 * (t - 0.45)))
        est_fpr = 0.05 * np.exp(-4 * t)
        
        n_fraud = monthly_volume * est_fraud_rate
        n_legit = monthly_volume * (1 - est_fraud_rate)
        
        fn = n_fraud * (1.0 - est_recall)
        fp = n_legit * est_fpr
        
        fraud_cost = fn * avg_fraud_loss
        friction_cost = fp * investigation_cost
        total_cost = fraud_cost + friction_cost
        
        sim_costs.append({
            "Threshold": round(t, 2),
            "Missed Fraud Cost ($)": fraud_cost,
            "False Positive Friction Cost ($)": friction_cost,
            "Total Operating Cost ($)": total_cost
        })
        
    df_sim = pd.DataFrame(sim_costs)
    opt_row = df_sim.loc[df_sim["Total Operating Cost ($)"].idxmin()]
    
    st.markdown(f"#### Optimal Economic Threshold: **{opt_row['Threshold']}** (Minimizes Total Business Loss to **${opt_row['Total Operating Cost ($)']:,.2f}**)")
    
    fig_sim = px.line(
        df_sim,
        x="Threshold",
        y=["Missed Fraud Cost ($)", "False Positive Friction Cost ($)", "Total Operating Cost ($)"],
        title="Cost Minimization vs Decision Threshold",
        labels={"value": "Cost ($)", "variable": "Expense Category"}
    )
    fig_sim.update_layout(height=400)
    st.plotly_chart(fig_sim, use_container_width=True)
