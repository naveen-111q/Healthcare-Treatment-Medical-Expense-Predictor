"""
Healthcare Treatment & Medical Expense Predictor
================================================
Interactive Streamlit Application
A modern, clinical-grade dashboard for individual expense forecasting,
lifestyle risk factor simulation, and machine learning model insights.
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from predict import load_model, predict_single, predict_batch

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MedExpense AI | Healthcare Treatment & Expense Forecaster",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern medical-tech visual aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.3);
    }
    
    .main-header h1 {
        font-weight: 800;
        font-size: 2.3rem;
        margin: 0;
        letter-spacing: -0.5px;
        color: #ffffff;
    }
    
    .main-header p {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #0f172a;
    }
    
    .badge-low {
        display: inline-block;
        padding: 0.25rem 0.85rem;
        border-radius: 9999px;
        background-color: #dcfce7;
        color: #15803d;
        font-weight: 700;
        font-size: 0.85rem;
    }
    
    .badge-moderate {
        display: inline-block;
        padding: 0.25rem 0.85rem;
        border-radius: 9999px;
        background-color: #e0f2fe;
        color: #0369a1;
        font-weight: 700;
        font-size: 0.85rem;
    }
    
    .badge-elevated {
        display: inline-block;
        padding: 0.25rem 0.85rem;
        border-radius: 9999px;
        background-color: #fef3c7;
        color: #b45309;
        font-weight: 700;
        font-size: 0.85rem;
    }
    
    .badge-high {
        display: inline-block;
        padding: 0.25rem 0.85rem;
        border-radius: 9999px;
        background-color: #fee2e2;
        color: #b91c1c;
        font-weight: 700;
        font-size: 0.85rem;
    }
    
    .savings-card {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 1px solid #a7f3d0;
        border-radius: 14px;
        padding: 1.4rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Resources
# -----------------------------------------------------------------------------
@st.cache_resource
def get_model():
    return load_model("artifacts/model_pipeline.joblib")

@st.cache_data
def get_metrics():
    if os.path.exists("artifacts/metrics.json"):
        with open("artifacts/metrics.json", "r") as f:
            return json.load(f)
    return {}

@st.cache_data
def get_feature_importances():
    if os.path.exists("artifacts/feature_importance.json"):
        with open("artifacts/feature_importance.json", "r") as f:
            return json.load(f)
    return {}

@st.cache_data
def get_sample_dataset():
    if os.path.exists("data/insurance.csv"):
        return pd.read_csv("data/insurance.csv")
    return pd.DataFrame()


try:
    model = get_model()
    metrics = get_metrics()
    feat_importances = get_feature_importances()
    raw_df = get_sample_dataset()
except Exception as e:
    st.error(f"Error loading system assets: {e}")
    st.stop()


# -----------------------------------------------------------------------------
# Header Banner
# -----------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <div style="display: flex; align-items: center; gap: 12px;">
        <span style="font-size: 2.2rem;">🏥</span>
        <div>
            <h1>Healthcare Treatment & Medical Expense Predictor</h1>
            <p>Production ML Pipeline with RandomForestRegressor & ColumnTransformer | Benchmark R² 0.88</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Sidebar Navigation & Patient Controls
# -----------------------------------------------------------------------------
st.sidebar.markdown("### 👤 Patient Clinical Profile")

age = st.sidebar.slider("Age (years)", min_value=18, max_value=85, value=35, step=1)
sex = st.sidebar.selectbox("Biological Sex", options=["Male", "Female"], index=0).lower()

# BMI Input mode selection
bmi_mode = st.sidebar.radio("BMI Calculation Mode", ["Direct BMI Entry", "Calculate from Height & Weight"], horizontal=True)

if bmi_mode == "Calculate from Height & Weight":
    col_h, col_w = st.sidebar.columns(2)
    with col_h:
        height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=230.0, value=175.0, step=0.5)
    with col_w:
        weight_kg = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=75.0, step=0.5)
    bmi = round(weight_kg / ((height_cm / 100.0) ** 2), 2)
    st.sidebar.info(f"Calculated BMI: **{bmi:.1f} kg/m²**")
else:
    bmi = st.sidebar.slider("Body Mass Index (BMI)", min_value=15.0, max_value=50.0, value=28.5, step=0.1)

# BMI category indicator
if bmi < 18.5:
    bmi_category = "Underweight (< 18.5)"
elif bmi < 25.0:
    bmi_category = "Normal (18.5 - 24.9)"
elif bmi < 30.0:
    bmi_category = "Overweight (25.0 - 29.9)"
else:
    bmi_category = "Obese (≥ 30.0)"
st.sidebar.caption(f"BMI Category: **{bmi_category}**")

children = st.sidebar.slider("Dependent Children", min_value=0, max_value=5, value=1, step=1)
smoker = st.sidebar.selectbox("Smoking Status", options=["No", "Yes"], index=0).lower()
region = st.sidebar.selectbox("U.S. Geographic Region", options=["Northeast", "Northwest", "Southeast", "Southwest"], index=0).lower()

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Pipeline Specifications")
st.sidebar.markdown(f"""
- **Model:** `RandomForestRegressor`
- **Preprocessing:** `ColumnTransformer`
- **Validation:** 5-Fold Cross Validation
- **Test Set $R^2$:** `{metrics.get('test_metrics', {}).get('r2_score', 0.8784):.3f}`
- **Test MAE:** `${metrics.get('test_metrics', {}).get('mae', 2491.01):,.2f}`
""")


# -----------------------------------------------------------------------------
# Main Content Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🩺 Patient Expense Forecaster",
    "💡 What-If Risk Simulator",
    "📊 Model Insights & Feature Importance",
    "📁 Batch Patient Analysis"
])


# -----------------------------------------------------------------------------
# TAB 1: Patient Expense Forecaster
# -----------------------------------------------------------------------------
with tab1:
    res = predict_single(
        age=age,
        sex=sex,
        bmi=bmi,
        children=children,
        smoker=smoker,
        region=region,
        model=model
    )
    
    annual_cost = res["predicted_annual_charges"]
    monthly_cost = res["predicted_monthly_charges"]
    risk_tier = res["risk_tier"]
    
    # Badge class lookup
    badge_class = {
        "Low": "badge-low",
        "Moderate": "badge-moderate",
        "Elevated": "badge-elevated",
        "High": "badge-high"
    }.get(risk_tier, "badge-low")
    
    # 3 Primary Metric KPI Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Predicted Annual Medical Expense</div>
            <div class="metric-value">${annual_cost:,.2f}</div>
            <div style="font-size: 0.8rem; color: #64748b; margin-top: 6px;">Total estimated annual insurer billing</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Monthly Amortized Cost</div>
            <div class="metric-value">${monthly_cost:,.2f}</div>
            <div style="font-size: 0.8rem; color: #64748b; margin-top: 6px;">Estimated monthly insurance liability</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Actuarial Risk Tier</div>
            <div style="margin: 0.5rem 0;"><span class="{badge_class}">{risk_tier.upper()} RISK</span></div>
            <div style="font-size: 0.8rem; color: #64748b;">Based on clinical demographic thresholds</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Breakdown & Comparison
    col_left, col_right = st.columns([1.2, 1])
    
    with col_left:
        st.subheader("🏥 Estimated Treatment Cost Distribution")
        
        # Standard actuarial distribution: Inpatient (~52%), Outpatient/Physician (~31%), Prescription Drugs (~17%)
        breakdown_data = pd.DataFrame({
            "Treatment Category": ["Inpatient Hospital Care", "Outpatient & Physician Visits", "Prescription Pharmaceuticals"],
            "Amount": [annual_cost * 0.52, annual_cost * 0.31, annual_cost * 0.17]
        })
        
        fig_pie = px.pie(
            breakdown_data,
            names="Treatment Category",
            values="Amount",
            hole=0.45,
            color_discrete_sequence=["#0284c7", "#0d9488", "#f59e0b"]
        )
        fig_pie.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate="<b>%{label}</b><br>Estimated Expense: $%{value:,.2f}<extra></extra>"
        )
        fig_pie.update_layout(
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            height=300
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_right:
        st.subheader("📈 Benchmark vs Population Averages")
        
        avg_cost = 13270.42 # Dataset population mean
        pct_diff = ((annual_cost - avg_cost) / avg_cost) * 100
        
        comp_df = pd.DataFrame({
            "Profile": ["This Patient", "Population Benchmark Average"],
            "Cost": [annual_cost, avg_cost],
            "Type": ["Patient", "Benchmark"]
        })
        
        fig_bar = px.bar(
            comp_df,
            x="Profile",
            y="Cost",
            color="Type",
            text="Cost",
            color_discrete_map={"Patient": "#2563eb", "Benchmark": "#94a3b8"}
        )
        fig_bar.update_traces(
            texttemplate='$%{text:,.0f}',
            textposition='outside'
        )
        fig_bar.update_layout(
            showlegend=False,
            yaxis_title="Annual Charges ($)",
            xaxis_title="",
            margin=dict(t=20, b=10, l=10, r=10),
            height=300
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        if pct_diff > 0:
            st.warning(f"⚠️ This patient's forecast is **{pct_diff:+.1f}% higher** than the nationwide benchmark average (${avg_cost:,.2f}).")
        else:
            st.success(f"✅ This patient's forecast is **{abs(pct_diff):.1f}% lower** than the nationwide benchmark average (${avg_cost:,.2f}).")


# -----------------------------------------------------------------------------
# TAB 2: What-If Risk Simulator
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("💡 Preventive Lifestyle Risk Factor Simulator")
    st.markdown("Quantify the direct financial return on investment (ROI) for preventive health interventions.")
    
    col_sim1, col_sim2 = st.columns(2)
    
    with col_sim1:
        st.markdown("#### 🚭 Smoking Cessation Impact")
        if smoker == "yes":
            savings_smoking = res["potential_annual_savings_quit_smoking"]
            ten_yr_savings = savings_smoking * 10
            
            st.markdown(f"""
            <div class="savings-card">
                <h3 style="color: #065f46; margin: 0;">🎉 Annual Financial Opportunity</h3>
                <div style="font-size: 2.2rem; font-weight: 800; color: #047857; margin: 0.5rem 0;">
                    -${savings_smoking:,.2f} / year
                </div>
                <p style="color: #065f46; margin: 0; font-size: 0.95rem;">
                    By quitting smoking, this patient is forecasted to reduce their medical treatment expenses by 
                    <strong>${savings_smoking:,.2f}</strong> annually, saving over <strong>${ten_yr_savings:,.2f}</strong> over the next decade!
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("✅ This patient is already a **non-smoker**. They avoid the ~68.7% cost surcharge penalty associated with nicotine use.")
            
    with col_sim2:
        st.markdown("#### ⚖️ Weight Management & BMI Reduction")
        target_bmi = st.slider("Target BMI Goal", min_value=18.5, max_value=float(max(bmi, 25.0)), value=float(min(bmi, 24.5)), step=0.5)
        
        target_df = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "bmi": target_bmi,
            "children": children,
            "smoker": smoker,
            "region": region
        }])
        target_cost = float(model.predict(target_df)[0])
        bmi_savings = max(0.0, annual_cost - target_cost)
        
        st.metric(
            label=f"Projected Expense at BMI {target_bmi:.1f}",
            value=f"${target_cost:,.2f}",
            delta=f"-${bmi_savings:,.2f} / yr" if bmi_savings > 0 else "Baseline"
        )
        
        if bmi_savings > 0:
            st.caption(f"Reducing BMI from {bmi:.1f} to {target_bmi:.1f} lowers expected medical liability by **${bmi_savings:,.2f} per year**.")
        else:
            st.caption("Patient's BMI is already in or below the target threshold.")

    st.markdown("---")
    st.subheader("📅 10-Year Longitudinal Cost Projection (Aging Effect)")
    
    # Project expenses across age + 10 years
    future_ages = list(range(age, min(age + 11, 85)))
    future_records = [
        {"age": a, "sex": sex, "bmi": bmi, "children": children, "smoker": smoker, "region": region}
        for a in future_ages
    ]
    future_costs = model.predict(pd.DataFrame(future_records))
    
    # Projection if non-smoker
    future_records_nonsmoker = [
        {"age": a, "sex": sex, "bmi": bmi, "children": children, "smoker": "no", "region": region}
        for a in future_ages
    ]
    future_costs_nonsmoker = model.predict(pd.DataFrame(future_records_nonsmoker))
    
    proj_df = pd.DataFrame({
        "Age": future_ages,
        "Current Trajectory": future_costs,
        "Non-Smoker Trajectory": future_costs_nonsmoker
    })
    
    fig_proj = go.Figure()
    fig_proj.add_trace(go.Scatter(
        x=proj_df["Age"],
        y=proj_df["Current Trajectory"],
        mode="lines+markers",
        name="Current Trajectory",
        line=dict(color="#ef4444" if smoker == "yes" else "#2563eb", width=3)
    ))
    if smoker == "yes":
        fig_proj.add_trace(go.Scatter(
            x=proj_df["Age"],
            y=proj_df["Non-Smoker Trajectory"],
            mode="lines+markers",
            name="If Smoke-Free",
            line=dict(color="#10b981", width=3, dash="dash")
        ))
    fig_proj.update_layout(
        xaxis_title="Patient Age (Years)",
        yaxis_title="Estimated Annual Charges ($)",
        hovermode="x unified",
        margin=dict(t=20, b=20, l=20, r=20),
        height=380
    )
    st.plotly_chart(fig_proj, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 3: Model Insights & Feature Importance
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("📊 Scikit-Learn Model Architecture & Feature Importance")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Test Set R² Score", f"{metrics.get('test_metrics', {}).get('r2_score', 0.8784):.4f}")
    with col_m2:
        st.metric("Test Root Mean Sq. Error", f"${metrics.get('test_metrics', {}).get('rmse', 4345.76):,.2f}")
    with col_m3:
        st.metric("Test Mean Absolute Error", f"${metrics.get('test_metrics', {}).get('mae', 2491.01):,.2f}")
    with col_m4:
        st.metric("MAE vs Linear Baseline", f"+{metrics.get('comparison', {}).get('mae_reduction_percentage', 40.4):.1f}%")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Feature Importance Plot
    col_imp, col_scatter = st.columns([1, 1.2])
    
    with col_imp:
        st.markdown("#### 🌳 Random Forest Gini Feature Importances")
        imp_df = pd.DataFrame(list(feat_importances.items()), columns=["Feature", "Importance"])
        imp_df["Importance_Pct"] = imp_df["Importance"] * 100
        
        # User-friendly names
        name_map = {
            "smoker_yes": "Smoking Status (Yes)",
            "bmi": "Body Mass Index (BMI)",
            "age": "Patient Age",
            "children": "Number of Dependents",
            "region_southeast": "Region: Southeast",
            "region_southwest": "Region: Southwest",
            "region_northwest": "Region: Northwest",
            "sex_male": "Sex: Male"
        }
        imp_df["Clean_Name"] = imp_df["Feature"].map(lambda x: name_map.get(x, x))
        imp_df = imp_df.sort_values(by="Importance", ascending=True)
        
        fig_imp = px.bar(
            imp_df,
            x="Importance_Pct",
            y="Clean_Name",
            orientation="h",
            text="Importance_Pct",
            color="Importance_Pct",
            color_continuous_scale="Viridis"
        )
        fig_imp.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside'
        )
        fig_imp.update_layout(
            xaxis_title="Importance Contribution (%)",
            yaxis_title="",
            coloraxis_showscale=False,
            margin=dict(t=10, b=10, l=10, r=30),
            height=380
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        
    with col_scatter:
        st.markdown("#### 🔍 The Smoker × BMI Non-Linear Multiplier Effect")
        if not raw_df.empty:
            fig_scatter = px.scatter(
                raw_df,
                x="bmi",
                y="charges",
                color="smoker",
                color_discrete_map={"yes": "#ef4444", "no": "#0284c7"},
                category_orders={"smoker": ["no", "yes"]},
                labels={"bmi": "Body Mass Index (BMI)", "charges": "Medical Charges ($)", "smoker": "Smoker"},
                hover_data=["age", "children", "region"],
                opacity=0.75
            )
            fig_scatter.add_vline(x=30.0, line_dash="dash", line_color="#f59e0b", annotation_text="Obesity Threshold (30)")
            fig_scatter.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                height=380,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption("Notice how when **BMI exceeds 30 in smokers**, medical expenses jump non-linearly from ~$15k to **over $40k**, demonstrating why tree ensembles outperform simple linear models.")


# -----------------------------------------------------------------------------
# TAB 4: Batch Patient Analysis
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("📁 Cohort & Actuarial Batch File Ingestion")
    st.markdown("Upload a cohort CSV of patient records to perform automated risk stratification and batch expense forecasting.")
    
    uploaded_file = st.file_uploader("Upload CSV File (Required columns: age, sex, bmi, children, smoker, region)", type=["csv"])
    
    cohort_df = None
    if uploaded_file is not None:
        try:
            cohort_df = pd.read_csv(uploaded_file)
            st.success(f"Uploaded cohort with {len(cohort_df)} records.")
        except Exception as e:
            st.error(f"Error parsing CSV: {e}")
    else:
        st.info("💡 No file uploaded yet. Click below to load a sample benchmark cohort of 10 patients.")
        if st.button("Load 10 Sample Patients from Benchmark"):
            if not raw_df.empty:
                cohort_df = raw_df.sample(10, random_state=42).copy()
                
    if cohort_df is not None:
        # Run batch predictions
        preds = model.predict(cohort_df[["age", "sex", "bmi", "children", "smoker", "region"]])
        cohort_df["Predicted_Annual_Charges"] = np.round(preds, 2)
        cohort_df["Predicted_Monthly_Charges"] = np.round(preds / 12.0, 2)
        
        # Risk categorization
        cohort_df["Risk_Category"] = pd.cut(
            cohort_df["Predicted_Annual_Charges"],
            bins=[0, 5000, 12000, 25000, np.inf],
            labels=["Low", "Moderate", "Elevated", "High"]
        )
        
        # Cohort Summary Statistics
        tot_liability = cohort_df["Predicted_Annual_Charges"].sum()
        avg_patient = cohort_df["Predicted_Annual_Charges"].mean()
        high_risk_count = (cohort_df["Risk_Category"] == "High").sum()
        
        b_c1, b_c2, b_c3 = st.columns(3)
        b_c1.metric("Total Cohort Liability", f"${tot_liability:,.2f}")
        b_c2.metric("Mean Patient Expense", f"${avg_patient:,.2f}")
        b_c3.metric("High Risk Patients", f"{high_risk_count} / {len(cohort_df)}")
        
        st.dataframe(cohort_df, use_container_width=True)
        
        # Download CSV export
        csv_data = cohort_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Scored Cohort CSV Report",
            data=csv_data,
            file_name="scored_patient_cohort.csv",
            mime="text/csv"
        )
