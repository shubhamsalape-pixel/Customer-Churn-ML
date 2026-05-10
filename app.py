# app.py

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# LOAD ARTIFACTS
# =========================================================

@st.cache_resource
def load_artifacts():
    model = joblib.load("artifacts/model.pkl")
    metadata = joblib.load("artifacts/metadata.pkl")
    return model, metadata

model, metadata = load_artifacts()

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-size: 16px;
    font-weight: bold;
}

.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.title("📊 Customer Churn Analytics")
st.caption("Advanced Churn Analytics with Explainable AI")

# =========================================================
# SIDEBAR INPUTS
# =========================================================

st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

senior_citizen = st.sidebar.selectbox(
    "Senior Citizen",
    [0, 1]
)

partner = st.sidebar.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.sidebar.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.sidebar.slider(
    "Tenure (Months)",
    0,
    72,
    12
)

phone_service = st.sidebar.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

multiple_lines = st.sidebar.selectbox(
    "Multiple Lines",
    ["Yes", "No"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.sidebar.selectbox(
    "Online Security",
    ["Yes", "No"]
)

online_backup = st.sidebar.selectbox(
    "Online Backup",
    ["Yes", "No"]
)

device_protection = st.sidebar.selectbox(
    "Device Protection",
    ["Yes", "No"]
)

tech_support = st.sidebar.selectbox(
    "Tech Support",
    ["Yes", "No"]
)

streaming_tv = st.sidebar.selectbox(
    "Streaming TV",
    ["Yes", "No"]
)

streaming_movies = st.sidebar.selectbox(
    "Streaming Movies",
    ["Yes", "No"]
)

contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.sidebar.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges = st.sidebar.slider(
    "Monthly Charges",
    0.0,
    200.0,
    70.0
)

total_charges = st.sidebar.slider(
    "Total Charges",
    0.0,
    10000.0,
    1000.0
)

predict_button = st.sidebar.button("Predict Churn")

# =========================================================
# CREATE INPUT DATAFRAME
# =========================================================

input_df = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [senior_citizen],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})

# =========================================================
# DASHBOARD METRICS
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Model",
        value="Random Forest"
    )

with col2:
    st.metric(
        label="Deployment",
        value="Production Ready"
    )

with col3:
    st.metric(
        label="Explainability",
        value="Enabled"
    )

# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    with st.spinner("Predicting customer churn..."):

        prediction = model.predict(input_df)[0]

        probability = model.predict_proba(input_df)[0][1]

        churn_label = (
            "Likely to Churn"
            if prediction == 1
            else "Not Likely to Churn"
        )

        confidence = probability * 100

    st.success("Prediction Completed Successfully!")

    # =====================================================
    # RESULT CARDS
    # =====================================================

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error(f"⚠️ {churn_label}")
        else:
            st.success(f"✅ {churn_label}")

    with result_col2:

        st.subheader("Confidence Score")

        st.metric(
            "Churn Probability",
            f"{confidence:.2f}%"
        )

    # =====================================================
    # RISK LEVEL
    # =====================================================

    if probability >= 0.75:
        risk = "High Risk"
    elif probability >= 0.50:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    st.info(f"Risk Category: {risk}")

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    try:

        feature_importance = model.named_steps[
            "model"
        ].feature_importances_

        feature_names = model.named_steps[
            "preprocessor"
        ].get_feature_names_out()

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": feature_importance
        })

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False
        ).head(10)

        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 10 Feature Importances"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    except Exception as e:
        st.warning(
            "Feature importance could not be generated."
        )

# =========================================================
# EDA SECTION
# =========================================================

st.header("📈 Analytics Dashboard")

eda_data = pd.DataFrame({
    "Contract Type": [
        "Month-to-month",
        "One year",
        "Two year"
    ],
    "Churn Rate": [42, 11, 3]
})

fig2 = px.pie(
    eda_data,
    names="Contract Type",
    values="Churn Rate",
    title="Contract Type vs Churn"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# =========================================================
# SHOW INPUT DATA
# =========================================================

with st.expander("View Customer Input Data"):
    st.dataframe(input_df)
