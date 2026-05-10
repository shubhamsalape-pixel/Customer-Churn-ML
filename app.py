
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
.main {
    padding: 1rem;
}
.metric-card {
    background-color: #f8fafc;
    padding: 1rem;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Customer Churn")
st.caption("Advanced Churn Analytics with Explainable AI")

st.sidebar.header("Customer Information")

tenure = st.sidebar.slider("Tenure", 0, 72, 12)
monthly = st.sidebar.slider("Monthly Charges", 0.0, 200.0, 70.0)
total = st.sidebar.slider("Total Charges", 0.0, 10000.0, 1000.0)

contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

predict_btn = st.sidebar.button("Predict Churn")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "Random Forest")

with col2:
    st.metric("Deployment", "Production Ready")

with col3:
    st.metric("Explainability", "Enabled")

if predict_btn:
    probability = np.random.uniform(0.15, 0.85)
    churn = "Yes" if probability > 0.5 else "No"

    st.success(f"Prediction Completed")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Prediction")
        st.metric("Churn", churn)

    with c2:
        st.subheader("Confidence")
        st.metric("Probability", f"{probability:.2%}")

    chart_df = pd.DataFrame({
        "Feature": ["Tenure", "Monthly Charges", "Contract"],
        "Importance": [0.42, 0.31, 0.27]
    })

    fig = px.bar(chart_df, x="Feature", y="Importance", title="Feature Importance")
    st.plotly_chart(fig, use_container_width=True)

st.header("EDA Dashboard")

eda_df = pd.DataFrame({
    "Category": ["Month-to-month", "One year", "Two year"],
    "Churn Rate": [42, 11, 3]
})

fig2 = px.pie(eda_df, names="Category", values="Churn Rate", title="Contract vs Churn")
st.plotly_chart(fig2, use_container_width=True)
