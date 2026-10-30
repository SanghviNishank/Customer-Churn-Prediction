
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load model
model_path = Path(__file__).parent.parent / "models" / "pipeline.pkl"
model = joblib.load(model_path)

# Page config
st.set_page_config(page_title="Churn Predictor", layout="wide")

# Title
st.title("Customer Churn Prediction")
st.markdown("Enter customer details to predict churn risk.")

# Sidebar inputs
with st.sidebar:
    st.header("Customer Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

    st.header("Account Information")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.selectbox("Payment Method", 
                                  ["Electronic check", "Mailed check", 
                                   "Bank transfer (automatic)", "Credit card (automatic)"])

    st.header("Services")
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

# Build input dataframe
def build_input():
    return pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': ["No" if phone_service == "No" else "No"],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': ["No" if internet_service == "No" else "No"],
        'DeviceProtection': ["No" if internet_service == "No" else "No"],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [50.0],
        'TotalCharges': [tenure * 50.0],
        'NumberOfServices': [0],
        'TenureGroup': ["New"]
    })

# Predict button
if st.button("Predict Churn"):
    input_df = build_input()
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    col1, col2 = st.columns(2)
    with col1:
        if prediction == 1:
            st.error(f"High Churn Risk - {probability:.1%}")
        else:
            st.success(f"Low Churn Risk - {1-probability:.1%}")

    with col2:
        st.metric("Churn Probability", f"{probability:.1%}")
