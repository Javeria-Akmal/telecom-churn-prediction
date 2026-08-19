import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Telecom Churn Predictor", page_icon="📞", layout="centered")

@st.cache_resource
def load_bundle():
    with open("churn_model.pkl", "rb") as f:
        return pickle.load(f)

bundle = load_bundle()
model = bundle["model"]
scaler = bundle["scaler"]
binary_cols = bundle["binary_cols"]
multi_cols = bundle["multi_cols"]
multi_col_categories = bundle["multi_col_categories"]
feature_order = bundle["feature_order"]

# NOTE: the LabelEncoders saved inside the bundle got fit on already-encoded
# 0/1 values (from a cell re-run in the notebook), so they can't map raw
# strings back to 0/1. We use the same alphabetical mapping LabelEncoder
# would have produced from the original Yes/No, Male/Female values instead.
BINARY_MAP = {"Female": 0, "Male": 1, "No": 0, "Yes": 1}
BINARY_OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "PaperlessBilling": ["No", "Yes"],
}

st.title("📞 Telecom Customer Churn Predictor")
st.write("Enter a customer's details to predict whether they are likely to churn.")

with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", BINARY_OPTIONS["gender"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", BINARY_OPTIONS["Partner"])
        dependents = st.selectbox("Dependents", BINARY_OPTIONS["Dependents"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", BINARY_OPTIONS["PhoneService"])
        multiple_lines = st.selectbox("Multiple Lines", multi_col_categories["MultipleLines"])
        internet_service = st.selectbox("Internet Service", multi_col_categories["InternetService"])
        online_security = st.selectbox("Online Security", multi_col_categories["OnlineSecurity"])
        online_backup = st.selectbox("Online Backup", multi_col_categories["OnlineBackup"])

    with col2:
        device_protection = st.selectbox("Device Protection", multi_col_categories["DeviceProtection"])
        tech_support = st.selectbox("Tech Support", multi_col_categories["TechSupport"])
        streaming_tv = st.selectbox("Streaming TV", multi_col_categories["StreamingTV"])
        streaming_movies = st.selectbox("Streaming Movies", multi_col_categories["StreamingMovies"])
        contract = st.selectbox("Contract", multi_col_categories["Contract"])
        paperless = st.selectbox("Paperless Billing", BINARY_OPTIONS["PaperlessBilling"])
        payment = st.selectbox("Payment Method", multi_col_categories["PaymentMethod"])
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
        total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    # Build a single-row raw dataframe exactly like the notebook's df (before encoding)
    raw = {
        "gender": gender, "SeniorCitizen": senior, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone_service,
        "MultipleLines": multiple_lines, "InternetService": internet_service,
        "OnlineSecurity": online_security, "OnlineBackup": online_backup,
        "DeviceProtection": device_protection, "TechSupport": tech_support,
        "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
        "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }
    row = pd.DataFrame([raw])

    # Step 1: encode binary columns (same 0/1 mapping LabelEncoder used at training time)
    for col in binary_cols:
        row[col] = row[col].map(BINARY_MAP)

    # Step 2: one-hot encode the multi-class columns manually against the
    # known feature_order (pd.get_dummies on a single row is unsafe here --
    # with only one category present it always treats it as the dropped
    # reference category, silently mis-encoding non-reference values).
    for col in multi_cols:
        val = row.at[0, col]
        dummy_col = f"{col}_{val}"
        for candidate in feature_order:
            if candidate.startswith(f"{col}_"):
                row[candidate] = 1 if candidate == dummy_col else 0
        row.drop(columns=[col], inplace=True)

    # Step 3: align columns to the exact training feature order (missing dummy cols -> 0)
    row = row.reindex(columns=feature_order, fill_value=0)

    # Step 4: scale
    X_scaled = scaler.transform(row)

    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0][1]

    st.divider()
    if pred == 1:
        st.error(f"⚠️ Likely to Churn — probability: {proba:.1%}")
    else:
        st.success(f"✅ Likely to Stay — churn probability: {proba:.1%}")

    st.progress(min(proba, 1.0))

st.divider()
st.caption("Model: Logistic Regression (from CUSTOMER_CHURN_PREDICTION.ipynb) | Accuracy 80.7% | ROC-AUC 84.16%")
