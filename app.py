import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Customer Churn Prediction")

st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn.")

# Load final saved files from the project folder
BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "final_churn_model.pkl")
preprocessor = joblib.load(BASE_DIR / "final_preprocessor.pkl")
feature_columns = joblib.load(BASE_DIR / "final_feature_columns.pkl")

# Customer details
credit_score = st.number_input("Credit Score", 300, 900, 650)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
age = st.number_input("Age", 18, 100, 35)
tenure = st.number_input("Tenure", 0, 10, 5)
balance = st.number_input("Balance", min_value=0.0, value=50000.0)
num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
active_member = st.selectbox("Active Member?", ["Yes", "No"])
salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

if st.button("Predict Churn"):
    input_data = pd.DataFrame([{
        "CreditScore": credit_score,
        "Geography": geography,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": 1 if has_card == "Yes" else 0,
        "IsActiveMember": 1 if active_member == "Yes" else 0,
        "EstimatedSalary": salary
    }])

    # Keep the same feature order used during training
    input_data = input_data.reindex(columns=feature_columns)

    input_processed = preprocessor.transform(input_data)
    churn_probability = model.predict_proba(input_processed)[0][1]

    st.subheader("Prediction Result")
    st.metric("Churn Probability", f"{churn_probability:.2%}")

    if churn_probability >= 0.5:
        st.error("High Churn Risk")
    else:
        st.success("Low Churn Risk")
