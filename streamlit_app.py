import streamlit as st
import pandas as pd
import joblib

# Load your trained model
model = joblib.load("model.pkl")

st.title("💳 Credit Card Fraud Detection App")

st.markdown("Enter the transaction values below:")

features = []
for i in range(1, 29):
    features.append(st.number_input(f"V{i}", value=0.0))

amount = st.number_input("Amount", value=0.0)
final_features = features + [amount]

if st.button("Predict Fraud"):
    df = pd.DataFrame([final_features], columns=[f"V{i}" for i in range(1, 29)] + ["Amount"])
    prediction = model.predict(df)[0]
    if prediction == 1:
        st.error("⚠️ Fraud Detected!")
    else:
        st.success("✅ Transaction is Safe")
