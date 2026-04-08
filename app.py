import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

model = load_model("model.keras")
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="FASTag Fraud Detection", layout="wide")

st.title("🚗 FASTag Fraud Detection System")
st.write("AI-powered fraud detection")

mode = st.sidebar.radio("Mode", ["Single Prediction", "Batch Prediction"])

if mode == "Single Prediction":
    t = st.number_input("Transaction Amount")
    p = st.number_input("Amount Paid")

    if st.button("Predict"):
        X = np.array([[t, p]])
        X = scaler.transform(X)
        pred = model.predict(X)[0][0]

        if pred > 0.5:
            st.error("Fraud Detected")
        else:
            st.success("Legitimate")

        st.write(f"Probability: {pred:.2f}")

else:
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        X = df[['Transaction_Amount', 'Amount_paid']]
        X = scaler.transform(X)

        preds = model.predict(X)
        df['Fraud_Probability'] = preds
        df['Prediction'] = (preds > 0.5).astype(int)

        st.dataframe(df)

        st.download_button("Download", df.to_csv(index=False), "results.csv")
