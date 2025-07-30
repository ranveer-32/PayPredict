# app/app.py
import streamlit as st
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
import pandas as pd
import os

st.title("💼 PayPredict: Smart Salary Estimator")

# Load model & encoders
model = joblib.load("E:\\payPredict\\models\\salary_model.pkl")
scaler = joblib.load("E:\\payPredict\\models\\scaler.pkl")
ohe = joblib.load("E:\\payPredict\\models\\ohe.pkl")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Get OHE categories for form
education_levels = sorted(
    [f.split('_', 1)[1] for f in ohe.get_feature_names_out() if f.startswith('education_')]
)

locations = sorted(
    [f.split('_', 1)[1] for f in ohe.get_feature_names_out() if f.startswith('location_')]
)

# User Inputs
st.subheader("📋 Enter Candidate Details")

experience = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, step=0.5)

education = st.selectbox("Education Level", education_levels)
location = st.selectbox("Location", locations)

uploaded_resume = st.file_uploader("Upload Resume (.txt)", type="txt")
uploaded_jd = st.file_uploader("Upload Job Description (.txt)", type="txt")

if st.button("Predict Salary"):
    if uploaded_resume is None or uploaded_jd is None:
        st.error("Please upload both Resume and Job Description files!")
    else:
        # Read text
        resume_text = uploaded_resume.read().decode("utf-8")
        jd_text = uploaded_jd.read().decode("utf-8")

        # Structured features
        experience_scaled = scaler.transform([[experience]])[0][0]
        df_temp = pd.DataFrame([[education, location]], columns=["education", "location"])
        ohe_features = ohe.transform(df_temp)

        # Text embeddings
        resume_emb = embedder.encode([resume_text])
        jd_emb = embedder.encode([jd_text])

        # Combine all
        X_structured = np.hstack([[experience_scaled], ohe_features.flatten()])
        X_text = np.hstack([resume_emb.flatten(), jd_emb.flatten()])
        X = np.hstack([X_structured, X_text]).reshape(1, -1)

        # Predict
        salary_pred = model.predict(X)[0]

        st.success(f"💰 Predicted Salary: ${salary_pred:,.2f}")
