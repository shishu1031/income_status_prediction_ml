from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "models" / "best_model.joblib"

st.set_page_config(page_title="Income Status Predictor", page_icon="📊", layout="wide")
st.title("📊 Income Status Predictor")
st.caption("Educational demo based on the Adult Census Income dataset")

if not MODEL_PATH.exists():
    st.warning("No trained model found. Run `python -m src.train` first.")
    st.stop()

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
threshold = float(artifact.get("threshold", 0.50))

with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 17, 90, 30)
        workclass = st.text_input("Workclass", "Private")
        education = st.text_input("Education", "Bachelors")
        educational_num = st.number_input("Education number", 1, 16, 13)
        marital_status = st.text_input("Marital status", "Never-married")
    with c2:
        occupation = st.text_input("Occupation", "Prof-specialty")
        relationship = st.text_input("Relationship", "Not-in-family")
        race = st.text_input("Race", "White")
        sex = st.selectbox("Sex", ["Female", "Male"])
        capital_gain = st.number_input("Capital gain", 0, 100000, 0)
    with c3:
        capital_loss = st.number_input("Capital loss", 0, 10000, 0)
        hours = st.number_input("Hours per week", 1, 100, 40)
        native_country = st.text_input("Native country", "United-States")
        fnlwgt = st.number_input("Final weight", 1000, 1000000, 189664)

    submitted = st.form_submit_button("Predict")

if submitted:
    row = pd.DataFrame([{
        "age": age,
        "workclass": workclass,
        "fnlwgt": fnlwgt,
        "education": education,
        "education-num": educational_num,
        "marital-status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours,
        "native-country": native_country,
    }])
    prob = float(model.predict_proba(row)[:, 1][0])
    pred = ">50K" if prob >= threshold else "<=50K"
    st.metric("Predicted status", pred)
    st.progress(min(max(prob, 0.0), 1.0), text=f"Estimated probability of >50K: {prob:.1%}")
    st.caption(f"Classification threshold used by the saved model: {threshold:.2f}")
    st.info("This demo is for educational use. The prediction should not be used as an automated high-impact decision.")
