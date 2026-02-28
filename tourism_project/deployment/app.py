import json
import pandas as pd
import streamlit as st
import joblib
from huggingface_hub import snapshot_download

st.set_page_config(page_title="VisitWithUs - Wellness Predictor", layout="centered")

st.title("Wellness Tourism Package Purchase Prediction")
st.write("Predict whether a customer will purchase the Wellness Tourism Package (ProdTaken).")

MODEL_REPO_ID = "premswan/visit-with-us-wellness-model"

@st.cache_resource
def load_model_and_schema():
    local_dir = snapshot_download(repo_id=MODEL_REPO_ID, repo_type="model")
    model = joblib.load(f"{local_dir}/model.joblib")
    with open(f"{local_dir}/feature_schema.json", "r") as f:
        schema = json.load(f)
    return model, schema

model, schema = load_model_and_schema()

st.subheader("Enter Customer Details")

inputs = {}
for col in schema["feature_columns"]:
    # simple UI: numeric -> number_input, categorical -> text_input
    if col in schema["numerical_columns"]:
        inputs[col] = st.number_input(col, value=0.0)
    else:
        inputs[col] = st.text_input(col, value="")

if st.button("Predict"):
    df = pd.DataFrame([inputs])
    proba = model.predict_proba(df)[:, 1][0]
    pred = int(proba >= 0.5)

    st.success(f"Prediction (ProdTaken): {pred}")
    st.info(f"Probability of purchase: {proba:.3f}")
