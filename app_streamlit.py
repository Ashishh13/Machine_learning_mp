# app_streamlit.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

MODEL_PATH = "models/credit_pipeline.joblib"
META_PATH = "models/metadata.joblib"

st.set_page_config(page_title="Credit Risk Demo", layout="centered")
st.title("Explainable Credit Risk Demo")

model = joblib.load(MODEL_PATH)
metadata = joblib.load(META_PATH)

numeric_cols = metadata['numeric_cols']
categorical_cols = metadata['categorical_cols']

st.sidebar.header("Applicant input")

# For each numeric field, show a number_input; for categorical, a selectbox with mode + a text field fallback
user_data = {}
for c in numeric_cols:
    v = st.sidebar.number_input(c, value=float(metadata['numeric_medians'].get(c, 0.0)))
    user_data[c] = v

for c in categorical_cols:
    default = metadata['categorical_modes'].get(c, "")
    # try to infer sample unique values - we can't; use text input
    user_val = st.sidebar.text_input(c, value=str(default))
    user_data[c] = user_val

st.write("### Input summary")
st.json(user_data)

if st.button("Predict"):
    X = pd.DataFrame([user_data])
    prob = model.predict_proba(X)[:, 1][0]
    pred = model.predict(X)[0]
    st.metric("Default probability", f"{prob:.3f}")
    st.write("Prediction:", "Default" if pred == 1 else "Good")

    # show top global importances
    feat_names = metadata['feature_names']
    importances = metadata['importances']
    imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False).head(10)
    st.write("### Top global important features")
    st.dataframe(imp_df)

    # simple local sign: compare numeric values to medians
    st.write("### Simple local reasoning (numeric features vs median)")
    rows = []
    for c in numeric_cols:
        median = metadata['numeric_medians'].get(c, 0.0)
        val = X.iloc[0][c]
        direction = "higher (increases risk)" if val > median else ("lower (decreases risk)" if val < median else "equal")
        rows.append({"feature": c, "value": float(val), "median": float(median), "note": direction})
    st.table(pd.DataFrame(rows))