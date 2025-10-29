# backend/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Credit Risk API")

MODEL_PATH = os.path.join("models", "credit_pipeline.joblib")
META_PATH = os.path.join("models", "metadata.joblib")

model = joblib.load(MODEL_PATH)
metadata = joblib.load(META_PATH)

numeric_cols = metadata['numeric_cols']
categorical_cols = metadata['categorical_cols']
feature_names = metadata['feature_names']

class Applicant(BaseModel):
    # dynamically accept any keys, but Pydantic needs fields. Keep a flexible schema instead:
    age: float = None
    # We'll accept arbitrary via dict endpoint below if needed.

@app.get("/")
def root():
    return {"message": "Backend running."}

@app.post("/predict")
def predict(payload: dict):
    # payload should be a dict of features matching training columns
    X = pd.DataFrame([payload])
    # ensure all columns exist
    for c in numeric_cols + categorical_cols:
        if c not in X.columns:
            X[c] = np.nan
    # fill missing with medians/modes
    for c in numeric_cols:
        if X[c].isnull().any():
            X[c] = X[c].fillna(metadata['numeric_medians'].get(c, 0.0))
    for c in categorical_cols:
        if X[c].isnull().any():
            X[c] = X[c].fillna(metadata['categorical_modes'].get(c, ""))

    proba = model.predict_proba(X)[:, 1].tolist()
    pred = model.predict(X).tolist()

    return {"prediction": pred[0], "default_probability": float(proba[0])}

@app.post("/explain")
def explain(payload: dict):
    X = pd.DataFrame([payload])
    # normalize missing as above
    for c in numeric_cols:
        if c not in X.columns:
            X[c] = metadata['numeric_medians'].get(c, 0.0)
    for c in categorical_cols:
        if c not in X.columns:
            X[c] = metadata['categorical_modes'].get(c, "")

    # Basic local explanation: compare numeric to median
    reasons = []
    imp = metadata['importances']
    for i, fname in enumerate(metadata['feature_names']):
        base_name = None
        # map ohe back to base column for categorical OHE names: find first '_' occurrence
        if "_" in fname and any(fname.startswith(col + "_") for col in categorical_cols):
            # example cat_col_value => base cat col name is before first _
            base_name = fname.split("_")[0]
        else:
            base_name = fname if fname in numeric_cols else fname

        # Use only numeric base features for a simple directionality rule
    # Simpler: return top global importances
    imp_pairs = list(zip(feature_names, imp))
    top = sorted(imp_pairs, key=lambda x: x[1], reverse=True)[:8]
    return {"top_features_by_importance": [{"feature": f, "importance": float(im)} for f, im in top]}