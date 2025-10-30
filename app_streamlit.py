# app_streamlit.py
"""
Streamlit app for the German Credit / Credit Risk model.
- Loads models/credit_pipeline.joblib and models/metadata.joblib
- Optionally loads models/metadata_categories.joblib to populate selectboxes
- Provides inputs, Predict button, probability + class label, and simple explanations
"""

import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

MODEL_PATH = "models/credit_pipeline.joblib"
META_PATH = "models/metadata.joblib"
CAT_META_PATH = "models/metadata_categories.joblib"

st.set_page_config(page_title="Credit Risk Demo", layout="wide")
st.title("Explainable Credit Risk Demo")

# Check artifacts
if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
    st.error("Model or metadata missing. Please run the training script (train_model.py) to produce models/credit_pipeline.joblib and models/metadata.joblib.")
    st.stop()

# Load model and metadata
model = joblib.load(MODEL_PATH)
meta = joblib.load(META_PATH)

numeric_cols = meta.get("numeric_cols", [])
categorical_cols = meta.get("categorical_cols", [])
feature_names = meta.get("feature_names", numeric_cols + categorical_cols)
importances = meta.get("importances", [0.0] * len(feature_names))
label_classes = meta.get("label_classes", None) # list of original class labels

# Load categorical choices if available
cat_choices = {}
if os.path.exists(CAT_META_PATH):
    try:
        cat_meta = joblib.load(CAT_META_PATH)
        cat_choices = cat_meta.get("categorical_uniques", {})
    except Exception:
        cat_choices = {}

st.sidebar.header("Applicant input")
st.sidebar.write("Provide applicant data and press Predict.")

# Build user input dictionary
user_input = {}

# Numeric inputs: use sensible defaults if possible
for col in numeric_cols:
    label = col
    # set smart defaults
    if "age" in col.lower():
        default = 35
        minv, maxv = 18, 100
        step = 1.0
    elif "credit" in col.lower():
        default = 2000.0
        minv, maxv = 0.0, 100000.0
        step = 50.0
    elif "duration" in col.lower():
        default = 12.0
        minv, maxv = 1.0, 240.0
        step = 1.0
    else:
        default = 0.0
        minv, maxv = -1e6, 1e6
        step = 1.0

    # Use slider for ranges that make sense else use number_input
    if maxv <= 1000 and minv >= 0:
        val = st.sidebar.number_input(label, value=float(default), min_value=float(minv), max_value=float(maxv), step=float(step))
    else:
        val = st.sidebar.number_input(label, value=float(default), step=float(step))
    user_input[col] = val

# Categorical inputs: prefer selectbox from cat_choices; fallback to text_input
for col in categorical_cols:
    options = cat_choices.get(col, None)
    mode_default = meta.get("categorical_modes", {}).get(col, "")
    if options and isinstance(options, (list, tuple)) and len(options) > 0:
        # pick the index of mode if present
        try:
            default_index = options.index(mode_default) if mode_default in options else 0
        except Exception:
            default_index = 0
        user_input[col] = st.sidebar.selectbox(col, options, index=default_index)
    else:
        user_input[col] = st.sidebar.text_input(col, value=str(mode_default))

# Advanced options
st.sidebar.markdown("---")
show_shap = st.sidebar.checkbox("Show SHAP explanation (if available)", value=False)
st.sidebar.markdown("When SHAP is enabled the app will attempt to compute local SHAP values (can be slow).")

# Predict button
if st.sidebar.button("Predict"):
    X = pd.DataFrame([user_input])
    st.subheader("Input summary")
    st.json(user_input)

    # Ensure numeric columns are numeric types
    for c in numeric_cols:
        try:
            X[c] = pd.to_numeric(X[c])
        except Exception:
            X[c] = float(X[c])

    # Fill missing columns the model expects (using metadata medians/modes if available)
    numeric_medians = meta.get("numeric_medians", {})
    categorical_modes = meta.get("categorical_modes", {})
    for c in numeric_cols:
        if c not in X.columns:
            X[c] = numeric_medians.get(c, 0.0)
    for c in categorical_cols:
        if c not in X.columns:
            X[c] = categorical_modes.get(c, "")

    # Model predict
    try:
        pred = model.predict(X)[0]
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        st.stop()

    # Map pred to label string if label_classes available
    pred_label = str(pred)
    if label_classes and isinstance(label_classes, (list, tuple)) and len(label_classes) > 0:
        try:
            pred_label = label_classes[int(pred)]
        except Exception:
            # fallback: if the model produced string label directly
            pred_label = pred

    # Probability if available
    prob = None
    try:
        prob = model.predict_proba(X)[:, 1][0]
    except Exception:
        prob = None

    st.subheader("Prediction")
    if prob is not None:
        st.metric(label="Default probability", value=f"{prob:.3f}")
    st.write("Predicted class (mapped):", pred_label)
    st.write("Raw model output:", str(pred))

    # Small local explanation using medians and global importances
    st.subheader("Local explanation (simple)")
    local_notes = []
    for c in numeric_cols:
        median = numeric_medians.get(c, None)
        if median is not None:
            val = float(X[c].iloc[0])
            if val > median:
                note = f"{c}: {val} (> median {median}) — may increase risk"
            elif val < median:
                note = f"{c}: {val} (< median {median}) — may decrease risk"
            else:
                note = f"{c}: {val} (= median)"
            local_notes.append(note)
    if local_notes:
        st.write("\n".join(local_notes))
    else:
        st.write("No numeric medians available for local comparisons.")

    # Global importances
    st.subheader("Top global features (model importances)")
    try:
        imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False).head(10).reset_index(drop=True)
        st.table(imp_df)
    except Exception:
        st.write("Global importances not available in metadata.")

    # Optional: SHAP (if user asked and shap installed)
    if show_shap:
        try:
            import shap
            st.subheader("SHAP local explanation (approx)")
            # attempt to get underlying tree model for fast explainer
            try:
                # If pipeline present, try to extract final estimator and preprocess the X
                if hasattr(model, "named_steps"):
                    preprocess = model.named_steps.get("preprocess", None)
                    clf = model.named_steps.get("clf", None)
                    if preprocess is not None:
                        X_proc = preprocess.transform(X)
                    else:
                        X_proc = X.values
                else:
                    clf = model
                    X_proc = X.values

                # use TreeExplainer for tree models
                expl = shap.TreeExplainer(clf)
                shap_vals = expl.shap_values(X_proc)
                # shap_values may be list for binary classifiers; pick class 1 shap values
                if isinstance(shap_vals, list) and len(shap_vals) >= 2:
                    sv = shap_vals[1]
                else:
                    sv = shap_vals
                # show as a small dataframe of feature contributions
                if hasattr(preprocess, "get_feature_names_out"):
                    feature_names_out = preprocess.get_feature_names_out()
                else:
                    # fallback: use feature_names from metadata
                    feature_names_out = feature_names
                shap_df = pd.DataFrame({"feature": feature_names_out, "shap_value": sv[0]})
                shap_df = shap_df.sort_values("shap_value", key=abs, ascending=False).head(10)
                st.table(shap_df)
                st.write("If you want full SHAP plots, compute offline and save images to display here.")
            except Exception as e:
                st.write("SHAP explanation failed (model may be incompatible). Error:", e)
        except Exception:
            st.warning("SHAP is not installed in this environment. To enable SHAP explanations, install the 'shap' Python package and restart the app.")

# Footer / tips
st.markdown("---")
st.info("Tip: If dropdowns show unexpected values, run `python generate_metadata_from_csv.py` locally and push the produced models/metadata_categories.joblib into your repo.")