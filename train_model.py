# train_model.py
"""
Train an explainable credit-risk classifier pipeline.
Saves:
 - models/credit_pipeline.joblib (the sklearn pipeline: preprocessing + classifier)
 - models/metadata.joblib (feature info, label encoder, importances)
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings("ignore")

# CONFIG
DATA_PATH = "data/credit.csv"
MODEL_PATH = "models/credit_pipeline.joblib"
META_PATH = "models/metadata.joblib"
RANDOM_STATE = 42
os.makedirs("models", exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
print("Loaded data shape:", df.shape)
print("Columns:", df.columns.tolist())

# Drop obvious index column if present
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])
    print("Dropped Unnamed: 0 column")

# Identify label column
label_col = None
for cand in ["target", "Risk", "risk", "label", "default"]:
    if cand in df.columns:
        label_col = cand
        break
if label_col is None:
    raise ValueError("No label column found. Add a column named 'target' or 'Risk' etc.")
print("Using label column:", label_col)

# Basic target encoding (label encoder)
y_raw = df[label_col].astype(str)
le = LabelEncoder()
y = le.fit_transform(y_raw)
print("Label classes:", list(le.classes_))

# Drop label from features
X = df.drop(columns=[label_col]).copy()

# Identify numeric and categorical columns
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
print("Numeric cols:", numeric_cols)
print("Categorical cols:", categorical_cols)

# Simple missing handling (if any)
for c in numeric_cols:
    if X[c].isnull().any():
        X[c] = X[c].fillna(X[c].median())
for c in categorical_cols:
    if X[c].isnull().any():
        X[c] = X[c].fillna(X[c].mode().iloc[0])

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

# use sparse_output=False for sklearn>=1.2
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
categorical_transformer = Pipeline(steps=[("ohe", ohe)])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# Choose classifier (we'll tune RandomForest; XGBoost can be used if installed)
clf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("clf", clf)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# Optionally apply SMOTE if imblearn is available; otherwise use class_weight balancing
use_smote = False
try:
    from imblearn.over_sampling import SMOTE
    use_smote = True
    print("imblearn installed: SMOTE will be used for resampling.")
except Exception:
    print("imblearn not installed: continuing without SMOTE; classifier will use class_weight='balanced' in tuning.")

# Hyperparameter search space for RandomForest
param_dist = {
    "clf__n_estimators": randint(100, 500),
    "clf__max_depth": randint(3, 20),
    "clf__min_samples_split": randint(2, 10),
    "clf__min_samples_leaf": randint(1, 6),
    # if imblearn not used, we try class_weight
    "clf__class_weight": [None, "balanced"]
}

# If SMOTE available, we will create a small pipeline with SMOTE for search
if use_smote:
    from imblearn.pipeline import Pipeline as ImbPipeline
    sm = SMOTE(random_state=RANDOM_STATE)
    imb_pipeline = ImbPipeline(steps=[("preprocess", preprocessor), ("smote", sm), ("clf", clf)])
    search_pipeline = imb_pipeline
    param_dist_smote = {
        "clf__n_estimators": randint(100, 500),
        "clf__max_depth": randint(3, 20),
        "clf__min_samples_split": randint(2, 10),
        "clf__min_samples_leaf": randint(1, 6),
    }
    param_search_space = param_dist_smote
else:
    search_pipeline = model
    param_search_space = param_dist

# Randomized search (faster than full grid)
rs = RandomizedSearchCV(
    estimator=search_pipeline,
    param_distributions=param_search_space,
    n_iter=30,
    scoring="f1",
    n_jobs=-1,
    cv=5,
    random_state=RANDOM_STATE,
    verbose=1
)

print("Starting hyperparameter search (this may take a while)...")
rs.fit(X_train, y_train)
print("Best params:", rs.best_params_)
best_model = rs.best_estimator_

# Evaluate on test set
y_pred = best_model.predict(X_test)
# for ROC AUC need predict_proba if available
y_prob = None
try:
    y_prob = best_model.predict_proba(X_test)[:, 1]
except Exception:
    # fallback: decision_function if available
    try:
        y_prob = best_model.decision_function(X_test)
    except Exception:
        y_prob = None

print("Classification report:")
print(classification_report(y_test, y_pred))
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
print(f"accuracy: {acc:.4f}")
print(f"f1: {f1:.4f}")
if roc is not None:
    print(f"ROC AUC: {roc:.4f}")

# Cross-validated performance on full training data (for robustness)
cv_scores = cross_val_score(rs.best_estimator_, X_train, y_train, cv=5, scoring='f1', n_jobs=-1)
print("CV f1 scores (train):", cv_scores, "mean:", cv_scores.mean())

# Save final pipeline and metadata
# If imblearn pipeline used, best_model contains preprocess+smote+clf; we save it as-is.
joblib.dump(best_model, MODEL_PATH)
print("Saved model pipeline to", MODEL_PATH)

# Compose metadata: feature names, numeric medians, cat modes, label encoder
# compute feature names after preprocessing (works for OneHotEncoder get_feature_names_out)
try:
    # get preprocessed feature names
    if use_smote:
        prep = best_model.named_steps['preprocess']
    else:
        prep = best_model.named_steps['preprocess']
    # numeric names first
    numeric_feature_names = numeric_cols
    ohe_transformer = prep.named_transformers_['cat'].named_steps['ohe']
    cat_feature_names = ohe_transformer.get_feature_names_out(categorical_cols).tolist() if hasattr(ohe_transformer, "get_feature_names_out") else []
    feature_names = numeric_feature_names + cat_feature_names
except Exception:
    feature_names = numeric_cols + categorical_cols

# feature importances attempt
try:
    clf_final = best_model.named_steps['clf'] if not use_smote else best_model.named_steps['clf']
    importances = clf_final.feature_importances_.tolist()
except Exception:
    importances = [0.0] * len(feature_names)

numeric_medians = {c: float(X_train[c].median()) for c in numeric_cols}
categorical_modes = {c: X_train[c].mode().iloc[0] if not X_train[c].mode().empty else None for c in categorical_cols}

metadata = {
    "feature_names": feature_names,
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
    "importances": importances,
    "numeric_medians": numeric_medians,
    "categorical_modes": categorical_modes,
    "label_classes": list(le.classes_)
}
joblib.dump(metadata, META_PATH)
print("Saved metadata to", META_PATH)
print("Done.")