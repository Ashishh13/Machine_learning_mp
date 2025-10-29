# train_model.py
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb

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

# Identify label column (try common names)
label_col = None
for cand in ["target", "Risk", "risk", "label", "default"]:
    if cand in df.columns:
        label_col = cand
        break

if label_col is None:
    raise ValueError("No label column found. Add a column named 'Risk' or 'target' or 'label' etc.")

print("Using label column:", label_col)

# Map string labels to integers if needed
y_raw = df[label_col]
if y_raw.dtype == object or y_raw.dtype.name == "category":
    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))
    print("Label classes:", le.classes_)
    # save label encoder
    joblib.dump(le, "models/label_encoder.joblib")
else:
    y = y_raw.values

# Remove label column from X
X_df = df.drop(columns=[label_col]).copy()

# Identify numerical and categorical columns
numeric_cols = X_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
print("Numeric cols:", numeric_cols)
print("Categorical cols:", categorical_cols)

# Fill simple missing values
for c in numeric_cols:
    if X_df[c].isnull().any():
        X_df[c] = X_df[c].fillna(X_df[c].median())
for c in categorical_cols:
    if X_df[c].isnull().any():
        X_df[c] = X_df[c].fillna(X_df[c].mode().iloc[0])

# Build preprocessing pipeline
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
# scikit-learn versions: use sparse_output=False to be safe
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
categorical_transformer = Pipeline(steps=[("ohe", ohe)])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# Build estimator pipeline
model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("clf", xgb.XGBClassifier(n_estimators=100, max_depth=4, use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y if len(np.unique(y))>1 else None)

# Fit
print("Training model...")
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1] if model.named_steps['clf'].n_classes_ > 1 else model.predict_proba(X_test)[:, 0]
print("Classification report:")
print(classification_report(y_test, y_pred))
try:
    auc = roc_auc_score(y_test, y_prob)
    print("ROC AUC:", auc)
except Exception:
    pass

# Save model pipeline
joblib.dump(model, MODEL_PATH)
print("Saved model pipeline to", MODEL_PATH)

# Build metadata: feature names and importances and simple medians/modes
# Get preprocessed feature names
numeric_feature_names = numeric_cols
cat_ohe = model.named_steps['preprocess'].named_transformers_['cat'].named_steps['ohe']
cat_feature_names = cat_ohe.get_feature_names_out(categorical_cols).tolist() if hasattr(cat_ohe, "get_feature_names_out") else []
feature_names = numeric_feature_names + cat_feature_names

# feature importances from XGB
clf = model.named_steps['clf']
try:
    importances = clf.feature_importances_.tolist()
except Exception:
    # fallback: zeros
    importances = [0.0] * len(feature_names)

# compute medians/modes for explanation baseline
numeric_medians = {c: float(X_train[c].median()) for c in numeric_cols}
categorical_modes = {c: X_train[c].mode().iloc[0] if not X_train[c].mode().empty else None for c in categorical_cols}

metadata = {
    "feature_names": feature_names,
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
    "importances": importances,
    "numeric_medians": numeric_medians,
    "categorical_modes": categorical_modes,
    "preprocess_columns_order": feature_names
}

joblib.dump(metadata, META_PATH)
print("Saved metadata to", META_PATH)
print("Done.")