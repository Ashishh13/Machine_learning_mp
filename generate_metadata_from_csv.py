# generate_metadata_from_csv.py
import os, joblib, pandas as pd

CSV_PATH = "data/credit.csv" # or data/german_credit_data.csv depending on your file
OUT_PATH = "models/metadata_categories.joblib"

os.makedirs("models", exist_ok=True)

df = pd.read_csv(CSV_PATH)

# drop obvious index if present
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# try to detect label/target so we don't include it in features
possible_targets = ['target','Target','TARGET','Risk','risk','credit_risk','CreditRisk','default','Default','class','Class','Loan_Status']
target_col = None
for c in possible_targets:
    if c in df.columns:
        target_col = c
        break
if target_col is not None:
    features_df = df.drop(columns=[target_col])
else:
    features_df = df.copy()

categorical_cols = features_df.select_dtypes(include=['object','category','bool']).columns.tolist()

cat_uniques = {}
for c in categorical_cols:
    # get unique strings sorted for nicer UI
    vals = features_df[c].dropna().astype(str).unique().tolist()
    vals_sorted = sorted(vals)
    cat_uniques[c] = vals_sorted

# Save
meta = {
    "categorical_cols": categorical_cols,
    "categorical_uniques": cat_uniques
}
joblib.dump(meta, OUT_PATH)
print("Wrote categorical metadata to", OUT_PATH)
print("Categorical columns:", categorical_cols)