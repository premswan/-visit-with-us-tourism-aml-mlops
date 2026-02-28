import os, json, joblib
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO_ID = os.getenv("DATASET_REPO_ID")
MODEL_REPO_ID = os.getenv("MODEL_REPO_ID")

if not HF_TOKEN or not DATASET_REPO_ID or not MODEL_REPO_ID:
    raise RuntimeError("Missing HF_TOKEN / DATASET_REPO_ID / MODEL_REPO_ID. Add them as GitHub Secrets.")

# Download train/test from HF dataset repo (robust)
train_local = hf_hub_download(repo_id=DATASET_REPO_ID, repo_type="dataset", filename="train.csv", token=HF_TOKEN)
test_local  = hf_hub_download(repo_id=DATASET_REPO_ID, repo_type="dataset", filename="test.csv", token=HF_TOKEN)

train_df = pd.read_csv(train_local)
test_df  = pd.read_csv(test_local)

TARGET = "ProdTaken"
X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET].astype(int)
X_test, y_test   = test_df.drop(columns=[TARGET]), test_df[TARGET].astype(int)

cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]
num_cols = [c for c in X_train.columns if c not in cat_cols]

preprocess = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
])

pipe = Pipeline([
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(random_state=42))
])

param_dist = {
    "model__n_estimators": [200, 400, 600],
    "model__max_depth": [5, 10, 15, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4]
}

search = RandomizedSearchCV(
    pipe,
    param_distributions=param_dist,
    n_iter=10,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
    random_state=42
)
search.fit(X_train, y_train)

best_model = search.best_estimator_
proba = best_model.predict_proba(X_test)[:, 1]
roc = float(roc_auc_score(y_test, proba))

# Save artifacts
os.makedirs("artifacts", exist_ok=True)

joblib.dump(best_model, "artifacts/model.joblib")

with open("artifacts/feature_schema.json", "w") as f:
    json.dump({
        "target": TARGET,
        "feature_columns": list(X_train.columns),
        "categorical_columns": cat_cols,
        "numerical_columns": num_cols
    }, f, indent=2)

with open("artifacts/metrics.json", "w") as f:
    json.dump({
        "roc_auc": roc,
        "best_params": search.best_params_
    }, f, indent=2)

# Upload to HF Model Hub (keyword-only args)
api = HfApi()
api.create_repo(repo_id=MODEL_REPO_ID, repo_type="model", exist_ok=True, private=False, token=HF_TOKEN)

api.upload_file(
    path_or_fileobj="artifacts/model.joblib",
    path_in_repo="model.joblib",
    repo_id=MODEL_REPO_ID,
    repo_type="model",
    token=HF_TOKEN
)

api.upload_file(
    path_or_fileobj="artifacts/feature_schema.json",
    path_in_repo="feature_schema.json",
    repo_id=MODEL_REPO_ID,
    repo_type="model",
    token=HF_TOKEN
)

api.upload_file(
    path_or_fileobj="artifacts/metrics.json",
    path_in_repo="metrics.json",
    repo_id=MODEL_REPO_ID,
    repo_type="model",
    token=HF_TOKEN
)

print("✅ Best model uploaded to:", MODEL_REPO_ID)
print("ROC_AUC:", roc)
