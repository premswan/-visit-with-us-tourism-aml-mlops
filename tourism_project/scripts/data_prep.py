import os
import pandas as pd
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split

HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO_ID = os.getenv("DATASET_REPO_ID")

if not HF_TOKEN or not DATASET_REPO_ID:
    raise RuntimeError("Missing HF_TOKEN or DATASET_REPO_ID. Add them as GitHub Secrets.")

# This file should already exist in the HF dataset repo after register_dataset.py
raw_url = f"https://huggingface.co/datasets/{DATASET_REPO_ID}/resolve/main/tourism.csv"

# Direct HTTP read (robust)
df = pd.read_csv(raw_url)

# Cleaning
for c in ["CustomerID", "Unnamed: 0", "index"]:
    if c in df.columns:
        df.drop(columns=[c], inplace=True)

df = df.drop_duplicates()

if "ProdTaken" not in df.columns:
    raise ValueError("Target column 'ProdTaken' not found in dataset!")

df["ProdTaken"] = df["ProdTaken"].astype(int)

# Split
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["ProdTaken"]
)

# Save locally in runner
train_path = "train.csv"
test_path = "test.csv"
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

# Upload back to HF dataset repo
api = HfApi()

api.upload_file(
    path_or_fileobj=train_path,
    path_in_repo="train.csv",
    repo_id=DATASET_REPO_ID,
    repo_type="dataset",
    token=HF_TOKEN
)

api.upload_file(
    path_or_fileobj=test_path,
    path_in_repo="test.csv",
    repo_id=DATASET_REPO_ID,
    repo_type="dataset",
    token=HF_TOKEN
)

print("✅ train.csv and test.csv uploaded to HF dataset repo:", DATASET_REPO_ID)
