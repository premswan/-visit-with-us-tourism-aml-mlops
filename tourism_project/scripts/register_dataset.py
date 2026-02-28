import os
from huggingface_hub import HfApi

HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO_ID = os.getenv("DATASET_REPO_ID")

if not HF_TOKEN or not DATASET_REPO_ID:
    raise RuntimeError("Missing HF_TOKEN or DATASET_REPO_ID. Add them as GitHub Secrets.")

csv_path = "tourism_project/data/tourism.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Missing {csv_path}. Commit tourism.csv to the repo at this path.")

api = HfApi()

api.create_repo(
    repo_id=DATASET_REPO_ID,
    repo_type="dataset",
    exist_ok=True,
    private=False,
    token=HF_TOKEN
)

api.upload_file(
    path_or_fileobj=csv_path,
    path_in_repo="tourism.csv",
    repo_id=DATASET_REPO_ID,
    repo_type="dataset",
    token=HF_TOKEN
)

print("✅ Dataset registered:", DATASET_REPO_ID)
