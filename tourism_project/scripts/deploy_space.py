import os
from huggingface_hub import HfApi

HF_TOKEN = os.getenv("HF_TOKEN")
SPACE_REPO_ID = os.getenv("SPACE_REPO_ID")

if not HF_TOKEN or not SPACE_REPO_ID:
    raise RuntimeError("Missing HF_TOKEN or SPACE_REPO_ID. Add them as GitHub Secrets.")

deploy_folder = "tourism_project/deployment"
if not os.path.isdir(deploy_folder):
    raise FileNotFoundError(f"Missing deployment folder: {deploy_folder}")

api = HfApi()

# Create Space repo (Docker Space)
api.create_repo(
    repo_id=SPACE_REPO_ID,
    repo_type="space",
    space_sdk="docker",
    exist_ok=True,
    private=False,
    token=HF_TOKEN
)

# Upload deployment folder to Space
api.upload_folder(
    folder_path=deploy_folder,
    repo_id=SPACE_REPO_ID,
    repo_type="space",
    token=HF_TOKEN
)

print("✅ Deployment pushed to HF Space:", SPACE_REPO_ID)
