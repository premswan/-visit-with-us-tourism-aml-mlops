import os
from huggingface_hub import HfApi

HF_TOKEN = os.getenv("HF_TOKEN") or "PASTE_YOUR_HF_TOKEN_HERE"
SPACE_REPO_ID = "premswan/visit-with-us-wellness-app"

api = HfApi()

# Create Space repo if needed
api.create_repo(
    repo_id=SPACE_REPO_ID,
    repo_type="space",
    space_sdk="docker",
    exist_ok=True,
    private=False
)

# Upload entire deployment folder
api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id=SPACE_REPO_ID,
    repo_type="space",
    token=HF_TOKEN
)

print("✅ Deployed files pushed to HF Space:", SPACE_REPO_ID)
