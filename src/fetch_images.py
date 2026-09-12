import os
import zipfile
from pathlib import Path

def download_algae_images():
    """
    Downloads the Kaggle Blue-Green Algae dataset using the Kaggle API.
    Expects ~/.kaggle/kaggle.json or KAGGLE_USERNAME and KAGGLE_KEY in environment.
    """
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    access_token_file = Path.home() / ".kaggle" / "access_token"
    
    if "KAGGLE_API_TOKEN" not in os.environ and access_token_file.exists():
        with open(access_token_file, 'r') as f:
            os.environ["KAGGLE_API_TOKEN"] = f.read().strip()

    has_token = access_token_file.exists() or "KAGGLE_API_TOKEN" in os.environ
    has_env = "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ

    dataset_name = "beyondstellaris/bluegreen-algae-dataset"
    target_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "images"))
    os.makedirs(target_folder, exist_ok=True)

    if not kaggle_json.exists() and not has_env and not has_token:
        print("[WARNING] Kaggle credentials not found at ~/.kaggle/access_token, ~/.kaggle/kaggle.json, or in environment.")
        print("To download the 2.3 GB dataset, place your token in ~/.kaggle/access_token")
        print("Using demo sample images in data/images/ for hackathon presentation.")
        return False

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        print(f"Downloading dataset '{dataset_name}' to {target_folder}...")
        api.dataset_download_files(dataset_name, path=target_folder, unzip=True)
        print("Images downloaded and extracted to data/images/")
        
        # Clean up any residual zip files if present
        for item in os.listdir(target_folder):
            if item.endswith(".zip"):
                try:
                    os.remove(os.path.join(target_folder, item))
                except Exception:
                    pass
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download Kaggle dataset: {e}")
        return False

if __name__ == "__main__":
    download_algae_images()

