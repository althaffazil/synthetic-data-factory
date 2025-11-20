from huggingface_hub import hf_hub_download
from config.settings import MODEL_NAME, MODEL_FILE

def download_model():
    print("Downloading model from HuggingFace...")
    model_path = hf_hub_download(
        repo_id=MODEL_NAME,
        filename=MODEL_FILE,
        local_dir=".",               # download into project folder
        local_dir_use_symlinks=False # ensure real file is saved
    )
    print(f"Model downloaded to: {model_path}")
    return model_path
