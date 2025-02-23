from huggingface_hub import hf_hub_download
import shutil
import os

cache_dir = os.path.expanduser("~/.cache/huggingface/hub/")
model_dir = os.path.join(cache_dir, "models--Yaminii--finetuned-mistral")

# Remove the cached model folder
if os.path.exists(model_dir):
    shutil.rmtree(model_dir)

print(f"Cache for {model_dir} cleared. Retry loading the model.")
