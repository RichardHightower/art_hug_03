import os

import torch
from dotenv import load_dotenv

"""Configuration module for examples."""

# Load environment variables from .env file
load_dotenv()

# HuggingFace Hub token - check multiple possible env var names
HF_TOKEN = (
    os.getenv("HUGGINGFACE_TOKEN")
    or os.getenv("HUGGING_FACE_HUB_TOKEN")
    or os.getenv("HF_TOKEN")
)

# Other configuration variables
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "bert-base-uncased")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
CACHE_DIR = os.getenv("CACHE_DIR", "./models")


def get_device():
    """Get the best available device for pipelines (returns int or str)."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return 0  # Return 0 for first CUDA device
    else:
        return -1  # Return -1 for CPU in pipeline API


DEVICE = get_device()
