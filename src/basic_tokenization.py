"""Basic Tokenization implementation."""

import torch
from transformers import AutoModel, AutoTokenizer

from config import DEFAULT_MODEL, get_device


def run_basic_tokenization_examples():
    """Run basic tokenization examples."""

    print(f"Loading model: {DEFAULT_MODEL}")
    device = get_device()
    print(f"Using device: {device}")

    # Example implementation
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    model = AutoModel.from_pretrained(DEFAULT_MODEL)

    # Example text
    text = "Hugging Face Transformers make NLP accessible to everyone!"

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    print(f"\nInput text: {text}")
    print(f"Tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].tolist())}")
    print(f"Token IDs: {inputs['input_ids'][0].tolist()}")

    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)

    print(f"\nModel output shape: {outputs.last_hidden_state.shape}")
    print("Example completed successfully!")


if __name__ == "__main__":
    print("=== Basic Tokenization Examples ===\n")
    run_basic_tokenization_examples()
