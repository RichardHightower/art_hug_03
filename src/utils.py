"""Utility functions for tokenization examples."""

from pathlib import Path
from typing import Any

import torch


def ensure_dirs():
    """Ensure necessary directories exist."""
    dirs = ["data", "models", "outputs"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)


def print_token_analysis(tokenizer, text: str, name: str = "Tokenizer"):
    """Print detailed token analysis for a given text."""
    print(f"\n{name} Analysis:")
    print(f"Text: {text}")

    # Tokenize
    encoding = tokenizer(text, add_special_tokens=True, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

    print(f"Tokens: {tokens}")
    print(f"Token IDs: {encoding['input_ids'][0].tolist()}")
    print(f"Token count: {len(tokens)}")

    # If attention mask is available
    if "attention_mask" in encoding:
        print(f"Attention mask: {encoding['attention_mask'][0].tolist()}")

    # If token type IDs are available
    if "token_type_ids" in encoding:
        print(f"Token type IDs: {encoding['token_type_ids'][0].tolist()}")

    return encoding


def compare_tokenizations(tokenizers: dict[str, Any], text: str):
    """Compare how different tokenizers handle the same text."""
    print(f"\nComparing tokenizations for: '{text}'")
    print("-" * 60)

    results = {}
    for name, tokenizer in tokenizers.items():
        tokens = tokenizer.tokenize(text)
        results[name] = {"tokens": tokens, "count": len(tokens)}
        print(f"{name:20} ({len(tokens):2} tokens): {tokens}")

    return results


def visualize_attention_mask(attention_mask: torch.Tensor, tokens: list[str]):
    """Visualize attention mask for tokens."""
    print("\nAttention Mask Visualization:")
    print("Token:     ", end="")
    for token in tokens:
        print(f"{token:>10}", end="")
    print("\nAttention: ", end="")
    for mask in attention_mask[0]:
        print(f"{mask.item():>10}", end="")
    print()


def save_tokenization_example(tokenizer, text: str, output_path: str):
    """Save tokenization example to file."""
    encoding = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

    with open(output_path, "w") as f:
        f.write(f"Original Text: {text}\n")
        f.write(f"Tokens: {tokens}\n")
        f.write(f"Token IDs: {encoding['input_ids'][0].tolist()}\n")
        if "attention_mask" in encoding:
            f.write(f"Attention Mask: {encoding['attention_mask'][0].tolist()}\n")

    print(f"Saved tokenization example to {output_path}")


def batch_tokenize_texts(tokenizer, texts: list[str], **kwargs):
    """Tokenize multiple texts in a batch."""
    default_kwargs = {"padding": True, "truncation": True, "return_tensors": "pt"}
    default_kwargs.update(kwargs)

    encodings = tokenizer(texts, **default_kwargs)

    print("\nBatch Tokenization Summary:")
    print(f"Number of texts: {len(texts)}")
    print(f"Batch shape: {encodings['input_ids'].shape}")
    print(f"Max sequence length in batch: {encodings['input_ids'].shape[1]}")

    return encodings


def demonstrate_special_tokens(tokenizer):
    """Demonstrate all special tokens for a tokenizer."""
    print(f"\nSpecial Tokens for {tokenizer.__class__.__name__}:")

    special_tokens = {
        "PAD": tokenizer.pad_token,
        "UNK": tokenizer.unk_token,
        "CLS": getattr(tokenizer, "cls_token", None),
        "SEP": getattr(tokenizer, "sep_token", None),
        "MASK": getattr(tokenizer, "mask_token", None),
        "BOS": getattr(tokenizer, "bos_token", None),
        "EOS": getattr(tokenizer, "eos_token", None),
    }

    for name, token in special_tokens.items():
        if token:
            token_id = tokenizer.convert_tokens_to_ids(token)
            print(f"  {name}: '{token}' (ID: {token_id})")

    return special_tokens
