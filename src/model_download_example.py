#!/usr/bin/env python3
"""
Model download example demonstrating how to download and cache models/tokenizers.
"""

import os

import torch
from transformers import (
    AutoModel,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from config import get_device


def download_sentiment_model():
    """Download a sentiment analysis model and tokenizer."""
    print("=== Downloading Sentiment Analysis Model ===")

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    print(f"Downloading model: {model_name}")
    print("This may take a moment on first run...")

    # Download the tokenizer for input text processing
    # (A tokenizer splits text into tokens the model understands)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Download the model weights and configuration
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Get model info
    print("✅ Model downloaded successfully!")
    print(f"   Model type: {model.config.model_type}")
    print(f"   Number of labels: {model.config.num_labels}")
    print(f"   Hidden size: {model.config.hidden_size}")

    # Test the model
    text = "I love using HuggingFace models!"
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    print(f"\nTest prediction for: '{text}'")
    print(f"   Negative: {predictions[0][0]:.3f}")
    print(f"   Positive: {predictions[0][1]:.3f}")
    print()

    return model, tokenizer


def download_multilingual_model():
    """Download a multilingual model."""
    print("=== Downloading Multilingual Model ===")

    model_name = "bert-base-multilingual-cased"

    print(f"Downloading model: {model_name}")
    print("This model supports 104 languages!")

    # Download tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    print("✅ Model downloaded successfully!")
    print(f"   Vocabulary size: {tokenizer.vocab_size}")

    # Test with multiple languages
    test_texts = [
        "Hello, world!",  # English
        "Bonjour le monde!",  # French
        "¡Hola mundo!",  # Spanish
        "你好世界！",  # Chinese
    ]

    print("\nTokenization examples:")
    for text in test_texts:
        tokens = tokenizer.tokenize(text)
        print(f"   '{text}' -> {tokens[:5]}...")
    print()

    return model, tokenizer


def download_specialized_models():
    """Download models for specific tasks."""
    print("=== Downloading Specialized Models ===")

    # Question Answering Model
    qa_model_name = "distilbert-base-cased-distilled-squad"
    print(f"\n1. Question Answering Model: {qa_model_name}")
    qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
    qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
    print("   ✅ QA model ready!")

    # Named Entity Recognition Model
    ner_model_name = "dslim/bert-base-NER"
    print(f"\n2. NER Model: {ner_model_name}")
    ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)
    ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
    print("   ✅ NER model ready!")
    print()

    return {"qa": (qa_model, qa_tokenizer), "ner": (ner_model, ner_tokenizer)}


def show_cache_info():
    """Show information about the model cache."""
    print("=== Model Cache Information ===")

    # Default cache directory
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

    if os.path.exists(cache_dir):
        # Count cached models
        model_dirs = [
            d
            for d in os.listdir(cache_dir)
            if os.path.isdir(os.path.join(cache_dir, d))
        ]
        print(f"Cache directory: {cache_dir}")
        print(f"Number of cached models: {len(model_dirs)}")

        import contextlib

        # Calculate cache size (simplified)
        total_size = 0
        for root, _dirs, files in os.walk(cache_dir):
            for file in files:
                with contextlib.suppress(Exception):
                    total_size += os.path.getsize(os.path.join(root, file))

        print(f"Total cache size: {total_size / (1024**3):.2f} GB")
    else:
        print("No cache directory found yet.")

    print("\nTip: Models are cached after first download for faster loading!")
    print()


def download_with_specific_revision():
    """Download a specific model revision/version."""
    print("=== Downloading Specific Model Revision ===")

    model_name = "gpt2"
    revision = "main"  # Can be a branch name, tag name, or commit id

    print(f"Downloading model: {model_name} (revision: {revision})")

    # Download with specific revision
    model = AutoModel.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=False,  # Security: don't execute remote code
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)

    print("✅ Model downloaded successfully!")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    return model, tokenizer


def main():
    """Run all model download examples."""
    print("HuggingFace Model Download Examples\n")

    # Check device
    device = get_device()
    if device == "mps":
        device_name = "MPS (Apple Silicon)"
    elif device == 0:
        device_name = "CUDA"
    else:
        device_name = "CPU"
    print(f"Device: {device_name}\n")

    # Show cache info first
    show_cache_info()

    # Download various models
    _sentiment_model, _sentiment_tokenizer = download_sentiment_model()
    _multilingual_model, _multilingual_tokenizer = download_multilingual_model()
    _specialized_models = download_specialized_models()
    _gpt2_model, _gpt2_tokenizer = download_with_specific_revision()

    # Final cache info
    print("\n=== After Downloads ===")
    show_cache_info()

    print("All model downloads completed!")


if __name__ == "__main__":
    main()
