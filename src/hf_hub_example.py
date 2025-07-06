#!/usr/bin/env python3
"""
HuggingFace Hub API example demonstrating model discovery and interaction.
"""

import os

from huggingface_hub import HfApi, list_models, model_info

from config import HF_TOKEN


def list_text_classification_models():
    """List models for text classification task."""
    print("=== Text Classification Models ===")

    api = HfApi()  # Create an API client for the Hugging Face Hub

    # List all models for the 'text-classification' task
    models = api.list_models(task="text-classification")

    # Convert to list to count and display
    model_list = list(models)
    print(f"Found {len(model_list)} text-classification models!")

    # Show top 10 most downloaded models
    print("\nTop 10 most downloaded text-classification models:")
    sorted_models = sorted(model_list, key=lambda x: x.downloads or 0, reverse=True)[
        :10
    ]

    for i, model in enumerate(sorted_models, 1):
        downloads = model.downloads or "N/A"
        likes = model.likes or 0
        print(
            f"{i:2d}. {model.modelId[:50]:<50} | Downloads: {downloads:>10} | "
            f"Likes: {likes:>5}"
        )
    print()


def explore_model_details():
    """Get detailed information about a specific model."""
    print("=== Model Details Example ===")

    # Popular sentiment analysis model
    model_id = "distilbert-base-uncased-finetuned-sst-2-english"

    try:
        # Get model information
        info = model_info(model_id)

        print(f"Model: {info.modelId}")
        print(f"Author: {info.author}")
        print(f"Downloads: {info.downloads:,}")
        print(f"Likes: {info.likes}")
        print(f"Pipeline tag: {info.pipeline_tag}")
        print(f"License: {info.license if hasattr(info, 'license') else 'N/A'}")

        # Show tags
        if info.tags:
            print(f"Tags: {', '.join(info.tags[:5])}")

        # Show model card extract
        if info.card_data:
            print("\nModel Card Extract:")
            if hasattr(info.card_data, "language"):
                print(f"  Language: {info.card_data.language}")
            if hasattr(info.card_data, "datasets"):
                print(f"  Datasets: {', '.join(info.card_data.datasets[:3])}")
    except Exception as e:
        print(f"Error getting model info: {e}")
    print()


def search_models_by_language():
    """Search for models by language."""
    print("=== Search Models by Language ===")

    # Search for French language models
    french_models = list_models(language="fr", task="text-generation", limit=5)

    print("French text generation models:")
    for i, model in enumerate(french_models, 1):
        print(f"{i}. {model.modelId}")
    print()


def list_models_by_library():
    """List models by library/framework."""
    print("=== Models by Library ===")

    # List PyTorch models for translation
    pytorch_translation = list_models(task="translation", library="pytorch", limit=5)

    print("PyTorch translation models:")
    for i, model in enumerate(pytorch_translation, 1):
        downloads = getattr(model, "downloads", "N/A")
        print(f"{i}. {model.modelId} (Downloads: {downloads})")
    print()


def search_dataset_models():
    """Search for models trained on specific datasets."""
    print("=== Models by Dataset ===")

    # Models trained on IMDB dataset
    imdb_models = list_models(search="imdb", task="text-classification", limit=5)

    print("Models trained on IMDB dataset:")
    for i, model in enumerate(imdb_models, 1):
        print(f"{i}. {model.modelId}")
    print()


def check_auth_status():
    """Check if user is authenticated with HuggingFace Hub."""
    print("=== Authentication Status ===")

    api = HfApi()
    # Use token from config or environment
    token = HF_TOKEN or os.environ.get("HUGGING_FACE_HUB_TOKEN") or api.token

    if token:
        try:
            # Set the token in the API
            api.token = token
            user = api.whoami(token=token)
            print(f"✅ Authenticated as: {user['name']}")
            print(
                print(
                    f"   Organizations: {', '.join(
                    org['name'] for org in user.get('orgs', [])
                )}"
                )
            )
        except Exception as e:
            print("❌ Invalid token or not authenticated")
            print(f"   Error: {str(e)[:100]}")
    else:
        print("❌ No authentication token found")
        print("   Run 'task hf-login' to authenticate")
        print("   Or add HUGGINGFACE_TOKEN to your .env file")
    print()


def main():
    """Run all HuggingFace Hub examples."""
    print("HuggingFace Hub API Examples\n")

    # Check authentication
    check_auth_status()

    # Run examples
    list_text_classification_models()
    explore_model_details()
    search_models_by_language()
    list_models_by_library()
    search_dataset_models()

    print("All Hub API examples completed!")


if __name__ == "__main__":
    main()
