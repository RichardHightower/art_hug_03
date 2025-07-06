#!/usr/bin/env python3
"""
Translation example demonstrating batch translation with HuggingFace Transformers.
"""

from transformers import pipeline
import torch
from config import get_device
import time


def basic_translation_example():
    """Basic translation from English to French."""
    print("=== Basic Translation Example (EN → FR) ===")
    
    device = get_device()
    
    # Create translation pipeline
    translator = pipeline(
        "translation_en_to_fr",
        model="Helsinki-NLP/opus-mt-en-fr",
        device=device
    )
    
    # Sentences to translate
    sentences = [
        "Hugging Face makes AI easy.",
        "Transformers are powerful.",
        "I love natural language processing!",
        "The weather is beautiful today."
    ]
    
    # Translate with batch processing
    print("Translating sentences...")
    start_time = time.time()
    translations = translator(sentences, batch_size=2, max_length=128)
    end_time = time.time()
    
    # Display results
    print(f"\nTranslation completed in {end_time - start_time:.2f} seconds")
    for original, result in zip(sentences, translations):
        translated = result['translation_text']
        print(f"EN: {original}")
        print(f"FR: {translated}\n")


def multilingual_translation_example():
    """Translate between multiple language pairs."""
    print("=== Multilingual Translation Examples ===")
    
    device = get_device()
    
    # Different language pairs
    language_pairs = [
        ("en", "de", "Helsinki-NLP/opus-mt-en-de"),  # English to German
        ("en", "es", "Helsinki-NLP/opus-mt-en-es"),  # English to Spanish
        ("en", "zh", "Helsinki-NLP/opus-mt-en-zh"),  # English to Chinese
    ]
    
    # Text to translate
    text = "Artificial intelligence is transforming the world."
    
    for src_lang, tgt_lang, model_name in language_pairs:
        print(f"\n{src_lang.upper()} → {tgt_lang.upper()}")
        
        # Create pipeline for this language pair
        translator = pipeline(
            f"translation_{src_lang}_to_{tgt_lang}",
            model=model_name,
            device=device
        )
        
        # Translate
        result = translator(text, max_length=128)
        print(f"Original: {text}")
        print(f"Translation: {result[0]['translation_text']}")


def batch_translation_performance():
    """Compare single vs batch translation performance."""
    print("\n=== Batch Translation Performance ===")
    
    device = get_device()
    
    # Create translator
    translator = pipeline(
        "translation_en_to_fr",
        model="Helsinki-NLP/opus-mt-en-fr",
        device=device
    )
    
    # Create test sentences
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a versatile programming language.",
        "Neural networks can learn complex patterns.",
        "Data science combines statistics and programming.",
        "Deep learning has revolutionized computer vision.",
        "Natural language processing enables machines to understand text.",
        "Transformers have become the foundation of modern NLP."
    ]
    
    # Single translation (one at a time)
    print("1. Single translation (one sentence at a time):")
    start_time = time.time()
    single_results = []
    for sentence in sentences:
        result = translator(sentence)
        single_results.append(result)
    single_time = time.time() - start_time
    print(f"   Time: {single_time:.2f} seconds")
    
    # Batch translation
    print("\n2. Batch translation (all sentences at once):")
    start_time = time.time()
    batch_results = translator(sentences, batch_size=4)
    batch_time = time.time() - start_time
    print(f"   Time: {batch_time:.2f} seconds")
    
    # Performance comparison
    speedup = single_time / batch_time
    print(f"\n📊 Speedup: {speedup:.2f}x faster with batching!")
    print(f"   Sentences processed: {len(sentences)}")
    print(f"   Avg time per sentence (single): {single_time/len(sentences):.3f}s")
    print(f"   Avg time per sentence (batch): {batch_time/len(sentences):.3f}s")


def translation_with_options():
    """Demonstrate translation with various options."""
    print("\n=== Translation with Options ===")
    
    device = get_device()
    
    translator = pipeline(
        "translation_en_to_fr",
        model="Helsinki-NLP/opus-mt-en-fr",
        device=device
    )
    
    text = "The artificial intelligence model performed exceptionally well on the challenging dataset."
    
    # Different max_length settings
    print("Original:", text)
    print("\nWith different max_length settings:")
    
    for max_len in [20, 50, 100]:
        result = translator(text, max_length=max_len)
        translation = result[0]['translation_text']
        print(f"  max_length={max_len}: {translation}")
    
    # Multiple translations (if supported)
    print("\nGenerating multiple translation candidates:")
    result = translator(
        text,
        max_length=100,
        num_beams=3,  # Beam search for better quality
        early_stopping=True
    )
    print(f"  Best translation: {result[0]['translation_text']}")


def reverse_translation_example():
    """Demonstrate back-translation (round-trip translation)."""
    print("\n=== Reverse Translation Example ===")
    
    device = get_device()
    
    # Create both direction translators
    en_to_fr = pipeline(
        "translation_en_to_fr",
        model="Helsinki-NLP/opus-mt-en-fr",
        device=device
    )
    
    fr_to_en = pipeline(
        "translation_fr_to_en",
        model="Helsinki-NLP/opus-mt-fr-en",
        device=device
    )
    
    # Original text
    original = "The secret to happiness is not in doing what you like, but in liking what you do."
    
    print("Back-translation (EN → FR → EN):")
    print(f"1. Original (EN): {original}")
    
    # Translate to French
    fr_result = en_to_fr(original)
    french_text = fr_result[0]['translation_text']
    print(f"2. French (FR): {french_text}")
    
    # Translate back to English
    en_result = fr_to_en(french_text)
    back_translated = en_result[0]['translation_text']
    print(f"3. Back to English: {back_translated}")
    
    # Compare
    print(f"\n📊 Similarity check:")
    print(f"   Original length: {len(original)} chars")
    print(f"   Back-translated length: {len(back_translated)} chars")


def main():
    """Run all translation examples."""
    print("HuggingFace Translation Examples\n")
    
    # Check device
    device = get_device()
    if device == "mps":
        device_name = "MPS (Apple Silicon)"
    elif device == 0:
        device_name = "CUDA"
    else:
        device_name = "CPU"
    print(f"Running on: {device_name}\n")
    
    # Run examples
    basic_translation_example()
    multilingual_translation_example()
    batch_translation_performance()
    translation_with_options()
    reverse_translation_example()
    
    print("\n✅ All translation examples completed!")


if __name__ == "__main__":
    main()