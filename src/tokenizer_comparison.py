"""Tokenizer Comparison examples."""

from transformers import AutoTokenizer
import tiktoken
from tabulate import tabulate
import time

def run_tokenizer_comparison_examples():
    """Compare different tokenizers on the same text."""
    
    print("Comparing different tokenizers on various texts:")
    
    # Initialize tokenizers
    tokenizers = {
        "BERT (WordPiece)": AutoTokenizer.from_pretrained("bert-base-uncased"),
        "GPT-2 (BPE)": AutoTokenizer.from_pretrained("gpt2"),
        "T5 (SentencePiece)": AutoTokenizer.from_pretrained("t5-small"),
        "RoBERTa (BPE)": AutoTokenizer.from_pretrained("roberta-base"),
    }
    
    # Add tiktoken
    tiktoken_enc = tiktoken.get_encoding("cl100k_base")
    
    # Test texts
    test_texts = {
        "Simple": "Hello world!",
        "Technical": "The transformer architecture revolutionized NLP in 2017.",
        "Multilingual": "Hello! Bonjour! ¡Hola! 你好! こんにちは!",
        "Code": "def tokenize(text): return text.split()",
        "Numbers": "The year 2024 has 365 days and 8,760 hours.",
        "Special": "Email: user@example.com, URL: https://example.com",
        "Long word": "Pneumonoultramicroscopicsilicovolcanoconiosis is a lung disease."
    }
    
    # Compare tokenization
    for text_name, text in test_texts.items():
        print(f"\n{'='*60}")
        print(f"Text: {text_name}")
        print(f"Content: {text}")
        print(f"{'='*60}")
        
        results = []
        
        # Process with each tokenizer
        for name, tokenizer in tokenizers.items():
            tokens = tokenizer.tokenize(text)
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            results.append([
                name,
                len(tokens),
                tokens[:5] + ['...'] if len(tokens) > 5 else tokens,
                str(token_ids[:5]) + '...' if len(token_ids) > 5 else str(token_ids)
            ])
        
        # Add tiktoken
        tiktoken_ids = tiktoken_enc.encode(text)
        tiktoken_tokens = [tiktoken_enc.decode([tid]) for tid in tiktoken_ids]
        results.append([
            "GPT-3.5/4 (Tiktoken)",
            len(tiktoken_tokens),
            tiktoken_tokens[:5] + ['...'] if len(tiktoken_tokens) > 5 else tiktoken_tokens,
            str(tiktoken_ids[:5]) + '...' if len(tiktoken_ids) > 5 else str(tiktoken_ids)
        ])
        
        # Display results
        headers = ["Tokenizer", "Token Count", "First 5 Tokens", "First 5 IDs"]
        print(tabulate(results, headers=headers, tablefmt="grid"))
    
    # Vocabulary size comparison
    print(f"\n{'='*60}")
    print("Vocabulary Size Comparison:")
    print(f"{'='*60}")
    
    vocab_sizes = []
    for name, tokenizer in tokenizers.items():
        vocab_sizes.append([name, f"{tokenizer.vocab_size:,}"])
    vocab_sizes.append(["GPT-3.5/4 (Tiktoken)", f"{tiktoken_enc.n_vocab:,}"])
    
    print(tabulate(vocab_sizes, headers=["Tokenizer", "Vocabulary Size"], tablefmt="grid"))
    
    # Performance comparison
    print(f"\n{'='*60}")
    print("Performance Comparison (encoding speed):")
    print(f"{'='*60}")
    
    long_text = " ".join(["This is a sample sentence for performance testing."] * 100)
    
    perf_results = []
    for name, tokenizer in tokenizers.items():
        start_time = time.time()
        for _ in range(100):
            _ = tokenizer.encode(long_text)
        elapsed = time.time() - start_time
        perf_results.append([name, f"{elapsed:.3f}s"])
    
    # Tiktoken performance
    start_time = time.time()
    for _ in range(100):
        _ = tiktoken_enc.encode(long_text)
    elapsed = time.time() - start_time
    perf_results.append(["GPT-3.5/4 (Tiktoken)", f"{elapsed:.3f}s"])
    
    print(tabulate(perf_results, headers=["Tokenizer", "Time (100 iterations)"], tablefmt="grid"))
    
    # Special features comparison
    print(f"\n{'='*60}")
    print("Special Features:")
    print(f"{'='*60}")
    
    features = []
    for name, tokenizer in tokenizers.items():
        feature_list = []
        if hasattr(tokenizer, 'cls_token') and tokenizer.cls_token:
            feature_list.append("CLS")
        if hasattr(tokenizer, 'sep_token') and tokenizer.sep_token:
            feature_list.append("SEP")
        if hasattr(tokenizer, 'mask_token') and tokenizer.mask_token:
            feature_list.append("MASK")
        if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token:
            feature_list.append("PAD")
        if hasattr(tokenizer, 'bos_token') and tokenizer.bos_token:
            feature_list.append("BOS")
        if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token:
            feature_list.append("EOS")
        
        features.append([name, ", ".join(feature_list) if feature_list else "None"])
    
    features.append(["GPT-3.5/4 (Tiktoken)", "None (pure encoding)"])
    
    print(tabulate(features, headers=["Tokenizer", "Special Tokens"], tablefmt="grid"))
    
    print("\nTokenizer comparison completed!")

if __name__ == "__main__":
    run_tokenizer_comparison_examples()