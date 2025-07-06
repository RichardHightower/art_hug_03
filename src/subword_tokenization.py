"""Subword Tokenization examples."""

import tiktoken
from transformers import AutoTokenizer


def run_subword_tokenization_examples():
    """Run subword tokenization examples with different tokenization methods."""

    print("Demonstrating different subword tokenization methods:")

    # Example text
    text = "Tokenization is fundamental to NLP. Let's explore BPE, WordPiece, and SentencePiece!"

    # 1. BPE (Byte Pair Encoding) - GPT-2
    print("\n1. BPE Tokenization (GPT-2):")
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gpt2_tokens = gpt2_tokenizer.tokenize(text)
    gpt2_ids = gpt2_tokenizer.encode(text)
    print(f"   Text: {text}")
    print(f"   Tokens: {gpt2_tokens}")
    print(f"   Token count: {len(gpt2_tokens)}")
    print(f"   Token IDs: {gpt2_ids[:10]}... (showing first 10)")

    # 2. WordPiece - BERT
    print("\n2. WordPiece Tokenization (BERT):")
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_tokens = bert_tokenizer.tokenize(text)
    bert_ids = bert_tokenizer.encode(text)
    print(f"   Text: {text}")
    print(f"   Tokens: {bert_tokens}")
    print(f"   Token count: {len(bert_tokens)}")
    print(f"   Token IDs: {bert_ids[:10]}... (showing first 10)")

    # 3. SentencePiece - T5
    print("\n3. SentencePiece Tokenization (T5):")
    t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")
    t5_tokens = t5_tokenizer.tokenize(text)
    t5_ids = t5_tokenizer.encode(text)
    print(f"   Text: {text}")
    print(f"   Tokens: {t5_tokens}")
    print(f"   Token count: {len(t5_tokens)}")
    print(f"   Token IDs: {t5_ids[:10]}... (showing first 10)")

    # 4. Tiktoken - GPT-3.5/4
    print("\n4. Tiktoken (cl100k_base - GPT-3.5/4):")
    encoding = tiktoken.get_encoding("cl100k_base")
    tiktoken_ids = encoding.encode(text)
    tiktoken_tokens = [encoding.decode([token_id]) for token_id in tiktoken_ids]
    print(f"   Text: {text}")
    print(f"   Tokens: {tiktoken_tokens}")
    print(f"   Token count: {len(tiktoken_tokens)}")
    print(f"   Token IDs: {tiktoken_ids[:10]}... (showing first 10)")

    # Comparison of unknown word handling
    print("\n5. Handling Unknown Words:")
    unknown_text = "Supercalifragilisticexpialidocious is quite extraordinary!"

    print(f"\n   Text with rare word: {unknown_text}")
    print(f"   GPT-2 tokens: {gpt2_tokenizer.tokenize(unknown_text)}")
    print(f"   BERT tokens: {bert_tokenizer.tokenize(unknown_text)}")
    print(f"   T5 tokens: {t5_tokenizer.tokenize(unknown_text)}")

    print("\nSubword tokenization examples completed!")


if __name__ == "__main__":
    run_subword_tokenization_examples()
