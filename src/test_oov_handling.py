#!/usr/bin/env python3
"""
Test how different tokenizers handle out-of-vocabulary (OOV) words.
"""

from transformers import AutoTokenizer


def test_oov_handling():
    """Test how different tokenizers handle unknown words."""
    # Test with made-up words
    test_texts = [
        "The flibbertigibbet jumped over the moon.",
        "Pneumonoultramicroscopicsilicovolcanoconiosis is a lung disease.",
        "The 🦄 and 🌈 are beautiful.",
    ]

    for model_name in ["bert-base-uncased", "gpt2", "roberta-base"]:
        print(f"\n{model_name}:")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Get the UNK token for this tokenizer (if it has one)
        unk_token = tokenizer.unk_token
        print(f"  UNK token: {unk_token}")

        for text in test_texts:
            tokens = tokenizer.tokenize(text)
            # Truncate display if text is long
            display_text = text if len(text) <= 30 else text[:30] + "..."
            print(f"  '{display_text}' -> {len(tokens)} tokens")

            # Check if UNK tokens are present
            if unk_token and unk_token in tokens:
                unk_count = tokens.count(unk_token)
                print(f"    Contains {unk_count} UNK token(s)!")

            # Show first few tokens
            print(f"    First tokens: {tokens[:5]}")


def detailed_oov_analysis():
    """Detailed analysis of OOV handling."""
    print("\n=== Detailed OOV Analysis ===")

    # Made-up word that should be broken into subwords
    text = "The supersupercalifragilisticexpialidocious word"

    tokenizers = {
        "BERT (WordPiece)": "bert-base-uncased",
        "GPT-2 (BPE)": "gpt2",
        "RoBERTa (BPE)": "roberta-base",
    }

    for name, model_name in tokenizers.items():
        print(f"\n{name}:")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Tokenize
        tokens = tokenizer.tokenize(text)

        # Get token IDs
        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Check for unknown token ID
        unk_token_id = (
            tokenizer.unk_token_id if hasattr(tokenizer, "unk_token_id") else None
        )

        print(f"  Tokens: {tokens}")
        print(f"  Token IDs: {token_ids}")

        if unk_token_id is not None:
            unk_count = token_ids.count(unk_token_id)
            if unk_count > 0:
                print(f"  Contains {unk_count} unknown token(s) (ID: {unk_token_id})")

        # Decode back
        decoded = tokenizer.decode(token_ids)
        print(f"  Decoded: '{decoded}'")


def test_special_characters():
    """Test how tokenizers handle special characters and emojis."""
    print("\n=== Special Characters and Emojis ===")

    test_cases = [
        "Hello 👋 World 🌍!",
        "Price: $99.99 | Discount: 50%",
        "Email: test@example.com",
        "Math: x² + y² = z²",
        "Unicode: ñ é ü ø å",
    ]

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    for text in test_cases:
        tokens = tokenizer.tokenize(text)
        print(f"\nText: '{text}'")
        print(f"Tokens: {tokens}")
        print(f"Token count: {len(tokens)}")

        # Check for UNK tokens
        if tokenizer.unk_token in tokens:
            print("⚠️  Contains unknown tokens!")


def main():
    """Run all OOV handling tests."""
    print("Out-of-Vocabulary (OOV) Word Handling Tests\n")

    test_oov_handling()
    detailed_oov_analysis()
    test_special_characters()

    print("\n✅ All tests completed!")


if __name__ == "__main__":
    main()
