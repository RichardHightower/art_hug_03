"""Advanced Tokenization examples."""

from transformers import AutoTokenizer


def run_advanced_tokenization_examples():
    """Run advanced tokenization examples including padding,
    truncation, and special tokens.
    """

    print("Demonstrating advanced tokenization concepts:")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # 1. Padding examples
    print("\n1. Padding Examples:")
    texts = [
        "Short text.",
        "This is a medium length sentence that demonstrates padding.",
        "This is a much longer sentence that will show how padding works with
        multiple sentences of different lengths in a batch.",
    ]

    # Padding to max length in batch
    batch_encoding = tokenizer(texts, padding=True, return_tensors="pt")
    print(f"   Original texts lengths: {[len(text.split()) for text in texts]}")
    print(f"   Padded sequence lengths: {batch_encoding['input_ids'].shape}")
    print(f"   Attention mask shape: {batch_encoding['attention_mask'].shape}")
    print(f"   Attention masks:\n{batch_encoding['attention_mask']}")

    # 2. Truncation examples
    print("\n2. Truncation Examples:")
    long_text = " ".join(
        ["This is a very long sentence."] * 50
    )  # Create a very long text

    # Without truncation (will be very long)
    tokens_no_trunc = tokenizer.tokenize(long_text)
    print(f"   Without truncation: {len(tokens_no_trunc)} tokens")

    # With truncation to max_length
    tokens_with_trunc = tokenizer(
        long_text, truncation=True, max_length=20, return_tensors="pt"
    )
    print(
        f"   With truncation (max_length=20): {tokens_with_trunc['input_ids'].shape[1]} "
        "tokens"
    )
    print(
        f"   Truncated tokens: {tokenizer.convert_ids_to_tokens(tokens_with_trunc['input_ids'][0])}"
    )

    # 3. Special tokens
    print("\n3. Special Tokens:")
    text = "Hello, how are you?"

    # Encode with special tokens
    tokens_with_special = tokenizer(text, add_special_tokens=True)
    tokens_without_special = tokenizer(text, add_special_tokens=False)

    print(f"   Original text: {text}")
    print(
        f"   With special tokens: {tokenizer.convert_ids_to_tokens(tokens_with_special['input_ids'])}"
    )
    print(
        f"   Without special tokens: {tokenizer.convert_ids_to_tokens(tokens_without_special['input_ids'])}"
    )
    print("   Special tokens mapping:")
    print(f"     - [CLS] token: {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})")
    print(f"     - [SEP] token: {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})")
    print(f"     - [PAD] token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"     - [UNK] token: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")

    # 4. Handling multiple sequences (for tasks like question answering)
    print("\n4. Multiple Sequences (Question-Answering):")
    question = "What is tokenization?"
    context = (
        "Tokenization is the process of breaking down text into smaller units "
        "called tokens."
    )

    # Encode question and context together
    qa_encoding = tokenizer(
        question, context, padding=True, truncation=True, return_tensors="pt"
    )
    tokens = tokenizer.convert_ids_to_tokens(qa_encoding["input_ids"][0])

    print(f"   Question: {question}")
    print(f"   Context: {context}")
    print(f"   Combined tokens: {tokens}")
    print(f"   Token type IDs: {qa_encoding['token_type_ids'][0].tolist()}")

    # 5. Batch encoding with different strategies
    print("\n5. Batch Encoding Strategies:")
    batch_texts = [
        "First sentence.",
        "Second sentence is a bit longer.",
        "Third sentence is the longest of all the sentences in this batch.",
    ]

    # Padding to longest sequence in batch
    dynamic_padding = tokenizer(batch_texts, padding="longest", return_tensors="pt")
    print(f"   Dynamic padding shape: {dynamic_padding['input_ids'].shape}")

    # Padding to specific length
    fixed_padding = tokenizer(
        batch_texts, padding="max_length", max_length=30, return_tensors="pt"
    )
    print(f"   Fixed padding shape (max_length=30): {fixed_padding['input_ids'].shape}")

    # 6. Offset mapping for token-to-character alignment
    print("\n6. Token-to-Character Mapping:")
    text = "Tokenization helps us process text!"
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)

    print(f"   Text: {text}")
    print(f"   Tokens: {tokenizer.convert_ids_to_tokens(encoding['input_ids'])}")
    print(f"   Character offsets: {encoding['offset_mapping']}")

    # Show which characters each token corresponds to
    for token, (start, end) in zip(
        tokenizer.convert_ids_to_tokens(encoding["input_ids"]),
        encoding["offset_mapping"], strict=False,
    ):
        print(f"     Token '{token}' -> Text '{text[start:end]}' (chars {start}-{end})")

    print("\nAdvanced tokenization examples completed!")


if __name__ == "__main__":
    run_advanced_tokenization_examples()
