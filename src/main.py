"""Main entry point for all examples."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from basic_tokenization import run_basic_tokenization_examples
from subword_tokenization import run_subword_tokenization_examples
from advanced_tokenization import run_advanced_tokenization_examples
from tokenizer_comparison import run_tokenizer_comparison_examples

def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")

def main():
    """Run all examples."""
    print_section("CHAPTER 03: SETTING UP YOUR HUGGING FACE ENVIRONMENT")
    print("Welcome! This script demonstrates all concepts from Chapter 3.")
    print("Including environment setup and tokenization fundamentals.\n")
    
    # Ask user what to run
    print("What would you like to run?")
    print("1. Environment Setup Examples (pipelines, Hub API, etc.)")
    print("2. Tokenization Examples")
    print("3. All Examples")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        run_environment_examples()
    elif choice == "2":
        run_tokenization_examples()
    elif choice == "3":
        run_environment_examples()
        run_tokenization_examples()
    elif choice == "4":
        print("Goodbye!")
        return
    else:
        print("Invalid choice. Running all examples...")
        run_environment_examples()
        run_tokenization_examples()
    
    print_section("CONCLUSION")
    print("These examples demonstrate the key concepts from Chapter 3.")
    print("Try modifying the code to experiment with different approaches!")

def run_environment_examples():
    """Run environment setup examples."""
    print_section("ENVIRONMENT SETUP EXAMPLES")
    
    print("\n1. Running verification script...")
    try:
        from verify_installation import main as verify_main
        verify_main()
    except Exception as e:
        print(f"Error running verification: {e}")
    
    print("\n2. Running pipeline examples...")
    try:
        from pipeline_example import main as pipeline_main
        pipeline_main()
    except Exception as e:
        print(f"Error running pipeline examples: {e}")
    
    print("\n3. Running HuggingFace Hub API examples...")
    try:
        from hf_hub_example import main as hub_main
        hub_main()
    except Exception as e:
        print(f"Error running Hub examples: {e}")
    
    print("\n4. Running model download examples...")
    try:
        from model_download_example import main as download_main
        download_main()
    except Exception as e:
        print(f"Error running download examples: {e}")
    
    print("\n5. Running translation examples...")
    try:
        from translation_example import main as translation_main
        translation_main()
    except Exception as e:
        print(f"Error running translation examples: {e}")
    
    print("\n6. Running speech recognition examples...")
    try:
        from speech_recognition_example import main as speech_main
        speech_main()
    except Exception as e:
        print(f"Error running speech examples: {e}")

def run_tokenization_examples():
    """Run tokenization examples."""
    print_section("TOKENIZATION EXAMPLES")
    
    print_section("1. BASIC TOKENIZATION")
    run_basic_tokenization_examples()
    
    print_section("2. SUBWORD TOKENIZATION")
    run_subword_tokenization_examples()
    
    print_section("3. ADVANCED TOKENIZATION")
    run_advanced_tokenization_examples()
    
    print_section("4. TOKENIZER COMPARISON")
    run_tokenizer_comparison_examples()
    
    print_section("5. OUT-OF-VOCABULARY HANDLING")
    try:
        from test_oov_handling import main as oov_main
        oov_main()
    except Exception as e:
        print(f"Error running OOV examples: {e}")

if __name__ == "__main__":
    main()