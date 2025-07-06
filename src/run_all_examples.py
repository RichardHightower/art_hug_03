"""Run all examples without user interaction."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from main import print_section, run_environment_examples, run_tokenization_examples

def main():
    """Run all examples automatically."""
    print_section("CHAPTER 03: SETTING UP YOUR HUGGING FACE ENVIRONMENT")
    print("Running all examples automatically...\n")
    
    # Run environment examples
    run_environment_examples()
    
    # Run tokenization examples  
    run_tokenization_examples()
    
    print_section("ALL EXAMPLES COMPLETED")
    print("✅ Successfully ran all Chapter 3 examples!")

if __name__ == "__main__":
    main()