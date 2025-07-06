#!/usr/bin/env python3
"""
Verify HuggingFace installation and environment setup.
"""

import importlib
import sys


def check_package(package_name: str) -> tuple[bool, str]:
    """Check if a package is installed and return its version."""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, "__version__", "unknown")
        return True, version
    except ImportError:
        return False, "not installed"


def main():
    """Main verification function."""
    print("=== HuggingFace Environment Verification ===\n")

    # Core packages to check
    packages = [
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("accelerate", "Accelerate"),
        ("torch", "PyTorch"),
        ("huggingface_hub", "HuggingFace Hub"),
        ("tokenizers", "Tokenizers"),
        ("sentencepiece", "SentencePiece"),
        ("tiktoken", "TikToken"),
    ]

    all_installed = True

    # Check each package
    for package, display_name in packages:
        installed, version = check_package(package)
        status = "✅" if installed else "❌"
        print(f"{status} {display_name:20} {version}")
        if not installed:
            all_installed = False

    print("\n=== Python Environment ===")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Python executable: {sys.executable}")

    # Check PyTorch device availability
    if check_package("torch")[0]:
        import torch

        print("\n=== PyTorch Device Info ===")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")

        # Check for Apple Silicon MPS
        if hasattr(torch.backends, "mps"):
            print(f"MPS available: {torch.backends.mps.is_available()}")

    # Summary
    print("\n=== Summary ===")
    if all_installed:
        print("✅ All required packages are installed!")
        print("You're ready to use HuggingFace tools.")
    else:
        print("❌ Some packages are missing.")
        print("Run 'task setup' to install dependencies.")

    return 0 if all_installed else 1


if __name__ == "__main__":
    sys.exit(main())
