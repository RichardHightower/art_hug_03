"""Unit tests for Chapter 03 examples."""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from basic_tokenization import run_basic_tokenization_examples
from config import get_device


def test_device_detection():
    """Test that device detection works."""
    device = get_device()
    assert device in ["cpu", "cuda", "mps"]


def test_basic_tokenization_runs():
    """Test that basic_tokenization examples run without errors."""
    # This is a basic smoke test
    try:
        run_basic_tokenization_examples()
    except Exception as e:
        pytest.fail(f"basic_tokenization examples failed: {e}")


def test_imports():
    """Test that all required modules can be imported."""
    import torch
    import transformers

    assert transformers.__version__
    assert torch.__version__
