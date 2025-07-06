# Setting Up Your Hugging Face Environment: Hands-On Foundations

Welcome to the world of Hugging Face, the cornerstone of modern AI development. This repository contains working examples for Chapter 3: **Setting Up Your Hugging Face Environment**.

## Your Gateway to the Exciting World of Transformers

Whether you're a seasoned machine learning engineer or just beginning your AI journey, this repository provides everything you need to create a professional-grade workspace for exploring, fine-tuning, and deploying state-of-the-art models. No more configuration headaches or dependency nightmares—we'll build a solid foundation together.

## What You'll Achieve

By working through these examples, you'll have:
- ✅ A fully configured environment optimized for AI development
- ✅ Direct access to thousands of pre-trained models from the Hugging Face Hub
- ✅ The ability to run powerful transformer models with just a few lines of code
- ✅ Understanding of best practices for reproducible AI workflows
- ✅ Working examples of pipelines, tokenization, and model management

## Prerequisites

- Python 3.10+ (3.12 recommended)
- Poetry for dependency management OR Conda
- Go Task for build automation
- (Optional) CUDA-capable GPU or Apple Silicon Mac for acceleration
- (Optional) HuggingFace account for model access

## 🚀 Quick Start: Choose Your Setup Path

### Option 1: Poetry Setup (Recommended for Projects)
```bash
# Complete setup with all dependencies
task setup-complete

# This automatically:
# - Sets up Python environment
# - Installs all Hugging Face libraries
# - Configures GPU support if available
# - Verifies your installation
```

### Option 2: Conda Setup (Alternative)
```bash
# Create and activate conda environment
task conda-setup
conda activate hf-env

# Install all dependencies
pip install -r requirements.txt
```

### Option 3: Manual Setup with pip
```bash
# Create virtual environment
python -m venv huggingface_env
source huggingface_env/bin/activate  # On Windows: huggingface_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 🖥️ Hardware Acceleration

**For NVIDIA GPUs (CUDA):**
```bash
task install-cuda
```

**For Apple Silicon (M1/M2/M3):**
```bash
task install-mps
```

The examples automatically detect and use the best available device (MPS for Apple Silicon, CUDA for NVIDIA, CPU fallback).

### 🔐 Hugging Face Hub Authentication

Authenticate to access private models and push your own:
```bash
task hf-login
# Or set HUGGINGFACE_TOKEN in your .env file
```

## ✅ Verify Your Installation

```bash
# Comprehensive environment verification
task verify-setup

# This checks:
# - All required packages
# - Device availability (GPU/MPS/CPU)
# - Hugging Face Hub connectivity
# - Model download capabilities
```

## 📁 Repository Structure

```
src/
├── config.py                    # Smart device detection and configuration
├── main.py                      # Interactive menu for all examples
├── verify_installation.py       # Comprehensive environment verification
│
├── Environment Setup Examples:
│   ├── pipeline_example.py          # Sentiment, QA, generation pipelines
│   ├── hf_hub_example.py           # Hub API: browsing models & datasets
│   ├── model_download_example.py    # Model downloading and caching
│   ├── translation_example.py       # Batch translation demonstrations
│   └── speech_recognition_example.py # Audio transcription with ASR
│
└── Tokenization Preview (Full coverage in Chapter 5):
    ├── basic_tokenization.py        # Quick tokenization examples
    ├── subword_tokenization.py      # BPE, WordPiece, SentencePiece
    ├── advanced_tokenization.py     # Padding, truncation, special tokens
    └── tokenizer_comparison.py      # Performance benchmarks

notebooks/
├── Chapter_03_Environment_Setup.ipynb  # Interactive environment examples
└── Chapter_03_Tokenization.ipynb      # Interactive tokenization preview
```

## 🎯 Running the Examples

### Interactive Menu (Recommended)
```bash
task run
# Choose from:
# 1. Environment Setup Examples (pipelines, Hub API, etc.)
# 2. Tokenization Preview Examples
# 3. All Examples
```

### Run Specific Examples

**First Steps with Pipelines:**
```bash
task run-pipeline  # Sentiment analysis, text generation, Q&A
```

**Model Hub Integration:**
```bash
task run-hub-api           # Browse and search models/datasets
task run-model-download    # Download and cache models
```

**Advanced Pipelines:**
```bash
task run-translation       # Batch translation, multiple languages
task run-speech           # Speech recognition demonstrations
```

**Tokenization Preview (Full coverage in Chapter 5):**
```bash
task run-basic-tokenization    # See text-to-token conversion
task run-subword-tokenization  # Compare BPE, WordPiece, SentencePiece
task run-advanced-tokenization # Padding, truncation, special tokens
task run-comparison           # Performance benchmarks
```

## Available Tasks

### Setup & Environment
- `task setup` - Basic Python/Poetry setup
- `task setup-complete` - Complete setup with all dependencies
- `task conda-setup` - Create conda environment
- `task export-requirements` - Export requirements.txt
- `task hf-login` - Login to HuggingFace Hub
- `task verify-setup` - Verify environment setup
- `task verify-imports` - Quick import check

### GPU Support
- `task install-cuda` - Install PyTorch with CUDA
- `task install-mps` - Install PyTorch for Apple Silicon

### Running Examples
- `task run` - Run all tokenization examples
- `task run-pipeline` - Run pipeline examples
- `task run-hub-api` - Run Hub API examples
- `task run-model-download` - Run model download examples
- `task run-translation` - Run translation examples
- `task run-speech` - Run speech recognition examples

### Development
- `task test` - Run unit tests
- `task format` - Format code with Black and Ruff
- `task clean` - Clean up generated files

## 🚀 Your First Hugging Face Pipeline

After setup, try this quick example:

```python
from transformers import pipeline

# Create a sentiment analysis pipeline
classifier = pipeline('text-classification')

# Analyze some text
result = classifier("Hugging Face is transforming the way we build AI!")
print(result)
# Output: [{'label': 'POSITIVE', 'score': 0.999}]
```

**That's it!** You're now using state-of-the-art AI with just three lines of code.

## 🛠️ Troubleshooting Common Issues

Based on real-world experience, here are solutions to common setup challenges:

### Installation Issues

**SentencePiece Build Errors (macOS):**
```bash
# Solution: Use a newer version
pip install sentencepiece>=0.2.0
```

**Poetry Lock File Issues:**
```bash
# Solution: Update the lock file
poetry lock
poetry install
```

### Performance Issues

**CUDA not available:**
- Ensure CUDA toolkit matches PyTorch version
- Visit https://pytorch.org/get-started/locally/ for correct installation

**MPS not available (Apple Silicon):**
- Update macOS to latest version
- Ensure PyTorch 2.0+ is installed

**Out of Memory:**
- Use smaller models (e.g., distilbert instead of bert-large)
- Reduce batch size in pipeline calls
- Clear GPU cache: `torch.cuda.empty_cache()`

### Environment Configuration

Create a `.env` file from the template:
```bash
cp .env.example .env
```

Configure your environment variables:
```bash
HUGGINGFACE_TOKEN=your_token_here     # For private models
DEFAULT_MODEL=bert-base-uncased       # Default model to use
MAX_LENGTH=512                        # Maximum sequence length
BATCH_SIZE=8                          # Batch processing size
CACHE_DIR=./models                    # Where to cache models
```

## 📚 Key Resources

**Essential Links:**
- 🤗 [Hugging Face Model Hub](https://huggingface.co/models) - Browse thousands of pre-trained models
- 📊 [Datasets Hub](https://huggingface.co/datasets) - Find datasets for training and evaluation
- 🚀 [Spaces](https://huggingface.co/spaces) - Interactive model demos
- 📖 [Documentation](https://huggingface.co/docs) - Official guides and tutorials

**Next Steps:**
- Chapter 4: Dive into transformer architecture and attention mechanisms
- Chapter 5: Comprehensive tokenization coverage (building on our preview)
- Chapter 8: Create custom pipelines and data processing workflows
- Chapter 11-13: Fine-tuning from basic to advanced RLHF techniques

## 🌟 Why This Environment Setup Matters

With this foundation, you can:
- **Prototype in minutes**: Test ideas with pre-trained models instantly
- **Scale effortlessly**: Same code works on CPU, GPU, or Apple Silicon
- **Collaborate seamlessly**: Share models and code with reproducible environments
- **Learn continuously**: Working examples guide you from basics to advanced techniques

## 💡 Your Journey Starts Here

You now have a professional-grade Hugging Face development environment. The combination of proper setup, working examples, and best practices positions you to tackle real-world AI challenges effectively.

**Remember:** This chapter focused on environment setup with a preview of tokenization. When you're ready for deep tokenization knowledge, Chapter 5 provides comprehensive coverage of all concepts introduced here.

---

*Built with ❤️ for the Hugging Face community*


---
## Documentation

- [Taskfile.md](docs/Taskfile.md)
- [__init__.md](docs/__init__.md)
- [advanced_tokenization.md](docs/advanced_tokenization.md)
- [article.md](docs/article.md)
- [article_orginal.md](docs/article_orginal.md)
- [basic_tokenization.md](docs/basic_tokenization.md)
- [changes.md](docs/changes.md)
- [config.md](docs/config.md)
- [hf_hub_example.md](docs/hf_hub_example.md)
- [main.md](docs/main.md)
- [model_download_example.md](docs/model_download_example.md)
- [pipeline_example.md](docs/pipeline_example.md)
- [project_analysis.md](docs/project_analysis.md)
- [run_all_examples.md](docs/run_all_examples.md)
- [speech_recognition_example.md](docs/speech_recognition_example.md)
- [subword_tokenization.md](docs/subword_tokenization.md)
- [test_oov_handling.md](docs/test_oov_handling.md)
- [tokenizer_comparison.md](docs/tokenizer_comparison.md)
- [translation_example.md](docs/translation_example.md)
- [utils.md](docs/utils.md)
- [verify_installation.md](docs/verify_installation.md)
