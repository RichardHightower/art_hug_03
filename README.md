# Tokenization and Text Processing Fundamentals

This project contains working examples for Chapter 03 of the Hugging Face Transformers book: **Setting Up Your Hugging Face Environment**.

## Overview

Learn how to set up and use the Hugging Face ecosystem with hands-on examples covering:
- Environment setup (Poetry, Conda, pip)
- Model pipelines for various NLP tasks
- HuggingFace Hub API integration
- Model downloading and caching
- Translation and speech recognition
- GPU/MPS optimization

## Prerequisites

- Python 3.10+ (3.12 recommended)
- Poetry for dependency management OR Conda
- Go Task for build automation
- (Optional) CUDA-capable GPU or Apple Silicon Mac for acceleration
- (Optional) HuggingFace account for model access

## Setup Options

### Option 1: Poetry Setup (Recommended)
```bash
# Full setup with Poetry
task setup-complete

# Or step by step:
task setup              # Install Python and dependencies
task hf-install-hub     # Install HuggingFace Hub
task export-requirements # Export requirements.txt
```

### Option 2: Conda Setup
```bash
# Create conda environment
task conda-setup
conda activate hf-env

# Install dependencies
pip install -r requirements.txt
```

### GPU Support

For NVIDIA GPUs (CUDA):
```bash
task install-cuda
```

For Apple Silicon (MPS):
```bash
task install-mps
```

### Authentication

To access private models or push to Hub:
```bash
task hf-login
```

## Verification

Verify your installation:
```bash
# Quick check
task verify-imports

# Detailed verification
task verify-setup
```

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── config.py                    # Configuration and device detection
│   ├── main.py                      # Entry point with all examples
│   ├── verify_installation.py       # Environment verification
│   ├── pipeline_example.py          # HF Pipeline demonstrations
│   ├── hf_hub_example.py           # HuggingFace Hub API usage
│   ├── model_download_example.py    # Model downloading and caching
│   ├── translation_example.py       # Translation pipelines
│   ├── speech_recognition_example.py # ASR examples
│   ├── basic_tokenization.py        # Basic tokenization
│   ├── subword_tokenization.py      # Subword tokenization
│   ├── advanced_tokenization.py     # Advanced tokenization
│   ├── tokenizer_comparison.py      # Tokenizer comparison
│   └── utils.py                     # Utility functions
├── tests/
│   └── test_examples.py             # Unit tests
├── docs/
│   └── changes.md                   # Chapter 3 additions
├── .env.example                     # Environment template
├── Taskfile.yml                     # Task automation
├── pyproject.toml                   # Poetry configuration
└── requirements.txt                 # Exported dependencies
```

## Running Examples

### Environment Examples
```bash
task verify-setup              # Verify installation
task run-pipeline              # Run pipeline examples
task run-hub-api              # Run Hub API examples
task run-model-download       # Run model download examples
task run-translation          # Run translation examples
task run-speech               # Run speech recognition examples
```

### Tokenization Examples
```bash
task run                      # Run all tokenization examples
task run-basic-tokenization   # Run basic tokenization
task run-subword-tokenization # Run subword tokenization
task run-advanced-tokenization # Run advanced tokenization
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

## Testing with Spaces

To test your models with HuggingFace Spaces:

1. **Create a Space**: Visit [huggingface.co/spaces](https://huggingface.co/spaces) and click "Create new Space"
2. **Choose SDK**: Select Gradio, Streamlit, or Static
3. **Deploy Your Model**: 
   ```python
   # Example app.py for Gradio Space
   import gradio as gr
   from transformers import pipeline
   
   classifier = pipeline("sentiment-analysis")
   
   def predict(text):
       return classifier(text)[0]
   
   iface = gr.Interface(
       fn=predict,
       inputs="text",
       outputs="json",
       title="Sentiment Analysis Demo"
   )
   
   iface.launch()
   ```
4. **Push to Space**: Use git or the web interface to deploy

## Troubleshooting

### Common Issues

1. **CUDA not available**: Ensure CUDA toolkit is installed and compatible with PyTorch
2. **MPS not available**: Update macOS and PyTorch to latest versions
3. **Model download fails**: Check internet connection and HF authentication
4. **Out of memory**: Use smaller models or reduce batch size

### Environment Variables

Set in `.env` file:
```bash
HUGGING_FACE_HUB_TOKEN=your_token_here
DEFAULT_MODEL=bert-base-uncased
MAX_LENGTH=512
BATCH_SIZE=8
CACHE_DIR=./models
```

## Learn More

- [Hugging Face Documentation](https://huggingface.co/docs)
- [Transformers Library](https://github.com/huggingface/transformers)
- [Model Hub](https://huggingface.co/models)
- [Datasets Hub](https://huggingface.co/datasets)
- [Spaces Documentation](https://huggingface.co/docs/hub/spaces)
