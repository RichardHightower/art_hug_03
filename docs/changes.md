# Chapter 3 Additions: Setting Up Your Hugging Face Environment

This document outlines all additions made to implement examples from Chapter 3 of the HuggingFace book.

## Taskfile.yml Updates

### Updated Tasks
- `run`: Now runs interactive menu from updated main.py
- `run-all`: New task for non-interactive execution of all examples

### Conda Setup Tasks
- `conda-create`: Create conda environment for Hugging Face
- `conda-setup`: Alternative setup using conda with Python 3.10

### Dependency Management
- `export-requirements`: Export requirements.txt from Poetry
- `hf-install-hub`: Install HuggingFace Hub package

### HuggingFace Authentication
- `hf-login`: Login to HuggingFace Hub using CLI

### Verification Tasks
- `verify-setup`: Verify HuggingFace environment setup (runs verify_installation.py)
- `verify-imports`: Quick verification of imports (inline Python)

### GPU Support
- `install-cuda`: Install PyTorch with CUDA support for NVIDIA GPUs
- `install-mps`: Install PyTorch for Apple Silicon (MPS)

### Example Running Tasks
- `run-pipeline`: Run pipeline examples
- `run-hub-api`: Run HuggingFace Hub API examples
- `run-model-download`: Run model download examples
- `run-translation`: Run translation examples
- `run-speech`: Run speech recognition examples

### Complete Setup
- `setup-complete`: Complete setup with all dependencies

## Updated and New Python Files in src/

### Updated Files

1. **`main.py`** - Enhanced to run ALL examples
   - Now includes interactive menu
   - Runs both environment setup and tokenization examples
   - Options to run specific categories or all examples
   - Includes all new Chapter 3 examples

### New Files

### 1. `verify_installation.py`
- Comprehensive environment verification script
- Checks all required packages and versions
- Displays PyTorch device information (CUDA/MPS)
- Shows Python environment details

### 2. `pipeline_example.py`
- Sentiment analysis pipeline
- Text generation with distilgpt2
- Zero-shot classification
- Question answering pipeline
- Device detection and usage

### 3. `hf_hub_example.py`
- List text classification models using task parameter
- Get detailed model information
- Search models by language
- Filter models by library/framework
- Search models trained on specific datasets
- Check authentication status
- Note: Updated to work with newer huggingface_hub API (ModelFilter removed)

### 4. `model_download_example.py`
- Download sentiment analysis model (distilbert)
- Download multilingual BERT model
- Download specialized models (QA, NER)
- Show cache directory information
- Download specific model revisions
- Display model parameters and configuration

### 5. `translation_example.py`
- Basic English to French translation
- Multilingual translation (EN→DE, EN→ES, EN→ZH)
- Batch translation performance comparison
- Translation with different options (max_length, beam search)
- Back-translation (round-trip) example

### 6. `speech_recognition_example.py`
- Basic speech recognition with wav2vec2
- Speech recognition with chunk processing
- Multilingual ASR model information
- Supported audio format documentation
- Real-world usage examples
- Performance considerations

### 7. `test_oov_handling.py`
- Test out-of-vocabulary word handling
- Compare BERT, GPT-2, and RoBERTa approaches
- Test special characters and emojis
- Detailed analysis of subword tokenization

### 8. `run_all_examples.py`
- Non-interactive script to run all examples
- Used by `task run-all` command
- Runs both environment and tokenization examples

## README.md Updates

### New Sections Added
1. **Comprehensive Overview**: Updated to reflect Chapter 3 content
2. **Setup Options**: Both Poetry and Conda setup instructions
3. **GPU Support**: Instructions for CUDA and Apple Silicon
4. **Authentication**: HuggingFace Hub login instructions
5. **Verification**: How to verify installation
6. **Extended Project Structure**: Added new example files
7. **Environment Examples**: New task commands for running examples
8. **Testing with Spaces**: Step-by-step guide for HuggingFace Spaces
9. **Troubleshooting**: Common issues and solutions
10. **Environment Variables**: Configuration options

## Jupyter Notebooks Added

### 1. `Chapter_03_Environment_Setup.ipynb`
A comprehensive notebook covering all Chapter 3 environment setup examples:
- Environment verification
- Basic pipeline examples (sentiment analysis)
- HuggingFace Hub API usage
- Model downloading and caching
- Translation pipelines
- Text generation
- Zero-shot classification
- Question answering
- Named Entity Recognition
- Model comparison
- Batch processing performance
- Cache information
- HuggingFace Spaces example code

### 2. `Chapter_03_Tokenization_Fixed.ipynb`
A complete tokenization tutorial notebook:
- Basic tokenization with BERT
- Token ID conversion
- Subword tokenization comparison (WordPiece, BPE, SentencePiece)
- Advanced features (padding, truncation, special tokens)
- Tokenizer comparison across models
- Visualization of token counts
- Special tokens and token types
- Out-of-vocabulary word handling
- Performance measurements

Note: The original `Chapter_03_Tokenization_Notebook.ipynb` was corrupted and has been renamed with `.corrupted` extension.

## Key Features Implemented

### 1. Environment Setup
- Multiple setup paths (Poetry, Conda, pip)
- Automatic Python version management with pyenv
- Dependency export to requirements.txt

### 2. Device Support
- Automatic device detection in config.py
- CUDA support for NVIDIA GPUs
- MPS support for Apple Silicon
- CPU fallback

### 3. Model Examples
- Various pipeline types (sentiment, generation, QA, etc.)
- Model downloading and caching
- Batch processing demonstrations
- Performance comparisons

### 4. HuggingFace Hub Integration
- API client usage
- Model discovery and filtering
- Authentication handling
- Model metadata access

### 5. Real-World Applications
- Translation with multiple languages
- Speech recognition setup
- Batch processing optimization
- Error handling examples

## Dependencies Added

All examples use the existing dependencies in pyproject.toml:
- transformers
- torch
- huggingface_hub
- datasets
- accelerate
- sentencepiece
- tokenizers

Additional dependencies for notebook visualizations:
- matplotlib - For creating plots and charts
- seaborn - For enhanced statistical visualizations
- pandas - For data manipulation (already present)

Core HuggingFace dependencies added:
- datasets - For loading and processing datasets
- accelerate - For distributed training and mixed precision

## Usage Instructions

1. **Setup Environment**:
   ```bash
   task setup-complete
   ```

2. **Verify Installation**:
   ```bash
   task verify-setup
   ```

3. **Run Examples**:
   ```bash
   task run-pipeline
   task run-hub-api
   task run-model-download
   task run-translation
   task run-speech
   ```

4. **Export Dependencies**:
   ```bash
   task export-requirements
   ```

All examples are self-contained and can be run independently. They include proper error handling and informative output for learning purposes.