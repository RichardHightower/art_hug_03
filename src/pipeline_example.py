#!/usr/bin/env python3
"""
Basic pipeline example demonstrating HuggingFace Transformers pipelines.
"""

from transformers import pipeline
import torch
from config import get_device


def sentiment_analysis_example():
    """Demonstrate sentiment analysis pipeline."""
    print("=== Sentiment Analysis Example ===")
    
    # Create pipeline with automatic model selection
    classifier = pipeline("sentiment-analysis", device=get_device())
    
    # Test sentences
    sentences = [
        "I love HuggingFace!",
        "This is terrible.",
        "The weather is okay today.",
        "Transformers make NLP so easy!",
    ]
    
    # Run analysis
    results = classifier(sentences)
    
    # Display results
    for sentence, result in zip(sentences, results):
        label = result['label']
        score = result['score']
        print(f'"{sentence}" -> {label} (confidence: {score:.3f})')
    print()


def text_generation_example():
    """Demonstrate text generation pipeline."""
    print("=== Text Generation Example ===")
    
    # Create pipeline for text generation
    generator = pipeline(
        "text-generation",
        model="distilgpt2",
        device=get_device(),
        max_length=50
    )
    
    prompts = [
        "HuggingFace is",
        "The future of AI is",
    ]
    
    for prompt in prompts:
        print(f'Prompt: "{prompt}"')
        result = generator(prompt, num_return_sequences=1)
        print(f'Generated: "{result[0]["generated_text"]}"')
        print()


def zero_shot_classification_example():
    """Demonstrate zero-shot classification pipeline."""
    print("=== Zero-Shot Classification Example ===")
    
    # Create zero-shot classifier
    classifier = pipeline(
        "zero-shot-classification",
        device=get_device()
    )
    
    # Text to classify
    text = "This tutorial explains how to use transformers for NLP tasks."
    
    # Candidate labels
    candidate_labels = ["education", "politics", "entertainment", "technology"]
    
    # Run classification
    result = classifier(text, candidate_labels)
    
    print(f'Text: "{text}"')
    print("\nClassification results:")
    for label, score in zip(result['labels'], result['scores']):
        print(f"  {label}: {score:.3f}")
    print()


def question_answering_example():
    """Demonstrate question answering pipeline."""
    print("=== Question Answering Example ===")
    
    # Create QA pipeline
    qa_pipeline = pipeline(
        "question-answering",
        device=get_device()
    )
    
    # Context and questions
    context = """
    HuggingFace is a company that develops tools for building applications 
    using machine learning. It is most notable for its Transformers library 
    built for natural language processing applications and its platform that 
    allows users to share machine learning models and datasets.
    """
    
    questions = [
        "What is HuggingFace?",
        "What is HuggingFace known for?",
        "What can users share on the platform?",
    ]
    
    # Answer questions
    for question in questions:
        result = qa_pipeline(question=question, context=context)
        print(f'Q: {question}')
        print(f'A: {result["answer"]} (score: {result["score"]:.3f})')
        print()


def main():
    """Run all pipeline examples."""
    print("HuggingFace Transformers Pipeline Examples\n")
    
    # Check device
    device = get_device()
    if device == "mps":
        device_name = "MPS (Apple Silicon)"
    elif device == 0:
        device_name = "CUDA"
    else:
        device_name = "CPU"
    print(f"Running on: {device_name}\n")
    
    # Run examples
    sentiment_analysis_example()
    text_generation_example()
    zero_shot_classification_example()
    question_answering_example()
    
    print("All examples completed!")


if __name__ == "__main__":
    main()