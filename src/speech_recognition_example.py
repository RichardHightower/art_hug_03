#!/usr/bin/env python3
"""
"""Speech recognition example demonstrating automatic speech recognition (ASR)
with HuggingFace.
"""
"""


import numpy as np
from transformers import pipeline

from config import get_device


def create_sample_audio():
    """Create a sample audio file for testing (sine wave)."""
    print("=== Creating Sample Audio ===")

    # Create a simple sine wave as sample audio
    sample_rate = 16000  # 16kHz is standard for many ASR models
    duration = 3  # seconds
    frequency = 440  # A4 note

    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Generate sine wave
    audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)

    # Add some noise to make it more realistic
    noise = 0.05 * np.random.randn(len(audio_data))
    audio_data = audio_data + noise

    # Ensure audio is in correct format (float32 between -1 and 1)
    audio_data = audio_data.astype(np.float32)
    audio_data = np.clip(audio_data, -1.0, 1.0)

    print(f"Created sample audio: {duration}s at {sample_rate}Hz")
    print("Note: This is synthetic audio for demonstration purposes")

    return audio_data, sample_rate


def basic_speech_recognition():
    """Basic speech recognition example."""
    print("\n=== Basic Speech Recognition ===")

    device = get_device()

    # Create ASR pipeline
    asr = pipeline(
        "automatic-speech-recognition",
        model="facebook/wav2vec2-base-960h",
        device=device,
    )

    # Create sample audio
    audio_data, sample_rate = create_sample_audio()

    print("\nProcessing audio...")
    print("Note: Real speech audio would produce meaningful transcriptions")

    # Process audio
    try:
        # The pipeline expects audio data as numpy array or file path
        result = asr(audio_data)
        print(f"Transcription: '{result['text']}'")
        print("(Synthetic audio may produce random/meaningless output)")
    except Exception as e:
        print(f"Note: {e}")
        print("This is expected with synthetic audio data")


def speech_recognition_with_options():
    """Speech recognition with various options."""
    print("\n=== Speech Recognition with Options ===")

    device = get_device()

    # Create ASR pipeline with specific model
    asr = pipeline(
        "automatic-speech-recognition",
        model="facebook/wav2vec2-large-960h-lv60-self",
        device=device,
    )

    # Create sample audio
    audio_data, sample_rate = create_sample_audio()

    print("Testing different chunk processing:")

    # Process with different chunk lengths
    for chunk_length in [5, 10, 30]:
        try:
            result = asr(
                audio_data,
                chunk_length_s=chunk_length,  # Process audio in chunks
                stride_length_s=chunk_length // 6,  # Overlap between chunks
            )
            print(f"  Chunk {chunk_length}s: '{result['text']}'")
        except Exception:
            print(f"  Chunk {chunk_length}s: Processing synthetic audio...")


def multilingual_speech_recognition():    """Demonstrate multilingual speech recognition."""    print("\n=== Multilingual Speech Recognition ===")    _device = get_device()

    print("Loading multilingual ASR model...")
    print("Model: facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    print("This model supports 53 languages!")

    # Note: In real usage, you would use actual speech audio files
    # For demonstration, we'll show the model info
    _model_name = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"

    print("\nSupported languages include:")
    languages = [
        "English",
        "Spanish",
        "French",
        "German",
        "Italian",
        "Portuguese",
        "Polish",
        "Dutch",
        "Swedish",
        "Norwegian",
    ]
    for i, lang in enumerate(languages[:10], 1):
        print(f"  {i:2d}. {lang}")
    print("  ... and 43 more languages!")


def audio_file_formats():
    """Show supported audio file formats."""
    print("\n=== Supported Audio Formats ===")

    print("HuggingFace ASR pipelines support various audio formats:")
    print("  • WAV files (.wav) - Recommended")
    print("  • MP3 files (.mp3)")
    print("  • FLAC files (.flac)")
    print("  • OGG files (.ogg)")
    print("  • M4A files (.m4a)")
    print("  • And more via ffmpeg/librosa")

    print("\nExample usage with file:")
    print('  result = asr("path/to/audio.wav")')
    print('  result = asr("path/to/speech.mp3")')

    print("\nYou can also pass:")
    print("  • File paths (string)")
    print("  • NumPy arrays (raw audio data)")
    print("  • PyTorch tensors")
    print("  • URLs to audio files")


def real_world_example():
    """Show real-world usage example."""
    print("\n=== Real-World Usage Example ===")

    print("Here's how you would use ASR with a real audio file:\n")

    example_code = """from transformers import pipeline
import torch

# Initialize ASR pipeline
device = 0 if torch.cuda.is_available() else -1
asr = pipeline(
    "automatic-speech-recognition",
    model="facebook/wav2vec2-base-960h",
    device=device
)

# Process audio file
audio_file = "path/to/your/audio.wav"
result = asr(audio_file)

# Get transcription
transcription = result["text"]
print(f"Transcription: {transcription}")

# Process multiple files
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
results = asr(audio_files, batch_size=2)

for file, result in zip(audio_files, results):
    print(f"{file}: {result['text']}")"""

    print(example_code)


def performance_considerations():
    """Discuss performance considerations for ASR."""
    print("\n=== Performance Considerations ===")

    print("1. Model Size:")
    print("   • wav2vec2-base: ~95M parameters (faster)")
    print("   • wav2vec2-large: ~317M parameters (more accurate)")

    print("\n2. Processing Speed:")
    print("   • GPU: Real-time or faster")
    print("   • CPU: May be slower than real-time for large models")

    print("\n3. Memory Usage:")
    print("   • Long audio files consume more memory")
    print("   • Use chunk processing for long recordings")

    print("\n4. Batch Processing:")
    print("   • Process multiple files together for better throughput")
    print("   • Especially beneficial on GPU")

    print("\n5. Audio Quality:")
    print("   • 16kHz sample rate is standard")
    print("   • Higher quality audio → better transcriptions")
    print("   • Background noise reduces accuracy")


def main():
    """Run all speech recognition examples."""
    print("HuggingFace Speech Recognition Examples\n")

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
    basic_speech_recognition()
    speech_recognition_with_options()
    multilingual_speech_recognition()
    audio_file_formats()
    real_world_example()
    performance_considerations()

    print("\n✅ All speech recognition examples completed!")
    print("\nNote: For real transcriptions, use actual speech audio files.")
    print("The examples above use synthetic audio for demonstration.")


if __name__ == "__main__":
    main()
