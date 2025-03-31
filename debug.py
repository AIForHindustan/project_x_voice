from project_x_voice.data_processor import BilingualDataset
import json
import pandas as pd
from pathlib import Path
import sys
import torch

sys.path.append('.')  # Ensure the project root is in Python path


# Load dataset
config_path = Path('config/config.json')
metadata_path = Path('data/metadata.csv')
wav_dir = Path('data/wavs')

with open(config_path, 'r') as f:
    config = json.load(f)

dataset = BilingualDataset(metadata_path, wav_dir, config)

# Debugging: Print the first dataset sample structure
sample = dataset[0]
print('DEBUG: dataset[0] structure:', sample)

# Process dataset
for i in range(len(dataset)):
    sample = dataset[i]

    if not isinstance(sample, tuple) or len(sample) < 2:
        print(f'Unexpected sample format at index {i}: {sample}')
        continue  # Skip if format is incorrect

    text_tensor = sample[0]  # Assuming text is stored in the first tensor
    # Assuming phonemes are stored in the second tensor
    phonemes_tensor = sample[1]

    # Convert tensor to list of indices
    text_indices = text_tensor.tolist()
    phoneme_indices = phonemes_tensor.tolist()

    # Convert indices to characters using a vocab lookup (modify based on
    # actual vocab)
    vocab = {109: "hello", 71: "world", 46: "this",
             123: "test"}  # Replace with actual vocab
    text = " ".join(vocab.get(idx, str(idx)) for idx in text_indices)

    if isinstance(phoneme_indices, int):  # Convert single int to a list
    phoneme_indices = [phoneme_indices]

    phonemes = " ".join(vocab.get(idx, str(idx)) for idx in phoneme_indices)

    if len(text.split()) != len(phonemes.split()):
        print(f'Mismatch at index {i}:')
        print(f'Original Text: {text}')
        print(f'Generated Phonemes: {phonemes}')
        print('-' * 50)
