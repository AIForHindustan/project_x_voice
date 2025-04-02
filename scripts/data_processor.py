import os
import json
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
from phonemizer.backend import EspeakBackend
from epitran import Epitran
from torch.nn.utils.rnn import pad_sequence
from indicnlp.normalize.indic_normalize import DevanagariNormalizer
from path_manager import get_base_dir  # Correct dynamic path management

# Constants for fixed input lengths
MAX_PHONEME_LEN = 44
FIXED_MEL_LEN = 517

# Get dynamic base directory
base_dir = Path(get_base_dir())
config_path = base_dir / "config" / "config.json"
phoneme_path = base_dir / "config" / "phonemes.json"
metadata_path = base_dir / "data" / "metadata.csv"
wav_dir = base_dir / "data" / "wavs"

# Load config dynamically
with open(config_path, "r") as f:
    config = json.load(f)


class MelProcessor:
    def __init__(self, config):
        self.sample_rate = config["data"]["sample_rate"]
        self.n_fft = config["data"]["n_fft"]
        self.hop_length = config["data"]["hop_length"]
        self.n_mels = config["model"]["n_mels"]

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            mel_scale="slaney",
            norm="slaney"
        )

    def wav_to_mel(self, wav_path):
        waveform, sr = torchaudio.load(str(wav_path))
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        mel = self.mel_transform(waveform)
        mel = mel.squeeze(0).T  # Shape: [T, n_mels]

        # Ensure fixed-length mel (pad/truncate to FIXED_MEL_LEN)
        if mel.size(0) < FIXED_MEL_LEN:
            mel = torch.nn.functional.pad(
                mel, (0, 0, 0, FIXED_MEL_LEN - mel.size(0)), value=-80.0)
        else:
            mel = mel[:FIXED_MEL_LEN]
        return mel


class BilingualDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.metadata_path = metadata_path
        self.phoneme_path = phoneme_path

        # Load metadata
        self.metadata = pd.read_csv(self.metadata_path)

        # Load phoneme mapping
        with open(self.phoneme_path, "r", encoding="utf-8") as f:
            self.phoneme_map = json.load(f)

        self.mel_processor = MelProcessor(config)
        self.epi_hin = Epitran('hin-Deva')
        self.normalizer = DevanagariNormalizer()
        self.espeak_backend = EspeakBackend(language="en-us")

        self.processed_data = self._preprocess_data()

    def _text_to_phonemes(self, text, language):
        if language.lower() == "hi":
            text = self.normalizer.normalize(text)
            phonemes = self.epi_hin.transliterate(text)
        else:
            phonemes = self.espeak_backend.phonemize([text])[0]

        phoneme_ids = [
            self.phoneme_map.get(
                p, self.phoneme_map.get(
                    "<unk>", 0)) for p in phonemes]

        # Pad/truncate phoneme sequence to MAX_PHONEME_LEN
        if len(phoneme_ids) < MAX_PHONEME_LEN:
            phoneme_ids.extend([0] * (MAX_PHONEME_LEN - len(phoneme_ids)))
        else:
            phoneme_ids = phoneme_ids[:MAX_PHONEME_LEN]

        return phoneme_ids

    def _preprocess_data(self):
        processed = []
        for idx, row in self.metadata.iterrows():
            try:
                # Ensure that your actual processing steps (e.g., data manipulations) are here.
                # For example:
                wav_path = wav_dir / row['audio_file']
                mel = self.mel_processor.wav_to_mel(wav_path)
                phonemes = self._text_to_phonemes(row['text'], row['language'])
                
                # Add more processing logic here as needed
                processed_sample = {
                    "mel": mel,
                    "phonemes": phonemes,
                    "text_length": torch.tensor(len(phonemes)),
                    "lang_id": torch.tensor(0 if row['language'] == 'en' else 1),  # Example: 0 for English, 1 for Hindi
                    "emotion_ids": torch.tensor(0),  # Assuming emotion IDs need to be defined
                    "mel_length": torch.tensor(mel.size(0))
                }
                processed.append(processed_sample)
            except Exception as e:
                logging.error(f"Skipping sample {row['audio_file']} due to error: {e}")
                continue  # Skip only the failed sample

        # Verify dataset size
        if len(processed) == 0:
            raise RuntimeError("No valid samples processed. Check data and preprocessing.")
        return processed

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        if idx >= len(self.processed_data):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self.processed_data)}")
        return self.processed_data[idx]


def collate_fn(batch):
    phonemes, text_lengths, lang_ids, mels, mel_lengths, emotion_ids = zip(*[
        (
            torch.tensor(item["phonemes"]),  # Convert to tensor
            torch.tensor(item["text_length"]),
            torch.tensor(item["lang_id"]),
            torch.tensor(item["mel"]),
            torch.tensor(item["mel_length"]),
            torch.tensor(item["emotion_ids"])
        )
        for item in batch
    ])

    phonemes_padded = torch.stack(phonemes)  # Now phonemes are tensors
    mels_padded = torch.stack(mels)  # Already fixed-length tensors
    text_lengths = torch.stack(text_lengths)
    lang_ids = torch.stack(lang_ids)
    mel_lengths = torch.stack(mel_lengths)
    emotion_ids = torch.stack(emotion_ids)

    return {
        "phonemes": phonemes_padded,
        "text_lengths": text_lengths,
        "lang_ids": lang_ids,
        "mel": mels_padded,
        "mel_lengths": mel_lengths,
        "emotion_ids": emotion_ids
    }

