import os
import json
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from phonemizer import phonemize

class BilingualDataset(Dataset):
    def __init__(self, metadata_path, data_dir, durations_path, config):
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadata_path)
        self.durations = pd.read_csv(durations_path)
        self.config = config
        self.phoneme_map = self._load_phoneme_map()

    def _load_phoneme_map(self):
        phoneme_path = Path(__file__).parent.parent / "configs" / "phonemes.json"
        with open(phoneme_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _phonemize(self, text, lang):
        return phonemize(text, language=lang, backend="espeak", preserve_punctuation=True)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        phonemes = self._phonemize(row["text"], row["language"])
        durations = self.durations[self.durations.utterance_id == row["utterance_id"]]
        
        return {
            "phonemes": torch.tensor([self.phoneme_map[p] for p in phonemes.split()], dtype=torch.long),
            "durations": torch.tensor(durations["duration_frames"].values, dtype=torch.long),
            "mel_path": str(self.data_dir / "mels" / f"{row['utterance_id']}.pt"),
            "language": row["language"],
            "speaker_id": torch.tensor(row["speaker_id"], dtype=torch.long)
        }

def collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    
    padded_phonemes = pad_sequence([item["phonemes"] for item in batch], batch_first=True, padding_value=0)
    padded_durations = pad_sequence([item["durations"] for item in batch], batch_first=True, padding_value=0)
    
    return {
        "phonemes": padded_phonemes,
        "durations": padded_durations,
        "mel": torch.stack([torch.load(item["mel_path"]) for item in batch]),
        "languages": [item["language"] for item in batch],
        "speaker_ids": torch.stack([item["speaker_id"] for item in batch])
    }