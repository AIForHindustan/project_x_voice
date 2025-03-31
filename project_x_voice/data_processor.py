import os
import json
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
from phonemizer import phonemize
from epitran import Epitran
from torch.nn.utils.rnn import pad_sequence
from indicnlp.normalize.indic_normalize import DevanagariNormalizer

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
        return mel.squeeze(0).T  # shape: (time, n_mels)

class BilingualDataset(Dataset):
    def __init__(self, metadata_path, wav_dir, config):
        self.config = config
        # Using the correct metadata path (one level up)
        self.metadata = pd.read_csv('../data/metadata.csv')
        self.wav_dir = Path(wav_dir)
        self.mel_processor = MelProcessor(config)
        self.epi_hin = Epitran('hin-Deva')
        self.normalizer = DevanagariNormalizer()
        
        # Load phoneme mapping
        phoneme_map_path = Path(__file__).resolve().parent.parent / "config" / "phonemes.json"
        with open(phoneme_map_path, "r", encoding="utf-8") as f:
            self.phoneme_map = json.load(f)
        
        self.processed_data = self._preprocess_data()

    def _text_to_phonemes(self, text, lang):
        try:
            if lang.lower() == "en":
                phonemes = phonemize(
                    text,
                    language='en-us',
                    backend='espeak',
                    strip=True,
                    preserve_punctuation=False,
                    with_stress=False
                ).split()
            elif lang.lower() == "hi":
                # Normalize the Hindi text
                normalized_text = self.normalizer.normalize(text)
                # Split the sentence into words
                words = normalized_text.split()
                phonemes = []
                for word in words:
                    # Use transliterate to get IPA string for the word
                    ipa_str = self.epi_hin.transliterate(word)
                    # Remove spaces and split into individual characters (tokens)
                    tokens = [char for char in ipa_str if char != ' ']
                    phonemes.extend(tokens)
            else:
                phonemes = []
            
            print(f"Processed phonemes for {lang}: {phonemes}")
            # Map phonemes to IDs; use <unk> if not found.
            return [self.phoneme_map.get(p, self.phoneme_map.get("<unk>", 0)) for p in phonemes]
        except Exception as e:
            print(f"Phonemization error for text '{text}' in language {lang}: {str(e)}")
            return [self.phoneme_map.get("<unk>", 0)]

    def _preprocess_data(self):
        processed = []
        for idx, row in self.metadata.iterrows():
            wav_path = self.wav_dir / row["audio_file"]
            phoneme_ids = self._text_to_phonemes(row["text"], row["language"])
            
            if len(phoneme_ids) == 0:
                phoneme_ids = [self.phoneme_map.get("<unk>", 0)]
            
            try:
                mel = self.mel_processor.wav_to_mel(wav_path)
            except Exception as e:
                print(f"Error processing MEL for {wav_path}: {e}")
                continue
            
            processed.append({
                "phonemes": torch.tensor(phoneme_ids, dtype=torch.long),
                "mel": mel,
                "lang_id": torch.tensor(1 if row["language"].lower() == "hi" else 0, dtype=torch.long),
                "text_length": torch.tensor(len(phoneme_ids), dtype=torch.long),
                "mel_length": torch.tensor(mel.size(0), dtype=torch.long)
            })
            
            if idx % 10 == 0:
                print(f"Processed sample {idx}/{len(self.metadata)}")
        return processed

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        item = self.processed_data[idx]
        return (
            item["phonemes"],
            item["text_length"],
            item["lang_id"],
            item["mel"],
            item["mel_length"]
        )

def collate_fn(batch):
    phonemes, text_lengths, lang_ids, mels, mel_lengths = zip(*batch)
    phonemes_padded = pad_sequence(phonemes, batch_first=True, padding_value=0)
    mels_padded = pad_sequence(mels, batch_first=True, padding_value=-80.0)
    text_lengths = torch.stack(text_lengths)
    lang_ids = torch.stack(lang_ids)
    mel_lengths = torch.stack(mel_lengths)
    return {
        "phonemes": phonemes_padded,
        "text_lengths": text_lengths,
        "lang_ids": lang_ids,
        "mel": mels_padded,
        "mel_lengths": mel_lengths
    }
