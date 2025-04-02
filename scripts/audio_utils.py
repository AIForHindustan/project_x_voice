import torch
import torchaudio
import numpy as np
from pathlib import Path


class MelProcessor:
    def __init__(self, config):
        self.sample_rate = config["data"]["sample_rate"]
        self.n_fft = config["data"]["n_fft"]
        self.hop_length = config["data"]["hop_length"]
        self.n_mels = config["model"]["n_mels"]
        self.mel_fmin = config["vocoder"]["mel_fmin"]
        self.mel_fmax = config["vocoder"]["mel_fmax"]

        # Create Mel filterbank
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.mel_fmin,
            f_max=self.mel_fmax,
            normalized=True
        )

        # Load normalization stats from config
        self.mean = torch.tensor(config["data"]["mel_mean"])
        self.std = torch.tensor(config["data"]["mel_std"])

    def _load_wav(self, path):
        waveform, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        return waveform

    def _normalize(self, mel):
        return (mel - self.mean) / self.std

    def wav2mel(self, wav_path):
        waveform = self._load_wav(wav_path)
        mel = self.mel_transform(waveform)
        return self._normalize(mel.squeeze(0)).T  # (time, n_mels)

# Usage in data_processor.py


def load_mels(batch_wav_paths):
    processor = MelProcessor(config)
    return torch.stack([processor.wav2mel(p) for p in batch_wav_paths])
