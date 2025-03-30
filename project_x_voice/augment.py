# scripts/augment_audio.py
import os
import argparse
import torchaudio
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch

def augment_audio(input_dir, output_dir, num_augments=3):
    os.makedirs(output_dir, exist_ok=True)
    
    augmenter = Compose([
        PitchShift(min_semitones=-4, max_semitones=4, p=0.8),
        TimeStretch(min_rate=0.85, max_rate=1.15, p=0.75),
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5)
    ])
    
    for file in os.listdir(input_dir):
        if file.endswith(".wav"):
            waveform, sr = torchaudio.load(os.path.join(input_dir, file))
            for i in range(num_augments):
                augmented = augmenter(waveform.numpy(), sample_rate=sr)
                new_name = f"aug_{i}_{file}"
                torchaudio.save(os.path.join(output_dir, new_name), torch.Tensor(augmented), sr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/wavs", help="Original WAV directory")
    parser.add_argument("--output_dir", default="data/augmented_wavs", help="Augmented output directory")
    parser.add_argument("--num_augments", type=int, default=3, help="Variants per file")
    args = parser.parse_args()
    
    augment_audio(args.input_dir, args.output_dir, args.num_augments)