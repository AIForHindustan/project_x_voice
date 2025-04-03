import json
import logging
import os
import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import GriffinLim

# Ensure project root is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import modules
from path_manager import get_base_dir
from data_processor import BilingualDataset, collate_fn
from model import BilingualTTS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def compute_loss(mel_pred, mel, mel_lengths):
    mel_pred = mel_pred[:, :, :mel.size(2)]  # Trim to match dimensions
    loss = F.l1_loss(mel_pred, mel, reduction='mean')
    return loss * mel_lengths.float().mean()

def train():
    project_root = Path(get_base_dir())
    config_path = project_root / "config" / "config.json"
    phoneme_path = project_root / "config" / "phonemes.json"
    metadata_path = project_root / "data" / "metadata.csv"
    wav_dir = project_root / "data" / "wavs"
    checkpoint_dir = project_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r") as f:
        config = json.load(f)
    with open(phoneme_path, "r") as f:
        phoneme_to_id = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = BilingualDataset(config=config)
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        collate_fn=collate_fn,
        shuffle=True
    )

    print(f"Dataset size: {len(dataset)}")

    model = BilingualTTS(config, phoneme_to_id=phoneme_to_id).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )

    for epoch in range(config["training"]["epochs"]):
        print(f"Starting Epoch {epoch+1}")
        model.train()
        total_loss = 0.0

        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()

            phonemes = batch["phonemes"].to(device)
            text_lengths = batch["text_lengths"].to(device)
            lang_ids = batch["lang_ids"].to(device)
            emotion_ids = batch["emotion_ids"].to(device)
            mel = batch["mel"].to(device).transpose(1, 2)

            mel_pred = model(phonemes, text_lengths, lang_ids, emotion_ids)

            if i % 100 == 0:
                print(f"Batch {i}: mel_pred shape {mel_pred.shape}, mel shape {mel.shape}")

            loss = compute_loss(mel_pred, mel, batch["mel_lengths"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            print(f"Epoch {epoch+1}/{config['training']['epochs']} | Batch Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Average Loss: {avg_loss:.4f}")

        checkpoint_path = checkpoint_dir / f"epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")

    # Inference and saving output audio
    test_phonemes = torch.randint(0, len(phoneme_to_id), (1, 20)).to(device)
    test_text_lengths = torch.tensor([20]).to(device)
    test_lang_ids = torch.tensor([0]).to(device)
    test_emotion_ids = torch.tensor([1]).to(device)

    with torch.no_grad():
        test_mel = model(test_phonemes, test_text_lengths, test_lang_ids, test_emotion_ids)

    # Convert mel spectrogram to waveform using Griffin-Lim
    griffin_lim = GriffinLim(n_iter=32)
    waveform = griffin_lim(test_mel.squeeze(0).cpu())

    # Save the waveform correctly
    torchaudio.save("output.wav", waveform.unsqueeze(0), sample_rate=22050, format="wav", encoding="PCM_S", bits_per_sample=16)
    print("Generated audio saved as output.wav")

if __name__ == "__main__":
    train()
