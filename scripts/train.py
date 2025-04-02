import json
import logging
import os
import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn.functional as F

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

# Compute loss function
def compute_loss(mel_pred, mel, mel_lengths):
    # Ensure the predicted mel spectrogram and the target mel spectrogram have the same dimensions
    mel_pred = mel_pred[:, :, :mel.size(2)]  # Trim to match the dimensions of mel
    
    # Calculate the L1 loss between predicted and actual mel spectrograms
    loss = F.l1_loss(mel_pred, mel, reduction='mean')

    # Optionally, if mel_lengths are provided, you can weight the loss by sequence lengths
    weighted_loss = loss * mel_lengths.float().mean()  # This is just an example of how you might use mel_lengths

    return weighted_loss

def train():
    # Get dynamic project root
    project_root = Path(get_base_dir())
    config_path = project_root / "config" / "config.json"
    phoneme_path = project_root / "config" / "phonemes.json"
    metadata_path = project_root / "data" / "metadata.csv"
    wav_dir = project_root / "data" / "wavs"
    checkpoint_dir = project_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load configurations
    with open(config_path, "r") as f:
        config = json.load(f)
    with open(phoneme_path, "r") as f:
        phoneme_to_id = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Initialize dataset
    dataset = BilingualDataset(config=config)
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        collate_fn=collate_fn,
        shuffle=True  # Optional: Change this to True for better training behavior
    )

    print(f"Dataset size: {len(dataset)}")

    # Initialize model
    model = BilingualTTS(config, phoneme_to_id=phoneme_to_id).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )

    # Training loop
    for epoch in range(config["training"]["epochs"]):
        print(f"Starting Epoch {epoch+1}")
        model.train()
        total_loss = 0.0

        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()

            # Prepare batch data
            phonemes = batch["phonemes"].to(device)
            text_lengths = batch["text_lengths"].to(device)
            lang_ids = batch["lang_ids"].to(device)
            emotion_ids = batch["emotion_ids"].to(device)
            mel = batch["mel"].to(device).transpose(1, 2)  # Ensure correct shape

            # Forward pass
            mel_pred = model(phonemes, text_lengths, lang_ids, emotion_ids)

            # Debugging tensor sizes every 100 batches
            if i % 100 == 0:
                print(f"Batch {i}: mel_pred shape {mel_pred.shape}, mel shape {mel.shape}")

            # Compute loss
            loss = compute_loss(mel_pred, mel, batch["mel_lengths"])

            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            logging.info(f"Epoch {epoch+1}/{config['training']['epochs']} | Batch Loss: {loss.item():.4f}")

        # Average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        logging.info(f"Epoch {epoch+1} | Average Loss: {avg_loss:.4f}")

        # Save model checkpoint
        checkpoint_path = checkpoint_dir / f"epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        logging.info(f"Model checkpoint saved at {checkpoint_path}")


if __name__ == "__main__":
    train()
