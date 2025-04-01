import torch
import json
from torch.utils.data import DataLoader
from pathlib import Path
from model import BilingualTTS  # Import model from model.py
from data_processor import BilingualDataset, collate_fn  # Import dataset and collate_fn from data_processor.py

def train():
    # Load config file from Google Drive
    config = json.load(open(Path('/content/drive/MyDrive/project_x_voice/config/config.json')))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the phoneme_to_id mapping from phoneme.json
    phoneme_to_id = json.load(open(Path('/content/drive/MyDrive/project_x_voice/config/phonemes.json')))
    
    # Initialize dataset
    dataset = BilingualDataset(
        metadata_path=Path('/content/drive/MyDrive/project_x_voice/data/metadata.csv'),
        wav_dir=Path('/content/drive/MyDrive/project_x_voice/data/wavs'),
        config=config
    )
    
    # Initialize the Bilingual TTS model with the phoneme_to_id mapping
    model = BilingualTTS(config, phoneme_to_id=phoneme_to_id).to(device)  # Pass phoneme_to_id here
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    
    # Training loop
    for epoch in range(config["training"]["epochs"]):
        model.train()
        total_loss = 0.0
        for batch in DataLoader(dataset, batch_size=config["training"]["batch_size"], collate_fn=collate_fn):
            optimizer.zero_grad()
            
            # Prepare the batch data
            phonemes = batch["phonemes"].to(device)
            text_lengths = batch["text_lengths"].to(device)
            lang_ids = batch["lang_ids"].to(device)
            emotion_ids = batch["emotion_ids"].to(device)  # Ensure this is in the batch
            mel = batch["mel"].to(device).transpose(1, 2)  # [B, n_mels, T]
            
            # Forward pass through the model
            mel_pred = model(
                phonemes, text_lengths, lang_ids, emotion_ids
            )
            
            # Compute loss (Mean Squared Error Loss)
            loss = torch.nn.functional.mse_loss(mel_pred, mel)
            
            # Backpropagate and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Log the loss for the current batch
            print(f"Epoch {epoch+1}/{config['training']['epochs']} | Batch Loss: {loss.item():.4f}")
        
        # Average loss for the epoch
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1} | Average Loss: {avg_loss:.4f}")
        
        # Save the model checkpoint after each epoch
        checkpoint_path = Path('/content/drive/MyDrive/project_x_voice/checkpoints') / f"epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")

if __name__ == "__main__":
    train()
