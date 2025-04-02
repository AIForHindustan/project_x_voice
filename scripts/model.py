import torch
import torch.nn as nn
import logging
import os
import sys

# Append the project root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from path_manager import get_base_dir  # Correct dynamic path management

# Define BilingualTTS Model
class BilingualTTS(nn.Module):
    def __init__(self, config, phoneme_to_id):
        super(BilingualTTS, self).__init__()
        self.n_mels = config["model"]["n_mels"]
        self.hidden_dim = config["model"]["hidden_dim"]

        # Embeddings
        self.phoneme_embedding = nn.Embedding(len(phoneme_to_id), self.hidden_dim)
        self.lang_embedding = nn.Embedding(len(config["model"]["languages"]), self.hidden_dim)
        self.emotion_embedding = nn.Embedding(config["model"]["num_emotions"], self.hidden_dim)

        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            batch_first=True
        )

        # Decoder to predict Mel-spectrograms
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.n_mels),
            nn.ReLU()
        )

        # Upsampling layer with corrected parameters
        self.upsample = nn.ConvTranspose1d(
            in_channels=self.n_mels,
            out_channels=self.n_mels,
            kernel_size=3,
            stride=12,
            padding=1,
            output_padding=0
        )

        logging.info("BilingualTTS model initialized.")

    def forward(self, phonemes, text_lengths, lang_ids, emotion_ids):
        print(f"Input phonemes shape: {phonemes.shape}")  # Debug print

        # Embed phonemes, language, and emotion
        phoneme_embeds = self.phoneme_embedding(phonemes)  # [B, T, H]
        print(f"Phoneme embeddings shape: {phoneme_embeds.shape}")

        lang_embeds = self.lang_embedding(lang_ids).unsqueeze(1)  # [B, 1, H]
        emotion_embeds = self.emotion_embedding(emotion_ids).unsqueeze(1)  # [B, 1, H]

        # Combine embeddings
        combined = phoneme_embeds + lang_embeds + emotion_embeds  # [B, T, H]
        print(f"Combined embeddings shape: {combined.shape}")

        # Process through LSTM
        lstm_out, _ = self.lstm(combined)  # [B, T, H]
        print(f"LSTM output shape: {lstm_out.shape}")

        # Decode to Mel-spectrograms
        mel_pred = self.decoder(lstm_out)  # [B, T, n_mels]
        mel_pred = mel_pred.transpose(1, 2)  # [B, n_mels, T]
        print(f"Mel prediction shape before upsampling: {mel_pred.shape}")

        # Upsample to target length
        mel_pred = self.upsample(mel_pred)  # [B, n_mels, 517]
        print(f"Mel prediction shape after upsampling: {mel_pred.shape}")

        return mel_pred

# Loss function with masking
def compute_loss(mel_pred, mel_target, mel_lengths):
    # Ensure shapes match: [B, n_mels, T]
    mel_target = mel_target.transpose(1, 2)
    assert mel_pred.shape == mel_target.shape, \
        f"Shape mismatch: Pred {mel_pred.shape}, Target {mel_target.shape}"

    # Create mask based on actual lengths
    mask = (torch.arange(mel_pred.size(2), device=mel_pred.device)
              .unsqueeze(0) < mel_lengths.unsqueeze(1)).float()

    # Apply mask to predictions and targets
    masked_pred = mel_pred * mask.unsqueeze(1)
    masked_target = mel_target * mask.unsqueeze(1)

    # Compute MSE loss
    loss = torch.nn.functional.mse_loss(masked_pred, masked_target)
    logging.info(f"Loss: {loss.item():.4f}")
    return loss
