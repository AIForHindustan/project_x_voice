import torch
import torch.nn as nn
import logging

# Define BilingualTTS Model
class BilingualTTS(nn.Module):
    def __init__(self, config, phoneme_to_id):
        super(BilingualTTS, self).__init__()
        
        # Model parameters from config
        self.n_mels = config["model"]["n_mels"]  # Number of Mel-spectrogram bins
        self.hidden_dim = config["model"]["hidden_dim"]  # Hidden layer dimension
        
        # Define the model layers
        self.phoneme_embedding = nn.Embedding(len(phoneme_to_id), self.hidden_dim)  # Embedding layer for phonemes
        self.lang_embedding = nn.Embedding(len(config["languages"]), self.hidden_dim)  # Embedding layer for languages
        self.emotion_embedding = nn.Embedding(len(config["emotions"]), self.hidden_dim)  # Embedding layer for emotions
        
        # LSTM layer to process embeddings (sequence modeling)
        self.lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, batch_first=True)
        
        # Fully connected layer to output Mel-spectrogram prediction
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.n_mels),
            nn.ReLU(),
        )

        logging.info("BilingualTTS model initialized.")

    def forward(self, phonemes, text_lengths, lang_ids, emotion_ids):
        """
        Forward pass through the model.

        Args:
            phonemes: Tensor of shape [B, T], containing phoneme indices.
            text_lengths: Tensor of shape [B], containing lengths of the input text sequences.
            lang_ids: Tensor of shape [B], containing language IDs.
            emotion_ids: Tensor of shape [B], containing emotion IDs.
        
        Returns:
            mel_pred: Predicted Mel-spectrogram with shape [B, n_mels, T]
        """
        # Embedding layers
        phoneme_embeds = self.phoneme_embedding(phonemes)  # [B, T, hidden_dim]
        lang_embeds = self.lang_embedding(lang_ids).unsqueeze(1)  # [B, 1, hidden_dim]
        emotion_embeds = self.emotion_embedding(emotion_ids).unsqueeze(1)  # [B, 1, hidden_dim]

        # Combine embeddings (concatenate across feature dimension)
        combined = phoneme_embeds + lang_embeds + emotion_embeds  # [B, T, hidden_dim]

        # LSTM processing
        lstm_out, _ = self.lstm(combined)  # LSTM output: [B, T, hidden_dim]

        # Decoder to predict Mel-spectrogram
        mel_pred = self.decoder(lstm_out)  # [B, T, n_mels]
        
        # Transpose to match expected output shape [B, n_mels, T]
        return mel_pred.transpose(1, 2)

# Compute loss function (Mean Squared Error Loss)
def compute_loss(mel_pred, mel_target):
    """
    Compute Mean Squared Error loss between predicted Mel-spectrogram and target.

    Args:
        mel_pred: Predicted Mel-spectrogram [B, n_mels, T].
        mel_target: Ground truth Mel-spectrogram [B, n_mels, T].
    
    Returns:
        loss: Computed loss value.
    """
    mel_target = mel_target.transpose(1, 2)  # Ensure correct shape: [B, n_mels, T]
    criterion = nn.MSELoss()
    loss = criterion(mel_pred, mel_target)
    logging.info(f"Computed loss: {loss.item():.4f}")
    return loss
