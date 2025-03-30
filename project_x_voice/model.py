import torch
import torch.nn as nn

class BilingualTTS(nn.Module):
    def __init__(self, config, phoneme_to_id):
        super().__init__()

        # Extract model config
        model_cfg = config["model"]

        # Define attributes
        self.vocab_size = len(phoneme_to_id)  # Use dynamic vocab size
        self.embedding = nn.Embedding(self.vocab_size, model_cfg["input_dim"])
        self.lang_embed = nn.Embedding(len(model_cfg["languages"]), model_cfg["language_embedding_dim"])
        self.emo_embed = nn.Embedding(model_cfg["num_emotions"], model_cfg["emotion_embedding_dim"])
        self.spk_embed = nn.Embedding(model_cfg["num_speakers"], model_cfg["speaker_embedding_dim"])

        # Text encoder
        self.encoder = nn.LSTM(
            input_size=model_cfg["input_dim"],
            hidden_size=model_cfg["hidden_dim"],
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg["dropout"],
            batch_first=True
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(model_cfg["hidden_dim"], model_cfg["n_mels"]),
            nn.ReLU()  # Better activation for mel spectrograms
        )

    def forward(self, phonemes, text_lens, langs, emotions, speakers):
        # Convert phoneme indices to embeddings
        embedded = self.embedding(phonemes)

        # Embeddings for language, emotion, and speaker
        lang_emb = self.lang_embed(langs).unsqueeze(1)  # Expand to [batch, 1, dim]
        emo_emb = self.emo_embed(emotions).unsqueeze(1)
        spk_emb = self.spk_embed(speakers).unsqueeze(1)

        # Pack and encode text
        text_lens_cpu = text_lens.cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lens_cpu, batch_first=True, enforce_sorted=False
        )
        encoded, _ = self.encoder(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True)

        # Combine features
        combined = encoded + lang_emb + emo_emb + spk_emb

        # Decode to mel spectrogram
        mel = self.decoder(combined).transpose(1, 2)
        return mel
