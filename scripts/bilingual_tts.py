import torch
import torch.nn as nn
import torch.nn.functional as F


class LocationAwareAttention(nn.Module):
    def __init__(self, hidden_dim, attn_dim):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, attn_dim)
        self.memory_proj = nn.Linear(hidden_dim, attn_dim)
        self.location_conv = nn.Conv1d(1, 10, kernel_size=31, padding=15)
        self.score_proj = nn.Linear(10, attn_dim)

    def forward(self, query, memory, attn_prev):
        energy = torch.tanh(
            self.query_proj(
                query.unsqueeze(1)) +
            self.memory_proj(memory))
        loc_energy = self.score_proj(
            self.location_conv(
                attn_prev.unsqueeze(1)).transpose(
                1, 2))
        energy = energy + loc_energy
        return F.softmax(energy, dim=1)


class BilingualTTS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(
            config["model"]["vocab_size"],
            config["model"]["hidden_dim"])
        self.encoder = nn.LSTM(
            config["model"]["hidden_dim"],
            config["model"]["hidden_dim"] // 2,
            bidirectional=True,
            batch_first=True)

        self.attention = LocationAwareAttention(
            config["model"]["hidden_dim"], 128)
        self.duration_predictor = DurationPredictor(
            config["model"]["hidden_dim"])

        self.decoder = nn.LSTMCell(
            config["model"]["hidden_dim"] +
            config["model"]["n_mels"],
            config["model"]["hidden_dim"])
        self.mel_proj = nn.Linear(
            config["model"]["hidden_dim"],
            config["model"]["n_mels"])

    def forward(self, phonemes, mels):
        embedded = self.embedding(phonemes)
        enc_out, _ = self.encoder(embedded)

        durations = self.duration_predictor(enc_out)
        attn_weights = []
        mel_outputs = []

        hx = torch.zeros(
            phonemes.size(0),
            self.decoder.hidden_size,
            device=phonemes.device)
        cx = torch.zeros_like(hx)
        prev_mel = torch.zeros(
            phonemes.size(0),
            self.config["model"]["n_mels"],
            device=phonemes.device)

        for t in range(mels.size(1)):
            context, attn = self.attention(hx, enc_out, prev_attn)
            decoder_in = torch.cat([context, prev_mel], dim=-1)
            hx, cx = self.decoder(decoder_in, (hx, cx))
            mel_out = self.mel_proj(hx)

            mel_outputs.append(mel_out)
            attn_weights.append(attn)
            prev_mel = mels[:, t] if mels is not None else mel_out

        return torch.stack(
            mel_outputs, 1), torch.stack(
            attn_weights, 1), durations
