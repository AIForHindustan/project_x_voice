import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BilingualTTS(nn.Module):
    def __init__(self, config):
        super(BilingualTTS, self).__init__()
        self.n_mels = config['n_mels']
        self.embedding_dim = config['embedding_dim']
        self.hidden_dim = config['hidden_dim']
        self.dropout = config.get('dropout', 0.1)

        self.text_embedding = nn.Embedding(config['vocab_size'], self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hidden_dim * 2, self.n_mels)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, phonemes, text_lengths, lang_ids, spk_ids):
        embedded = self.text_embedding(phonemes)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        mel_outputs = self.fc(self.dropout_layer(unpacked))
        return {"mel_pred": mel_outputs}


def collate_fn(batch):
    text_indices, text_lengths, lang_ids, emo_ids, spk_ids, mels = zip(*batch)
    text_padded = torch.nn.utils.rnn.pad_sequence(text_indices, batch_first=True, padding_value=0)
    text_lengths = torch.stack(text_lengths).squeeze(1)
    lang_ids = torch.stack(lang_ids).squeeze(1)
    emo_ids = torch.stack(emo_ids).squeeze(1)
    spk_ids = torch.stack(spk_ids).squeeze(1)
    mel_padded = torch.nn.utils.rnn.pad_sequence(mels, batch_first=True, padding_value=-80.0)
    return text_padded, text_lengths, lang_ids, emo_ids, spk_ids, mel_padded


def train_step(model, batch, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()
    
    text_padded, text_lengths, lang_ids, emo_ids, spk_ids, mel_padded = [b.to(device) for b in batch]
    outputs = model(text_padded, text_lengths, lang_ids, spk_ids)
    loss = criterion(outputs["mel_pred"], mel_padded)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item()


def synthesize_test(model, sample, device, vocoder):
    model.eval()
    with torch.no_grad():
        inputs = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}
        mel_pred = model(**inputs)["mel_pred"]
        audio = vocoder(mel_pred.cpu())
        return audio
