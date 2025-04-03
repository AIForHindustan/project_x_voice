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
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import modules
from path_manager import get_base_dir
from data_processor import BilingualDataset, collate_fn
from model import BilingualTTS

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Attention Mechanism
class Attention(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = torch.nn.Linear(hidden_dim, hidden_dim)
        self.key = torch.nn.Linear(hidden_dim, hidden_dim)
        self.value = torch.nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, decoder_state, encoder_outputs):
        attn_weights = torch.matmul(self.query(decoder_state), self.key(encoder_outputs).transpose(-2, -1))
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights, self.value(encoder_outputs)), attn_weights

# Enhanced Loss Calculation
def compute_loss(mel_pred, mel, mel_lengths, attention_weights, config):
    mel_pred = (mel_pred - mel_pred.mean()) / (mel_pred.std() + 1e-5)  # Normalize
    loss = F.l1_loss(mel_pred, mel, reduction='mean')
    attn_loss = attention_weights.mean()  # Simple attention loss
    return loss + config["loss_weights"].get("attention", 1.0) * attn_loss


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
    dataset = BilingualDataset(config=config)
    dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"], collate_fn=collate_fn, shuffle=True)
    model = BilingualTTS(config, phoneme_to_id=phoneme_to_id).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["training"].get("lr_decay", 0.98))

    for epoch in range(config["training"]["epochs"]):
        model.train()
        total_loss = 0.0

        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            phonemes = batch["phonemes"].to(device)
            text_lengths = batch["text_lengths"].to(device)
            lang_ids = batch["lang_ids"].to(device)
            emotion_ids = batch["emotion_ids"].to(device)
            mel = batch["mel"].to(device).transpose(1, 2)
            
            mel_pred, attention_weights = model(phonemes, text_lengths, lang_ids, emotion_ids)
            loss = compute_loss(mel_pred, mel, batch["mel_lengths"], attention_weights, config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"].get("grad_clip_thresh", 1.0))
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if i % 50 == 0:
                plt.imshow(mel_pred[0].cpu().detach().numpy())
                plt.savefig(f"mel_epoch{epoch}_batch{i}.png")
                print(f"Batch {i}: Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Average Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), checkpoint_dir / f"epoch_{epoch+1}.pt")

    # Inference and saving output audio
    test_phonemes = torch.randint(0, len(phoneme_to_id), (1, 20)).to(device)
    test_text_lengths = torch.tensor([20]).to(device)
    test_lang_ids = torch.tensor([0]).to(device)
    test_emotion_ids = torch.tensor([1]).to(device)
    
    with torch.no_grad():
        test_mel, _ = model(test_phonemes, test_text_lengths, test_lang_ids, test_emotion_ids)
    
    griffin_lim = GriffinLim(n_fft=512, n_iter=32, win_length=400, hop_length=200)
    test_mel = test_mel.squeeze(0).cpu()
    waveform = griffin_lim(test_mel)
    sample_rate = config["data"].get("sample_rate", 22050)
    torchaudio.save("output.wav", waveform.unsqueeze(0), sample_rate=sample_rate, format="wav", encoding="PCM_S", bits_per_sample=16)
    print("Generated audio saved as output.wav")

if __name__ == "__main__":
    train()
