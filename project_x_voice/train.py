import json
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Local imports
from data.processor import BilingualDataset, collate_fn
from models.tts import BilingualTTS

def load_config():
    config_path = Path(__file__).parent.parent / "configs" / "train_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def guided_attention_loss(attn_weights, text_lens, mel_lens, sigma=0.2):
    """Vectorized guided attention loss with per-sample length awareness"""
    B, max_T, max_M = attn_weights.shape
    device = attn_weights.device
    
    # Create grid for entire batch using broadcasting
    t = torch.arange(max_T, device=device).float()[None, :, None]  # [1, T, 1]
    m = torch.arange(max_M, device=device).float()[None, None, :]  # [1, 1, M]
    
    # Normalize by actual lengths (shape: [B, 1, 1])
    text_ratio = t / text_lens.view(B, 1, 1)
    mel_ratio = m / mel_lens.view(B, 1, 1)
    
    # Compute Gaussian weights
    distance = text_ratio - mel_ratio
    grid = 1 - torch.exp(-(distance**2)/(2*sigma**2))
    
    return (attn_weights * grid).mean()

def train():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize components
    dataset = BilingualDataset(
        metadata_path=Path(config["data"]["metadata_path"]),
        data_dir=Path(config["data"]["root_dir"]),
        durations_path=Path(config["data"]["durations_path"]),
        config=config
    )
    
    model = BilingualTTS(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    writer = SummaryWriter(log_dir=config["training"]["log_dir"])
    global_step = 0
    
    # Training loop
    for epoch in range(config["training"]["epochs"]):
        model.train()
        for batch in DataLoader(dataset, batch_size=config["training"]["batch_size"], collate_fn=collate_fn):
            optimizer.zero_grad()
            
            # Prepare inputs with proper tensor shapes
            inputs = {
                "phonemes": batch["phonemes"].to(device),
                "text_lengths": batch["text_lengths"].squeeze().to(device),  # [B]
                "lang_ids": batch["lang_ids"].squeeze().to(device),          # [B]
                "spk_ids": batch["spk_ids"].squeeze().to(device),            # [B]
                "mel_input": batch["mel"].to(device)[:, :-1]  # Teacher forcing
            }
            
            # Forward pass
            outputs = model(**inputs)
            
            # Loss calculation
            mel_target = batch["mel"].to(device)[:, 1:]
            mse_loss = F.mse_loss(outputs["mel_pred"], mel_target)
            
            # Log-space duration loss with numerical stability
            dur_loss = F.mse_loss(
                outputs["log_durations"],
                torch.log(batch["durations"].to(device).float() + 1e-8)
            )
            
            # Attention loss with actual lengths
            ga_loss = guided_attention_loss(
                outputs["attn_weights"],
                text_lens=inputs["text_lengths"],
                mel_lens=batch["mel_lengths"].squeeze().to(device)
            )
            
            total_loss = (
                config["loss_weights"]["mse"] * mse_loss +
                config["loss_weights"]["duration"] * dur_loss +
                config["loss_weights"]["attention"] * ga_loss
            )
            
            # Backprop with gradient clipping
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Enhanced logging with global step
            writer.add_scalar("Loss/total", total_loss.item(), global_step)
            writer.add_scalars("Loss/components", {
                "mse": mse_loss.item(),
                "duration": dur_loss.item(),
                "attention": ga_loss.item()
            }, global_step)
            global_step += 1
        
        # Save checkpoint every epoch
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }, f"{config['training']['checkpoint_dir']}/checkpoint_{epoch}.pt")
        
        # First test synthesis after 500 steps (Placeholder)
        if global_step >= 500:
            print("Running first synthesis test...")
            # synthesize_test_sample(model, dataset[0])  # Uncomment and implement
    
    writer.close()

if __name__ == "__main__":
    train()
