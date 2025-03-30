import torch
import numpy as np
from vocoder.hifigan.models import Generator
from scipy.io.wavfile import write

def load_hifigan(config_path, checkpoint_path, device="cuda"):
    config = {
        "resblock": "1",
        "num_gpus": 0,
        "batch_size": 16,
        "learning_rate": 0.0002,
        "adam_b1": 0.8,
        "adam_b2": 0.99,
        "lr_decay": 0.999,
        "seed": 1234,
        "upsample_rates": [8,8,2,2],
        "upsample_kernel_sizes": [16,16,4,4],
        "upsample_initial_channel": 512,
        "resblock_kernel_sizes": [3,7,11],
        "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
        "num_mels": 80,
        "sampling_rate": 22050
    }
    
    model = Generator(config)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict["generator"])
    model.eval()
    model.remove_weight_norm()
    return model.to(device)

def mel2wav(vocoder, mel, sr=22050):
    with torch.no_grad():
    mel = torch.log(torch.clamp(mel, min=1e-5))    
    if len(mel.shape) == 2:
            mel = mel.unsqueeze(0)
        audio = vocoder(mel).squeeze()
    audio = audio.cpu().numpy()
    audio = audio * 32768.0
    return audio.astype('int16')