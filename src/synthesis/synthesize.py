# Add at the top
import sys
sys.path.append("vocoder/hifigan")
from vocoder.utils import load_hifigan, mel2wav

# Modify synthesize function
def synthesize(text, lang, emotion, checkpoint_path, output_file):
    # ... [previous code until mel generation]
    
    # Load vocoder
    vocoder = load_hifigan(
        config_path=None,  # Using hardcoded config
        checkpoint_path="vocoder/pretrained/hifigan_universal.pth",
        device=device
    )
    
    # Convert mel to waveform
    audio = mel2wav(vocoder, mel.squeeze(1))
    
    # Save with proper scaling
    write(output_file, config["data"]["sample_rate"], audio)