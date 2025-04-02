import json
import os
import sys
from pathlib import Path

# ✅ Define the function first
def get_base_dir():
    """Returns the project root directory dynamically based on execution environment."""
    if 'COLAB_GPU' in os.environ:
        return Path("/content/drive/MyDrive/project_x_voice")  # Google Drive path for Colab
    return Path(__file__).resolve().parent  # Local execution: points to project_x_voice

# ✅ Now call it after definition
PROJECT_ROOT = get_base_dir()
CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"
PHONEME_PATH = PROJECT_ROOT / "config" / "phonemes.json"

# ✅ Load config file with error handling
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
else:
    raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")

# ✅ Dynamically update paths
config['paths'] = {
    "base_dir": str(PROJECT_ROOT),
    "metadata_path": str(PROJECT_ROOT / "data" / "metadata.csv"),
    "wav_dir": str(PROJECT_ROOT / "data" / "wavs"),
    "checkpoint_dir": str(PROJECT_ROOT / "checkpoints"),
    "log_dir": str(PROJECT_ROOT / "logs"),
    "phoneme_path": str(PHONEME_PATH)
}

# ✅ Print updated paths to verify
print("Updated Config Paths:")
print(json.dumps(config['paths'], indent=4))

# ✅ Add project root to sys.path for module resolution
sys.path.append(str(PROJECT_ROOT))
