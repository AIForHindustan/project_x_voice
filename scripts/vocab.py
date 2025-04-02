import os
import json
from pathlib import Path


def load_phoneme_mapping():
    # Adjust path: vocab.py is in project_x_voice/project_x_voice, config is
    # one level up.

    # Get the absolute path based on script location
    config_path = os.path.join(
        os.path.dirname(__file__),
        "../config/phonemes.json"
    )

    with open(config_path, "r") as f:  # Corrected indentation
        mapping = json.load(f)

    return mapping  # Corrected indentation


# Export the mapping as phoneme_to_id
phoneme_to_id = load_phoneme_mapping()

if __name__ == "__main__":
    print("Phoneme mapping loaded:", phoneme_to_id)
