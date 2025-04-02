Project X Voice - Daily Development Log

Objective

Develop a bilingual TTS (Text-to-Speech) model with emotion embedding, ensuring high-quality synthesis with efficient processing.
Current Tree Structure
arduino
Copy
Edit
.
├── README.md
├── config
│   ├── config.json
│   └── phonemes.json
├── config.yaml
├── data
│   ├── durations.json
│   ├── metadata.csv
│   └── wavs
│       ├── angry_en_1.wav
│       ├── angry_en_2.wav
│       └── ... (list all other wavs)
├── debug_dataset.py
├── project_structure.txt
├── scripts
│   ├── __init__.py
│   ├── audio_utils.py
│   ├── augment.py
│   ├── bilingual_tts.py
│   ├── check_and_run.sh
│   ├── collate_fn.py
│   ├── data_processor.py
│   ├── model.py
│   ├── synthesis.py
│   ├── train.py
│   ├── validation.py
│   └── vocab.py
├── pyproject.toml
├── requirements.txt
├── setup.cfg
└── vocoder
    ├── __init__.py
    └── utils.py

Functions of Key Scripts

model.py - Defines the bilingual TTS model architecture, including embeddings and LSTM layers.

train.py - Handles the training process for the model.

validation.py - Performs validation checks on trained models.

data_processor.py - Processes input data, including text and audio formatting.

audio_utils.py - Contains audio processing utilities.

synthesis.py - Handles text-to-speech synthesis.

augment.py - Implements data augmentation techniques.

vocab.py - Manages phonemes and vocabulary processing.

collate_fn.py - Defines batching and collation functions for training.

bilingual_tts.py - Wrapper script for TTS synthesis.

check_and_run.sh - Shell script for running automated tasks.

Colab Workflow Notes:
Step-by-Step Approach: Each time working in Colab, we’ll follow this specific process:

Clone the Latest Repo from GitHub:

bash
Copy
Edit
git clone https://github.com/AIForHindustan/project_x_voice.git
If the repo already exists, use git pull to get the latest updates:

cd project_x_voice
git pull
Google Drive Setup (if needed for persistence): Mount Google Drive in Colab:

from google.colab import drive
drive.mount('/content/drive')
Run Training Script (or any desired script):
python3 train.py
Common Issues and Fixes:
Git Clone Error: If you try to clone into a directory that already exists, it will fail. Instead, either remove the existing folder or use git pull to fetch the latest changes.

Comment Lines in Terminal: When running commands directly in the terminal, ensure comments (lines starting with #) are removed or handled in script files, as the terminal interprets them as commands, leading to errors.

Missing Python Module (indicnlp):

The error ModuleNotFoundError: No module named 'indicnlp' can be fixed by installing the necessary package:


pip install indic-nlp-library
Daily Updates

Date: [YYYY-MM-DD]

1) Progress Made

[Key updates on what was implemented today]

2) Key Issues Resolved

[Summary of major debugging and fixes]

3) Enhancements Incorporated

[New features or optimizations added]

4) Testing Results

[Summary of tests performed and outcomes]

5) Next Steps

[Planned work for the next session]

Project Journal - April 1, 2025
Timestamp: 12:15 PM IST

Work Completed:
Config Fixes:

The issue with the config.json file (regarding incorrect syntax) was identified and corrected. Specifically, the trailing commas after JSON elements were removed to comply with proper JSON formatting.

After this fix, the configuration file was reloaded without errors.

Model Architecture Review & Updates:

The model.py file was reviewed and updated to handle phoneme embeddings, language embeddings, and emotion embeddings effectively. The model was structured to use LSTM layers followed by a decoder for predicting the Mel spectrogram.

Ensured the model outputs were aligned with the target Mel spectrogram format: [B, n_mels, T] where B is batch size, n_mels is Mel bins, and T is time steps.

Issue Debugging:

Encountered an issue where the target size (mel) mismatched the predicted size (mel_pred). The warning indicated that there were differences in the time steps: mel_pred had T=44 and mel had T=517.

Troubleshot the issue by checking the model's output shape and ensuring that the mel_pred and mel align in time steps. This is a critical point for preventing broadcasting errors.

Discussed potential fixes such as padding, truncating, or aligning the target Mel spectrogram data during preprocessing.

GitHub Push:

Changes made to the repository were pushed to GitHub:

Fixes for the config.json.

Updates in the model.py (for handling embeddings and LSTM outputs).

Various minor adjustments to the training pipeline.

Commit hash after pushing: 709df51.

Challenges Faced:
Mismatch in Mel Spectrogram Time Steps:

The core issue was the discrepancy in the time steps of the target Mel spectrogram and the predicted Mel spectrogram. Despite updates to the model, the time step mismatch remains unresolved.

The mel_pred output shape is [B, T, n_mels], but the target mel has an unexpected time dimension. This could be due to improper sequence handling during data preparation.

Configuration File Debugging:

The configuration file initially caused issues due to incorrect formatting (trailing commas). This was resolved after a detailed inspection of the file.

Next Steps / Pending Issues:
Further Investigation into Time Steps Mismatch:

Continue debugging the target Mel spectrogram generation and ensure that its sequence length aligns with the predicted Mel spectrogram from the model.

Check the data preprocessing pipeline to ensure that Mel spectrograms are generated with the correct dimensions.

Add padding/truncation where necessary to align sequence lengths.

Data Pipeline Inspection:

Examine the data generation process for the Mel spectrograms, ensuring consistent time steps across the entire pipeline.

Performance Optimizations:

Explore GPU time efficiency, as current debug iterations might be consuming excessive resources.
Journal Entry: April 1, 2025 (Post 12:15 PM)
Objective:
Resolve issues in the BilingualTTS model to address the shape mismatch error during training.

Actions Taken:

Error Identified:

Problem: The model is throwing a runtime error during training due to a mismatch in the expected dimensions of the predicted Mel-spectrogram (mel_pred) and the target Mel-spectrogram (mel).

The error specifically states:

java
Copy
Edit
RuntimeError: The size of tensor a (44) must match the size of tensor b (517) at non-singleton dimension 2.
Proposed Solution:

We attempted several changes to adjust the dimensions of mel_pred to match the target dimensions. This involved:

Upsampling Layer: A ConvTranspose1d layer was introduced to adjust the time dimension of mel_pred from ~44 to 517.

Transpose: We ensured that the tensor is transposed to match the expected shape [B, n_mels, T] before applying the upsampling.

Trimming: After upsampling, the tensor was trimmed to exactly 517 time steps (mel_pred = mel_pred[:, :, :517]).

Further Exploration:

We applied DeepSeek's recommendation for debugging the dimensions by adding print statements and verifying tensor sizes right before the interpolation or upsampling layers.

The shape of the tensor was printed before the upsampling layer, but the error persisted, indicating that there was still an issue with dimension alignment or the method of interpolation used.

Conclusion of Actions:

Despite implementing the changes to upsample and trim the Mel-spectrogram, the error persisted, indicating that the mismatch issue was not resolved. The current dimensions of mel_pred and mel were not aligned in such a way that would allow the MSE loss to compute correctly.

Further investigation and changes are required, including revisiting the architecture and layer configuration tomorrow.

Changes Made (but not effective yet):

Upsample Layer (ConvTranspose1d) was introduced to match dimensions.

Tensor Transposition to ensure proper dimension order before upsampling.

Trimming to ensure that the time dimension of mel_pred is exactly 517 steps.

Unchanged:

The underlying issue with tensor dimension misalignment remains unresolved, as the error persists even after the changes.

The MSE loss function still causes errors due to dimension mismatch between mel_pred and mel.

Next Steps:

We will revisit the issue tomorrow and consider alternate methods for dimension alignment or potential fixes in the architecture of the model.

Possibly involve more detailed debugging and try changing layer parameters or the way tensor dimensions are handled.




