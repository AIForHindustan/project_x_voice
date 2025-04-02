import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    """
    Collates a batch of data samples into padded tensors.

    Args:
        batch (list of tuples): Each tuple contains:
            (text_indices, text_length, lang_id, emo_id, spk_id, mel)

    Returns:
        tuple: Padded and stacked tensors for batch training.
    """
    # Unpack batch
    text_indices, text_lengths, lang_ids, emo_ids, spk_ids, mels = zip(*batch)

    # Pad text sequences
    text_padded = pad_sequence(text_indices, batch_first=True, padding_value=0)

    # Convert lists to tensors
    text_lengths = torch.stack(text_lengths).squeeze(1)  # [batch]
    lang_ids = torch.stack(lang_ids).squeeze(1)          # [batch]
    emo_ids = torch.stack(emo_ids).squeeze(1)            # [batch]
    spk_ids = torch.stack(spk_ids).squeeze(1)            # [batch]

    # Pad mel spectrograms along time dimension
    mel_padded = pad_sequence(mels, batch_first=True, padding_value=-80.0)

    return text_padded, text_lengths, lang_ids, emo_ids, spk_ids, mel_padded
