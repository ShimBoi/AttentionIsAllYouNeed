import torch
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    d_model: int = 512
    d_ff: int = 2048
    num_heads: int = 8
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
    vocab_size: int = 58101
    dropout_prob: float = 0.1
    positional_encoding: str = "sinusoidal"
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_k: int = None  # Add if your Transformer expects these
    d_v: int = None


def create_copy_task_data(n_samples, seq_len, vocab_size):
    """
    Simple task: copy input sequence
    Input:  [1, 2, 3, 4, <sep>]
    Target: [1, 2, 3, 4, <eos>]
    """
    data = []
    for _ in range(n_samples):
        # Random sequence
        seq = torch.randint(1, vocab_size - 2, (seq_len,))
        data.append((seq, seq))

    return data
