import math
import torch


def sinusoidal_positional_encoding(x):
    """
    Implementation of sinusoidal positional embeddings
    """

    bs, seqlen, d_model = x.shape
    device = x.device

    positional_encoding = torch.zeros(seqlen, d_model, device=device)
    position = torch.arange(0, seqlen, dtype=torch.float32, device=device)
    position = position.unsqueeze(1)  # [seqlen, 1]
    denom = torch.pow(
        10000.0,
        torch.arange(0, d_model, 2, dtype=torch.float32, device=device) / d_model,
    )

    positional_encoding[:, ::2] = torch.sin(position / denom)
    positional_encoding[:, 1::2] = torch.cos(position / denom)

    return x + positional_encoding.unsqueeze(0)
