import torch
import matplotlib.pyplot as plt

"""
This file focuses on understanding the role of wavelenght in sinusoidal embeddings and how both extremes affect position encoding.
"""


# modified from positional_encoders.py
def sinusoidal_positional_encoding(seqlen, d_model, device, wavelength):
    positional_encoding = torch.zeros(seqlen, d_model, device=device)
    position = torch.arange(0, seqlen, dtype=torch.float32, device=device)
    position = position.unsqueeze(1)  # [seqlen, 1]

    power = torch.arange(0, d_model, 2, dtype=torch.float32, device=device) / d_model
    denom = torch.pow(wavelength, power)

    positional_encoding[:, ::2] = torch.sin(position / denom)
    positional_encoding[:, 1::2] = torch.cos(position / denom)

    return positional_encoding


seqlen = 50
d_model = 512
device = "cpu"

# Very small → repeats quickly
pe_small = sinusoidal_positional_encoding(seqlen, d_model, wavelength=50, device=device)
# Default from Transformer
pe_default = sinusoidal_positional_encoding(
    seqlen, d_model, wavelength=10000, device=device
)
# Very large → nearly flat
pe_large = sinusoidal_positional_encoding(
    seqlen, d_model, wavelength=1e8, device=device
)

low_dims = [0, 1, 2, 3, 4]
mid_dims = [248, 249, 250, 251, 252]
high_dims = [507, 508, 509, 510, 511]


def plot_hierarchical_subplots(pe_small, pe_default, pe_large, dims_groups):
    import matplotlib.pyplot as plt

    pes = [pe_small, pe_default, pe_large]
    wavelength_labels = ["Small λ=50", "Default λ=10k", "Large λ=1e8"]
    group_names = list(dims_groups.keys())

    fig, axes = plt.subplots(len(pes), len(dims_groups), figsize=(18, 12), sharex=True)

    if len(dims_groups) == 1:
        axes = axes[:, None]

    for row_idx, (pe, wl_label) in enumerate(zip(pes, wavelength_labels)):
        for col_idx, group_name in enumerate(group_names):
            ax = axes[row_idx, col_idx]
            for dim in dims_groups[group_name]:
                ax.plot(pe[:, dim], label=f"dim {dim}")
            if row_idx == 0:
                ax.set_title(group_name)
            if col_idx == 0:
                ax.set_ylabel(wl_label)
            ax.grid(True)
            ax.legend(fontsize=8)

    axes[-1, 0].set_xlabel("Position")
    plt.tight_layout()
    plt.show()


# Example usage
dims_groups = {"Low dims": low_dims, "Mid dims": mid_dims, "High dims": high_dims}
plot_hierarchical_subplots(pe_small, pe_default, pe_large, dims_groups)
