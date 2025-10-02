import torch
from dataclasses import dataclass
import matplotlib.pyplot as plt


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


def save_checkpoint(model, optimizer, scheduler, epoch, loss, cfg, filepath):
    """Save model checkpoint"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "loss": loss,
        "config": cfg,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def plot_loss_curve(train_losses, save_path):
    """Plot and save training loss curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training Loss Curve", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Loss curve saved to {save_path}")
    plt.close()
