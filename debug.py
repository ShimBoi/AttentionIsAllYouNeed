import torch
import torch.nn as nn
from transformer import Transformer
from utils import TransformerConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create tiny model
cfg_tiny = TransformerConfig(
    d_model=32,
    d_ff=64,
    num_heads=4,
    n_encoder_layers=2,
    n_decoder_layers=2,
    vocab_size=100,
    dropout_prob=0.0,  # Turn off dropout for debugging
    positional_encoding="sinusoidal",
    d_k=8,
    d_v=8,
    device=device,
)

model = Transformer(cfg_tiny).to(device)
model.train()

# Single example to overfit
src = torch.tensor([[5, 12, 3, 8, 2, 7]], device=device)
tgt_input = torch.tensor([[5, 12, 3, 8, 2]], device=device)  # shifted right
tgt_output = torch.tensor([[12, 3, 8, 2, 7]], device=device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Overfitting single example...")
for step in range(200):
    optimizer.zero_grad()

    logits = model(src, tgt_input)

    loss = criterion(
        logits.view(-1, cfg_tiny.vocab_size),
        tgt_output.view(-1),
    )

    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        print(f"Step {step}: Loss = {loss.item():.4f}")

    # Success: loss should go below 0.01
    if loss.item() < 0.01:
        print(f"Successfully overfitted at step {step}!")
        break
else:
    print("Failed to overfit")
