import os
import torch
import torch.nn as nn
from transformer import Transformer
from datasets import load_dataset
from torch.utils.data import DataLoader
from utils import TransformerConfig
from wmt14_dataset import WMTDataset
from transformers import MarianTokenizer
from scheduler import AIAYNScheduler
from datetime import datetime
from utils import save_checkpoint, plot_loss_curve
from tqdm import tqdm


def train(cfg):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"checkpoints/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    device = cfg["device"]
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]

    split = f"train[:{train_cfg['samples']}]"
    if train_cfg["samples"] == -1:
        split = "train"

    dataset = load_dataset("wmt14", "de-en", split=split)
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")

    train_data = WMTDataset(
        dataset, tokenizer=tokenizer, src_language="de", tgt_language="en"
    )
    train_loader = DataLoader(
        train_data,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=train_data.get_collate_fn(),
    )

    model = Transformer(model_cfg).to(device)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],  # ignored by scheduler
        betas=train_cfg["betas"],
        eps=train_cfg["eps"],
    )
    scheduler = AIAYNScheduler(
        optimizer=optimizer,
        d_model=model_cfg.d_model,
        warmup_steps=train_cfg["warmup_steps"],
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id, label_smoothing=0.1
    )

    accumulation_steps = train_cfg.get("grad_accum_steps", 1)
    losses = []
    for epoch in range(train_cfg["epochs"]):
        total_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            input_ids = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            padding_mask = batch["src_padding_mask"].to(device)
            self_attn_mask = batch["tgt_mask"].to(device)

            logits = model(
                input=input_ids,
                target=tgt[:, :-1],
                input_padding_mask=padding_mask,
                self_attn_mask=self_attn_mask[:, :, :-1, :-1],
            )

            loss = criterion(
                logits.reshape(-1, tokenizer.vocab_size), tgt[:, 1:].reshape(-1)
            )

            # Scale loss by accumulation steps
            loss = loss / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                print(
                    f"Epoch {epoch}, Batch {batch_idx}, Loss: \
                      {loss.item():.4f}, LR: {scheduler.get_lr():.6f}"
                )

            total_loss += loss.item() * accumulation_steps  # unscale for logging

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch} finished, Average Loss: {avg_loss:.4f}")

        if (epoch + 1) % train_cfg["ckpt_interval"] == 0:
            checkpoint_path = os.path.join(
                output_dir, f"checkpoint_epoch_{epoch+1}_{avg_loss}.pt"
            )
            save_checkpoint(
                model, optimizer, scheduler, epoch, avg_loss, cfg, checkpoint_path
            )

    # save final
    final_model_path = os.path.join(output_dir, "final_model.pt")
    save_checkpoint(
        model,
        optimizer,
        scheduler,
        train_cfg["epochs"] - 1,
        losses[-1],
        cfg,
        final_model_path,
    )

    # Plot loss curve at the end
    loss_curve_path = os.path.join(output_dir, "loss_curve.png")
    plot_loss_curve(losses, loss_curve_path)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = {
        "device": device,
        "train": {
            "epochs": 15,  # 100_000 / (4_500_000 / 16) = ~0.36
            "samples": -1,
            "batch_size": 16,
            "grad_accum_steps": 8,
            "lr": 3e-4,
            "betas": (0.9, 0.98),
            "eps": 1e-9,
            "warmup_steps": 4000,
            "ckpt_interval": 1,
        },
        "model": TransformerConfig(
            d_model=512,
            d_ff=2048,
            num_heads=8,
            n_encoder_layers=6,
            n_decoder_layers=6,
            vocab_size=58101,  # MarianTokenizer vocab size
            dropout_prob=0.1,
            positional_encoding="sinusoidal",
            device=device,
        ),
    }

    ### TEST CONFIG
    # cfg = {
    #     "device": device,
    #     "train": {
    #         "epochs": 100,
    #         "samples": 10000,
    #         "batch_size": 32,
    #         "lr": 3e-4,
    #         "betas": (0.9, 0.98),
    #         "eps": 1e-9,
    #         "warmup_steps": 100,
    #     },
    #     "model": TransformerConfig(
    #         d_model=256,
    #         d_ff=1024,
    #         num_heads=4,
    #         n_encoder_layers=3,
    #         n_decoder_layers=3,
    #         vocab_size=58101,
    #         dropout_prob=0.1,
    #         positional_encoding="sinusoidal",
    #         device=device,
    #     ),
    # }

    train(cfg)
