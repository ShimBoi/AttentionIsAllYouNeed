import sacrebleu
from datasets import load_dataset
from transformers import MarianTokenizer
import torch
from transformer import Transformer
from tqdm import tqdm


def load_checkpoint(filepath, device):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    # Recreate model with saved config
    model_cfg = checkpoint["config"]["model"]
    model_cfg.dropout_prob = 0.0  # no dropout during eval

    model = Transformer(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Training loss was: {checkpoint['loss']:.4f}")

    return model, checkpoint


def evaluate_bleu(model, tokenizer, device, num_samples=1000, max_length=256):
    """Evaluate model BLEU score on test set"""
    model.eval()

    # Load test set
    split = f"test[:{num_samples}]"
    if num_samples == -1:
        split = "test"

    test_dataset = load_dataset("wmt14", "de-en", split=split)

    references = []
    hypotheses = []

    with torch.no_grad():
        for example in tqdm(test_dataset, total=len(test_dataset)):
            src_text = example["translation"]["de"]
            ref_text = example["translation"]["en"]

            # Tokenize source
            src_tokens = tokenizer.encode(
                src_text, max_length=max_length, truncation=True
            )
            src_tensor = torch.tensor([src_tokens]).to(device)

            # Generate translation (greedy decoding)
            translation = greedy_decode(
                model, src_tensor, tokenizer, max_length, device
            )

            references.append(ref_text)
            hypotheses.append(translation)

    # Calculate BLEU score
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])

    print(f"BLEU score: {bleu.score:.2f}")
    print(f"Evaluated on {num_samples} samples")

    # Show some examples
    print("\nSample translations:")
    for i in range(min(5, len(hypotheses))):
        print(f"\nSource: {test_dataset[i]['translation']['de']}")
        print(f"Reference: {references[i]}")
        print(f"Hypothesis: {hypotheses[i]}")

    return bleu.score


def greedy_decode(model, src, tokenizer, max_length, device):
    """Generate translation using greedy decoding"""
    # Start with BOS token
    tgt_tokens = [tokenizer.eos_token_id]

    for i in range(max_length):
        tgt_tensor = torch.tensor([tgt_tokens]).to(device)

        tgt_len = len(tgt_tokens)
        tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len))  # FIX: wrong triangle mask
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0).to(device)

        logits = model(src, tgt_tensor, self_attn_mask=tgt_mask)
        next_token = logits[0, -1, :].argmax().item()

        # Stop if EOS
        if next_token == tokenizer.eos_token_id:
            break

        tgt_tokens.append(next_token)

    # Decode to text
    translation = tokenizer.decode(tgt_tokens, skip_special_tokens=True)
    return translation


if __name__ == "__main__":
    folder = "checkpoints/20251004_124049"
    model = "final_model.pt"
    checkpoint_path = f"{folder}/{model}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(checkpoint_path, device)

    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")

    bleu_score = evaluate_bleu(model, tokenizer, device, num_samples=-1, max_length=256)
