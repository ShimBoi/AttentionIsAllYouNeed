import torch
from torch.utils.data import Dataset
from transformers import MarianTokenizer


class WMTDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer=None,
        src_language: str = "de",
        tgt_language: str = "en",
    ):
        self.dataset = dataset
        self.src_language = src_language
        self.tgt_language = tgt_language

        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = MarianTokenizer.from_pretrained(
                "Helsinki-NLP/opus-mt-de-en"
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        pair = self.dataset[idx]["translation"]

        src_ids = self.tokenizer.encode(
            pair[self.src_language], max_length=512, truncation=True
        )
        tgt_ids = self.tokenizer.encode(
            pair[self.tgt_language], max_length=512, truncation=True
        )
        tgt_ids = [self.tokenizer.eos_token_id] + tgt_ids

        return {
            "src": src_ids,
            "tgt": tgt_ids,
        }

    def get_collate_fn(self):
        pad_token_id = self.tokenizer.pad_token_id

        def collate_fn(batch):
            src_list = [torch.tensor(item["src"]) for item in batch]
            tgt_list = [torch.tensor(item["tgt"]) for item in batch]

            src_padded = torch.nn.utils.rnn.pad_sequence(
                src_list, batch_first=True, padding_value=pad_token_id
            )
            tgt_padded = torch.nn.utils.rnn.pad_sequence(
                tgt_list, batch_first=True, padding_value=pad_token_id
            )

            src_padding_mask = (src_padded != pad_token_id).unsqueeze(1).unsqueeze(2)
            tgt_len = tgt_padded.size(1)
            causal_mask = torch.tril(torch.ones(1, tgt_len, tgt_len))
            tgt_padding_mask = (tgt_padded != pad_token_id).unsqueeze(1).unsqueeze(2)
            combined_tgt_mask = causal_mask * tgt_padding_mask

            return {
                "src": src_padded,
                "tgt": tgt_padded,
                "src_padding_mask": src_padding_mask,
                "tgt_mask": combined_tgt_mask,
            }

        return collate_fn


if __name__ == "__main__":
    from datasets import load_dataset

    dataset = load_dataset("wmt14", "de-en", split="train")
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")

    train_data = WMTDataset(
        dataset, tokenizer=tokenizer, src_language="de", tgt_language="en"
    )

    batch = train_data[0]
    batch = train_data.get_collate_fn()([batch])

    for key, value in batch.items():
        print(f"{key}: {value.shape}")
        print(value)
