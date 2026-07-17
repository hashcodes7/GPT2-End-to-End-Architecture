import torch
from torch.utils.data import Dataset


class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, context_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(
            text,
            allowed_special={"<|endoftext|>"}
        )

        print("=" * 50)
        print(f"Characters      : {len(text)}")
        print(f"Tokens          : {len(token_ids)}")
        print("=" * 50)

        assert len(token_ids) > context_length, (
            "Tokenized text must contain at least context_length + 1 tokens."
        )

        for i in range(0, len(token_ids) - context_length, stride):
            self.input_ids.append(
                torch.tensor(token_ids[i:i + context_length])
            )

            self.target_ids.append(
                torch.tensor(token_ids[i + 1:i + context_length + 1])
            )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]