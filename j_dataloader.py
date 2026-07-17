import tiktoken
from torch.utils.data import DataLoader

from i_gpt_dataset import GPTDataset


def create_dataloader(
    text,
    batch_size=4,
    context_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDataset(
        text=text,
        tokenizer=tokenizer,
        context_length=context_length,
        stride=stride,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )