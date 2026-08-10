import torch
from pathlib import Path
from tokenizer import CharTokenizer
from bpe_tokenizer import BPETokenizer


class TextDataset:
    def __init__(self, filepath: str, train_split: float = 0.9, tokenizer_type: str = "char", num_merges: int = 500):
        text = Path(filepath).read_text()
        if tokenizer_type == "bpe":
            self.tokenizer = BPETokenizer().fit(text, num_merges=num_merges)
        else:
            self.tokenizer = CharTokenizer().fit(text)
        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        n = int(len(data) * train_split)
        self.train_data = data[:n]
        self.val_data = data[n:]

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def get_batch(
        self, split: str, batch_size: int, block_size: int, device: str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)