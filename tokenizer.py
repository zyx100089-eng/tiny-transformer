import json
from pathlib import Path


class CharTokenizer:
    def __init__(self):
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}
        self.vocab_size: int = 0

    def fit(self, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        return self

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps({"stoi": self.stoi}))

    def load(self, path: str) -> "CharTokenizer":
        data = json.loads(Path(path).read_text())
        self.stoi = {k: int(v) for k, v in data["stoi"].items()}
        self.itos = {v: k for k, v in self.stoi.items()}
        self.vocab_size = len(self.stoi)
        return self
