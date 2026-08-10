import json
from pathlib import Path


class BPETokenizer:
    """Byte-pair encoding tokenizer implemented from scratch.

    Learns merges greedily: find the most frequent adjacent pair, merge it,
    repeat. At encode time, apply learned merges in order.
    """

    def __init__(self):
        self.merges: list[tuple[str, str]] = []
        self.vocab: list[str] = []
        self.stoi: dict[str, int] = {}
        self.vocab_size: int = 0
        self.unk_token: str = "<unk>"
        self.unk_id: int = 0

    @staticmethod
    def _get_pairs(tokens: list[str]) -> dict[tuple[str, str], int]:
        counts: dict[tuple[str, str], int] = {}
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def fit(self, text: str, num_merges: int = 500) -> "BPETokenizer":
        tokens = list(text)
        unique_chars = sorted(set(tokens))
        self.vocab = list(unique_chars)
        self.merges = []

        for _ in range(num_merges):
            pair_counts = self._get_pairs(tokens)
            if not pair_counts:
                break
            best_pair = max(pair_counts, key=pair_counts.get)
            merged = best_pair[0] + best_pair[1]

            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            self.merges.append(best_pair)
            self.vocab.append(merged)

        # Reserve id 0 for <unk> so unseen characters map to a real token
        # rather than silently substituting the first character of the input.
        self.vocab = [self.unk_token] + self.vocab
        self.stoi = {tok: i for i, tok in enumerate(self.vocab)}
        self.unk_id = 0
        self.vocab_size = len(self.vocab)
        return self

    def encode(self, text: str) -> list[int]:
        tokens = list(text)
        for pair in self.merges:
            merged = pair[0] + pair[1]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return [self.stoi.get(tok, self.unk_id) for tok in tokens]

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            tok = self.vocab[i] if 0 <= i < len(self.vocab) else self.unk_token
            if tok == self.unk_token:
                out.append(self.unk_token)  # represent unseen chars explicitly
            else:
                out.append(tok)
        return "".join(out)

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps({
            "merges": self.merges,
            "vocab": self.vocab,
            "unk_token": self.unk_token,
            "unk_id": self.unk_id,
        }))

    def load(self, path: str) -> "BPETokenizer":
        data = json.loads(Path(path).read_text())
        self.merges = [tuple(m) for m in data["merges"]]
        self.vocab = data["vocab"]
        self.unk_token = data.get("unk_token", "<unk>")
        self.unk_id = data.get("unk_id", 0)
        self.stoi = {tok: i for i, tok in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        return self