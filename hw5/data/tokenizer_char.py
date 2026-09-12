"""
Zero-dependency character-level tokenizer.

Every unique character in the training text becomes one token. This is
the simplest possible tokenizer -- no merges, no library -- and it makes
the vocab_size essentially free (see configs.py's char_cmp_6m, where the
embedding table is <1% of total params), at the cost of much longer
token sequences per story than a BPE tokenizer would produce.
"""

import json
from pathlib import Path


class CharTokenizer:
    def __init__(self, stoi: dict):
        self.stoi = stoi
        self.itos = {i: ch for ch, i in stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @classmethod
    def train(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        return cls(stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.stoi), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        stoi = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(stoi)
