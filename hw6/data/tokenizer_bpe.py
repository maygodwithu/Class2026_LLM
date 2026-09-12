"""
Byte-level BPE tokenizer, trained with the HuggingFace `tokenizers`
library rather than hand-rolled.

We use byte-level pre-tokenization (each of the 256 possible bytes is a
base token) specifically so the tokenizer can represent *any* input --
there's no [UNK] token and nothing can fail to encode, which matters
once students start feeding it text the trainer never saw.

Note: `tiktoken` (already installed) can only *load* fixed pretrained
vocabularies (e.g. GPT-2's 50257), it has no training API -- which is
exactly why it's not used here, where the whole point is training a
small (4000-8000 token) vocabulary sized for a 1-15M parameter model.
"""

from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers


class BpeTokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self._tok = tokenizer

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @classmethod
    def train(cls, corpus_path: str | Path, vocab_size: int) -> "BpeTokenizer":
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<|endoftext|>"],
            show_progress=True,
        )
        tok.train(files=[str(corpus_path)], trainer=trainer)
        return cls(tok)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)

    def save(self, path: str | Path):
        self._tok.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "BpeTokenizer":
        return cls(Tokenizer.from_file(str(path)))
