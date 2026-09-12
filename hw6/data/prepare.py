"""
Turn a raw TinyStories .txt file into train.bin / val.bin token arrays.

Pipeline: split raw text into individual stories (TinyStories separates
them with the literal "<|endoftext|>" marker) -> shuffle + split into
train/val -> train the chosen tokenizer on the TRAIN split only (so the
vocabulary never "sees" validation text) -> encode both splits.

Token ids are written as uint16 (every vocab_size we use is < 65536) via
small flushed chunks rather than one giant Python list, so memory stays
bounded even for the much larger "real run" TinyStories train file later
-- a 400M-token list of Python ints would be tens of GB, but writing in
1M-token chunks keeps this script's memory flat regardless of corpus size.
train.py then reads these files back with np.memmap, so at no point does
the full token stream need to fit in RAM at once.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from data.tokenizer_char import CharTokenizer
from data.tokenizer_bpe import BpeTokenizer

SEP = "<|endoftext|>"


def load_stories(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [s.strip() for s in text.split(SEP) if s.strip()]


def train_val_split(stories: list[str], val_fraction: float, seed: int = 42):
    stories = stories.copy()
    random.Random(seed).shuffle(stories)
    n_val = max(1, int(len(stories) * val_fraction))
    return stories[n_val:], stories[:n_val]


def build_tokenizer(kind: str, train_stories: list[str], vocab_size: int, out_dir: Path,
                     reuse_tokenizer_path: str | None = None):
    if reuse_tokenizer_path is not None:
        # Fine-tuning MUST reuse the base model's exact tokenizer: token id
        # 17 in the fine-tuning data has to mean the same sub-word it meant
        # during pretraining, or the (frozen and unfrozen) embeddings the
        # model already learned are meaningless for the new data. So here
        # we skip training entirely and just load the existing tokenizer.
        tok = CharTokenizer.load(reuse_tokenizer_path) if kind == "char" else BpeTokenizer.load(reuse_tokenizer_path)
        tok.save(out_dir / f"tokenizer_{kind}.json")
        return tok
    if kind == "char":
        # Re-include SEP so its characters are guaranteed to be in-vocab
        # when we later encode "story + SEP" for every story.
        sample_text = SEP.join(train_stories) + SEP
        tok = CharTokenizer.train(sample_text)
        tok.save(out_dir / "tokenizer_char.json")
    elif kind == "bpe":
        corpus_path = out_dir / "train_corpus.txt"
        corpus_path.write_text(SEP.join(train_stories), encoding="utf-8")
        tok = BpeTokenizer.train(corpus_path, vocab_size=vocab_size)
        tok.save(out_dir / "tokenizer_bpe.json")
    else:
        raise ValueError(f"unknown tokenizer kind: {kind}")
    return tok


def encode_split(stories: list[str], tokenizer, out_path: Path, flush_every: int = 1_000_000) -> int:
    total = 0
    buffer: list[int] = []
    with open(out_path, "wb") as f:
        for story in stories:
            buffer.extend(tokenizer.encode(story + SEP))
            if len(buffer) >= flush_every:
                np.array(buffer, dtype=np.uint16).tofile(f)
                total += len(buffer)
                buffer = []
        if buffer:
            np.array(buffer, dtype=np.uint16).tofile(f)
            total += len(buffer)
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="raw TinyStories .txt file")
    parser.add_argument("--tokenizer", choices=["char", "bpe"], required=True)
    parser.add_argument("--vocab-size", type=int, default=6000, help="target vocab size (bpe only)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--reuse-tokenizer", default=None,
                         help="path to an existing tokenizer_{char,bpe}.json to reuse instead of "
                              "training a new one -- required for fine-tuning on new data with a "
                              "model that already has pretrained embeddings")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stories = load_stories(Path(args.input))
    train_stories, val_stories = train_val_split(stories, args.val_fraction)
    print(f"stories: {len(stories):,} total -> {len(train_stories):,} train / {len(val_stories):,} val")

    tok = build_tokenizer(args.tokenizer, train_stories, args.vocab_size, out_dir, args.reuse_tokenizer)
    print(f"tokenizer: {args.tokenizer}, actual vocab_size = {tok.vocab_size:,}")

    n_train = encode_split(train_stories, tok, out_dir / "train.bin")
    n_val = encode_split(val_stories, tok, out_dir / "val.bin")
    print(f"train.bin: {n_train:,} tokens  ({n_train / max(len(train_stories), 1):.1f} tokens/story avg)")
    print(f"val.bin:   {n_val:,} tokens")

    meta = {
        "source_file": str(args.input),
        "tokenizer_kind": args.tokenizer,
        "tokenizer_path": str(out_dir / f"tokenizer_{args.tokenizer}.json"),
        "vocab_size": tok.vocab_size,
        "n_train_stories": len(train_stories),
        "n_val_stories": len(val_stories),
        "n_train_tokens": n_train,
        "n_val_tokens": n_val,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote to: {out_dir}")


if __name__ == "__main__":
    main()
