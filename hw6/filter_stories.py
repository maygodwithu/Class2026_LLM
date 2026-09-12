"""
Given -- utility to carve a small "fine-tuning domain" out of the
TinyStories corpus: every story containing a keyword (e.g. "dragon"),
written back out in the same "<|endoftext|>"-separated format that
data/prepare.py expects.

This is what makes the fine-tuning exercise concrete: you pretrain a
general model on all of TinyStories (HW5), then fine-tune a copy of it
on just the "dragon stories" and compare generations before/after.

Usage:
    python filter_stories.py --input data_cache/raw/TinyStoriesV2-GPT4-valid.txt \
        --keyword dragon --output data_cache/raw/dragon_stories.txt
"""

import argparse
from pathlib import Path

SEP = "<|endoftext|>"


def filter_stories(input_path: Path, keyword: str) -> list[str]:
    text = input_path.read_text(encoding="utf-8")
    stories = [s.strip() for s in text.split(SEP) if s.strip()]
    return [s for s in stories if keyword.lower() in s.lower()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--keyword", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    matched = filter_stories(Path(args.input), args.keyword)
    Path(args.output).write_text(SEP.join(matched) + SEP, encoding="utf-8")
    print(f"{len(matched):,} stories contain '{args.keyword}' -> wrote {args.output}")
