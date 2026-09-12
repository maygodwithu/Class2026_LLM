import json
from pathlib import Path

from data.tokenizer_char import CharTokenizer
from data.tokenizer_bpe import BpeTokenizer


def load_meta(data_dir: str | Path) -> dict:
    return json.loads((Path(data_dir) / "meta.json").read_text(encoding="utf-8"))


def load_tokenizer(data_dir: str | Path):
    """Load whichever tokenizer (char or bpe) was used to prepare data_dir,
    based on the meta.json written by data/prepare.py."""
    meta = load_meta(data_dir)
    if meta["tokenizer_kind"] == "char":
        return CharTokenizer.load(meta["tokenizer_path"])
    return BpeTokenizer.load(meta["tokenizer_path"])
