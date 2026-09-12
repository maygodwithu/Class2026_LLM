"""
Fetch a raw TinyStories text file from the roneneldan/TinyStories dataset
on HuggingFace.

Default is the GPT-4-generated validation split (~22.5MB) -- small enough
to download, tokenize, and smoke-test train against in minutes. The full
GPT-4 training split (~2.2GB) is available via --file for a later "real"
run; it is NOT downloaded by default.
"""

import argparse
from pathlib import Path

REPO_ID = "roneneldan/TinyStories"
DEFAULT_FILE = "TinyStoriesV2-GPT4-valid.txt"
RESOLVE_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{{filename}}"


def download(filename: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / filename
    if dest.exists():
        print(f"already downloaded: {dest} ({dest.stat().st_size:,} bytes)")
        return dest

    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=REPO_ID, repo_type="dataset", filename=filename,
            local_dir=str(out_dir),
        )
        dest = Path(path)
    except ImportError:
        # Zero-new-dependency fallback: plain streamed HTTP download.
        import requests

        url = RESOLVE_URL.format(filename=filename)
        print(f"huggingface_hub not available, falling back to requests: {url}")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)

    print(f"downloaded: {dest} ({dest.stat().st_size:,} bytes)")
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=DEFAULT_FILE,
                         help="e.g. TinyStoriesV2-GPT4-valid.txt (smoke test, default) "
                              "or TinyStoriesV2-GPT4-train.txt (2.2GB, real run)")
    parser.add_argument("--out-dir", default="data_cache/raw")
    args = parser.parse_args()
    download(args.file, Path(args.out_dir))
