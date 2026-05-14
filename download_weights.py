"""Download model weights from GitHub Releases.

Usage:
    python download_weights.py
"""

import os
import sys
import urllib.request
from pathlib import Path

WEIGHTS_DIR = Path(__file__).parent / "weights"
RELEASE_BASE = "https://github.com/rasaidarkh/astana-tree-detection/releases/download/v1.0"

FILES = [
    ("deepforest_astana.pl", "DeepForest fine-tuned on Astana (245 MB)"),
]


def download(url: str, dest: Path) -> None:
    print(f"Downloading {dest.name} ...")
    tmp = dest.with_suffix(".tmp")
    try:
        def progress(block_num, block_size, total):
            downloaded = block_num * block_size
            if total > 0:
                pct = min(100, downloaded * 100 // total)
                mb = downloaded / 1_048_576
                total_mb = total / 1_048_576
                print(f"\r  {mb:.1f} / {total_mb:.1f} MB  ({pct}%)", end="", flush=True)
        urllib.request.urlretrieve(url, tmp, reporthook=progress)
        print()
        tmp.rename(dest)
        print(f"  Saved → {dest}")
    except Exception as e:
        tmp.unlink(missing_ok=True)
        print(f"\n  ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    WEIGHTS_DIR.mkdir(exist_ok=True)
    for filename, description in FILES:
        dest = WEIGHTS_DIR / filename
        if dest.exists():
            print(f"  {filename} already exists, skipping.")
            continue
        print(f"\n{description}")
        download(f"{RELEASE_BASE}/{filename}", dest)
    print("\nDone. Start the server: uvicorn backend.main:app --host 127.0.0.1 --port 8000")


if __name__ == "__main__":
    main()
