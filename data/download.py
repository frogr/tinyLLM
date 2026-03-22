"""
Download the tiny Shakespeare dataset.

This is the training corpus for all chapters — about 1MB of Shakespeare's
collected works. It's small enough to train on in minutes, but rich enough
to learn interesting patterns.

Usage:
    python data/download.py
"""

import os
import requests

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(DATA_DIR, "input.txt")


def download():
    if os.path.exists(OUTPUT_PATH):
        print(f"Dataset already exists at {OUTPUT_PATH}")
        print(f"File size: {os.path.getsize(OUTPUT_PATH):,} bytes")
        return

    print(f"Downloading tiny Shakespeare dataset...")
    response = requests.get(URL)
    response.raise_for_status()

    with open(OUTPUT_PATH, "w") as f:
        f.write(response.text)

    size = os.path.getsize(OUTPUT_PATH)
    print(f"Downloaded {size:,} bytes to {OUTPUT_PATH}")

    # Quick peek at the data
    lines = response.text.split("\n")
    print(f"Total lines: {len(lines):,}")
    print(f"Total characters: {len(response.text):,}")
    print(f"\nFirst 5 lines:")
    for line in lines[:5]:
        print(f"  {line}")

    # Show unique characters (this becomes our vocabulary)
    chars = sorted(set(response.text))
    print(f"\nUnique characters (vocabulary): {len(chars)}")
    print(f"Characters: {''.join(chars)}")


if __name__ == "__main__":
    download()
