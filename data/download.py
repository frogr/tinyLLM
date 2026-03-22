"""
Downloads the Tiny Shakespeare dataset.

This is ~1MB of Shakespeare's complete works, concatenated into a single text file.
It's the "hello world" dataset for character-level language models.
"""

import os
import requests

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(DATA_DIR, "input.txt")


def download():
    if os.path.exists(OUTPUT_PATH):
        print(f"Dataset already exists at {OUTPUT_PATH}")
        print(f"File size: {os.path.getsize(OUTPUT_PATH):,} bytes")
        return

    print(f"Downloading Tiny Shakespeare dataset...")
    response = requests.get(DATA_URL)
    response.raise_for_status()

    with open(OUTPUT_PATH, "w") as f:
        f.write(response.text)

    print(f"Saved to {OUTPUT_PATH}")
    print(f"File size: {len(response.text):,} characters")
    print(f"First 200 characters:\n{response.text[:200]}")


if __name__ == "__main__":
    download()
