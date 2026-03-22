"""
Chapter 6 Sandbox — Make It Yours!
===================================

This is your playground. You've built a GPT from scratch.
Now make it do something fun!

Ideas:
- Train on your own writing (emails, messages, blog posts)
- Train on song lyrics from your favorite artist
- Train on Python/JavaScript code
- Train on cooking recipes
- Combine multiple authors and see what comes out
- Try different model sizes and compare
- Implement beam search
- Add a simple web interface with Flask

Usage:
    python sandbox.py                    # Default: Shakespeare
    python sandbox.py path/to/data.txt   # Custom data
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import os

device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# You can import the full model and tokenizers from exercises.py or solution.py:
#
#   from solution import GPT, CharTokenizer, BPETokenizer, load_text, train_model
#
# Or copy-paste the parts you need and modify them freely.

# Your experiments here!
