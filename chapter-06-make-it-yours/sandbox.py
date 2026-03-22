"""
Chapter 6 — Sandbox

Your open-ended playground. This is YOUR chapter — experiment freely!

Ideas to try:
  - Train on your own writing (emails, Slack messages, code comments)
  - Train on song lyrics, poetry, or recipes
  - Try different model sizes and see the quality/speed tradeoff
  - Implement beam search as another generation strategy
  - Compare character-level vs BPE tokenization on the same data
  - Try different positional encoding schemes
  - Implement a simple chat interface that generates responses
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Your experiments here...
