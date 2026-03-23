"""
Chapter 1 — Sandbox

Your playground for experimentation. Try things here without
worrying about breaking the exercises.

Ideas to try:
  - What happens with different learning rates?
  - What if you change batch_size or block_size?
  - Can you print out the embedding table and see what it learned?
  - What does the probability distribution look like for specific characters?
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
