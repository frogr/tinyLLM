"""
Chapter 3 — Sandbox

Your playground for experimentation. Try things here without
worrying about breaking the exercises.

Ideas to try:
  - Change n_head from 4 to 1 or 8 — how does it affect loss and generation?
  - Change n_layer from 1 to 2 or 3 — does stacking blocks help?
  - Add dropout=0.1 — does regularization help at this small scale?
  - Visualize attention weights from different heads — do they specialize?
  - Remove residual connections — how badly does training break?
  - Remove layer norm — what happens to the loss curve?
  - Try post-norm instead of pre-norm — is it less stable?
  - Compare parameter counts across chapters
  - Print tensor shapes at every step in the block to verify your understanding
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
