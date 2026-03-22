"""
Chapter 3 Sandbox
=================

Experiment with the transformer architecture! Ideas:
- Try different numbers of heads (1, 2, 4, 8)
- Change the feedforward multiplier (4x → 2x or 8x)
- Remove residual connections and see what happens
- Remove LayerNorm and see what happens
- Print intermediate tensor shapes to verify your understanding
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# Your experiments here!
