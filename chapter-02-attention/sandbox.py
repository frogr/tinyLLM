"""
Chapter 2 Sandbox
=================

Experiment with attention! Ideas:
- Visualize attention weights: what does the model attend to?
- Try different head_size values (4, 16, 32, 64)
- Remove position embeddings — what happens?
- Print the Q, K, V matrices for a tiny example
- Plot attention patterns with matplotlib
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
