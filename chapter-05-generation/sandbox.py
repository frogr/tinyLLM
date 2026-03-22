"""
Chapter 5 Sandbox
=================

Experiment with generation strategies! Ideas:
- Find the "sweet spot" temperature for Shakespeare
- Compare top-k vs top-p at equivalent restrictiveness
- Try combining temperature with top-k or top-p
- Generate completions for famous Shakespeare openings
- Measure how repetitive each strategy is
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
