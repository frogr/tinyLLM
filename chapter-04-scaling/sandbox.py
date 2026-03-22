"""
Chapter 4 Sandbox
=================

Experiment with scaling! Ideas:
- Try different model sizes and compare loss curves
- Plot train vs val loss to see overfitting
- Experiment with dropout values
- Try different learning rate schedules
- Compare generation quality at different training checkpoints
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
