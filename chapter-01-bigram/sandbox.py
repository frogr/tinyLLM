"""
Chapter 1 Sandbox
=================

Your playground for experimentation. Try things, break things, learn things.

Some ideas to get you started:
- Change the learning rate and see what happens
- Try different block_size values
- Look at the embedding table values before and after training
- Count bigram frequencies manually and compare to the model
- Plot the loss curve with matplotlib
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
