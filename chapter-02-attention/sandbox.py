"""
Chapter 2 — Sandbox

Your playground for experimentation. Try things here without
worrying about breaking the exercises.

Ideas to try:
  - Visualize the attention mask (tril) — print it for different sizes
  - Create Q, K, V matrices by hand and compute attention step by step
  - See what happens to softmax with very large or very small inputs
  - Remove position embeddings and see how generation changes
  - Plot attention weights as a heatmap (see example below)
  - Try different head_size values (8, 16, 32, 64) — how does it affect learning?
  - Compare the attention model's loss to the bigram model's loss
  - What happens if you remove the 1/sqrt(d_k) scaling?
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

# ============================================================================
# Experiment: Visualize the causal mask
# ============================================================================
# Uncomment to run:
#
# T = 8
# tril = torch.tril(torch.ones(T, T))
# print("Causal mask (tril):")
# print(tril)
# print()
# # What softmax does to masked vs unmasked scores:
# scores = torch.randn(T, T)
# scores = scores.masked_fill(tril == 0, float('-inf'))
# weights = F.softmax(scores, dim=-1)
# print("Attention weights after masking + softmax:")
# print(weights)
# print("Each row sums to:", weights.sum(dim=-1))

# ============================================================================
# Experiment: Plot attention weights
# ============================================================================
# Requires matplotlib: pip install matplotlib
# Uncomment to run after training a model:
#
# import matplotlib.pyplot as plt
#
# # Get attention weights from a trained Head
# # (you'd need to modify Head.forward to also return `wei`)
# # Example with random weights for illustration:
# T = 8
# tril = torch.tril(torch.ones(T, T))
# scores = torch.randn(T, T)
# scores = scores.masked_fill(tril == 0, float('-inf'))
# weights = F.softmax(scores, dim=-1)
#
# plt.figure(figsize=(6, 5))
# plt.imshow(weights.detach().numpy(), cmap='Blues')
# plt.colorbar(label='Attention Weight')
# plt.xlabel('Key position (attending TO)')
# plt.ylabel('Query position (attending FROM)')
# plt.title('Single-Head Attention Weights')
# plt.tight_layout()
# plt.savefig('attention_weights.png')
# plt.show()
# print("Saved attention_weights.png")

# ============================================================================
# Experiment: Softmax behavior with scaling
# ============================================================================
# Uncomment to run:
#
# # Without scaling — values get extreme
# x_unscaled = torch.tensor([10.0, 20.0, 30.0])
# print("Unscaled softmax:", F.softmax(x_unscaled, dim=-1))
# # One value dominates — gradient is near zero for the others
#
# # With scaling — values stay manageable
# head_size = 32
# x_scaled = x_unscaled / (head_size ** 0.5)
# print("Scaled softmax:  ", F.softmax(x_scaled, dim=-1))
# # More spread out — gradients flow to all positions

# Your experiments here...
