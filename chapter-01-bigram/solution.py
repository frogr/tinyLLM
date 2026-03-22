"""
Chapter 1: The Bigram Language Model — SOLUTION
=================================================

This is the completed version of exercises.py with all TODOs filled in.
If you get stuck, check the specific TODO here, but try to solve it yourself first!

Run this file to see the full working model:
    python solution.py

Dependencies: Make sure you've run `python ../data/download.py` first!
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# ============================================================================
# DEVICE SETUP
# ============================================================================
device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
eval_iters = 200

torch.manual_seed(1337)

# ============================================================================
# DATA LOADING
# ============================================================================

import os
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'input.txt')

# SOLUTION TODO 1: Read the data file
with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Dataset size: {len(text):,} characters")
print(f"First 100 characters:\n{text[:100]}")

# SOLUTION TODO 2: Build the vocabulary
chars = sorted(set(text))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

print(f"\nVocabulary size: {vocab_size} characters")
print(f"Characters: {''.join(chars)}")
print(f"Example: '{chars[0]}' -> {stoi[chars[0]]}")

# Helper functions for encoding/decoding
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# SOLUTION TODO 3: Encode the dataset and create train/val split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"\nEncoded data shape: {data.shape}")
print(f"Train size: {len(train_data):,}, Val size: {len(val_data):,}")
print(f"First 20 encoded values: {data[:20].tolist()}")
print(f"Decoded back: '{decode(data[:20].tolist())}'")

# ============================================================================
# BATCH LOADER
# ============================================================================

def get_batch(split):
    """
    Generate a random batch of training examples.

    Returns:
        x: input tensor,  shape (batch_size, block_size)
        y: target tensor, shape (batch_size, block_size)
    """
    source = train_data if split == 'train' else val_data

    # SOLUTION TODO 4: Implement the batch loader
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i:i+block_size] for i in ix])
    y = torch.stack([source[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# ============================================================================
# LOSS ESTIMATION
# ============================================================================

@torch.no_grad()
def estimate_loss():
    """Average the loss over many batches for a more stable estimate."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# ============================================================================
# THE BIGRAM MODEL
# ============================================================================

class BigramLanguageModel(nn.Module):
    """
    The simplest possible language model.

    It has ONE learnable component: an embedding table.
    Given a character (as an integer), it looks up that row in the table
    and uses it directly as the prediction for what comes next.
    """

    def __init__(self, vocab_size):
        super().__init__()
        # SOLUTION TODO 5: Create the embedding table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """
        Forward pass: given input indices, produce predictions.

        Args:
            idx: tensor of token indices, shape (batch_size, block_size)
            targets: tensor of target indices, shape (batch_size, block_size), or None

        Returns:
            logits: raw predictions, shape (batch_size, block_size, vocab_size)
            loss: cross-entropy loss (only if targets provided)
        """
        # SOLUTION TODO 6: Implement the forward pass
        logits = self.token_embedding_table(idx)  # (B, T, C)
        loss = None

        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)    # (B*T, C)
            targets_flat = targets.view(B * T)      # (B*T,)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Generate new text by repeatedly predicting the next character.

        Args:
            idx: starting context, shape (batch_size, current_length)
            max_new_tokens: how many new characters to generate

        Returns:
            idx: extended sequence, shape (batch_size, current_length + max_new_tokens)
        """
        # SOLUTION TODO 7: Implement text generation
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]              # (B, C)
            probs = F.softmax(logits, dim=-1)      # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


# ============================================================================
# CREATE THE MODEL
# ============================================================================

model = BigramLanguageModel(vocab_size).to(device)
print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")

# Test generation before training (should be random gibberish)
print("\n--- Before Training ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=100)
print(f"Generated text: '{decode(generated[0].tolist())}'")

# ============================================================================
# TRAINING LOOP
# ============================================================================

# SOLUTION TODO 8: Create the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# SOLUTION TODO 9: Implement the training loop
print("\n--- Training ---")
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter:5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Final loss
losses = estimate_loss()
print(f"Step {max_iters:5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

# ============================================================================
# GENERATE TEXT
# ============================================================================

print("\n--- After Training ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500)
print(f"Generated text:\n{decode(generated[0].tolist())}")
print("\n(This will be gibberish, but it should look more Shakespeare-ish than before training!)")
