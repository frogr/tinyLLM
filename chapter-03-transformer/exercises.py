"""
Chapter 3: The Full Transformer Block — Exercises
===================================================

We're building the complete transformer block:
  1. Multi-Head Attention (multiple attention heads in parallel)
  2. FeedForward network (two linear layers with ReLU)
  3. Transformer Block (attention + feedforward + residual + layernorm)
  4. Full language model using the transformer block

There are 5 TODOs. Work through them in order.

Run this file after each TODO — it's designed to work at every stage.
  python chapter-03-transformer/exercises.py

Hyperparameters for this chapter:
  n_embd     = 64   (embedding dimension)
  n_head     = 4    (number of attention heads)
  head_size  = 16   (n_embd // n_head, each head's dimension)
  block_size = 32   (context length — how many characters the model sees)
  n_layer    = 1    (just 1 transformer block for now)
  dropout    = 0.0  (no dropout yet — introduced in chapter 4)
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import os

# ============================================================================
# Hyperparameters
# ============================================================================

batch_size = 32         # how many sequences to process in parallel
block_size = 32         # context length (increased from ch2's 8)
max_iters = 5000        # training iterations
eval_interval = 500     # how often to evaluate
learning_rate = 1e-3    # optimizer learning rate
eval_iters = 200        # how many batches to average for eval loss
n_embd = 64             # embedding dimension
n_head = 4              # number of attention heads
n_layer = 1             # number of transformer blocks (just 1 for now)
dropout = 0.0           # dropout rate (0 = no dropout, introduced in ch4)

device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# ============================================================================
# Data Loading (same as chapters 1 & 2 — already complete)
# ============================================================================

# Read the dataset
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "input.txt")
with open(data_path, "r") as f:
    text = f.read()

# Build vocabulary from all unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size}")
print(f"Characters: {''.join(chars)}")

# Create mappings between characters and integers
stoi = {ch: i for i, ch in enumerate(chars)}  # string to integer
itos = {i: ch for i, ch in enumerate(chars)}  # integer to string
encode = lambda s: [stoi[c] for c in s]       # encode string → list of ints
decode = lambda l: "".join([itos[i] for i in l])  # decode list of ints → string

# Train/val split (90% train, 10% val)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Batch generator
def get_batch(split):
    """Generate a batch of inputs (x) and targets (y)."""
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])         # (B, T)
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix]) # (B, T)
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    """Estimate loss on train and val splits."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ============================================================================
# Head class (from Chapter 2 — already complete, no TODOs)
# ============================================================================

class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)    # (n_embd, head_size)
        self.query = nn.Linear(n_embd, head_size, bias=False)  # (n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size, bias=False)  # (n_embd, head_size)
        # register_buffer: a tensor that's part of the module but not a parameter
        # (not updated by the optimizer). We use it for the causal mask.
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape                                      # (batch, time, channels)
        k = self.key(x)                                         # (B, T, head_size)
        q = self.query(x)                                       # (B, T, head_size)
        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)            # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # causal mask
        wei = F.softmax(wei, dim=-1)                            # (B, T, T)
        # Weighted aggregation of values
        v = self.value(x)                                       # (B, T, head_size)
        out = wei @ v                                           # (B, T, head_size)
        return out

# ============================================================================
# TODO 1: Multi-Head Attention
# ============================================================================
#
# Create multiple attention heads that run in parallel, then concatenate
# their outputs and project back to n_embd dimensions.
#
# Architecture:
#   Input (B, T, n_embd)
#     → Split into n_head independent heads, each of size head_size
#     → Each head computes attention independently: (B, T, head_size)
#     → Concatenate all heads: (B, T, n_head * head_size) = (B, T, n_embd)
#     → Linear projection: (B, T, n_embd) → (B, T, n_embd)
#
# Parameters to create in __init__:
#   self.heads : nn.ModuleList of n_head Head instances, each with head_size
#   self.proj  : nn.Linear(n_embd, n_embd) — projection after concatenation
#
# In forward(self, x):
#   1. Run each head on x, collecting outputs in a list
#   2. Concatenate along the last dimension (dim=-1)
#   3. Pass through self.proj
#   4. Return result
#
# Shape trace:
#   x:          (B, T, n_embd)    e.g. (32, 32, 64)
#   per head:   (B, T, head_size) e.g. (32, 32, 16)
#   concat:     (B, T, n_embd)    e.g. (32, 32, 64)  ← 4 heads × 16 = 64
#   projected:  (B, T, n_embd)    e.g. (32, 32, 64)

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        # ---- TODO 1: Create the heads and projection ----
        # self.heads = nn.ModuleList([...])
        # self.proj = nn.Linear(...)
        pass  # Remove this line when you fill in the TODO

    def forward(self, x):
        # ---- TODO 1: Implement the forward pass ----
        # 1. Run each head: [h(x) for h in self.heads]
        # 2. Concatenate: torch.cat(..., dim=-1)
        # 3. Project: self.proj(...)
        # 4. Return
        return x  # Placeholder — replace with your implementation

# ============================================================================
# TODO 2: FeedForward Network
# ============================================================================
#
# A simple two-layer network with ReLU activation.
# This is applied to each token position independently.
#
# Architecture:
#   Input (B, T, n_embd)
#     → Linear(n_embd, 4 * n_embd)     — expand to larger hidden dimension
#     → ReLU()                          — nonlinearity
#     → Linear(4 * n_embd, n_embd)     — project back to n_embd
#
# The 4x expansion is a convention from the original transformer paper.
# The feedforward network gives each token a chance to "think" about the
# information gathered by attention.
#
# Shape trace:
#   x:        (B, T, n_embd)       e.g. (32, 32, 64)
#   expand:   (B, T, 4 * n_embd)   e.g. (32, 32, 256)
#   relu:     (B, T, 4 * n_embd)   e.g. (32, 32, 256)  ← negative values zeroed
#   project:  (B, T, n_embd)       e.g. (32, 32, 64)

class FeedForward(nn.Module):
    """A simple feedforward network: expand → ReLU → compress."""

    def __init__(self, n_embd):
        super().__init__()
        # ---- TODO 2: Create the feedforward network ----
        # Use nn.Sequential with:
        #   nn.Linear(n_embd, 4 * n_embd),
        #   nn.ReLU(),
        #   nn.Linear(4 * n_embd, n_embd),
        #
        # self.net = nn.Sequential(...)
        pass  # Remove this line when you fill in the TODO

    def forward(self, x):
        # ---- TODO 2: Implement the forward pass ----
        # return self.net(x)
        return x  # Placeholder — replace with your implementation

# ============================================================================
# TODO 3: Transformer Block
# ============================================================================
#
# Combine multi-head attention and feedforward with residual connections
# and layer normalization. This is the core building block of GPT.
#
# Architecture (pre-norm):
#   x = x + MultiHeadAttention(LayerNorm(x))    ← attention + residual
#   x = x + FeedForward(LayerNorm(x))           ← feedforward + residual
#
# Parameters to create in __init__:
#   self.sa    : MultiHeadAttention(n_head, head_size)
#   self.ffwd  : FeedForward(n_embd)
#   self.ln1   : nn.LayerNorm(n_embd)    — layer norm before attention
#   self.ln2   : nn.LayerNorm(n_embd)    — layer norm before feedforward
#
# In forward(self, x):
#   1. Apply layer norm, then attention, then add residual:
#      x = x + self.sa(self.ln1(x))
#   2. Apply layer norm, then feedforward, then add residual:
#      x = x + self.ffwd(self.ln2(x))
#   3. Return x
#
# IMPORTANT: This is "pre-norm" — LayerNorm comes BEFORE each sub-layer.
#
# Shape trace (all shapes stay the same — that's the beauty of residuals):
#   x:                (B, T, n_embd)    e.g. (32, 32, 64)
#   ln1(x):           (B, T, n_embd)    e.g. (32, 32, 64)
#   sa(ln1(x)):       (B, T, n_embd)    e.g. (32, 32, 64)
#   x + sa(ln1(x)):   (B, T, n_embd)    e.g. (32, 32, 64)  ← same shape!
#   ln2(x):           (B, T, n_embd)    e.g. (32, 32, 64)
#   ffwd(ln2(x)):     (B, T, n_embd)    e.g. (32, 32, 64)
#   x + ffwd(ln2(x)): (B, T, n_embd)    e.g. (32, 32, 64)  ← same shape!

class TransformerBlock(nn.Module):
    """Transformer block: attention + feedforward with residuals and layer norm."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        # ---- TODO 3: Create the sub-layers ----
        # self.sa = MultiHeadAttention(n_head, head_size)
        # self.ffwd = FeedForward(n_embd)
        # self.ln1 = nn.LayerNorm(n_embd)
        # self.ln2 = nn.LayerNorm(n_embd)
        pass  # Remove this line when you fill in the TODO

    def forward(self, x):
        # ---- TODO 3: Implement the forward pass with residual connections ----
        # x = x + self.sa(self.ln1(x))     # attention + residual
        # x = x + self.ffwd(self.ln2(x))   # feedforward + residual
        # return x
        return x  # Placeholder — replace with your implementation

# ============================================================================
# TODO 4 & 5: Full Language Model
# ============================================================================
#
# The complete model that brings everything together:
#   1. Token embedding table        — look up embedding for each character
#   2. Position embedding table     — add positional information
#   3. Transformer block(s)         — process the sequence
#   4. Final layer norm             — normalize before the output head
#   5. Linear head                  — project to vocabulary size for predictions
#
# TODO 4 is in __init__: Create all the layers.
# TODO 5 is in forward: Wire them together.
#
# Shape trace through the full model:
#   idx:             (B, T)           e.g. (32, 32)        — input token indices
#   tok_emb:         (B, T, n_embd)   e.g. (32, 32, 64)   — token embeddings
#   pos_emb:         (T, n_embd)      e.g. (32, 64)       — position embeddings
#                                                            (broadcast over batch)
#   x = tok + pos:   (B, T, n_embd)   e.g. (32, 32, 64)   — combined embeddings
#   after block(s):  (B, T, n_embd)   e.g. (32, 32, 64)   — processed
#   after ln_f:      (B, T, n_embd)   e.g. (32, 32, 64)   — normalized
#   logits:          (B, T, vocab)    e.g. (32, 32, 65)    — prediction scores

class TransformerLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        # ---- TODO 4: Create the model layers ----
        #
        # self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        #     Maps each token (integer) to an n_embd-dimensional vector.
        #
        # self.position_embedding_table = nn.Embedding(block_size, n_embd)
        #     Maps each position (0 to block_size-1) to an n_embd-dimensional vector.
        #
        # self.blocks = nn.Sequential(
        #     *[TransformerBlock(n_embd, n_head=n_head) for _ in range(n_layer)]
        # )
        #     Stack of n_layer transformer blocks. For now n_layer=1, but this
        #     generalizes to any number of blocks.
        #
        # self.ln_f = nn.LayerNorm(n_embd)
        #     Final layer norm applied after all transformer blocks.
        #
        # self.lm_head = nn.Linear(n_embd, vocab_size)
        #     Projects from n_embd to vocab_size to get prediction scores.
        #

        # Placeholder layers so the model can be created before TODO 4 is done
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        # Uncomment these when you implement TODO 4:
        # self.blocks = nn.Sequential(
        #     *[TransformerBlock(n_embd, n_head=n_head) for _ in range(n_layer)]
        # )
        # self.ln_f = nn.LayerNorm(n_embd)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # ---- TODO 5: Implement the forward pass ----
        #
        # Step 1: Get token embeddings
        tok_emb = self.token_embedding_table(idx)          # (B, T, n_embd)
        #
        # Step 2: Get position embeddings
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)                 # (T,)
        )                                                   # (T, n_embd)
        #
        # Step 3: Combine token and position embeddings
        x = tok_emb + pos_emb                               # (B, T, n_embd)
        #
        # Step 4: Pass through transformer block(s)
        #   x = self.blocks(x)                              # (B, T, n_embd)
        #
        # Step 5: Apply final layer norm
        #   x = self.ln_f(x)                                # (B, T, n_embd)
        #
        # Step 6: Project to vocabulary size
        logits = self.lm_head(x)                            # (B, T, vocab_size)

        # Compute loss if targets are provided
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)             # (B*T, C)
            targets_flat = targets.view(B * T)              # (B*T,)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Generate new tokens autoregressively."""
        for _ in range(max_new_tokens):
            # Crop context to block_size (can't exceed position embeddings)
            idx_cond = idx[:, -block_size:]                 # (B, T) where T <= block_size
            # Get predictions
            logits, loss = self(idx_cond)                   # (B, T, vocab_size)
            # Take only the last time step
            logits = logits[:, -1, :]                       # (B, vocab_size)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)               # (B, vocab_size)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)         # (B, T+1)
        return idx

# ============================================================================
# Training Loop (already complete)
# ============================================================================

if __name__ == "__main__":
    model = TransformerLanguageModel()
    model = model.to(device)

    # Print model size
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print()

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    print("Training...")
    print("-" * 50)
    for iter in range(max_iters):
        # Every eval_interval steps, evaluate on train and val
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
            print(f"step {iter:5d} | "
                  f"train loss {losses['train']:.4f} | "
                  f"val loss {losses['val']:.4f}")

        # Get a batch
        xb, yb = get_batch("train")

        # Forward pass
        logits, loss = model(xb, yb)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("-" * 50)
    print(f"Final train loss: {losses['train']:.4f}")
    print(f"Final val loss:   {losses['val']:.4f}")

    # Generate some text
    print("\n" + "=" * 50)
    print("GENERATED TEXT:")
    print("=" * 50)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=500)
    print(decode(generated[0].tolist()))
