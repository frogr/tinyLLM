"""
Chapter 5: Generation and Sampling Strategies — Exercises
=========================================================

We have a trained transformer. Now we learn to CONTROL the generation.

This chapter is different from previous ones: the model code is provided
complete (no TODOs). Your job is to implement the generation strategies:
temperature, top-k, and top-p sampling.

The model is smaller than Chapter 4 so training is fast (2-5 minutes).
After training, you'll experiment with different sampling strategies to
see how they affect the generated text.

TODOs:
  1. Implement temperature scaling
  2. Implement top-k sampling
  3. Implement top-p (nucleus) sampling
  4. Create a comparison function
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import os

# ---------------------------------------------------------------------------
# Hyperparameters — smaller model for fast training
# ---------------------------------------------------------------------------
batch_size = 64
block_size = 128          # context window
n_embd = 128              # embedding dimension
n_head = 4                # number of attention heads
n_layer = 4               # number of transformer blocks
dropout = 0.2
learning_rate = 3e-4
max_iters = 3000
eval_interval = 500
eval_iters = 200

device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# Data loading (same as previous chapters)
# ---------------------------------------------------------------------------
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "input.txt")
if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"Data file not found at {data_path}\n"
        "Run: python data/download.py"
    )

with open(data_path, "r") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
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


# ---------------------------------------------------------------------------
# Model (complete — no TODOs here)
# ---------------------------------------------------------------------------
class Head(nn.Module):
    """Single head of self-attention."""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Simple feedforward network with ReLU."""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: attention followed by feedforward."""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """
        Generate tokens one at a time, using the specified sampling strategy.

        Args:
            idx: Starting token indices, shape (B, T)
            max_new_tokens: How many tokens to generate
            temperature: Temperature for scaling logits (default 1.0)
            top_k: If set, only sample from the top k most likely tokens
            top_p: If set, only sample from the smallest set of tokens
                   whose cumulative probability exceeds p (nucleus sampling)

        Returns:
            idx: Token indices including generated tokens, shape (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx[:, -block_size:]
            # Forward pass
            logits, _ = self(idx_cond)
            # Get logits for the last position
            logits = logits[:, -1, :]  # (B, vocab_size)

            # ------------------------------------------------------------------
            # TODO 1: Temperature scaling
            # Divide logits by temperature. This reshapes the probability
            # distribution — lower temperature makes it sharper (more confident),
            # higher temperature makes it flatter (more random).
            #
            # Be careful: temperature of 0 would cause division by zero.
            # Use a small epsilon (1e-8) as a floor.
            #
            # Your code (1 line):
            # logits = ...
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # TODO 2: Top-k sampling
            # If top_k is specified:
            #   1. Find the top_k largest logit values using torch.topk()
            #   2. Find the minimum value among those top_k values
            #   3. Set all logits below that minimum to -infinity (float('-inf'))
            #      so they get zero probability after softmax
            #
            # Hint: torch.topk(logits, k) returns (values, indices)
            #       You only need the values to find the threshold.
            #       Use logits[logits < threshold] = float('-inf')
            #       But be careful with batched tensors — use the last value
            #       from topk as the threshold: values[:, [-1]]
            #
            # Your code (3-4 lines inside an if block):
            # if top_k is not None:
            #     ...
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # TODO 3: Top-p (nucleus) sampling
            # If top_p is specified:
            #   1. Convert logits to probabilities with softmax
            #   2. Sort probabilities in descending order (torch.sort with descending=True)
            #   3. Compute cumulative sum of sorted probabilities (torch.cumsum)
            #   4. Create a mask for tokens where cumsum > top_p
            #      (but shift the mask right by 1 so we always keep at least one token)
            #   5. Use the sorted indices to map back: set masked logits to -inf
            #
            # This is trickier than top-k! Here's a step-by-step approach:
            #
            #   sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            #   cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            #
            #   # Remove tokens with cumulative probability above threshold
            #   # Shift right so we keep the token that crosses the threshold
            #   sorted_mask = cumulative_probs - sorted_probs >= top_p
            #
            #   # Zero out the masked probabilities in sorted order
            #   sorted_probs[sorted_mask] = 0.0
            #
            #   # Scatter back to original order
            #   probs.scatter_(1, sorted_indices, sorted_probs)
            #
            #   # Sample from the filtered distribution
            #   idx_next = torch.multinomial(probs, num_samples=1)
            #   idx = torch.cat((idx, idx_next), dim=1)
            #   continue  # Skip the normal sampling below
            #
            # Your code (10-12 lines inside an if block):
            # if top_p is not None:
            #     ...
            # ------------------------------------------------------------------

            # Standard sampling (used when top_p is not specified,
            # or after temperature/top_k filtering)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
print("=" * 60)
print("Training a small GPT model on Shakespeare...")
print(f"Parameters: {n_layer} layers, {n_head} heads, {n_embd} embedding dim")
print(f"Training for {max_iters} iterations")
print("=" * 60)

model = GPTLanguageModel().to(device)
param_count = sum(p.numel() for p in model.parameters())
print(f"Model has {param_count:,} parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        print(f"Step {iter:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("\nTraining complete!")


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------
def generate_text(prompt="", max_tokens=200, temperature=1.0, top_k=None, top_p=None):
    """Generate text from the model with the given sampling settings."""
    if prompt:
        context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)

    with torch.no_grad():
        output = model.generate(
            context,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
    return decode(output[0].tolist())


# ---------------------------------------------------------------------------
# TODO 4: Comparison function
# ---------------------------------------------------------------------------
# Implement a function that generates text with different settings and prints
# them side by side for easy comparison.
#
# The function should:
#   1. Print a "Temperature Comparison" section with temperatures 0.5, 1.0, 1.5
#   2. Print a "Top-k Comparison" section with k values 5, 20, 50
#   3. Print a "Top-p Comparison" section with p values 0.5, 0.9, 0.95
#   4. Print a "Combined Strategies" section with a few combinations
#
# Use generate_text() for each generation. Use a consistent prompt like
# "ROMEO:" and generate 200 tokens each time.
#
# Format example:
#   === Temperature Comparison ===
#   --- Temperature 0.5 ---
#   [generated text]
#
#   --- Temperature 1.0 ---
#   [generated text]
#   ...
#
# Your code here:
# def compare_sampling_strategies():
#     prompt = "ROMEO:"
#     max_tokens = 200
#     ...

# ------------------------------------------------------------------
# Run the comparison
# ------------------------------------------------------------------
# Uncomment after implementing TODO 4:
# compare_sampling_strategies()

# ---------------------------------------------------------------------------
# Quick test — vanilla generation (works even without completing any TODOs)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Sample generation (vanilla sampling, temperature=1.0):")
print("=" * 60)
print(generate_text(prompt="ROMEO:", max_tokens=200))
