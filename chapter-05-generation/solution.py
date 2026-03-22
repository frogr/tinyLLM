"""
Chapter 5: Generation and Sampling Strategies — Solution
========================================================

Complete working implementation of temperature, top-k, and top-p sampling.
Run this file to see all sampling strategies compared side by side.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import os

# ---------------------------------------------------------------------------
# Hyperparameters — smaller model for fast training
# ---------------------------------------------------------------------------
batch_size = 64
block_size = 128
n_embd = 128
n_head = 4
n_layer = 4
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
# Data loading
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
# Model
# ---------------------------------------------------------------------------
class Head(nn.Module):
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
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
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
        Generate tokens with temperature, top-k, and/or top-p sampling.

        Args:
            idx: Starting token indices, shape (B, T)
            max_new_tokens: How many tokens to generate
            temperature: Divide logits by this value (default 1.0)
            top_k: If set, only sample from top k tokens
            top_p: If set, use nucleus sampling with this threshold
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx[:, -block_size:]
            # Forward pass
            logits, _ = self(idx_cond)
            # Get logits for the last position
            logits = logits[:, -1, :]  # (B, vocab_size)

            # SOLUTION 1: Temperature scaling
            # Divide logits by temperature to reshape the distribution.
            # Use max() to prevent division by zero.
            logits = logits / max(temperature, 1e-8)

            # SOLUTION 2: Top-k sampling
            # Keep only the top k logits, set the rest to -infinity.
            if top_k is not None:
                top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                threshold = top_k_values[:, [-1]]  # smallest value in top-k
                logits[logits < threshold] = float("-inf")

            # SOLUTION 3: Top-p (nucleus) sampling
            # Keep the smallest set of tokens whose cumulative probability >= p.
            if top_p is not None:
                probs = F.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # Remove tokens with cumulative probability above threshold
                # Shift right by 1 so the token that crosses the threshold is kept
                sorted_mask = cumulative_probs - sorted_probs >= top_p
                sorted_probs[sorted_mask] = 0.0

                # Scatter back to original order
                probs.scatter_(1, sorted_indices, sorted_probs)

                # Sample from filtered distribution
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
                continue  # Skip the normal sampling below

            # Standard sampling (after temperature and optional top-k filtering)
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
# SOLUTION 4: Comparison function
# ---------------------------------------------------------------------------
def compare_sampling_strategies():
    """Generate text with different sampling strategies for comparison."""
    prompt = "ROMEO:"
    max_tokens = 200

    # --- Temperature comparison ---
    print("\n" + "=" * 60)
    print("=== Temperature Comparison ===")
    print("=" * 60)
    for temp in [0.5, 1.0, 1.5]:
        print(f"\n--- Temperature {temp} ---")
        print(generate_text(prompt=prompt, max_tokens=max_tokens, temperature=temp))

    # --- Top-k comparison ---
    print("\n" + "=" * 60)
    print("=== Top-k Comparison ===")
    print("=" * 60)
    for k in [5, 20, 50]:
        print(f"\n--- Top-k {k} ---")
        print(generate_text(prompt=prompt, max_tokens=max_tokens, top_k=k))

    # --- Top-p comparison ---
    print("\n" + "=" * 60)
    print("=== Top-p Comparison ===")
    print("=" * 60)
    for p in [0.5, 0.9, 0.95]:
        print(f"\n--- Top-p {p} ---")
        print(generate_text(prompt=prompt, max_tokens=max_tokens, top_p=p))

    # --- Combined strategies ---
    print("\n" + "=" * 60)
    print("=== Combined Strategies ===")
    print("=" * 60)

    combos = [
        {"temperature": 0.7, "top_p": 0.9},
        {"temperature": 0.5, "top_k": 10},
        {"temperature": 1.0, "top_k": 40, "top_p": 0.95},
        {"temperature": 0.01},  # Near-greedy
    ]
    for settings in combos:
        label = ", ".join(f"{k}={v}" for k, v in settings.items())
        print(f"\n--- {label} ---")
        print(generate_text(prompt=prompt, max_tokens=max_tokens, **settings))

    # --- Greedy vs creative ---
    print("\n" + "=" * 60)
    print("=== Greedy (T=0.01) vs Creative (T=1.5) ===")
    print("=" * 60)
    print("\n--- Greedy (Temperature 0.01) ---")
    print(generate_text(prompt="To be, or not to be", max_tokens=300, temperature=0.01))
    print("\n--- Creative (Temperature 1.5, top_p=0.95) ---")
    print(generate_text(prompt="To be, or not to be", max_tokens=300, temperature=1.5, top_p=0.95))


# ---------------------------------------------------------------------------
# Run the comparison
# ---------------------------------------------------------------------------
compare_sampling_strategies()
