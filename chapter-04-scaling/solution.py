"""
Chapter 4: Scaling Up — Solution
=================================

Complete working version of the scaled-up GPT model.
After training (~5-10 min), this produces recognizably Shakespeare-like text.

This model has ~10M parameters, uses 6 transformer layers, 384-dim embeddings,
6 attention heads, dropout regularization, and a cosine learning rate schedule.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import math
import os

# --------------------------------------------------------------------------
# Hyperparameters
# --------------------------------------------------------------------------
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
eval_iters = 200
learning_rate = 3e-4
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
warmup_iters = 500
min_lr = 3e-5

device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "input.txt")
if not os.path.exists(data_path):
    print(f"Data file not found at {data_path}")
    print("Run: python data/download.py")
    exit(1)

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
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


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


# ==========================================================================
# Model Components — with dropout
# ==========================================================================

class Head(nn.Module):
    """Single head of self-attention with dropout."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        # SOLUTION 1a: Dropout on attention weights
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        # SOLUTION 1b: Apply dropout to attention weights
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of attention with projection and dropout."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        # SOLUTION 1c: Dropout after projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        # SOLUTION 1d: Apply dropout to projected output
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """Feed-forward network with dropout."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            # SOLUTION 1e: Dropout after activation
            nn.Dropout(dropout),
            nn.Linear(4 * n_embd, n_embd),
            # SOLUTION 1f: Dropout after output projection
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: attention + feedforward with residual connections."""

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


# ==========================================================================
# SOLUTION 2: Full GPT Model
# ==========================================================================

class GPTLanguageModel(nn.Module):
    """Full scaled-up GPT model with 6 layers, dropout, and proper init."""

    def __init__(self):
        super().__init__()

        # SOLUTION 2a: Model components
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.dropout_layer = nn.Dropout(dropout)

        # SOLUTION 2b: Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # SOLUTION 2c: Forward pass
        tok_emb = self.token_embedding_table(idx)            # (B, T, n_embd)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)                   # (T,)
        )                                                     # (T, n_embd)
        x = tok_emb + pos_emb                                # (B, T, n_embd)
        x = self.dropout_layer(x)                            # dropout on embeddings
        x = self.blocks(x)                                   # (B, T, n_embd)
        x = self.ln_f(x)                                     # (B, T, n_embd)
        logits = self.lm_head(x)                             # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ==========================================================================
# SOLUTION 3: Learning Rate Schedule
# ==========================================================================

def get_lr(it):
    """Cosine learning rate schedule with linear warmup."""
    # Phase 1: Linear warmup
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters

    # Phase 3: After decay, stay at min_lr
    if it > max_iters:
        return min_lr

    # Phase 2: Cosine decay from learning_rate to min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# ==========================================================================
# SOLUTION 4: Training Loop
# ==========================================================================

def train():
    model = GPTLanguageModel()
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params:,} parameters ({n_params/1e6:.1f}M)")
    print(f"Training for {max_iters} steps with batch_size={batch_size}")
    print(f"Context window: {block_size} characters")
    print(f"LR schedule: warmup {warmup_iters} steps, cosine decay to {min_lr}")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    start_time = time.time()

    for iter in range(max_iters):

        # SOLUTION 4a: Update learning rate
        lr = get_lr(iter)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Evaluate periodically
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
            elapsed = time.time() - start_time
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"step {iter:5d} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | "
                f"lr {current_lr:.2e} | "
                f"elapsed {elapsed:.0f}s"
            )

        # SOLUTION 4b: Training step
        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.0f}s ({total_time/60:.1f} min)")

    return model


# ==========================================================================
# SOLUTION 5: Generate Text
# ==========================================================================

def generate_text(model, num_chars=1000):
    """Generate text from the trained model."""
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=num_chars)
    text = decode(generated[0].tolist())
    model.train()
    return text


# ==========================================================================
# Main
# ==========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 4: Scaling Up — Solution")
    print("=" * 60)
    print(f"Vocab size: {vocab_size}")
    print(f"Training data: {len(train_data):,} characters")
    print(f"Validation data: {len(val_data):,} characters")
    print()

    model = train()

    # Generate a long passage
    print("\n" + "=" * 60)
    print("Generated Shakespeare (1000 characters):")
    print("=" * 60)
    output = generate_text(model, num_chars=1000)
    print(output)
    print("=" * 60)

    # Generate a few shorter samples to compare quality
    print("\nAdditional samples (200 chars each):")
    for i in range(3):
        print(f"\n--- Sample {i+1} ---")
        sample = generate_text(model, num_chars=200)
        print(sample)
