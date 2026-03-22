"""
Chapter 3: The Full Transformer Block — Solution
==================================================

This is the complete, working solution with all TODOs filled in.
Run it to see the transformer block in action:

  python chapter-03-transformer/solution.py

Expected results after 5000 iterations:
  - Train loss: ~1.7-1.9
  - Val loss:   ~1.9-2.1
  - Generated text: rough Shakespeare-like structure with recognizable words

This is a significant improvement over the single-head model from Chapter 2!
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import os

# ============================================================================
# Hyperparameters
# ============================================================================

batch_size = 32
block_size = 32
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 1
dropout = 0.0

device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# ============================================================================
# Data Loading
# ============================================================================

data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "input.txt")
with open(data_path, "r") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    """Generate a batch of inputs (x) and targets (y)."""
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
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
# Head (from Chapter 2)
# ============================================================================

class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)                                         # (B, T, head_size)
        q = self.query(x)                                       # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)            # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)                            # (B, T, T)
        v = self.value(x)                                       # (B, T, head_size)
        out = wei @ v                                           # (B, T, head_size)
        return out

# ============================================================================
# SOLUTION — TODO 1: Multi-Head Attention
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        # Create num_heads independent attention heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # Projection layer to mix information from different heads
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        # Run all heads in parallel, concatenate their outputs
        out = torch.cat([h(x) for h in self.heads], dim=-1)    # (B, T, n_embd)
        # Project back — this lets heads interact with each other
        out = self.proj(out)                                     # (B, T, n_embd)
        return out

# ============================================================================
# SOLUTION — TODO 2: FeedForward Network
# ============================================================================

class FeedForward(nn.Module):
    """A simple feedforward network: expand → ReLU → compress."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),     # expand: (n_embd) → (4 * n_embd)
            nn.ReLU(),                           # nonlinearity
            nn.Linear(4 * n_embd, n_embd),      # compress: (4 * n_embd) → (n_embd)
        )

    def forward(self, x):
        return self.net(x)                       # (B, T, n_embd) → (B, T, n_embd)

# ============================================================================
# SOLUTION — TODO 3: Transformer Block
# ============================================================================

class TransformerBlock(nn.Module):
    """Transformer block: attention + feedforward with residuals and layer norm."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)          # layer norm before attention
        self.ln2 = nn.LayerNorm(n_embd)          # layer norm before feedforward

    def forward(self, x):
        # Pre-norm: LayerNorm BEFORE each sub-layer
        # Residual: ADD input back to output
        x = x + self.sa(self.ln1(x))              # attention + residual
        x = x + self.ffwd(self.ln2(x))            # feedforward + residual
        return x                                    # (B, T, n_embd)

# ============================================================================
# SOLUTION — TODO 4 & 5: Full Language Model
# ============================================================================

class TransformerLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        # TODO 4: All model layers
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[TransformerBlock(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)           # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # TODO 5: Forward pass
        tok_emb = self.token_embedding_table(idx)               # (B, T, n_embd)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)                      # (T,)
        )                                                        # (T, n_embd)
        x = tok_emb + pos_emb                                    # (B, T, n_embd)
        x = self.blocks(x)                                       # (B, T, n_embd)
        x = self.ln_f(x)                                         # (B, T, n_embd)
        logits = self.lm_head(x)                                 # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Generate new tokens autoregressively."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ============================================================================
# Training Loop
# ============================================================================

if __name__ == "__main__":
    model = TransformerLanguageModel()
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("Training...")
    print("-" * 50)
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
            print(f"step {iter:5d} | "
                  f"train loss {losses['train']:.4f} | "
                  f"val loss {losses['val']:.4f}")

        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("-" * 50)
    print(f"Final train loss: {losses['train']:.4f}")
    print(f"Final val loss:   {losses['val']:.4f}")

    print("\n" + "=" * 50)
    print("GENERATED TEXT:")
    print("=" * 50)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=500)
    print(decode(generated[0].tolist()))
