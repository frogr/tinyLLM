"""
Chapter 3 — The Transformer Block (Solution)

Complete working implementation of a transformer-based character-level language model.
Try to work through exercises.py first before looking at this!

This builds on Chapter 2 by adding:
  - Multi-head attention (4 heads running in parallel)
  - Feedforward network (linear → ReLU → linear)
  - Residual connections (skip connections)
  - Layer normalization (pre-norm formulation)

Run with: python solution.py
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# --- Device Setup ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# --- Hyperparameters ---
batch_size = 32
block_size = 8
n_embd = 32
n_head = 4
n_layer = 1
dropout = 0.0
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
eval_iters = 200

torch.manual_seed(1337)

# --- Load Data ---
with open('../data/input.txt', 'r') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Dataset: {len(text):,} chars | Vocab: {vocab_size} | Train: {len(train_data):,} | Val: {len(val_data):,}")


def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
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


# --- Single Attention Head ---
class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape                                           # (B, T, n_embd)
        k = self.key(x)                                              # (B, T, head_size)
        q = self.query(x)                                            # (B, T, head_size)

        # Compute attention scores
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)      # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # mask future
        wei = F.softmax(wei, dim=-1)                                 # (B, T, T)
        wei = self.dropout(wei)

        # Weighted aggregation of values
        v = self.value(x)                                            # (B, T, head_size)
        out = wei @ v                                                # (B, T, head_size)
        return out


# --- Multi-Head Attention ---
class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention running in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)       # output projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Run all heads in parallel, concatenate along last dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_embd)
        out = self.dropout(self.proj(out))                     # (B, T, n_embd)
        return out


# --- Feedforward Network ---
class FeedForward(nn.Module):
    """Two linear layers with ReLU — the model's 'thinking' step."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),     # expand: 32 → 128
            nn.ReLU(),                           # non-linearity
            nn.Linear(4 * n_embd, n_embd),      # compress: 128 → 32
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)                       # (B, T, n_embd) → (B, T, n_embd)


# --- Transformer Block ---
class Block(nn.Module):
    """Transformer block: communication (attention) followed by computation (ffn)."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head             # 32 // 4 = 8
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffn = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))             # attend + residual
        x = x + self.ffn(self.ln2(x))            # think + residual
        return x                                  # (B, T, n_embd)


# --- Full Transformer Language Model ---
class TransformerLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)     # (65, 32)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  # (8, 32)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)           # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)  # (32, 65)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)                            # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, n_embd)
        x = tok_emb + pos_emb                                               # (B, T, n_embd)
        x = self.blocks(x)                                                   # (B, T, n_embd)
        x = self.ln_f(x)                                                     # (B, T, n_embd)
        logits = self.lm_head(x)                                            # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)           # (B*T, C)
            targets = targets.view(B*T)             # (B*T,)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Generate new tokens autoregressively."""
        for _ in range(max_new_tokens):
            # Crop context to block_size (model can only handle block_size positions)
            idx_cond = idx[:, -block_size:]                # (B, T) where T <= block_size
            logits, loss = self(idx_cond)                  # (B, T, vocab_size)
            logits = logits[:, -1, :]                      # (B, vocab_size) — last position
            probs = F.softmax(logits, dim=-1)              # (B, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)        # (B, T+1)
        return idx


# --- Create Model ---
model = TransformerLanguageModel().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel has {n_params:,} parameters")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# --- Generate Before Training ---
print("\n--- Generated text BEFORE training ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=100)[0].tolist()))
print("--- End ---\n")

# --- Training Loop ---
print("Training...")
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {iter:5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

losses = estimate_loss(model)
print(f"Final:      train loss {losses['train']:.4f}, val loss {losses['val']:.4f}\n")

# --- Generate After Training ---
print("--- Generated text AFTER training ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
print("--- End ---")
