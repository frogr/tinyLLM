"""
Chapter 2: Self-Attention from Scratch — SOLUTION
====================================================

Complete working implementation with all TODOs filled in.
Compare your exercises.py against this if you get stuck.
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
n_embd = 32
head_size = 16

torch.manual_seed(1337)

# ============================================================================
# DATA LOADING
# ============================================================================
import os
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'input.txt')

with open(data_path, 'r') as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Vocabulary size: {vocab_size}")
print(f"Dataset size: {len(data):,} characters")

def get_batch(split):
    source = train_data if split == 'train' else val_data
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i:i+block_size] for i in ix])
    y = torch.stack([source[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
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
# SELF-ATTENTION HEAD
# ============================================================================

class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size):
        super().__init__()
        # SOLUTION TODO 1: Q, K, V projection layers
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        # SOLUTION TODO 2: Compute Q, K, V
        k = self.key(x)        # shape: (B, T, head_size)
        q = self.query(x)      # shape: (B, T, head_size)
        v = self.value(x)      # shape: (B, T, head_size)

        # SOLUTION TODO 3: Compute attention scores (scaled dot product)
        wei = q @ k.transpose(-2, -1) * head_size**-0.5  # (B, T, T)

        # SOLUTION TODO 4: Mask future positions + softmax
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        # SOLUTION TODO 5: Weighted sum of values
        out = wei @ v  # (B, T, T) @ (B, T, head_size) → (B, T, head_size)

        return out


# ============================================================================
# THE MODEL
# ============================================================================

class AttentionLanguageModel(nn.Module):
    """Bigram model upgraded with self-attention and position embeddings."""

    def __init__(self):
        super().__init__()
        # SOLUTION TODO 6: Model components
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(head_size)
        self.lm_head = nn.Linear(head_size, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # SOLUTION TODO 7: Forward pass with attention
        tok_emb = self.token_embedding_table(idx)                                # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, n_embd)
        x = tok_emb + pos_emb                                                    # (B, T, n_embd)
        x = self.sa_head(x)                                                      # (B, T, head_size)
        logits = self.lm_head(x)                                                 # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ============================================================================
# TRAINING
# ============================================================================

model = AttentionLanguageModel().to(device)
param_count = sum(p.numel() for p in model.parameters())
print(f"\nModel created with {param_count:,} parameters")
print(f"(Compare to chapter 1's bigram model which had {vocab_size * vocab_size:,} parameters)")

print("\n--- Before Training ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=100)
print(f"Generated: '{decode(generated[0].tolist())}'")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

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

losses = estimate_loss()
print(f"Step {max_iters:5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

print("\n--- After Training ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500)
print(f"Generated text:\n{decode(generated[0].tolist())}")
print("\n(Should be noticeably better than the bigram model!)")
