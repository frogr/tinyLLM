"""
Chapter 4 — Sandbox

Your playground for experimenting with hyperparameters and training dynamics.
This file has the full working model from solution.py, plus ideas for experiments.

Run with: python sandbox.py

Ideas to try:
  - What happens when you change n_layer, n_embd, n_head?
  - How does dropout affect overfitting?
  - Can you plot loss curves?
  - What's the minimum model size that still produces recognizable Shakespeare?
  - What happens if you train for 10,000 or 20,000 steps?
"""

import time
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

# ============================================================================
# Experiment: Change these and see what happens!
# ============================================================================
# Try different configurations and compare the results.

# "Tiny" config — fast training, lower quality
# batch_size, block_size, n_embd, n_head, n_layer = 32, 128, 128, 4, 4
# dropout, max_iters, learning_rate = 0.2, 3000, 3e-4

# "Standard" config — the nanoGPT defaults (same as solution.py)
batch_size, block_size, n_embd, n_head, n_layer = 64, 256, 384, 6, 6
dropout, max_iters, learning_rate = 0.2, 5000, 3e-4

# "No dropout" config — watch it overfit!
# batch_size, block_size, n_embd, n_head, n_layer = 64, 256, 384, 6, 6
# dropout, max_iters, learning_rate = 0.0, 5000, 3e-4

# "Bigger" config — more capacity (slower, needs more memory)
# batch_size, block_size, n_embd, n_head, n_layer = 64, 256, 512, 8, 8
# dropout, max_iters, learning_rate = 0.2, 5000, 3e-4

eval_interval = 500
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


# --- Model (same as solution.py) ---

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
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
        out = self.dropout(self.proj(out))
        return out


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
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = self.dropout(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

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


# --- Train and Record Loss Curve ---
model = GPTLanguageModel().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nConfig: n_embd={n_embd}, n_head={n_head}, n_layer={n_layer}, dropout={dropout}")
print(f"Model has {n_params:,} parameters")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Record losses for plotting
train_losses = []
val_losses = []
steps = []

print("\nTraining...")
start_time = time.time()

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        elapsed = time.time() - start_time
        print(f"step {iter:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | "
              f"gap {losses['val'] - losses['train']:.4f} | {elapsed:.1f}s")
        train_losses.append(losses['train'].item())
        val_losses.append(losses['val'].item())
        steps.append(iter)

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

total_time = time.time() - start_time
print(f"\nTraining complete in {total_time:.1f}s")

# --- Plot Loss Curves (if matplotlib is available) ---
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend (works without a display)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_losses, label='Train Loss', marker='o')
    plt.plot(steps, val_losses, label='Val Loss', marker='s')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title(f'Loss Curves (n_embd={n_embd}, n_layer={n_layer}, dropout={dropout})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add the overfitting gap
    gaps = [v - t for t, v in zip(train_losses, val_losses)]
    plt.twinx()
    plt.plot(steps, gaps, label='Val-Train Gap', color='red', linestyle='--', alpha=0.5)
    plt.ylabel('Overfitting Gap', color='red')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('loss_curves.png', dpi=150)
    print("Loss curves saved to loss_curves.png")
except ImportError:
    print("\nmatplotlib not installed — skipping loss curve plot.")
    print("Install with: pip install matplotlib")
    print("\nLoss data (for manual plotting):")
    for s, t, v in zip(steps, train_losses, val_losses):
        print(f"  step {s:5d}: train={t:.4f}, val={v:.4f}, gap={v-t:.4f}")

# --- Generate Sample ---
print("\n" + "="*70)
print("GENERATED TEXT")
print("="*70 + "\n")

model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500)[0].tolist()
print(decode(generated))
print("="*70)

# ============================================================================
# Experiment Ideas
# ============================================================================
print("""
=== EXPERIMENT IDEAS ===

1. DROPOUT COMPARISON
   Run with dropout=0.0, 0.1, 0.2, 0.4 and compare the val-train gap.
   Higher dropout = smaller gap but potentially higher overall loss.

2. MODEL SIZE SWEEP
   Try these configs and compare val loss and training time:
     Tiny:   n_embd=64,  n_layer=2, n_head=2  (~0.2M params)
     Small:  n_embd=128, n_layer=4, n_head=4  (~1.5M params)
     Medium: n_embd=384, n_layer=6, n_head=6  (~10.8M params)
     Large:  n_embd=512, n_layer=8, n_head=8  (~25M params)

3. LEARNING RATE SWEEP
   Try 1e-2, 1e-3, 3e-4, 1e-4, 1e-5. Which converges fastest?
   Which gives the best final loss?

4. CONTEXT LENGTH
   Try block_size=32, 64, 128, 256. How does longer context
   affect the quality of generated text?

5. LONGER TRAINING
   Set max_iters=10000 or 20000. Does the model keep improving
   or does it plateau? When does overfitting start?
""")
