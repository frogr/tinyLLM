"""
Chapter 4 — Scaling Up (Solution)

Complete working implementation of the scaled-up GPT model.
This produces semi-coherent Shakespeare output after ~5000 training steps.

Try to work through exercises.py first before looking at this!

Run with: python solution.py
Expected training time: ~2-5 min on Apple Silicon, ~10-20 min on CPU.
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

# --- Hyperparameters ---
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


# ============================================================================
# Model Components
# ============================================================================

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
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # Attention scores
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)  # Dropout on attention weights

        # Weighted aggregation of values
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v      # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

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
    """Two-layer MLP: expand to 4x, ReLU, project back, dropout."""

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
    """Transformer block: layernorm -> attention -> residual -> layernorm -> ffwd -> residual."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))    # Residual connection around attention
        x = x + self.ffwd(self.ln2(x))  # Residual connection around feedforward
        return x


class GPTLanguageModel(nn.Module):
    """The full GPT model: embeddings -> N transformer blocks -> layernorm -> linear head."""

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # Final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)          # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, n_embd)
        x = self.dropout(tok_emb + pos_emb)                # (B, T, n_embd) — dropout on embeddings
        x = self.blocks(x)                                  # (B, T, n_embd)
        x = self.ln_f(x)                                    # (B, T, n_embd)
        logits = self.lm_head(x)                            # (B, T, vocab_size)

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
            # Crop context to block_size (position embeddings only go up to block_size)
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]           # (B, C)
            probs = F.softmax(logits, dim=-1)   # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)             # (B, T+1)
        return idx


# --- Create Model ---
model = GPTLanguageModel().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel has {n_params:,} parameters")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# --- Generate Before Training (should be random gibberish) ---
print("\n--- Generated text BEFORE training ---")
model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=200)[0].tolist()))
print("--- End ---\n")
model.train()

# --- Training Loop ---
print("Training...")
start_time = time.time()

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        elapsed = time.time() - start_time
        print(f"step {iter:5d} | train loss {losses['train']:.4f} | "
              f"val loss {losses['val']:.4f} | time {elapsed:.1f}s")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

total_time = time.time() - start_time
losses = estimate_loss(model)
print(f"\nTraining complete in {total_time:.1f}s")
print(f"Final: train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

# --- Generate After Training ---
print("\n" + "="*70)
print("GENERATED SHAKESPEARE (1000 characters)")
print("="*70 + "\n")

model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=1000)[0].tolist()
print(decode(generated))

print("\n" + "="*70)
print(f"\nModel: {n_params:,} parameters | Trained for {max_iters} steps in {total_time:.1f}s")
print("Expected val loss: ~1.48-1.55 (lower = better, random would be ~4.17)")
print("\nNotice the output has:")
print("  - Real English words (not random characters)")
print("  - Character names followed by colons (like ROMEO:)")
print("  - Line breaks in roughly the right places")
print("  - Something resembling verse or dialogue structure")
print("\nIt's not perfect — it's a 10M parameter model trained on 1MB of text.")
print("But it's unmistakably Shakespeare-flavored. That's the power of transformers.")
