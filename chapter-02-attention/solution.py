"""
Chapter 2 — Self-Attention (Solution)

Complete working implementation of a character-level language model
with a single self-attention head and position embeddings.

Try to work through exercises.py first before looking at this!

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
head_size = 32
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
eval_iters = 200

torch.manual_seed(1337)

# --- Load Data ---
with open('../data/input.txt', 'r') as f:
    text = f.read()
print(f"Dataset has {len(text):,} characters")

# --- Tokenizer ---
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
print(f"Vocabulary size: {vocab_size} unique characters")

# --- Train/Val Split ---
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"Train: {len(train_data):,} chars | Val: {len(val_data):,} chars\n")


# --- Batch Loader ---
def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])       # shape: (B, T)
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])   # shape: (B, T)
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


# --- Single Self-Attention Head ---
class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)   # what do I contain?
        self.query = nn.Linear(n_embd, head_size, bias=False)   # what am I looking for?
        self.value = nn.Linear(n_embd, head_size, bias=False)   # what do I provide?
        # The causal mask — lower triangular matrix
        # register_buffer: part of model state, but not a learned parameter
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape                                          # (B, T, n_embd)
        k = self.key(x)                                             # (B, T, head_size)
        q = self.query(x)                                           # (B, T, head_size)

        # Compute attention scores ("affinities")
        # q @ k^T: each query dot-products with every key
        # Scale by 1/sqrt(head_size) to keep variance stable
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)                # (B, T, T)

        # Mask: set future positions to -inf so softmax gives them 0 weight
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)

        # Convert scores to probabilities
        wei = F.softmax(wei, dim=-1)                                # (B, T, T)

        # Weighted aggregation of values
        v = self.value(x)                                           # (B, T, head_size)
        out = wei @ v                                               # (B, T, head_size)
        return out


# --- The Model ---
class AttentionLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Two embedding tables: WHAT (token) + WHERE (position)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)     # (65, 32)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  # (8, 32)
        # One self-attention head
        self.sa_head = Head(head_size)
        # Project from attention output back to vocabulary
        self.lm_head = nn.Linear(head_size, vocab_size)                   # (32, 65)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Token embeddings: what each character IS
        tok_emb = self.token_embedding_table(idx)                         # (B, T, n_embd)

        # Position embeddings: where each character SITS
        pos = torch.arange(T, device=device)                              # (T,)
        pos_emb = self.position_embedding_table(pos)                      # (T, n_embd)

        # Combine: element-wise addition (pos_emb broadcasts over batch)
        x = tok_emb + pos_emb                                             # (B, T, n_embd)

        # Self-attention: each position gathers info from previous positions
        x = self.sa_head(x)                                               # (B, T, head_size)

        # Project to vocabulary size for predictions
        logits = self.lm_head(x)                                          # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)        # (B*T, C)
            targets = targets.view(B*T)          # (B*T,)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Crop to last block_size tokens (position embeddings only go up to block_size)
            idx_cond = idx[:, -block_size:]                         # (B, min(T, block_size))
            logits, _ = self(idx_cond)                              # (B, T', vocab_size)
            logits = logits[:, -1, :]                               # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)                       # (B, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)      # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)                 # (B, T+1)
        return idx


# --- Create Model ---
model = AttentionLanguageModel().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model has {n_params:,} parameters")
print(f"(Compare to bigram's {vocab_size * vocab_size:,} parameters)\n")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# --- Generate Before Training ---
print("--- Generated text BEFORE training ---")
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
