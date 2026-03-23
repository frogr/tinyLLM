"""
Chapter 1 — The Bigram Model (Solution)

Complete working implementation of a bigram character-level language model.
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
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
eval_iters = 200

torch.manual_seed(1337)

# --- Load Data ---
with open('../data/input.txt', 'r') as f:
    text = f.read()
print(f"Dataset has {len(text):,} characters")
print(f"First 200 characters:\n{text[:200]}\n")

# --- Tokenizer ---
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
print(f"Vocabulary size: {vocab_size} unique characters")
print(f"Characters: {''.join(chars)}\n")

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


# --- The Model ---
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # The entire model is a single embedding table.
        # It maps each character to a vector of scores for the next character.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        # shape: (vocab_size, vocab_size) = (65, 65) = 4,225 parameters

    def forward(self, idx, targets=None):
        # idx shape: (B, T) — batch of character index sequences
        logits = self.token_embedding_table(idx)  # shape: (B, T, C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)     # shape: (B*T, C) — flatten for cross_entropy
            targets = targets.view(B*T)       # shape: (B*T,) — flatten to match
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx shape: (B, T) — current context
        for _ in range(max_new_tokens):
            logits, _ = self(idx)              # (B, T, C)
            logits = logits[:, -1, :]          # (B, C) — last position only
            probs = F.softmax(logits, dim=-1)  # (B, C) — convert to probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1) — sample
            idx = torch.cat((idx, idx_next), dim=1)             # (B, T+1) — append
        return idx


# --- Create Model ---
model = BigramLanguageModel(vocab_size).to(device)
print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters\n")
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
