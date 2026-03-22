"""
Chapter 6 — Make It Yours (Solution)

Complete working code with:
  - BPE tokenizer implemented from scratch
  - Custom dataset support
  - Full GPT model with generation

This demonstrates all the TODOs from exercises.py filled in.

Run with: python solution.py
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import re
from collections import Counter

# ============================================================================
# Device Setup
# ============================================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ============================================================================
# Hyperparameters
# ============================================================================
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

# ============================================================================
# Dataset Loading
# ============================================================================

def prepare_text(filepath, max_chars=None):
    """Load and clean a text file for training."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text)
    if max_chars is not None:
        text = text[:max_chars]
    return text


# Change this path to train on different text
data_path = '../data/input.txt'
text = prepare_text(data_path)
print(f"Loaded {len(text):,} characters from {data_path}")
print(f"First 200 characters:\n{text[:200]}\n")


# ============================================================================
# BPE Tokenizer — Complete Implementation
# ============================================================================

class CharTokenizer:
    """Simple character-level tokenizer (baseline)."""

    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        return ''.join([self.itos[i] for i in ids])


class SimpleBPE:
    """A minimal BPE tokenizer built from scratch.

    The algorithm:
      1. Start with individual characters as tokens
      2. Find the most frequent adjacent pair
      3. Merge it into a new token
      4. Repeat until desired vocabulary size
    """

    def __init__(self, text, num_merges=256):
        # Step 1: Build base vocabulary from individual characters
        self.chars = sorted(list(set(text)))
        self.base_vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        # Convert text to token IDs
        token_ids = [self.stoi[c] for c in text]

        # Step 2: Perform merges
        self.merges = {}  # (id1, id2) -> new_id

        print(f"Starting BPE training: {self.base_vocab_size} base chars, {num_merges} merges")

        for i in range(num_merges):
            # Count all adjacent pairs
            pair_counts = Counter()
            for j in range(len(token_ids) - 1):
                pair = (token_ids[j], token_ids[j + 1])
                pair_counts[pair] += 1

            if not pair_counts:
                print(f"No more pairs after {i} merges.")
                break

            # Find the most common pair
            best_pair, best_count = pair_counts.most_common(1)[0]

            if best_count < 2:
                print(f"Most common pair appears only {best_count} time(s). Stopping.")
                break

            # Create new token for the merged pair
            new_id = self.base_vocab_size + i
            self.merges[best_pair] = new_id
            self.itos[new_id] = self.itos[best_pair[0]] + self.itos[best_pair[1]]
            self.stoi[self.itos[new_id]] = new_id

            # Replace all occurrences of best_pair with new_id
            new_ids = []
            j = 0
            while j < len(token_ids):
                if (j < len(token_ids) - 1
                        and token_ids[j] == best_pair[0]
                        and token_ids[j + 1] == best_pair[1]):
                    new_ids.append(new_id)
                    j += 2
                else:
                    new_ids.append(token_ids[j])
                    j += 1
            token_ids = new_ids

            # Progress reporting
            if (i + 1) % 50 == 0 or i < 5:
                merged_str = self.itos[new_id]
                ratio = len(text) / len(token_ids)
                print(f"  Merge {i+1:4d}: {self.itos[best_pair[0]]!r} + {self.itos[best_pair[1]]!r} "
                      f"-> {merged_str!r} (count: {best_count:,}, "
                      f"tokens: {len(token_ids):,}, ratio: {ratio:.2f}x)")

        self.vocab_size = self.base_vocab_size + len(self.merges)
        final_ratio = len(text) / len(token_ids)
        print(f"BPE complete. Vocab: {self.vocab_size}, "
              f"Compression: {final_ratio:.2f}x ({len(text):,} chars -> {len(token_ids):,} tokens)\n")

    def encode(self, text):
        """Encode text to BPE token IDs by applying learned merges in order."""
        ids = [self.stoi[c] for c in text if c in self.stoi]
        for pair, new_id in self.merges.items():
            new_ids = []
            j = 0
            while j < len(ids):
                if j < len(ids) - 1 and ids[j] == pair[0] and ids[j + 1] == pair[1]:
                    new_ids.append(new_id)
                    j += 2
                else:
                    new_ids.append(ids[j])
                    j += 1
            ids = new_ids
        return ids

    def decode(self, ids):
        """Decode BPE token IDs back to text."""
        return ''.join([self.itos[i] for i in ids])


# ============================================================================
# Choose tokenizer — toggle to compare character-level vs BPE
# ============================================================================
USE_BPE = True
NUM_BPE_MERGES = 256

if USE_BPE:
    print("--- Building BPE tokenizer ---")
    tokenizer = SimpleBPE(text, num_merges=NUM_BPE_MERGES)
else:
    print("--- Using character-level tokenizer ---")
    tokenizer = CharTokenizer(text)

vocab_size = tokenizer.vocab_size
print(f"Vocabulary size: {vocab_size}")

# Verify round-trip
test_str = text[:100]
encoded = tokenizer.encode(test_str)
decoded = tokenizer.decode(encoded)
assert decoded == test_str, f"Round-trip failed!\n  Original: {test_str!r}\n  Decoded:  {decoded!r}"
print(f"Round-trip OK: {len(test_str)} chars -> {len(encoded)} tokens "
      f"({len(test_str)/len(encoded):.2f}x compression)\n")


# ============================================================================
# Data Preparation
# ============================================================================
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"Train: {len(train_data):,} tokens | Val: {len(val_data):,} tokens\n")


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
# The Full GPT Model
# ============================================================================

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
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


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
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
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ============================================================================
# Training
# ============================================================================
model = GPT().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model has {n_params:,} parameters\n")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Generate before training
print("--- Generated text BEFORE training ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(tokenizer.decode(model.generate(context, max_new_tokens=200)[0].tolist()))
print("--- End ---\n")

# Training loop
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


# ============================================================================
# Generation — compare settings
# ============================================================================
print("=" * 60)
print("GENERATION RESULTS")
print("=" * 60)

# Main generation
print("\n--- Temperature 0.8 (default) ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500, temperature=0.8)[0].tolist()
print(tokenizer.decode(generated))

# Temperature comparison
print("\n--- Temperature comparison (100 tokens each) ---")
for temp in [0.3, 0.8, 1.0, 1.5]:
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=100, temperature=temp)[0].tolist()
    print(f"\nTemp={temp}:")
    print(tokenizer.decode(generated))

# Top-k comparison
print("\n--- Top-k comparison (100 tokens each) ---")
for k in [5, 20, 50]:
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=100, temperature=0.8, top_k=k)[0].tolist()
    print(f"\nTop-k={k}:")
    print(tokenizer.decode(generated))

print("\n--- End ---")

# ============================================================================
# Tokenizer stats
# ============================================================================
if USE_BPE:
    print("\n--- BPE Tokenizer Stats ---")
    sample = text[:1000]
    char_tokens = len(sample)
    bpe_tokens = len(tokenizer.encode(sample))
    print(f"Sample of {char_tokens} characters encodes to {bpe_tokens} BPE tokens")
    print(f"Compression ratio: {char_tokens / bpe_tokens:.2f}x")
    print(f"Vocabulary size: {tokenizer.vocab_size} "
          f"({tokenizer.base_vocab_size} base + {len(tokenizer.merges)} merges)")

    # Show some learned tokens
    print("\nSome learned BPE tokens (most common merges first):")
    for i, ((id1, id2), new_id) in enumerate(tokenizer.merges.items()):
        if i >= 20:
            break
        print(f"  {tokenizer.itos[new_id]!r}")

print("\nDone!")
