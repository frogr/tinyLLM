"""
Chapter 6 -- Make It Yours (Solution)

Complete working implementation with:
  - Custom data loading with text preparation
  - BPE tokenizer built from scratch
  - Full GPT model with generation
  - Support for switching between character-level and BPE tokenization

Run with: python solution.py

To use your own data, change DATA_PATH below.
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
# Configuration
# ============================================================================
DATA_PATH = '../data/input.txt'  # Change this to your own text file
USE_BPE = True                   # True = BPE tokenization, False = character-level
NUM_BPE_MERGES = 300             # Number of BPE merges (bigger = larger vocab, more compression)

# Hyperparameters
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
# Data Loading
# ============================================================================
def prepare_text_file(filepath):
    """Load and clean a text file."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = text.replace('\x00', '')
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    text = text.strip()
    return text


print(f"Loading data from: {DATA_PATH}")
text = prepare_text_file(DATA_PATH)
print(f"Dataset size: {len(text):,} characters")
print(f"First 200 characters:\n{text[:200]}\n")

# ============================================================================
# BPE Tokenizer -- Complete Implementation
# ============================================================================

class SimpleBPE:
    """A minimal BPE tokenizer for learning purposes.

    This implements the core BPE algorithm:
    1. Start with character-level tokens
    2. Repeatedly find and merge the most common adjacent pair
    3. Build up a vocabulary of subword tokens

    Production tokenizers (tiktoken, SentencePiece) do the same thing but
    operate on bytes, handle Unicode edge cases, and are written in C/Rust
    for speed.
    """

    def __init__(self, num_merges=200):
        self.num_merges = num_merges
        self.merges = {}         # (token_a, token_b) -> new_token_id
        self.vocab = {}          # token_id -> string
        self.inverse_vocab = {}  # string -> token_id

    def _get_pair_counts(self, token_ids):
        """Count how often each pair of adjacent tokens appears.

        This is the core operation of BPE training. We scan through the entire
        token sequence and count every adjacent pair. The most frequent pair
        will be our next merge.

        Example:
            [1, 2, 3, 1, 2] -> {(1,2): 2, (2,3): 1, (3,1): 1}
        """
        counts = Counter()
        for i in range(len(token_ids) - 1):
            pair = (token_ids[i], token_ids[i + 1])
            counts[pair] += 1
        return counts

    def _merge(self, token_ids, pair, new_id):
        """Replace all occurrences of a pair with a new token ID.

        Walk through the list left to right. When we see pair[0] followed by
        pair[1], we replace both with new_id and skip ahead. Otherwise, we
        keep the current token.

        Example:
            [1, 2, 3, 1, 2], pair=(1,2), new_id=99 -> [99, 3, 99]
        """
        new_ids = []
        i = 0
        while i < len(token_ids):
            # Check if current position matches the pair
            if (i < len(token_ids) - 1 and
                    token_ids[i] == pair[0] and
                    token_ids[i + 1] == pair[1]):
                new_ids.append(new_id)
                i += 2  # Skip both tokens in the pair
            else:
                new_ids.append(token_ids[i])
                i += 1
        return new_ids

    def train(self, text):
        """Learn BPE merges from the training text.

        The algorithm:
        1. Build a character-level vocabulary (every unique character gets an ID)
        2. Convert the entire text to character-level token IDs
        3. Find the most frequent adjacent pair
        4. Merge it into a new token, add to vocabulary
        5. Repeat steps 3-4 for num_merges iterations
        """
        # Step 1: Build initial character-level vocabulary
        chars = sorted(list(set(text)))
        self.vocab = {i: ch for i, ch in enumerate(chars)}
        self.inverse_vocab = {ch: i for i, ch in enumerate(chars)}
        base_vocab_size = len(chars)

        # Step 2: Convert entire text to character-level token IDs
        token_ids = [self.inverse_vocab[ch] for ch in text]
        print(f"BPE training: starting with {base_vocab_size} character tokens")
        print(f"Text length: {len(token_ids):,} tokens before merging")

        # Step 3-5: Iteratively merge the most common pair
        for i in range(self.num_merges):
            # Count all adjacent pairs
            counts = self._get_pair_counts(token_ids)
            if not counts:
                print(f"No more pairs to merge after {i} merges")
                break

            # Find the most frequent pair
            best_pair = max(counts, key=counts.get)

            # Create a new token ID for the merged pair
            new_id = base_vocab_size + i

            # Replace all occurrences of the pair in our token list
            token_ids = self._merge(token_ids, best_pair, new_id)

            # Record the merge rule and update vocabulary
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # Print progress every 50 merges
            if (i + 1) % 50 == 0:
                merged_str = self.vocab[new_id]
                pair_str = f"'{self.vocab[best_pair[0]]}' + '{self.vocab[best_pair[1]]}'"
                print(f"  Merge {i+1:3d}/{self.num_merges}: "
                      f"{pair_str:30s} -> '{merged_str}' "
                      f"(count: {counts[best_pair]})")

        actual_vocab_size = len(self.vocab)
        compression = len(text) / max(len(token_ids), 1)
        print(f"\nBPE training complete:")
        print(f"  Vocabulary size: {actual_vocab_size} tokens")
        print(f"  Text length: {len(token_ids):,} tokens after merging")
        print(f"  Compression ratio: {compression:.2f}x")

        # Show some interesting merged tokens
        print(f"\n  Sample merged tokens:")
        merge_items = list(self.merges.items())
        # Show first few and last few merges
        for idx in [0, 1, 2, len(merge_items)//2, -3, -2, -1]:
            if abs(idx) <= len(merge_items):
                pair, new_id = merge_items[idx]
                token_str = self.vocab[new_id]
                print(f"    Token {new_id}: '{token_str}'")

        return actual_vocab_size

    def encode(self, text):
        """Encode text into token IDs using learned merge rules.

        Important: merges must be applied in the same order as they were learned
        during training. The first merge (most frequent pair) is applied first,
        then the second, and so on.
        """
        # Start with character-level tokens
        token_ids = []
        for ch in text:
            if ch in self.inverse_vocab:
                token_ids.append(self.inverse_vocab[ch])
            # Skip unknown characters

        # Apply merges in training order
        for pair, new_id in self.merges.items():
            token_ids = self._merge(token_ids, pair, new_id)

        return token_ids

    def decode(self, token_ids):
        """Decode token IDs back to text."""
        return ''.join(self.vocab.get(id, '?') for id in token_ids)

    @property
    def vocab_size(self):
        return len(self.vocab)


# ============================================================================
# Choose Tokenizer
# ============================================================================
if USE_BPE:
    print("\n--- Training BPE tokenizer ---")
    tokenizer = SimpleBPE(num_merges=NUM_BPE_MERGES)
    vocab_size = tokenizer.train(text)
    encode = tokenizer.encode
    decode = tokenizer.decode

    # Verify round-trip
    test_str = text[:100]
    encoded = encode(test_str)
    decoded = decode(encoded)
    assert decoded == test_str, f"Round-trip failed!\n  Original: {test_str}\n  Decoded:  {decoded}"
    print(f"\nRound-trip test passed (100 chars -> {len(encoded)} tokens -> 100 chars)")
else:
    print("Using character-level tokenizer")
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

print(f"\nFinal vocabulary size: {vocab_size}")


# ============================================================================
# Data Preparation
# ============================================================================
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"Train: {len(train_data):,} tokens | Val: {len(val_data):,} tokens")


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
print("\n--- Creating model ---")
model = GPT().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model has {n_params:,} parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("\nTraining...")
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
print(f"Final:      train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

# ============================================================================
# Generation
# ============================================================================
print("\n--- Generated text (temperature=0.8, top_k=50) ---")
model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500, temperature=0.8, top_k=50)
print(decode(generated[0].tolist()))
print("--- End ---")

# Compare different temperatures
print("\n--- Temperature comparison ---")
for temp in [0.5, 0.8, 1.0, 1.5]:
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=200, temperature=temp, top_k=50)
    output = decode(generated[0].tolist())
    # Show first 100 chars
    preview = output[:100].replace('\n', ' ')
    print(f"  temp={temp}: {preview}...")

# ============================================================================
# Save the model (optional)
# ============================================================================
save_path = 'model.pt'
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab_size': vocab_size,
    'n_embd': n_embd,
    'n_head': n_head,
    'n_layer': n_layer,
    'block_size': block_size,
    'use_bpe': USE_BPE,
}, save_path)
print(f"\nModel saved to {save_path}")
print("To load later: checkpoint = torch.load('model.pt')")
