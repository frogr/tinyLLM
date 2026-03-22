"""
Chapter 6 — Make It Yours (Exercises)

This file gives you the full GPT model from Chapter 4 as a starting point,
plus guided TODOs for:
  - Loading and preparing a custom dataset
  - Implementing a simple BPE tokenizer from scratch
  - Training on custom data
  - Experimenting with hyperparameters

The TODOs are more open-ended than previous chapters. There's no single
"right answer" — the goal is experimentation.

Run with: python exercises.py
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
# These are the Chapter 4 defaults. TODO 4 is about changing them.
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
# TODO 1: Load and Prepare a Custom Dataset
# ============================================================================
# By default, we load Tiny Shakespeare. Your task: load YOUR OWN text data.
#
# Steps:
#   1. Get a text file (any plain text: emails, code, recipes, lyrics, etc.)
#   2. Save it to ../data/my_corpus.txt (or whatever name you like)
#   3. Update the file path below
#   4. Run the file and see what the model learns!
#
# The helper function below handles common text cleaning tasks.
# Modify it for your specific data if needed.

def prepare_text(filepath, max_chars=None):
    """Load and clean a text file for training.

    Args:
        filepath: Path to a .txt file
        max_chars: Optional limit on text length (useful for quick experiments)

    Returns:
        Cleaned text as a string
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    # Basic cleaning: normalize whitespace, remove very rare characters
    # You might want to customize this for your data
    text = text.replace('\r\n', '\n')  # Normalize line endings
    text = text.replace('\r', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text)  # Collapse excessive blank lines

    if max_chars is not None:
        text = text[:max_chars]

    return text


# --- Load dataset ---
# Change this path to use your own data!
data_path = '../data/input.txt'   # <-- TODO: change to your own file

try:
    text = prepare_text(data_path)
    print(f"Loaded dataset from {data_path}")
except FileNotFoundError:
    print(f"File not found: {data_path}")
    print("Using placeholder text. Run 'python ../data/download.py' or provide your own file.\n")
    text = "First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n" * 500

print(f"Dataset has {len(text):,} characters")
print(f"First 200 characters:\n{text[:200]}\n")


# ============================================================================
# TODO 2: Implement a Simple BPE Tokenizer
# ============================================================================
# Our Chapter 1-5 tokenizer was character-level: each character = one token.
# BPE (Byte Pair Encoding) creates a smarter vocabulary by merging common pairs.
#
# The algorithm:
#   1. Start with a vocabulary of individual characters
#   2. Find the most frequent pair of adjacent tokens in the corpus
#   3. Merge that pair into a new token
#   4. Repeat until you reach the desired vocabulary size
#
# Below is a skeleton BPE tokenizer. Fill in the TODO sections.
# Each step has detailed hints.

class CharTokenizer:
    """The simple character-level tokenizer from Chapters 1-5."""

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

    This follows the same algorithm used by GPT-2's tokenizer, just simplified.
    The real GPT-2 tokenizer works on bytes (not characters) and has special
    handling for spaces, but the core merge logic is identical.
    """

    def __init__(self, text, num_merges=256):
        """Build BPE vocabulary by performing num_merges merge operations.

        Args:
            text: Training text to learn merges from
            num_merges: How many merge operations to perform.
                        Final vocab_size = num_base_chars + num_merges
        """
        # Step 1: Start with individual characters as the base vocabulary
        self.chars = sorted(list(set(text)))
        self.base_vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        # Convert the entire text into a list of token IDs (initially, character IDs)
        token_ids = [self.stoi[c] for c in text]

        # Step 2: Perform merges
        # Each merge finds the most common pair and combines them into one token
        self.merges = {}  # maps (id1, id2) -> new_id

        print(f"Starting BPE training with {self.base_vocab_size} base characters...")
        print(f"Performing {num_merges} merges...")

        for i in range(num_merges):
            # --- TODO 2a: Count all adjacent pairs ---
            # Scan through token_ids and count how often each pair (id1, id2) appears.
            #
            # HINT:
            #   pair_counts = Counter()
            #   for j in range(len(token_ids) - 1):
            #       pair = (token_ids[j], token_ids[j+1])
            #       pair_counts[pair] += 1
            #
            # If pair_counts is empty, we're done (no more pairs to merge).

            # --- YOUR CODE HERE ---
            pair_counts = Counter()  # placeholder — replace with pair counting
            # --- END TODO 2a ---

            if not pair_counts:
                print(f"No more pairs to merge after {i} merges.")
                break

            # --- TODO 2b: Find the most common pair ---
            # Use pair_counts.most_common(1) to get the most frequent pair.
            #
            # HINT:
            #   best_pair = pair_counts.most_common(1)[0][0]
            #   best_count = pair_counts.most_common(1)[0][1]

            # --- YOUR CODE HERE ---
            best_pair = (0, 0)  # placeholder
            best_count = 0      # placeholder
            # --- END TODO 2b ---

            if best_count < 2:
                print(f"Most common pair only appears {best_count} time(s). Stopping.")
                break

            # Create a new token ID for the merged pair
            new_id = self.base_vocab_size + i
            self.merges[best_pair] = new_id

            # Record what this new token represents (concatenation of the two tokens)
            self.itos[new_id] = self.itos[best_pair[0]] + self.itos[best_pair[1]]
            self.stoi[self.itos[new_id]] = new_id

            # --- TODO 2c: Replace all occurrences of best_pair in token_ids ---
            # Scan through token_ids and wherever you see best_pair[0] followed by
            # best_pair[1], replace them with new_id.
            #
            # HINT: Build a new list:
            #   new_ids = []
            #   j = 0
            #   while j < len(token_ids):
            #       if j < len(token_ids) - 1 and token_ids[j] == best_pair[0] and token_ids[j+1] == best_pair[1]:
            #           new_ids.append(new_id)
            #           j += 2  # skip both tokens
            #       else:
            #           new_ids.append(token_ids[j])
            #           j += 1
            #   token_ids = new_ids

            # --- YOUR CODE HERE ---
            pass  # placeholder — replace with merge logic
            # --- END TODO 2c ---

            if (i + 1) % 50 == 0 or i < 5:
                merged_str = self.itos[new_id]
                print(f"  Merge {i+1:4d}: ({self.itos[best_pair[0]]!r}, {self.itos[best_pair[1]]!r}) "
                      f"-> {merged_str!r} (count: {best_count:,})")

        self.vocab_size = self.base_vocab_size + len(self.merges)
        print(f"BPE training complete. Vocabulary size: {self.vocab_size}")

    def encode(self, text):
        """Encode text into BPE token IDs.

        This applies the same merges (in the same order) that were learned
        during training.
        """
        # Start with character-level tokens
        ids = [self.stoi[c] for c in text if c in self.stoi]

        # Apply merges in the order they were learned
        # (This is important! Earlier merges represent more common patterns.)
        for pair, new_id in self.merges.items():
            new_ids = []
            j = 0
            while j < len(ids):
                if j < len(ids) - 1 and ids[j] == pair[0] and ids[j+1] == pair[1]:
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
# Choose your tokenizer
# ============================================================================
# Toggle between character-level and BPE tokenization.
# Start with character-level (it works out of the box).
# Switch to BPE after completing TODO 2.

USE_BPE = False  # <-- Change to True after implementing TODO 2
NUM_BPE_MERGES = 256  # Number of merges. More merges = larger vocab, fewer tokens.

if USE_BPE:
    print("\n--- Building BPE tokenizer ---")
    tokenizer = SimpleBPE(text, num_merges=NUM_BPE_MERGES)
else:
    print("\n--- Using character-level tokenizer ---")
    tokenizer = CharTokenizer(text)

vocab_size = tokenizer.vocab_size
print(f"Vocabulary size: {vocab_size}")

# Quick test
test_str = text[:50]
encoded = tokenizer.encode(test_str)
decoded = tokenizer.decode(encoded)
print(f"Encode test: {len(test_str)} chars -> {len(encoded)} tokens")
print(f"Decode test: round-trip {'OK' if decoded == test_str else 'FAILED'}")
if USE_BPE:
    compression = len(test_str) / len(encoded)
    print(f"Compression ratio: {compression:.2f}x")
print()


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
# The Full GPT Model (from Chapter 4)
# ============================================================================
# This is the complete model you built in earlier chapters.
# No TODOs here — it's your working starting point.

class Head(nn.Module):
    """One head of self-attention."""
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
        out = wei @ v
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
    """A simple feed-forward network with one hidden layer."""
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
    """Transformer block: attention followed by feedforward."""
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
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
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
# TODO 3: Train on Your Custom Data
# ============================================================================
# The training loop below is ready to go. After you've set up your data
# (TODO 1) and optionally your BPE tokenizer (TODO 2), just run the file.
#
# Things to watch for:
#   - Does the loss go down? (If not, check your data and hyperparameters)
#   - Does the generated text look like your training data? (Style, vocabulary)
#   - What's the gap between train and val loss? (Large gap = overfitting)

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
# TODO 4: Experiment with Different Hyperparameters
# ============================================================================
# Now that training works, try changing things and observing the effect.
# Here are some experiments to try (change the values at the top of the file):
#
# Experiment A: Reduce model size for faster iteration
#   n_embd = 128, n_head = 4, n_layer = 4, max_iters = 3000
#   -> Faster training, lower quality. Good for quick experiments.
#
# Experiment B: Longer context
#   block_size = 512 (or even 1024 if you have enough memory)
#   -> Does the model produce more coherent long-range text?
#
# Experiment C: More training
#   max_iters = 10000
#   -> Does the loss keep going down, or does it plateau?
#
# Experiment D: Different learning rates
#   learning_rate = 1e-3 (10x higher) or learning_rate = 1e-4 (3x lower)
#   -> How does learning rate affect convergence and final quality?
#
# Experiment E: Use BPE tokenization
#   USE_BPE = True, NUM_BPE_MERGES = 500
#   -> Compare output quality and training speed to character-level.
#
# Record your observations! What worked? What didn't? Why?


# ============================================================================
# Generate from the trained model
# ============================================================================
print("--- Generated text AFTER training ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500, temperature=0.8)[0].tolist()
print(tokenizer.decode(generated))
print("--- End ---\n")

# Try different temperatures
print("--- Temperature comparison ---")
for temp in [0.5, 0.8, 1.0, 1.5]:
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=100, temperature=temp)[0].tolist()
    print(f"\nTemperature = {temp}:")
    print(tokenizer.decode(generated))
print("--- End ---")

print("\nDone! Check the README for more experiment ideas.")
