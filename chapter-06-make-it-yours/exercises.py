"""
Chapter 6 -- Make It Yours (Exercises)

This chapter is more open-ended than previous ones. You have a full GPT model
(copied from Chapter 4) and three guided experiments:

  TODO 1: Load and prepare a custom dataset
  TODO 2: Implement a simple BPE tokenizer from scratch
  TODO 3: Train on your custom data
  TODO 4: Experiment with different hyperparameters

The model code is complete and ready to use -- your job is to experiment with
the data pipeline, tokenization, and training configuration.

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
# These are the Chapter 4 defaults. TODO 4 asks you to experiment with them.
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
# By default, this loads Tiny Shakespeare. Your task: swap in your own data.
#
# Steps:
#   1. Find or create a text file (at least 50KB, ideally 500KB+)
#   2. Update the file path below to point to your file
#   3. Run the script and see what happens
#
# HELPER: The function below cleans common text issues. Use it if you want.
#
# Ideas for data sources:
#   - Export your Slack/Discord messages
#   - Concatenate source code files: find ~/project -name "*.py" -exec cat {} \;
#   - Download a book from Project Gutenberg (gutenberg.org)
#   - Copy-paste song lyrics, recipes, legal text, anything

def prepare_text_file(filepath):
    """Load and do basic cleaning on a text file.

    Handles common issues:
    - Strips leading/trailing whitespace
    - Normalizes line endings to \n
    - Removes null bytes
    - Collapses runs of 3+ blank lines into 2
    """
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Remove null bytes
    text = text.replace('\x00', '')
    # Collapse excessive blank lines
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


# --- YOUR CODE HERE ---
# Change this path to point to your own text file.
# If you don't have one yet, the Shakespeare default works fine for testing.
data_path = '../data/input.txt'

try:
    text = prepare_text_file(data_path)
    print(f"Loaded data from: {data_path}")
except FileNotFoundError:
    print(f"File not found: {data_path}")
    print("Falling back to placeholder text...\n")
    text = "To be or not to be, that is the question.\n" * 5000
# --- END TODO 1 ---

print(f"Dataset size: {len(text):,} characters")
print(f"First 200 characters:\n{text[:200]}\n")


# ============================================================================
# TODO 2: Implement a Simple BPE Tokenizer from Scratch
# ============================================================================
# Our model currently uses character-level tokenization (each character = 1 token).
# Real LLMs use BPE (Byte Pair Encoding) for much more efficient tokenization.
#
# YOUR TASK: Complete the SimpleBPE class below.
#
# The algorithm:
#   1. Start with a vocabulary of individual characters
#   2. Find the most frequent pair of adjacent tokens in the corpus
#   3. Merge that pair into a new token
#   4. Repeat until you reach the desired vocabulary size
#
# We provide the skeleton. Fill in the methods marked with TODO.
#
# NOTE: This is a simplified version. Production BPE (like tiktoken) operates
# on bytes and handles Unicode properly. Ours works on characters for clarity.

class SimpleBPE:
    """A minimal BPE tokenizer for learning purposes.

    Usage:
        bpe = SimpleBPE(num_merges=200)
        bpe.train(text)
        tokens = bpe.encode("hello world")
        decoded = bpe.decode(tokens)
    """

    def __init__(self, num_merges=200):
        self.num_merges = num_merges
        self.merges = {}       # (token_a, token_b) -> new_token_id
        self.vocab = {}        # token_id -> string
        self.inverse_vocab = {}  # string -> token_id

    def _get_pair_counts(self, token_ids):
        """Count how often each pair of adjacent tokens appears.

        Args:
            token_ids: list of integer token IDs

        Returns:
            Counter mapping (id_a, id_b) -> count

        Example:
            token_ids = [1, 2, 3, 1, 2]
            returns: {(1,2): 2, (2,3): 1, (3,1): 1}
        """
        # TODO 2a: Count adjacent pairs
        # HINT: Loop through token_ids and count pairs (token_ids[i], token_ids[i+1])
        # --- YOUR CODE HERE ---
        counts = Counter()
        # for i in range(len(token_ids) - 1):
        #     pair = (token_ids[i], token_ids[i + 1])
        #     counts[pair] += 1
        return counts
        # --- END TODO 2a ---

    def _merge(self, token_ids, pair, new_id):
        """Replace all occurrences of pair with new_id in token_ids.

        Args:
            token_ids: list of integer token IDs
            pair: tuple of (id_a, id_b) to replace
            new_id: the new token ID to replace the pair with

        Returns:
            new list with pairs merged

        Example:
            token_ids = [1, 2, 3, 1, 2]
            pair = (1, 2)
            new_id = 99
            returns: [99, 3, 99]
        """
        # TODO 2b: Merge pairs
        # HINT: Walk through the list. When you see pair[0] followed by pair[1],
        #       append new_id and skip the next token. Otherwise, append the current token.
        # --- YOUR CODE HERE ---
        new_ids = []
        # i = 0
        # while i < len(token_ids):
        #     if i < len(token_ids) - 1 and token_ids[i] == pair[0] and token_ids[i + 1] == pair[1]:
        #         new_ids.append(new_id)
        #         i += 2
        #     else:
        #         new_ids.append(token_ids[i])
        #         i += 1
        return new_ids if new_ids else token_ids
        # --- END TODO 2b ---

    def train(self, text):
        """Learn BPE merges from the training text.

        This is where the magic happens:
        1. Convert text to a list of character-level token IDs
        2. Repeatedly find the most common pair and merge it
        3. Store the merge rules so we can apply them during encoding
        """
        # Step 1: Build initial character-level vocabulary
        chars = sorted(list(set(text)))
        self.vocab = {i: ch for i, ch in enumerate(chars)}
        self.inverse_vocab = {ch: i for i, ch in enumerate(chars)}
        base_vocab_size = len(chars)

        # Step 2: Convert entire text to character-level token IDs
        token_ids = [self.inverse_vocab[ch] for ch in text]
        print(f"BPE training: starting with {base_vocab_size} character tokens")
        print(f"Text length: {len(token_ids)} tokens before merging")

        # Step 3: Perform merges
        # TODO 2c: Complete the merge loop
        # HINT:
        #   for i in range(self.num_merges):
        #       counts = self._get_pair_counts(token_ids)
        #       if not counts:
        #           break
        #       best_pair = max(counts, key=counts.get)
        #       new_id = base_vocab_size + i
        #       token_ids = self._merge(token_ids, best_pair, new_id)
        #       self.merges[best_pair] = new_id
        #       self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
        #       if (i + 1) % 50 == 0:
        #           print(f"  Merge {i+1}/{self.num_merges}: "
        #                 f"'{self.vocab[best_pair[0]]}' + '{self.vocab[best_pair[1]]}' "
        #                 f"-> '{self.vocab[new_id]}' (appeared {counts[best_pair]} times)")
        # --- YOUR CODE HERE ---
        pass
        # --- END TODO 2c ---

        actual_vocab_size = len(self.vocab)
        print(f"BPE training complete: {actual_vocab_size} tokens in vocabulary")
        print(f"Text length: {len(token_ids)} tokens after merging")
        compression = len(text) / max(len(token_ids), 1)
        print(f"Compression ratio: {compression:.2f}x")

        return actual_vocab_size

    def encode(self, text):
        """Encode text into token IDs using learned merge rules.

        Apply the same merges in the same order as training.
        """
        # Start with character-level tokens
        token_ids = []
        for ch in text:
            if ch in self.inverse_vocab:
                token_ids.append(self.inverse_vocab[ch])
            else:
                # Unknown character -- skip it
                continue

        # Apply merges in order (the order matters!)
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
# Choose your tokenizer
# ============================================================================
# Option A: Character-level (what we've been using)
# Option B: BPE (what you'll implement above)
#
# Start with character-level so the model runs immediately.
# Switch to BPE once you've completed TODO 2.

USE_BPE = False  # Set to True after completing TODO 2
NUM_BPE_MERGES = 200  # How many merges to learn (more = bigger vocabulary)

if USE_BPE:
    print("\n--- Training BPE tokenizer ---")
    tokenizer = SimpleBPE(num_merges=NUM_BPE_MERGES)
    vocab_size = tokenizer.train(text)
    encode = tokenizer.encode
    decode = tokenizer.decode
    print(f"\nSample BPE tokens for 'the cat sat':")
    sample = "the cat sat" if "the cat sat"[0] in text else text[:20]
    encoded_sample = encode(sample)
    print(f"  '{sample}' -> {encoded_sample} ({len(encoded_sample)} tokens)")
    print(f"  Decoded back: '{decode(encoded_sample)}'")
else:
    print("Using character-level tokenizer (set USE_BPE=True to use BPE)")
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

print(f"Vocabulary size: {vocab_size}")


# ============================================================================
# Train/Val Split
# ============================================================================
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"Train: {len(train_data):,} tokens | Val: {len(val_data):,} tokens")


# ============================================================================
# Data Loading
# ============================================================================
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
# This is the complete model. You don't need to modify it -- but you can!
# See TODO 4 for ideas on architectural experiments.

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
        return wei @ v


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Simple feedforward network: linear -> ReLU -> linear -> dropout."""
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
    """Transformer block: attention + feedforward with residual connections."""
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
# TODO 3: Train on Your Custom Data
# ============================================================================
# The training code below is ready to go. Once you've:
#   - Set data_path to your own text file (TODO 1), OR
#   - Enabled BPE tokenization (TODO 2), OR
#   - Both!
# ...just run this script and watch it train.
#
# Look at the generated output and ask yourself:
#   - Does it capture the style of my data?
#   - What patterns did it learn first? (Look at early vs late generation)
#   - Where does it fall apart?

print("\n--- Creating model ---")
model = GPT().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model has {n_params:,} parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# --- YOUR CODE HERE (TODO 3) ---
# The training loop is provided. You might want to:
#   - Change max_iters for faster/slower training
#   - Add early stopping if val loss stops improving
#   - Save the model weights with torch.save(model.state_dict(), 'model.pt')

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
# --- END TODO 3 ---


# ============================================================================
# Generate text
# ============================================================================
print("\n--- Generated text ---")
model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500, temperature=0.8, top_k=50)
print(decode(generated[0].tolist()))
print("--- End ---")


# ============================================================================
# TODO 4: Experiment with Hyperparameters
# ============================================================================
# Go back to the hyperparameters section at the top and try different values.
# Here are some experiments to try:
#
# Experiment 1: Smaller model, faster training
#   n_embd=128, n_head=4, n_layer=4, max_iters=3000
#   -> How much worse is the output? How much faster is training?
#
# Experiment 2: Bigger context window
#   block_size=512 (or 1024 if you have enough memory)
#   -> Does the model generate more coherent long-range text?
#   -> How does training speed change? (Attention is O(T^2))
#
# Experiment 3: More training
#   max_iters=10000 or 20000
#   -> Does the output keep improving? When does it plateau?
#   -> Watch the gap between train and val loss (overfitting)
#
# Experiment 4: Learning rate
#   Try 1e-3 (higher) or 1e-4 (lower)
#   -> Higher: faster learning but might be unstable
#   -> Lower: slower but more stable convergence
#
# Experiment 5: Dropout
#   Try dropout=0.0 (no regularization) vs dropout=0.4 (heavy regularization)
#   -> With 0.0: does the model overfit faster?
#   -> With 0.4: does training loss struggle to go down?
#
# Write your observations in notes.md!
