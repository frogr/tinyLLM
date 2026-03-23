"""
Chapter 2 — Self-Attention (Exercises)

Work through the TODOs in order. Each one builds on the previous.
This file is runnable at every stage — incomplete TODOs use placeholder values
so the script won't crash.

Run with: python exercises.py
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# ============================================================================
# Device Setup
# ============================================================================
# PyTorch can run on CPU, NVIDIA GPU (cuda), or Apple Silicon GPU (mps).
# This picks the best available option automatically.
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
batch_size = 32       # How many sequences to process in parallel
block_size = 8        # Context length (how many characters the model sees)
n_embd = 32           # Embedding dimension (vector size for each character)
head_size = 32        # Size of the attention head's key/query/value vectors
max_iters = 5000      # Training steps
eval_interval = 500   # How often to print loss
learning_rate = 1e-3  # Optimization step size
eval_iters = 200      # Batches to average when estimating loss

torch.manual_seed(1337)

# ============================================================================
# Data Loading (reused from Chapter 1)
# ============================================================================
try:
    with open('../data/input.txt', 'r') as f:
        text = f.read()
except FileNotFoundError:
    print("Dataset not found! Run 'python ../data/download.py' first.")
    print("Using placeholder text for now...\n")
    text = "First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n" * 100

print(f"Dataset has {len(text):,} characters")

# --- Tokenizer (same as Chapter 1) ---
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
print(f"Vocabulary size: {vocab_size} unique characters")

# --- Train/Val Split (same as Chapter 1) ---
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"Train: {len(train_data):,} chars | Val: {len(val_data):,} chars\n")


# --- Batch Loader (same as Chapter 1) ---
def get_batch(split):
    """Get a random batch of inputs (x) and targets (y)."""
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])       # shape: (B, T)
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])   # shape: (B, T)
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    """Average loss over many batches for a stable measurement."""
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
# TODO 1: Understand Token + Position Embeddings
# ============================================================================
# In Chapter 1, we had ONE embedding table: token_embedding_table.
# It told the model WHAT each character is, but not WHERE it sits.
#
# Now we add a SECOND embedding table: position_embedding_table.
# It tells the model the POSITION of each character (0, 1, 2, ..., T-1).
#
# We ADD the two embeddings together:
#   x = tok_emb + pos_emb
#
# This gives each character a combined vector that encodes both
# "what I am" and "where I am."
#
# Look at the model's __init__ and forward methods below. You'll implement
# these two embedding tables and their addition in TODO 3.
#
# For now, just make sure you understand the concept:
#   - token_embedding_table:    (vocab_size, n_embd) — maps char ID to vector
#   - position_embedding_table: (block_size, n_embd) — maps position to vector
#   - We add them: x = tok_emb + pos_emb    shape: (B, T, n_embd)
# ============================================================================
print("TODO 1: Token + Position embeddings")
print(f"  Token embedding table shape:    ({vocab_size}, {n_embd})")
print(f"  Position embedding table shape: ({block_size}, {n_embd})")
print(f"  Combined embedding shape:       (B, {block_size}, {n_embd})")
print()


# ============================================================================
# TODO 2: Implement a Single Self-Attention Head
# ============================================================================
# This is the core of the chapter. You'll build the Head class.
#
# A Head takes in x of shape (B, T, n_embd) and outputs (B, T, head_size).
#
# Steps inside forward():
#   1. Project x into queries, keys, and values using three nn.Linear layers
#   2. Compute attention scores: Q @ K^T / sqrt(head_size)
#   3. Mask out future positions (can't look ahead!)
#   4. Apply softmax to get attention weights
#   5. Multiply weights by values to get the output
#
# HINTS:
#   In __init__:
#     self.key   = nn.Linear(n_embd, head_size, bias=False)
#     self.query = nn.Linear(n_embd, head_size, bias=False)
#     self.value = nn.Linear(n_embd, head_size, bias=False)
#     self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
#
#   In forward:
#     k = self.key(x)                                          # (B, T, head_size)
#     q = self.query(x)                                        # (B, T, head_size)
#     wei = q @ k.transpose(-2, -1) * (head_size ** -0.5)      # (B, T, T)
#     wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
#     wei = F.softmax(wei, dim=-1)                             # (B, T, T)
#     v = self.value(x)                                        # (B, T, head_size)
#     out = wei @ v                                            # (B, T, head_size)
#     return out

class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size):
        super().__init__()
        # TODO 2a: Create the three linear projection layers (key, query, value)
        # Each one takes n_embd inputs and produces head_size outputs.
        # Use bias=False (standard for attention projections).
        # --- YOUR CODE HERE ---
        self.key   = nn.Linear(n_embd, head_size, bias=False)   # PLACEHOLDER — replace or keep
        self.query = nn.Linear(n_embd, head_size, bias=False)   # PLACEHOLDER — replace or keep
        self.value = nn.Linear(n_embd, head_size, bias=False)   # PLACEHOLDER — replace or keep
        # --- END TODO 2a ---

        # The mask: a lower-triangular matrix.
        # register_buffer means "this is part of the model state but NOT a learnable parameter."
        # We'll use this to prevent looking at future positions.
        #
        # Visualize it:
        #   [[1, 0, 0, 0, 0],
        #    [1, 1, 0, 0, 0],
        #    [1, 1, 1, 0, 0],
        #    [1, 1, 1, 1, 0],
        #    [1, 1, 1, 1, 1]]
        # Row i can only "see" columns 0..i
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape  # C = n_embd

        # TODO 2b: Compute keys, queries, and values
        # --- YOUR CODE HERE ---
        k = self.key(x)    # shape: (B, T, head_size)
        q = self.query(x)  # shape: (B, T, head_size)
        # --- END TODO 2b ---

        # TODO 2c: Compute attention scores
        # Dot product of queries with keys, scaled by sqrt(head_size).
        #
        # q @ k.transpose(-2, -1) gives shape (B, T, T)
        # Each entry [b][i][j] = "how much should token i attend to token j?"
        #
        # Scale by 1/sqrt(head_size) to prevent extreme values before softmax.
        # --- YOUR CODE HERE ---
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)  # shape: (B, T, T)  # PLACEHOLDER
        # --- END TODO 2c ---

        # TODO 2d: Apply the causal mask
        # Set all entries where tril==0 to -inf. After softmax, -inf becomes 0.
        # This prevents position i from attending to position j where j > i.
        # --- YOUR CODE HERE ---
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # shape: (B, T, T)  # PLACEHOLDER
        # --- END TODO 2d ---

        # TODO 2e: Apply softmax to get attention weights, then compute output
        # softmax converts scores to probabilities (each row sums to 1)
        # Then multiply by values to get the weighted combination.
        # --- YOUR CODE HERE ---
        wei = F.softmax(wei, dim=-1)  # shape: (B, T, T)           # PLACEHOLDER
        v = self.value(x)             # shape: (B, T, head_size)    # PLACEHOLDER
        out = wei @ v                 # shape: (B, T, head_size)    # PLACEHOLDER
        # --- END TODO 2e ---

        return out


# ============================================================================
# TODO 3: Build the Full Model
# ============================================================================
# The model class needs:
#   __init__:
#     - token_embedding_table:    nn.Embedding(vocab_size, n_embd)
#     - position_embedding_table: nn.Embedding(block_size, n_embd)
#     - sa_head: a Head(head_size)
#     - lm_head: nn.Linear(head_size, vocab_size) — projects attention output to logits
#
#   forward(idx, targets=None):
#     1. Get token embeddings:    tok_emb = token_embedding_table(idx)      (B, T, n_embd)
#     2. Get position embeddings: pos_emb = position_embedding_table(pos)   (T, n_embd)
#        where pos = torch.arange(T, device=device)
#     3. Add them:                x = tok_emb + pos_emb                     (B, T, n_embd)
#     4. Pass through attention:  x = sa_head(x)                            (B, T, head_size)
#     5. Project to vocab:        logits = lm_head(x)                       (B, T, vocab_size)
#     6. Compute loss if targets given (same as Chapter 1)
#
#   generate(idx, max_new_tokens):
#     Same as Chapter 1, BUT now we need to crop the input to the last
#     block_size characters because position embeddings only go up to block_size.
#
# HINT for generate — the key difference from Chapter 1:
#     idx_cond = idx[:, -block_size:]   # crop to last block_size tokens
#     logits, _ = self(idx_cond)        # use cropped input

class AttentionLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO 3a: Create embedding tables
        # Token embedding: maps each character ID to a vector of size n_embd
        # Position embedding: maps each position (0 to block_size-1) to a vector
        # --- YOUR CODE HERE ---
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)       # shape: (65, 32)  # PLACEHOLDER
        self.position_embedding_table = nn.Embedding(block_size, n_embd)    # shape: (8, 32)   # PLACEHOLDER
        # --- END TODO 3a ---

        # TODO 3b: Create the attention head and the output projection
        # sa_head: our Head class from TODO 2
        # lm_head: a Linear layer that maps head_size -> vocab_size (the "language model head")
        # --- YOUR CODE HERE ---
        self.sa_head = Head(head_size)                      # PLACEHOLDER
        self.lm_head = nn.Linear(head_size, vocab_size)     # PLACEHOLDER
        # --- END TODO 3b ---

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # TODO 3c: Compute embeddings and pass through attention
        # Step 1: Token embeddings
        # Step 2: Position embeddings (need to create position indices first!)
        # Step 3: Add them together
        # Step 4: Pass through the attention head
        # Step 5: Project to vocabulary size
        # --- YOUR CODE HERE ---
        tok_emb = self.token_embedding_table(idx)                                   # shape: (B, T, n_embd)
        pos = torch.arange(T, device=device)                                        # shape: (T,)
        pos_emb = self.position_embedding_table(pos)                                # shape: (T, n_embd)
        x = tok_emb + pos_emb                                                       # shape: (B, T, n_embd) — pos_emb broadcasts
        x = self.sa_head(x)                                                          # shape: (B, T, head_size)
        logits = self.lm_head(x)                                                     # shape: (B, T, vocab_size)
        # --- END TODO 3c ---

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)       # shape: (B*T, C) — flatten for cross_entropy
            targets = targets.view(B*T)         # shape: (B*T,) — flatten to match
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx shape: (B, T)
        for _ in range(max_new_tokens):
            # IMPORTANT: crop to the last block_size tokens!
            # Position embeddings only handle up to block_size positions.
            # In Ch1 we didn't need this because the bigram model only looked at one char.
            idx_cond = idx[:, -block_size:]                     # shape: (B, block_size) at most
            logits, _ = self(idx_cond)                          # shape: (B, T', vocab_size)
            logits = logits[:, -1, :]                           # shape: (B, vocab_size) — last position
            probs = F.softmax(logits, dim=-1)                   # shape: (B, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)  # shape: (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)             # shape: (B, T+1)
        return idx


# ============================================================================
# Create Model and Optimizer
# ============================================================================
model = AttentionLanguageModel().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model has {n_params:,} parameters")
print(f"(Compare to bigram's {vocab_size * vocab_size:,} parameters)\n")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# ============================================================================
# Generate BEFORE training
# ============================================================================
print("--- Generated text BEFORE training ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=100)[0].tolist()))
print("--- End ---\n")


# ============================================================================
# TODO 4: Training Loop
# ============================================================================
# This is almost identical to Chapter 1's training loop.
# The only difference is that we're training our new AttentionLanguageModel
# instead of the BigramLanguageModel.
#
# HINT:
#   for iter in range(max_iters):
#       if iter % eval_interval == 0:
#           losses = estimate_loss(model)
#           print(f"step {iter:5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
#       xb, yb = get_batch('train')
#       logits, loss = model(xb, yb)
#       optimizer.zero_grad(set_to_none=True)
#       loss.backward()
#       optimizer.step()

print("Starting training...")
# --- YOUR CODE HERE ---
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {iter:5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
# --- END TODO 4 ---

# Print final loss
losses = estimate_loss(model)
print(f"Final:      train loss {losses['train']:.4f}, val loss {losses['val']:.4f}\n")


# ============================================================================
# TODO 5: Generate and Compare
# ============================================================================
# Generate text from your trained model. Compare it to the bigram output
# from Chapter 1. Is it better? How?
#
# HINT:
#   context = torch.zeros((1, 1), dtype=torch.long, device=device)
#   print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

print("--- Generated text AFTER training ---")
# --- YOUR CODE HERE ---
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500)[0].tolist()
print(decode(generated))
# --- END TODO 5 ---
print("--- End ---")

print("\nDone! Compare this output to Chapter 1's bigram model.")
print("You should see slightly more structure — maybe more word-like patterns.")
print("But it's still pretty rough. In Chapter 3, we'll add multiple attention")
print("heads, stack layers, and add feedforward networks to really boost quality.")
