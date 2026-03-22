"""
Chapter 3 — The Transformer Block (Exercises)

Work through the TODOs in order. Each one builds on the previous.
This file is runnable at every stage — incomplete TODOs use placeholder code.

We're building on Chapter 2's single attention head by adding:
  - Multi-head attention (multiple heads in parallel)
  - Feedforward network (two linear layers with ReLU)
  - Residual connections (skip connections)
  - Layer normalization (stabilize training)

Run with: python exercises.py
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

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
batch_size = 32        # How many sequences to process in parallel
block_size = 8         # Context length (how many characters the model sees)
n_embd = 32            # Embedding dimension (size of each token's vector)
n_head = 4             # Number of attention heads
n_layer = 1            # Number of transformer blocks (we start with 1)
dropout = 0.0          # Dropout rate (0.0 = no dropout, we'll add it later)
max_iters = 5000       # Training steps
eval_interval = 500    # How often to print loss
learning_rate = 1e-3   # Optimizer step size
eval_iters = 200       # Batches to average when estimating loss

torch.manual_seed(1337)

# ============================================================================
# Data Loading (same as previous chapters — no TODOs here)
# ============================================================================
try:
    with open('../data/input.txt', 'r') as f:
        text = f.read()
except FileNotFoundError:
    print("Dataset not found! Run 'python ../data/download.py' first.")
    print("Using placeholder text for now...\n")
    text = "First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n" * 100

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
    """Get a random batch of inputs (x) and targets (y)."""
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
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
# Single Attention Head (from Chapter 2 — provided for you)
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
        B, T, C = x.shape                                           # (B, T, n_embd)
        k = self.key(x)                                              # (B, T, head_size)
        q = self.query(x)                                            # (B, T, head_size)

        # Compute attention scores
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)      # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # mask future
        wei = F.softmax(wei, dim=-1)                                 # (B, T, T)
        wei = self.dropout(wei)

        # Weighted aggregation of values
        v = self.value(x)                                            # (B, T, head_size)
        out = wei @ v                                                # (B, T, head_size)
        return out


# ============================================================================
# TODO 1: Multi-Head Attention
# ============================================================================
# Run multiple attention heads in parallel and concatenate their outputs.
#
# If n_embd=32 and n_head=4, each head has head_size = 32 // 4 = 8.
# After running 4 heads, each producing (B, T, 8), we concatenate along the
# last dimension to get (B, T, 32) — back to our embedding size.
#
# Then apply an output projection (nn.Linear) to let the model learn how to
# combine information from all heads.
#
# ARCHITECTURE:
#   x (B, T, 32) → [Head1, Head2, Head3, Head4] → concat → (B, T, 32) → proj → (B, T, 32)
#
# HINTS:
#   - self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
#   - self.proj = nn.Linear(n_embd, n_embd)       # output projection
#   - self.dropout = nn.Dropout(dropout)
#   - In forward: out = torch.cat([h(x) for h in self.heads], dim=-1)
#   - Then: out = self.dropout(self.proj(out))

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention running in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        # --- YOUR CODE HERE ---
        # Create a list of Head modules and an output projection
        # Remember: use nn.ModuleList, not a plain Python list!
        # (nn.ModuleList registers the heads as submodules so PyTorch
        #  can find their parameters for training)
        pass
        # --- END TODO 1 __init__ ---

    def forward(self, x):
        # --- YOUR CODE HERE ---
        # 1. Run each head on x
        # 2. Concatenate outputs along the last dimension (dim=-1)
        # 3. Apply the output projection and dropout
        #
        # Expected shapes:
        #   x:                    (B, T, n_embd)    e.g., (32, 8, 32)
        #   each head output:     (B, T, head_size) e.g., (32, 8, 8)
        #   after concat:         (B, T, n_embd)    e.g., (32, 8, 32)
        #   after proj:           (B, T, n_embd)    e.g., (32, 8, 32)
        return x  # placeholder — replace with your implementation
        # --- END TODO 1 forward ---


# ============================================================================
# TODO 2: Feedforward Network
# ============================================================================
# Two linear layers with a ReLU activation in between.
# This is the "thinking" step — it processes the information that attention gathered.
#
# Architecture: Linear(n_embd, 4*n_embd) → ReLU → Linear(4*n_embd, n_embd) → Dropout
#
# The inner dimension is 4x the embedding size — this gives the model more
# "scratch space" for intermediate computation.
#
# HINTS:
#   self.net = nn.Sequential(
#       nn.Linear(n_embd, 4 * n_embd),
#       nn.ReLU(),
#       nn.Linear(4 * n_embd, n_embd),
#       nn.Dropout(dropout),
#   )
#
# WHY nn.Sequential? It's like a pipeline — data flows through each layer in order.
# It's equivalent to calling each layer manually in forward(), just cleaner.

class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity."""

    def __init__(self, n_embd):
        super().__init__()
        # --- YOUR CODE HERE ---
        # Create nn.Sequential with:
        #   1. nn.Linear(n_embd, 4 * n_embd)   — expand
        #   2. nn.ReLU()                         — non-linearity
        #   3. nn.Linear(4 * n_embd, n_embd)   — compress back
        #   4. nn.Dropout(dropout)               — regularization
        pass
        # --- END TODO 2 __init__ ---

    def forward(self, x):
        # --- YOUR CODE HERE ---
        # Pass x through self.net
        #
        # Expected shapes:
        #   x:           (B, T, n_embd)     e.g., (32, 8, 32)
        #   after net:   (B, T, n_embd)     e.g., (32, 8, 32)
        #   (internally: first linear produces (32, 8, 128), then back to (32, 8, 32))
        return x  # placeholder — replace with your implementation
        # --- END TODO 2 forward ---


# ============================================================================
# TODO 3: Transformer Block
# ============================================================================
# This is the core building block. It combines:
#   1. Multi-head attention (communication between tokens)
#   2. Feedforward network (processing per token)
#   3. Residual connections (skip connections around each sublayer)
#   4. Layer normalization (stabilize inputs to each sublayer)
#
# We use the PRE-NORM formulation:
#   x = x + MultiHeadAttention(LayerNorm(x))
#   x = x + FeedForward(LayerNorm(x))
#
# Notice: LayerNorm is applied BEFORE each sublayer, and the residual
# connection adds the sublayer output BACK to the original input.
#
# HINTS:
#   self.sa = MultiHeadAttention(n_head, head_size)
#   self.ffn = FeedForward(n_embd)
#   self.ln1 = nn.LayerNorm(n_embd)
#   self.ln2 = nn.LayerNorm(n_embd)
#
#   In forward:
#     x = x + self.sa(self.ln1(x))     # attend + residual
#     x = x + self.ffn(self.ln2(x))    # think + residual
#     return x

class Block(nn.Module):
    """Transformer block: communication (attention) followed by computation (ffn)."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head   # 32 // 4 = 8
        # --- YOUR CODE HERE ---
        # Create:
        #   1. MultiHeadAttention (self.sa)
        #   2. FeedForward (self.ffn)
        #   3. Two LayerNorm layers (self.ln1, self.ln2)
        pass
        # --- END TODO 3 __init__ ---

    def forward(self, x):
        # --- YOUR CODE HERE ---
        # Apply pre-norm transformer block:
        #   x = x + self.sa(self.ln1(x))     # communication + residual
        #   x = x + self.ffn(self.ln2(x))    # computation + residual
        #
        # Expected shapes: everything stays (B, T, n_embd) throughout
        return x  # placeholder — replace with your implementation
        # --- END TODO 3 forward ---


# ============================================================================
# TODO 4: Full Transformer Language Model
# ============================================================================
# Put it all together:
#   1. Token embedding table: vocab_size → n_embd
#   2. Position embedding table: block_size → n_embd
#   3. Transformer block(s)
#   4. Final layer norm
#   5. Linear head: n_embd → vocab_size (to produce logits)
#
# Forward pass:
#   tok_emb = token_embedding(idx)           # (B, T, n_embd)
#   pos_emb = position_embedding(positions)  # (T, n_embd)
#   x = tok_emb + pos_emb                   # (B, T, n_embd)
#   x = blocks(x)                            # (B, T, n_embd)  — through transformer
#   x = layer_norm(x)                        # (B, T, n_embd)  — final norm
#   logits = linear_head(x)                  # (B, T, vocab_size)
#
# HINTS for __init__:
#   self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
#   self.position_embedding_table = nn.Embedding(block_size, n_embd)
#   self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
#   self.ln_f = nn.LayerNorm(n_embd)    # final layer norm
#   self.lm_head = nn.Linear(n_embd, vocab_size)
#
# HINTS for forward:
#   B, T = idx.shape
#   tok_emb = self.token_embedding_table(idx)                           # (B, T, n_embd)
#   pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, n_embd)
#   x = tok_emb + pos_emb                                              # (B, T, n_embd)
#   x = self.blocks(x)                                                  # (B, T, n_embd)
#   x = self.ln_f(x)                                                    # (B, T, n_embd)
#   logits = self.lm_head(x)                                           # (B, T, vocab_size)

class TransformerLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # --- YOUR CODE HERE ---
        # Create all the layers listed above:
        #   - token_embedding_table
        #   - position_embedding_table
        #   - blocks (nn.Sequential of Block modules)
        #   - ln_f (final layer norm)
        #   - lm_head (linear projection to vocab_size)
        pass
        # --- END TODO 4 __init__ ---

    def forward(self, idx, targets=None):
        # idx shape: (B, T) — batch of character index sequences
        # --- YOUR CODE HERE ---
        # 1. Get token embeddings                          (B, T, n_embd)
        # 2. Get position embeddings                       (T, n_embd)
        # 3. Add them together                             (B, T, n_embd)
        # 4. Pass through transformer blocks               (B, T, n_embd)
        # 5. Apply final layer norm                        (B, T, n_embd)
        # 6. Project to vocab size with lm_head            (B, T, vocab_size)
        #
        # Then compute loss if targets are provided:
        #   B, T, C = logits.shape
        #   logits = logits.view(B*T, C)
        #   targets = targets.view(B*T)
        #   loss = F.cross_entropy(logits, targets)

        B, T = idx.shape
        # Placeholder: just use a simple embedding (replace with full implementation!)
        logits = torch.zeros(B, T, vocab_size, device=device)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
        # --- END TODO 4 forward ---

    def generate(self, idx, max_new_tokens):
        """Generate new tokens autoregressively."""
        for _ in range(max_new_tokens):
            # Crop context to block_size (the model can only handle block_size positions)
            idx_cond = idx[:, -block_size:]                # (B, T) where T <= block_size
            logits, loss = self(idx_cond)                  # (B, T, vocab_size)
            logits = logits[:, -1, :]                      # (B, vocab_size) — last position
            probs = F.softmax(logits, dim=-1)              # (B, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)        # (B, T+1)
        return idx


# ============================================================================
# TODO 5: Training Loop and Generation
# ============================================================================
# This is the same training loop from previous chapters. Create the model,
# optimizer, train, and generate.
#
# HINTS:
#   model = TransformerLanguageModel().to(device)
#   optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#
#   for iter in range(max_iters):
#       if iter % eval_interval == 0:
#           losses = estimate_loss(model)
#           print(f"step {iter:5d}: train {losses['train']:.4f}, val {losses['val']:.4f}")
#       xb, yb = get_batch('train')
#       logits, loss = model(xb, yb)
#       optimizer.zero_grad(set_to_none=True)
#       loss.backward()
#       optimizer.step()
#
#   # Generate text
#   context = torch.zeros((1, 1), dtype=torch.long, device=device)
#   print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

# --- YOUR CODE HERE ---
print("\n--- Creating model ---")
model = TransformerLanguageModel().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model has {n_params:,} parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Generate before training
print("\n--- Generated text BEFORE training ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=100)[0].tolist()))
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

# Final loss
losses = estimate_loss(model)
print(f"Final:      train loss {losses['train']:.4f}, val loss {losses['val']:.4f}\n")

# Generate after training
print("--- Generated text AFTER training ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
print("--- End ---")

print("\nCongratulations! You've built a full transformer block.")
print("Compare this output to Chapter 2's single-head model.")
print("Next up: Chapter 4, where we scale up with more layers and dimensions.")
# --- END TODO 5 ---
