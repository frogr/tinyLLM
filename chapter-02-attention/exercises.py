"""
Chapter 2: Self-Attention from Scratch
========================================

Last chapter we built a bigram model — it predicts the next character by
looking at only the current character. That's like autocomplete that only
sees the last letter you typed. Not great.

Now we're adding ATTENTION: a mechanism that lets each token "look at"
previous tokens and decide which ones are relevant for predicting what
comes next. This is the key innovation behind transformers and GPT.

Run this file after completing each TODO:
    python exercises.py
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# ============================================================================
# DEVICE SETUP
# ============================================================================
device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
eval_iters = 200
n_embd = 32         # NEW: embedding dimension (how "rich" each token's representation is)
head_size = 16       # NEW: size of attention head (dimension of Q, K, V)

torch.manual_seed(1337)

# ============================================================================
# DATA LOADING (same as chapter 1)
# ============================================================================
import os
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'input.txt')

with open(data_path, 'r') as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Vocabulary size: {vocab_size}")
print(f"Dataset size: {len(data):,} characters")

def get_batch(split):
    """Generate a random batch of training examples."""
    source = train_data if split == 'train' else val_data
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i:i+block_size] for i in ix])
    y = torch.stack([source[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    """Average the loss over many batches for a more stable estimate."""
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
# SELF-ATTENTION HEAD
# ============================================================================

class Head(nn.Module):
    """
    One head of self-attention.

    This is where the magic happens. Each token produces three vectors:
      - Query (Q): "What am I looking for?"
      - Key (K):   "What do I have to offer?"
      - Value (V): "What information do I actually provide?"

    The attention mechanism:
    1. Compute how much each token should attend to every other token (Q @ K)
    2. Mask out future tokens (can't cheat by looking ahead)
    3. Normalize with softmax (turn scores into probabilities)
    4. Use those probabilities to compute a weighted sum of Values

    Think of it as a search engine running inside the model at every position.
    """

    def __init__(self, head_size):
        super().__init__()
        # TODO 1: Create Q, K, V projection layers
        # ------------------------------------------
        # Each of these is a linear layer (matrix multiplication) that transforms
        # a token's embedding into a query, key, or value vector.
        #
        # Input dimension:  n_embd (the token's embedding size, 32)
        # Output dimension: head_size (the attention dimension, 16)
        #
        # We use bias=False because the original "Attention Is All You Need"
        # paper does, and it works well empirically.
        #
        # Hint:
        #   self.key   = nn.Linear(n_embd, head_size, bias=False)
        #   self.query = nn.Linear(n_embd, head_size, bias=False)
        #   self.value = nn.Linear(n_embd, head_size, bias=False)
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # The triangular mask. register_buffer means it's part of the module
        # but NOT a learnable parameter — it's a constant.
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        """
        Args:
            x: input embeddings, shape (B, T, C) where C = n_embd

        Returns:
            out: attention output, shape (B, T, head_size)
        """
        B, T, C = x.shape

        # TODO 2: Compute Q, K, V
        # -------------------------
        # Pass x through each linear layer to get queries, keys, values.
        # Each will have shape (B, T, head_size).
        #
        # Hint:
        #   k = self.key(x)    # shape: (B, T, head_size)
        #   q = self.query(x)  # shape: (B, T, head_size)
        #   v = self.value(x)  # shape: (B, T, head_size)
        k = self.key(x)        # shape: (B, T, head_size)
        q = self.query(x)      # shape: (B, T, head_size)
        v = self.value(x)      # shape: (B, T, head_size)

        # TODO 3: Compute attention scores
        # ----------------------------------
        # The attention score between token i and token j is:
        #   score = Q[i] · K[j] / sqrt(head_size)
        #
        # In matrix form: scores = Q @ K^T / sqrt(head_size)
        #
        # The sqrt(head_size) scaling prevents dot products from getting too large.
        # Without it, softmax produces very peaked distributions (almost one-hot),
        # and the gradients vanish — the model stops learning.
        #
        # Hint:
        #   wei = q @ k.transpose(-2, -1) * head_size**-0.5
        #   # shapes: (B, T, head_size) @ (B, head_size, T) → (B, T, T)
        wei = torch.zeros(B, T, T, device=x.device)  # <-- Replace with actual computation

        # TODO 4: Apply the causal mask
        # ------------------------------
        # We can't let tokens attend to future positions — that would be cheating.
        # The mask is a lower-triangular matrix: position i can see positions 0..i only.
        #
        # We set masked positions to negative infinity BEFORE softmax.
        # softmax(−∞) = 0, so those positions get zero attention weight.
        #
        # Hint:
        #   wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        #   wei = F.softmax(wei, dim=-1)  # shape: (B, T, T)
        pass  # <-- Replace with mask + softmax

        # TODO 5: Compute weighted sum of values
        # ----------------------------------------
        # Now we have attention weights (who to listen to) and values (what they say).
        # The output for each position is a weighted average of all value vectors,
        # using the attention weights.
        #
        # Hint:
        #   out = wei @ v  # (B, T, T) @ (B, T, head_size) → (B, T, head_size)
        out = torch.zeros(B, T, head_size, device=x.device)  # <-- Replace

        return out


# ============================================================================
# THE MODEL (upgraded from bigram)
# ============================================================================

class AttentionLanguageModel(nn.Module):
    """
    Like the bigram model, but now we add:
    1. A richer token embedding (n_embd dimensions instead of vocab_size)
    2. Position embeddings (so the model knows WHERE each token is)
    3. A self-attention head (so tokens can look at each other)
    4. A final linear layer to project back to vocab_size for predictions

    Architecture:
        Token Embedding + Position Embedding → Attention Head → Linear → Logits
    """

    def __init__(self):
        super().__init__()
        # TODO 6: Create the model components
        # -------------------------------------
        # 1. Token embedding: maps token IDs to vectors of size n_embd
        #    (In ch1 it mapped to vocab_size; now we use a richer representation)
        #
        # 2. Position embedding: maps positions (0, 1, ..., block_size-1) to
        #    vectors of size n_embd. This tells the model WHERE each token is.
        #    Without this, attention treats "H at position 0" the same as
        #    "H at position 5."
        #
        # 3. Self-attention head: the Head class we defined above
        #
        # 4. Final linear layer: projects from head_size back to vocab_size
        #    so we can get a prediction for every possible next character
        #
        # Hint:
        #   self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        #   self.position_embedding_table = nn.Embedding(block_size, n_embd)
        #   self.sa_head = Head(head_size)
        #   self.lm_head = nn.Linear(head_size, vocab_size)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(head_size)
        self.lm_head = nn.Linear(head_size, vocab_size)

    def forward(self, idx, targets=None):
        """
        Forward pass with attention.

        Args:
            idx: token indices, shape (B, T)
            targets: target indices, shape (B, T), or None for generation

        Returns:
            logits: predictions, shape (B, T, vocab_size)
            loss: cross-entropy loss, or None
        """
        B, T = idx.shape

        # TODO 7: Implement the forward pass
        # -------------------------------------
        # 1. Get token embeddings:    shape (B, T, n_embd)
        # 2. Get position embeddings: shape (T, n_embd)
        #    Use torch.arange(T, device=device) to create [0, 1, 2, ..., T-1]
        # 3. Add them: x = tok_emb + pos_emb  (broadcasting adds pos_emb to each batch)
        # 4. Pass through attention head: shape (B, T, head_size)
        # 5. Project to vocabulary:        shape (B, T, vocab_size)
        #
        # Hint:
        #   tok_emb = self.token_embedding_table(idx)                                    # (B, T, n_embd)
        #   pos_emb = self.position_embedding_table(torch.arange(T, device=device))      # (T, n_embd)
        #   x = tok_emb + pos_emb                                                        # (B, T, n_embd)
        #   x = self.sa_head(x)                                                          # (B, T, head_size)
        #   logits = self.lm_head(x)                                                     # (B, T, vocab_size)

        # Placeholder — produces wrong results but won't crash
        logits = self.token_embedding_table(idx)  # (B, T, n_embd) — wrong shape for loss
        if logits.shape[-1] != vocab_size:
            logits = torch.zeros(B, T, vocab_size, device=device)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)     # (B*T, C)
            targets_flat = targets.view(B * T)       # (B*T,)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Generate new tokens by repeatedly predicting the next one."""
        for _ in range(max_new_tokens):
            # Crop to last block_size tokens (position embedding only goes that far)
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]              # last position only: (B, C)
            probs = F.softmax(logits, dim=-1)      # convert to probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # sample: (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)             # append: (B, T+1)
        return idx


# ============================================================================
# TRAINING
# ============================================================================

model = AttentionLanguageModel().to(device)
param_count = sum(p.numel() for p in model.parameters())
print(f"\nModel created with {param_count:,} parameters")
print(f"(Compare to chapter 1's bigram model which had {vocab_size * vocab_size:,} parameters)")

# Generate before training (should be random gibberish)
print("\n--- Before Training ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=100)
print(f"Generated: '{decode(generated[0].tolist())}'")

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("\n--- Training ---")
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter:5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

losses = estimate_loss()
print(f"Step {max_iters:5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

# Generate after training
print("\n--- After Training ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500)
print(f"Generated text:\n{decode(generated[0].tolist())}")
print("\n(Should be slightly better than the bigram model — attention lets tokens look at context!)")
