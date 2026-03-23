"""
Chapter 4 — Scaling Up (Exercises)

Work through the TODOs in order. Each one builds on the previous.
This file is runnable at every stage — incomplete TODOs use placeholder values.

By the end, you'll have a model that produces semi-coherent Shakespeare.
This is where it goes from "toy" to "wow, this actually works."

Run with: python exercises.py
Expected training time: ~2-5 min on Apple Silicon, ~10-20 min on CPU.
"""

import time
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
# Hyperparameters — The "Real" nanoGPT Settings for Tiny Shakespeare
# ============================================================================
# These are the values from Karpathy's nanoGPT, tuned for this dataset.
# Compare to Chapter 1: batch_size=32, block_size=8, n_embd=vocab_size(65).
# We're going much bigger now.

batch_size = 64        # 32 -> 64: more sequences per step = more stable gradients
block_size = 256       # 8 -> 256: the model can now "see" ~50 words of context
max_iters = 5000       # Total training steps
eval_interval = 500    # Print loss every 500 steps
eval_iters = 200       # Average loss over 200 batches for stable estimates
learning_rate = 3e-4   # 1e-3 -> 3e-4: smaller steps for a bigger model
n_embd = 384           # Embedding dimension (was vocab_size=65 in Ch1)
n_head = 6             # Number of attention heads (384 / 6 = 64 dims per head)
n_layer = 6            # Number of transformer blocks (was 1 in Ch3)
dropout = 0.2          # Dropout rate: randomly zero 20% of activations

# NOTE: If you get MPS memory errors, try these reduced settings:
#   block_size = 128
#   batch_size = 32
#   n_embd = 128
#   n_layer = 4

torch.manual_seed(1337)

# ============================================================================
# Data Loading (same as previous chapters)
# ============================================================================
try:
    with open('../data/input.txt', 'r') as f:
        text = f.read()
except FileNotFoundError:
    print("Dataset not found! Run 'python ../data/download.py' first.")
    print("Using placeholder text for now...\n")
    text = "First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n" * 1000

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
# TODO 1: Build the Full Transformer Model
# ============================================================================
# Copy/adapt your code from Chapter 3. You need all these components:
#
#   1. Head — a single attention head
#   2. MultiHeadAttention — runs multiple heads in parallel and concatenates
#   3. FeedForward — two-layer MLP with ReLU
#   4. Block — one transformer block (attention + feedforward + residuals + layernorm)
#   5. GPTLanguageModel — the full model that stacks everything together
#
# Key differences from Chapter 3:
#   - n_embd is now 384 (not 32 or 64)
#   - n_head is 6 (not 4)
#   - n_layer is 6 (not 1)
#   - head_size = n_embd // n_head = 64
#   - FeedForward inner dimension is 4 * n_embd = 1536
#
# The architecture is identical — we're just making it bigger.
#
# HINTS:
#
# class Head(nn.Module):
#     """One head of self-attention."""
#     def __init__(self, head_size):
#         super().__init__()
#         self.key   = nn.Linear(n_embd, head_size, bias=False)
#         self.query = nn.Linear(n_embd, head_size, bias=False)
#         self.value = nn.Linear(n_embd, head_size, bias=False)
#         self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
#         # TODO 2 adds dropout here
#
#     def forward(self, x):
#         B, T, C = x.shape
#         k = self.key(x)
#         q = self.query(x)
#         wei = q @ k.transpose(-2, -1) * (C ** -0.5)
#         wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
#         wei = F.softmax(wei, dim=-1)
#         # TODO 2 applies dropout to wei here
#         v = self.value(x)
#         out = wei @ v
#         return out
#
# class MultiHeadAttention(nn.Module):
#     def __init__(self, num_heads, head_size):
#         super().__init__()
#         self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
#         self.proj = nn.Linear(n_embd, n_embd)
#         # TODO 2 adds dropout here
#
#     def forward(self, x):
#         out = torch.cat([h(x) for h in self.heads], dim=-1)
#         out = self.proj(out)
#         # TODO 2 applies dropout to out here
#         return out
#
# class FeedForward(nn.Module):
#     def __init__(self, n_embd):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(n_embd, 4 * n_embd),
#             nn.ReLU(),
#             nn.Linear(4 * n_embd, n_embd),
#             # TODO 2 adds dropout here
#         )
#
#     def forward(self, x):
#         return self.net(x)
#
# class Block(nn.Module):
#     def __init__(self, n_embd, n_head):
#         super().__init__()
#         head_size = n_embd // n_head
#         self.sa = MultiHeadAttention(n_head, head_size)
#         self.ffwd = FeedForward(n_embd)
#         self.ln1 = nn.LayerNorm(n_embd)
#         self.ln2 = nn.LayerNorm(n_embd)
#
#     def forward(self, x):
#         x = x + self.sa(self.ln1(x))    # residual + attention
#         x = x + self.ffwd(self.ln2(x))  # residual + feedforward
#         return x

# --- YOUR CODE FOR TODO 1 HERE ---

# Placeholder model (same as Chapter 1 bigram) — replace with full transformer!
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
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
            # Crop context to block_size (important now that we have position embeddings!)
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- END TODO 1 ---


# ============================================================================
# TODO 2: Add Dropout
# ============================================================================
# Go back to your model code above and add dropout in three places:
#
# 1. In Head.__init__ and Head.forward:
#        self.dropout = nn.Dropout(dropout)
#        # In forward, after softmax:
#        wei = self.dropout(wei)
#
# 2. In MultiHeadAttention.__init__ and MultiHeadAttention.forward:
#        self.dropout = nn.Dropout(dropout)
#        # In forward, after projection:
#        out = self.dropout(self.proj(out))
#
# 3. In FeedForward — add nn.Dropout(dropout) as the last layer in nn.Sequential
#
# 4. In GPTLanguageModel.__init__ — add dropout after embedding sum:
#        self.dropout = nn.Dropout(dropout)
#        # In forward, after adding token + position embeddings:
#        x = self.dropout(tok_emb + pos_emb)
#
# WHY: Without dropout, the model will memorize the training data. You'll see
# training loss drop very low but validation loss stay high or increase.
# With dropout=0.2, the model learns more general patterns.


# ============================================================================
# Create Model and Print Stats
# ============================================================================
model = GPTLanguageModel()
model = model.to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel has {n_params:,} parameters")
print(f"  (For reference: Chapter 1 bigram had 4,225 parameters)")
print(f"  (The full transformer should have ~10.8M parameters)")

if n_params < 10000:
    print("\n  *** Still using the placeholder model! ***")
    print("  *** Complete TODO 1 to build the real transformer. ***\n")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# ============================================================================
# TODO 3: Training Loop with Timing
# ============================================================================
# Implement the training loop. This is the same structure as Chapter 1, but
# we add timing so you can see how long it takes.
#
# HINT:
#   start_time = time.time()
#
#   for iter in range(max_iters):
#       if iter % eval_interval == 0 or iter == max_iters - 1:
#           losses = estimate_loss(model)
#           elapsed = time.time() - start_time
#           print(f"step {iter:5d} | train loss {losses['train']:.4f} | "
#                 f"val loss {losses['val']:.4f} | time {elapsed:.1f}s")
#
#       xb, yb = get_batch('train')
#       logits, loss = model(xb, yb)
#       optimizer.zero_grad(set_to_none=True)
#       loss.backward()
#       optimizer.step()
#
#   total_time = time.time() - start_time
#   print(f"\nTraining complete in {total_time:.1f}s")

print("Starting training...")

# --- YOUR CODE FOR TODO 3 HERE ---
start_time = time.time()
# (Replace this placeholder with the real training loop)
print("TODO 3: Implement the training loop!")
print("(See the hints above for the full code)")
total_time = time.time() - start_time
print(f"Training took {total_time:.1f}s")
# --- END TODO 3 ---


# ============================================================================
# TODO 4: Generate Text and Evaluate Quality
# ============================================================================
# Generate 1000 tokens and see what you get!
#
# With the placeholder bigram model, you'll get the same quality as Chapter 1.
# With the full transformer + dropout + training, you should see:
#   - Real English words
#   - Character names followed by colons (ROMEO:, JULIET:, etc.)
#   - Something resembling verse structure
#   - Occasional coherent phrases
#
# HINT:
#   model.eval()  # IMPORTANT: turn off dropout for generation!
#   context = torch.zeros((1, 1), dtype=torch.long, device=device)
#   generated = model.generate(context, max_new_tokens=1000)[0].tolist()
#   print(decode(generated))
#   model.train()

print("\n" + "="*70)
print("GENERATED SHAKESPEARE (1000 characters)")
print("="*70)

# --- YOUR CODE FOR TODO 4 HERE ---
print("TODO 4: Generate text from the trained model!")
print("(Make sure to call model.eval() before generating)")
# --- END TODO 4 ---

print("="*70)
print("\nDone! If you completed all TODOs with the full transformer,")
print("you should see semi-coherent Shakespeare above.")
print("If it's still gibberish, check:")
print("  - Did you complete TODO 1? (model should have ~10.8M params)")
print("  - Did you complete TODO 2? (dropout prevents overfitting)")
print("  - Did you complete TODO 3? (training loop needs to run)")
print("  - Did you call model.eval() in TODO 4? (dropout hurts generation)")
