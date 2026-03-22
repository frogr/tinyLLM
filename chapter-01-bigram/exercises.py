"""
Chapter 1 — The Bigram Model (Exercises)

Work through the TODOs in order. Each one builds on the previous.
This file is runnable at every stage — incomplete TODOs use placeholder values.

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
# Hyperparameters (settings that control training)
# ============================================================================
batch_size = 32    # How many sequences to process in parallel
block_size = 8     # How many characters of context (sequence length)
max_iters = 5000   # How many training steps to run
eval_interval = 500  # How often to print loss
learning_rate = 1e-3  # How big each optimization step is
eval_iters = 200   # How many batches to average when estimating loss

torch.manual_seed(1337)  # For reproducibility — same random numbers every run

# ============================================================================
# TODO 1: Load and Explore the Dataset
# ============================================================================
# Read the tiny shakespeare file. It's just a big string of text.
#
# HINT: Use open() to read ../data/input.txt
# After loading, print:
#   - The total number of characters
#   - The first 200 characters so you can see what it looks like

# --- YOUR CODE HERE ---
try:
    with open('../data/input.txt', 'r') as f:
        text = f.read()
except FileNotFoundError:
    print("Dataset not found! Run 'python ../data/download.py' first.")
    print("Using placeholder text for now...\n")
    text = "First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n" * 100

# TODO 1: Print the length of the text and the first 200 characters
# print(f"Dataset has {len(text)} characters")
# print(f"First 200 characters:\n{text[:200]}")
# --- END TODO 1 ---

print(f"Dataset has {len(text)} characters")

# ============================================================================
# TODO 2: Build the Character Tokenizer
# ============================================================================
# We need to convert characters to numbers and back.
# Steps:
#   1. Find all unique characters in the text (the "vocabulary")
#   2. Create stoi: a dict mapping each character to a unique integer
#   3. Create itos: a dict mapping each integer back to its character
#   4. Create encode() and decode() functions
#
# HINT:
#   chars = sorted(list(set(text)))
#   stoi = {ch: i for i, ch in enumerate(chars)}
#   itos = {i: ch for i, ch in enumerate(chars)}
#   encode = lambda s: [stoi[c] for c in s]
#   decode = lambda l: ''.join([itos[i] for i in l])

# --- YOUR CODE HERE ---
# Placeholder tokenizer (just uses ASCII values) — replace with the real one!
_placeholder_tokenizer = True
chars = sorted(list(set(text)))
vocab_size = len(chars) if not _placeholder_tokenizer else len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
# --- END TODO 2 ---

print(f"Vocabulary size: {vocab_size} unique characters")
print(f"Characters: {''.join(chars)}")

# Quick test — encode and decode should round-trip perfectly
test_str = "hello"
encoded = encode(test_str)
decoded = decode(encoded)
print(f"Encode/decode test: '{test_str}' -> {encoded} -> '{decoded}'")
assert decoded == test_str, "Encode/decode round-trip failed!"

# ============================================================================
# TODO 3: Create Train/Val Splits
# ============================================================================
# We split the data into training (90%) and validation (10%).
# Training data is what the model learns from.
# Validation data is what we use to check if the model is actually learning
# general patterns vs. just memorizing the training data.
#
# Think of it like a test suite — you don't test with the same data you developed with.
#
# HINT:
#   data = torch.tensor(encode(text), dtype=torch.long)
#   n = int(0.9 * len(data))
#   train_data = data[:n]
#   val_data = data[n:]

# --- YOUR CODE HERE ---
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
# --- END TODO 3 ---

print(f"Train split: {len(train_data):,} characters")
print(f"Val split:   {len(val_data):,} characters")


# ============================================================================
# TODO 4: Build the Batch Loader
# ============================================================================
# This function grabs random chunks from the dataset.
#
# It returns two tensors:
#   x — the input sequences      shape: (batch_size, block_size)
#   y — the target sequences      shape: (batch_size, block_size)
#
# y is x shifted right by one (the "next character" for each position).
#
# EXAMPLE with block_size=4:
#   If our data contains: [18, 47, 56, 57, 58, 1, 15, 47, 58]
#   And we randomly pick starting index 2:
#     x = [56, 57, 58, 1]    (characters at positions 2,3,4,5)
#     y = [57, 58, 1, 15]    (characters at positions 3,4,5,6) — shifted by one!
#
# HINT:
#   ix = torch.randint(len(data) - block_size, (batch_size,))
#   x = torch.stack([data[i:i+block_size] for i in ix])
#   y = torch.stack([data[i+1:i+block_size+1] for i in ix])
#   return x.to(device), y.to(device)

def get_batch(split):
    """Get a random batch of inputs (x) and targets (y)."""
    data_split = train_data if split == 'train' else val_data

    # --- YOUR CODE HERE ---
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    # --- END TODO 4 ---

    return x, y


# Quick test — check shapes
xb, yb = get_batch('train')
print(f"\nBatch shapes: x={xb.shape}, y={yb.shape}")
print(f"First input sequence:  {xb[0].tolist()}")
print(f"First target sequence: {yb[0].tolist()}")


# ============================================================================
# Loss Estimation Helper
# ============================================================================
# This averages loss over many batches for a more stable measurement.
# Don't worry about the @torch.no_grad() decorator — it just tells PyTorch
# "we're not training right now, don't track gradients" (saves memory).

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()  # Set model to evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # Set model back to training mode
    return out


# ============================================================================
# TODO 5 & 6: Define the Bigram Language Model
# ============================================================================
# This is the entire model. It's just an embedding table.
#
# nn.Embedding(vocab_size, vocab_size) creates a table with shape (65, 65).
# When we look up character i, we get 65 numbers — the model's "scores" for
# which character comes next.
#
# TODO 5: Create the embedding table in __init__
# TODO 6: Implement the forward pass
#
# HINTS for __init__:
#   self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
#
# HINTS for forward:
#   logits = self.token_embedding_table(idx)  # shape: (B, T, vocab_size)
#
#   For loss calculation:
#     B, T, C = logits.shape
#     logits = logits.view(B*T, C)      # reshape for cross_entropy
#     targets = targets.view(B*T)        # reshape for cross_entropy
#     loss = F.cross_entropy(logits, targets)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # TODO 5: Create the embedding table
        # --- YOUR CODE HERE ---
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        # --- END TODO 5 ---

    def forward(self, idx, targets=None):
        # idx shape: (B, T) where B=batch_size, T=block_size
        # Each value in idx is a character index (0 to vocab_size-1)

        # TODO 6: Look up the embeddings and compute loss
        # --- YOUR CODE HERE ---
        logits = self.token_embedding_table(idx)  # shape: (B, T, C) where C=vocab_size

        if targets is None:
            loss = None
        else:
            # Reshape for cross_entropy: it expects (N, C) and (N,)
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)       # shape: (B*T, C)
            targets_flat = targets.view(B*T)          # shape: (B*T,)
            loss = F.cross_entropy(logits_flat, targets_flat)
        # --- END TODO 6 ---

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Generate new tokens by repeatedly predicting the next character."""
        # idx shape: (B, T) — current sequence of character indices
        for _ in range(max_new_tokens):
            # Get predictions
            logits, loss = self(idx)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # shape: (B, C)
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)  # shape: (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # shape: (B, 1)
            # Append to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # shape: (B, T+1)
        return idx


# ============================================================================
# Create the model and optimizer
# ============================================================================
model = BigramLanguageModel(vocab_size)
m = model.to(device)

# Count parameters
n_params = sum(p.numel() for p in m.parameters())
print(f"\nModel has {n_params:,} parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# ============================================================================
# Generate text BEFORE training (should be random gibberish)
# ============================================================================
print("\n--- Generated text BEFORE training ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))
print("--- End generated text ---\n")


# ============================================================================
# TODO 7: The Training Loop
# ============================================================================
# This is where the model actually learns. For each iteration:
#   1. Get a batch of data
#   2. Forward pass: compute predictions and loss
#   3. Backward pass: compute gradients
#   4. Optimizer step: update parameters
#
# HINT:
#   for iter in range(max_iters):
#       # Every eval_interval steps, print the loss
#       if iter % eval_interval == 0:
#           losses = estimate_loss(model)
#           print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
#
#       # Get a batch
#       xb, yb = get_batch('train')
#
#       # Forward pass
#       logits, loss = model(xb, yb)
#
#       # Backward pass
#       optimizer.zero_grad(set_to_none=True)
#       loss.backward()
#
#       # Update parameters
#       optimizer.step()

print("Starting training...")
# --- YOUR CODE HERE ---
for iter in range(max_iters):
    # Every eval_interval steps, estimate and print loss
    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {iter:5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Get a batch of training data
    xb, yb = get_batch('train')

    # Forward pass: get predictions and loss
    logits, loss = model(xb, yb)

    # Backward pass: compute gradients
    optimizer.zero_grad(set_to_none=True)  # Clear old gradients
    loss.backward()                         # Compute new gradients

    # Update parameters: nudge them to reduce the loss
    optimizer.step()
# --- END TODO 7 ---

# Print final loss
losses = estimate_loss(model)
print(f"Final:      train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


# ============================================================================
# TODO 8: Generate Text from the Trained Model
# ============================================================================
# Now generate text and see how it compares to the pre-training output!
#
# HINT: Same code as the pre-training generation, just more tokens.
#   context = torch.zeros((1, 1), dtype=torch.long, device=device)
#   generated = m.generate(context, max_new_tokens=500)[0].tolist()
#   print(decode(generated))

print("\n--- Generated text AFTER training ---")
# --- YOUR CODE HERE ---
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_ids = m.generate(context, max_new_tokens=500)[0].tolist()
print(decode(generated_ids))
# --- END TODO 8 ---
print("--- End generated text ---")

print("\nCongratulations! You've built your first language model.")
print("The output is gibberish, but it's SHAKESPEARE-FLAVORED gibberish.")
print("Notice it learned things like: character names followed by colons,")
print("roughly correct word lengths, and that lines end with newlines.")
print("\nNext up: Chapter 2, where we teach the model to look at more than")
print("one character at a time using self-attention.")
