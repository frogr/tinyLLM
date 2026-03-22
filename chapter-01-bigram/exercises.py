"""
Chapter 1: The Bigram Language Model
=====================================

We're building the simplest possible language model: given one character,
predict the next character. That's it. No memory, no context, just
"I see an 'H', so the next character is probably 'e'."

This is intentionally simple — it lets us learn the entire training pipeline
(data loading, model, loss, training loop, generation) without any complex
architecture getting in the way.

Run this file after completing each TODO to see your progress:
    python exercises.py

Dependencies: Make sure you've run `python ../data/download.py` first!
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# ============================================================================
# DEVICE SETUP
# ============================================================================
# PyTorch can run on CPU, NVIDIA GPU (cuda), or Apple Silicon GPU (mps).
# MPS = Metal Performance Shaders, Apple's GPU framework.
# This picks the best available option automatically.
device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
# These are the knobs we can turn. Don't worry about picking perfect values —
# part of learning is seeing what happens when you change them.
batch_size = 32        # How many independent sequences to process in parallel
block_size = 8         # Maximum context length (how many characters the model sees)
max_iters = 5000       # How many training steps to run
eval_interval = 500    # How often to print the loss
learning_rate = 1e-3   # How big of a step to take when updating weights
eval_iters = 200       # How many batches to average over when estimating loss

# For reproducibility — same random seed = same results every time
torch.manual_seed(1337)

# ============================================================================
# DATA LOADING
# ============================================================================
# Read the tiny Shakespeare dataset. This is about 1MB of text — every play,
# sonnet, and poem concatenated together.

import os
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'input.txt')

# TODO 1: Read the data file
# -------------------------
# Read the contents of the file at data_path into a string variable called `text`.
# This is just regular Python file I/O — nothing ML-specific yet.
#
# Hint: Use open() and .read()
# Expected: text should be a string with ~1 million characters
text = ""  # <-- Replace this with actual file reading
if text:
    print(f"Dataset size: {len(text):,} characters")
    print(f"First 100 characters:\n{text[:100]}")
else:
    print("TODO 1: Read the data file (text is empty)")

# TODO 2: Build the vocabulary
# ----------------------------
# A "vocabulary" is just the set of all unique characters in our text.
# We need two mappings:
#   - stoi (string to integer): given a character, what's its number?
#   - itos (integer to string): given a number, what character is it?
#
# Think of this as building a codec — we're encoding characters as integers
# so the neural network can work with them (neural nets only understand numbers).
#
# Hint: sorted(set(text)) gives you all unique characters in order
# Expected: ~65 unique characters (letters, punctuation, spaces, newlines)
chars = sorted(set(text)) if text else []
vocab_size = len(chars) if chars else 65  # fallback so file doesn't crash

stoi = {}  # <-- Build this: {char: index} mapping
itos = {}  # <-- Build this: {index: char} mapping

if stoi:
    print(f"\nVocabulary size: {vocab_size} characters")
    print(f"Characters: {''.join(chars)}")
    print(f"Example: '{chars[0]}' -> {stoi.get(chars[0], '?')}")
else:
    print("\nTODO 2: Build the vocabulary (stoi/itos are empty)")

# Helper functions for encoding/decoding
# encode: string -> list of integers
# decode: list of integers -> string
encode = lambda s: [stoi[c] for c in s] if stoi else []
decode = lambda l: ''.join([itos[i] for i in l]) if itos else ""

# TODO 3: Encode the dataset and create train/val split
# -----------------------------------------------------
# Convert the entire text into a tensor of integers using our encode function.
# Then split it: 90% for training, 10% for validation.
#
# Why split? Same reason as in any ML: we want to check if the model is
# actually learning patterns vs. just memorizing the training data.
# The validation set is data the model never trains on.
#
# Hint: torch.tensor(encode(text)) converts the encoded text to a tensor
# Expected: data should be a 1D tensor with ~1M elements
data = torch.tensor(encode(text), dtype=torch.long) if text and stoi else torch.zeros(100, dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

if text and stoi:
    print(f"\nEncoded data shape: {data.shape}")  # shape: (num_characters,)
    print(f"Train size: {len(train_data):,}, Val size: {len(val_data):,}")
    print(f"First 20 encoded values: {data[:20].tolist()}")
    print(f"Decoded back: '{decode(data[:20].tolist())}'")
else:
    print("\nTODO 3: Encode the dataset (waiting on TODOs 1 & 2)")

# ============================================================================
# BATCH LOADER
# ============================================================================

def get_batch(split):
    """
    Generate a random batch of training examples.

    This grabs `batch_size` random chunks of text, each `block_size` characters long.
    For each chunk, the input is the first block_size characters and the target
    is the same chunk shifted by one position (because we're predicting the NEXT
    character at each position).

    Returns:
        x: input tensor,  shape (batch_size, block_size)
        y: target tensor, shape (batch_size, block_size)
    """
    source = train_data if split == 'train' else val_data

    # TODO 4: Implement the batch loader
    # -----------------------------------
    # 1. Pick `batch_size` random starting positions in the data
    #    (make sure you leave room for block_size characters after the start)
    # 2. For each starting position, grab a chunk of block_size characters as input (x)
    # 3. Grab the same chunk shifted by 1 as the target (y)
    # 4. Stack them into tensors and move to device
    #
    # Hint: torch.randint(low, high, (size,)) gives random integers
    # Hint: torch.stack([list of tensors]) combines them into a 2D tensor
    #
    # The shift-by-one is the key insight:
    #   data = [18, 47, 56, 57, 58,  1, 15, 47, 58]
    #   x    = [18, 47, 56, 57, 58,  1, 15, 47]     (positions 0-7)
    #   y    = [47, 56, 57, 58,  1, 15, 47, 58]     (positions 1-8)
    #
    # So when the model sees [18], it should predict 47
    # When it sees [18, 47], it should predict 56
    # etc.

    # Placeholder that returns random data so the file doesn't crash
    x = torch.zeros(batch_size, block_size, dtype=torch.long, device=device)
    y = torch.zeros(batch_size, block_size, dtype=torch.long, device=device)
    return x, y


# ============================================================================
# LOSS ESTIMATION
# ============================================================================

@torch.no_grad()  # Tells PyTorch "don't track gradients here" — saves memory
def estimate_loss():
    """Average the loss over many batches for a more stable estimate."""
    out = {}
    model.eval()  # Set model to evaluation mode (matters more in later chapters)
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # Set back to training mode
    return out


# ============================================================================
# THE BIGRAM MODEL
# ============================================================================

class BigramLanguageModel(nn.Module):
    """
    The simplest possible language model.

    It has ONE learnable component: an embedding table.
    Given a character (as an integer), it looks up that row in the table
    and uses it directly as the prediction for what comes next.

    That's it. No attention, no hidden layers, no magic.
    Think of it as: "Every time I see the letter 'q', predict 'u'."
    """

    def __init__(self, vocab_size):
        super().__init__()
        # TODO 5: Create the embedding table
        # -----------------------------------
        # nn.Embedding(num_embeddings, embedding_dim) creates a lookup table.
        # For a bigram model, we want each token to directly predict the next token,
        # so embedding_dim should equal vocab_size (each row is a probability
        # distribution over all possible next characters).
        #
        # Hint: self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        # Shape of the table: (vocab_size, vocab_size)
        # Row i contains the model's beliefs about what comes after character i
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """
        Forward pass: given input indices, produce predictions.

        Args:
            idx: tensor of token indices, shape (batch_size, block_size)
            targets: tensor of target indices, shape (batch_size, block_size), or None

        Returns:
            logits: raw predictions, shape (batch_size, block_size, vocab_size)
            loss: cross-entropy loss (only if targets provided)
        """
        # TODO 6: Implement the forward pass
        # ------------------------------------
        # Step 1: Look up the embeddings for each input token
        #   logits = self.token_embedding_table(idx)
        #   This takes idx of shape (B, T) and returns (B, T, C)
        #   where B=batch_size, T=block_size (time), C=vocab_size (channels)
        #
        # Step 2: If targets are provided, compute the loss
        #   PyTorch's cross_entropy expects shapes (N, C) and (N,)
        #   so we need to reshape:
        #     logits from (B, T, C) → (B*T, C)
        #     targets from (B, T)   → (B*T,)
        #
        #   loss = F.cross_entropy(logits_reshaped, targets_reshaped)
        #
        # Hint for reshaping: tensor.view(B*T, C) or tensor.view(-1, C)
        #   -1 means "figure out this dimension automatically"

        logits = self.token_embedding_table(idx)  # shape: (B, T, C)
        loss = None

        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)    # shape: (B*T, C)
            targets_flat = targets.view(B * T)      # shape: (B*T,)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Generate new text by repeatedly predicting the next character.

        Args:
            idx: starting context, shape (batch_size, current_length)
            max_new_tokens: how many new characters to generate

        Returns:
            idx: extended sequence, shape (batch_size, current_length + max_new_tokens)
        """
        # TODO 7: Implement text generation
        # -----------------------------------
        # For each new token we want to generate:
        # 1. Run the forward pass to get logits
        # 2. Take only the logits for the LAST time step: logits[:, -1, :]
        #    (shape goes from (B, T, C) to (B, C))
        # 3. Convert logits to probabilities with softmax
        # 4. Sample from the probability distribution
        # 5. Append the sampled token to the sequence
        #
        # Hint:
        #   logits, _ = self(idx)              # forward pass
        #   logits = logits[:, -1, :]          # last time step, shape: (B, C)
        #   probs = F.softmax(logits, dim=-1)  # convert to probabilities
        #   idx_next = torch.multinomial(probs, num_samples=1)  # sample, shape: (B, 1)
        #   idx = torch.cat((idx, idx_next), dim=1)  # append to sequence

        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]              # shape: (B, C)
            probs = F.softmax(logits, dim=-1)      # shape: (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # shape: (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # shape: (B, T+1)
        return idx


# ============================================================================
# CREATE THE MODEL
# ============================================================================

model = BigramLanguageModel(vocab_size).to(device)
print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")

# Test generation before training (should be random gibberish)
print("\n--- Before Training ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # Start with token 0
generated = model.generate(context, max_new_tokens=100)
print(f"Generated text: '{decode(generated[0].tolist())}'")

# ============================================================================
# TRAINING LOOP
# ============================================================================

# TODO 8: Create the optimizer
# ----------------------------
# The optimizer is what actually updates the model's parameters based on gradients.
# AdamW is the go-to optimizer — it's like gradient descent but smarter
# (it adapts the learning rate for each parameter individually).
#
# Hint: optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# TODO 9: Implement the training loop
# ------------------------------------
# This is the core learning cycle. For each iteration:
# 1. Get a batch of training data
# 2. Forward pass: compute predictions and loss
# 3. Backward pass: compute gradients (how to adjust each parameter)
# 4. Optimizer step: actually adjust the parameters
# 5. Zero the gradients (PyTorch accumulates gradients by default)
#
# The pattern is ALWAYS:
#   loss = model(inputs)
#   optimizer.zero_grad(set_to_none=True)
#   loss.backward()
#   optimizer.step()
#
# Print the loss every eval_interval steps to track progress.
# You should see the loss decrease from ~4.1 to ~2.5 over 5000 steps.

print("\n--- Training ---")
for iter in range(max_iters):
    # Every eval_interval steps, estimate and print the loss
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter:5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Get a batch of training data
    xb, yb = get_batch('train')

    # Forward pass
    logits, loss = model(xb, yb)

    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # Update parameters
    optimizer.step()

# Final loss
losses = estimate_loss()
print(f"Step {max_iters:5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

# ============================================================================
# GENERATE TEXT
# ============================================================================

print("\n--- After Training ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500)
print(f"Generated text:\n{decode(generated[0].tolist())}")
print("\n(This will be gibberish, but it should look more Shakespeare-ish than before training!)")
