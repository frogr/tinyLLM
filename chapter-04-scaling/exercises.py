"""
Chapter 4: Scaling Up — Exercises
=================================

We take our transformer block from Chapter 3 and scale it up:
- 6 layers instead of 1
- 384-dim embeddings instead of 32
- Dropout for regularization
- Learning rate schedule with warmup

After training, this model will generate recognizably Shakespeare-like text.

There are 5 TODOs to complete. Each one builds on the previous.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import math
import os

# --------------------------------------------------------------------------
# Hyperparameters — these are the "scaled up" settings for this chapter
# --------------------------------------------------------------------------
batch_size = 64       # sequences per training step
block_size = 256      # context window (characters of context)
max_iters = 5000      # training steps
eval_interval = 500   # how often to estimate loss
eval_iters = 200      # batches to average for loss estimate
learning_rate = 3e-4  # peak learning rate (note: smaller than ch1's 1e-3)
n_embd = 384          # embedding dimension (was 32)
n_head = 6            # attention heads (was 4)
n_layer = 6           # transformer blocks (was 1)
dropout = 0.2         # dropout rate (new!)
warmup_iters = 500    # LR warmup steps
min_lr = 3e-5         # minimum learning rate (10x smaller than peak)

device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# --------------------------------------------------------------------------
# Data loading — same as previous chapters
# --------------------------------------------------------------------------
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "input.txt")
if not os.path.exists(data_path):
    print(f"Data file not found at {data_path}")
    print("Run: python data/download.py")
    exit(1)

with open(data_path, "r") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# Train/val split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ==========================================================================
# Model components — mostly from Chapter 3, but now with dropout
# ==========================================================================

class Head(nn.Module):
    """Single head of self-attention, now with dropout."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        # TODO 1a: Add a dropout layer for attention weights
        # Hint: self.dropout = nn.Dropout(...)
        # YOUR CODE HERE

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # Compute attention scores
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        # TODO 1b: Apply dropout to attention weights (wei)
        # This randomly zeros out some attention connections during training,
        # preventing the model from relying too heavily on specific positions.
        # YOUR CODE HERE

        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of attention in parallel, with output projection and dropout."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

        # TODO 1c: Add dropout after the projection
        # self.dropout = nn.Dropout(...)
        # YOUR CODE HERE

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)

        # TODO 1d: Apply dropout to the projected output
        # YOUR CODE HERE

        return out


class FeedForward(nn.Module):
    """Feed-forward network with dropout."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            # TODO 1e: Add a Dropout layer here (inside the Sequential)
            # This goes after the activation, before the output projection.
            # YOUR CODE HERE
            nn.Linear(4 * n_embd, n_embd),
            # TODO 1f: Add another Dropout layer here (after the output projection)
            # YOUR CODE HERE
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication (attention) + computation (feedforward)."""

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


# ==========================================================================
# TODO 2: Build the full GPT model
# ==========================================================================

class GPTLanguageModel(nn.Module):
    """
    The full scaled-up model.

    This is like the BigramLanguageModel from Chapter 1, but with:
    - Token AND position embeddings
    - Multiple (n_layer) transformer blocks
    - Layer normalization
    - Proper weight initialization

    Architecture:
        token_embedding + position_embedding
        -> dropout
        -> Block 1 -> Block 2 -> ... -> Block n_layer
        -> LayerNorm
        -> Linear (to vocab_size)
    """

    def __init__(self):
        super().__init__()

        # TODO 2a: Create the model components:
        # 1. token_embedding_table: nn.Embedding(vocab_size, n_embd)
        # 2. position_embedding_table: nn.Embedding(block_size, n_embd)
        # 3. blocks: nn.Sequential of n_layer Block instances
        #    Hint: nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        # 4. ln_f: Final LayerNorm(n_embd)
        # 5. lm_head: nn.Linear(n_embd, vocab_size)
        # 6. dropout: nn.Dropout(dropout) — applied after embeddings
        #
        # YOUR CODE HERE

        # TODO 2b: Initialize weights
        # This helps training converge faster and more reliably.
        # Apply self._init_weights to all submodules:
        #   self.apply(self._init_weights)
        #
        # YOUR CODE HERE
        pass

    def _init_weights(self, module):
        """Initialize weights with small random values.

        Linear layers: normal distribution with std=0.02
        Embedding layers: normal distribution with std=0.02
        LayerNorm: bias=0, weight=1 (these are the defaults, but explicit is good)
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # TODO 2c: Implement the forward pass
        # 1. Get token embeddings: self.token_embedding_table(idx)  -> (B, T, n_embd)
        # 2. Get position embeddings: self.position_embedding_table(torch.arange(T, device=device))
        # 3. Add them together: tok_emb + pos_emb
        # 4. Apply dropout
        # 5. Pass through all blocks: self.blocks(x)
        # 6. Apply final layer norm: self.ln_f(x)
        # 7. Get logits: self.lm_head(x)  -> (B, T, vocab_size)
        #
        # Then compute loss if targets are provided (same as previous chapters):
        #   B, T, C = logits.shape
        #   logits_flat = logits.view(B*T, C)
        #   targets_flat = targets.view(B*T)
        #   loss = F.cross_entropy(logits_flat, targets_flat)
        #
        # YOUR CODE HERE
        logits = None  # replace this
        loss = None

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Generate new tokens autoregressively."""
        for _ in range(max_new_tokens):
            # Crop context to block_size (important now that block_size=256!)
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ==========================================================================
# TODO 3: Learning rate schedule
# ==========================================================================

def get_lr(it):
    """
    Learning rate schedule with linear warmup and cosine decay.

    The schedule has three phases:
    1. Warmup (0 to warmup_iters): linearly increase from 0 to learning_rate
    2. Cosine decay (warmup_iters to max_iters): smoothly decrease to min_lr
    3. After max_iters: stay at min_lr

    Args:
        it: current training iteration

    Returns:
        lr: the learning rate for this iteration

    Hint:
    - Warmup: lr = learning_rate * (it / warmup_iters)
    - Cosine decay: use cosine schedule between learning_rate and min_lr
      decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
      coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
      lr = min_lr + coeff * (learning_rate - min_lr)
    """
    # TODO 3: Implement the learning rate schedule
    # YOUR CODE HERE
    return learning_rate  # replace this with the actual schedule


# ==========================================================================
# TODO 4: Training loop
# ==========================================================================

def train():
    model = GPTLanguageModel()
    model = model.to(device)

    # Print parameter count
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params:,} parameters ({n_params/1e6:.1f}M)")
    print(f"Training for {max_iters} steps with batch_size={batch_size}")
    print(f"Context window: {block_size} characters")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    start_time = time.time()

    for iter in range(max_iters):

        # TODO 4a: Update the learning rate using get_lr()
        # Set the learning rate for all parameter groups:
        #   lr = get_lr(iter)
        #   for param_group in optimizer.param_groups:
        #       param_group["lr"] = lr
        # YOUR CODE HERE

        # Evaluate loss periodically
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
            elapsed = time.time() - start_time
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"step {iter:5d} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | "
                f"lr {current_lr:.2e} | "
                f"elapsed {elapsed:.0f}s"
            )

        # TODO 4b: Standard training step
        # 1. Get a batch: xb, yb = get_batch("train")
        # 2. Forward pass: logits, loss = model(xb, yb)
        # 3. Backward pass: optimizer.zero_grad(set_to_none=True), loss.backward()
        # 4. Update weights: optimizer.step()
        # YOUR CODE HERE

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.0f}s ({total_time/60:.1f} min)")

    return model


# ==========================================================================
# TODO 5: Generate text and evaluate quality
# ==========================================================================

def generate_text(model, num_chars=1000):
    """
    Generate text from the trained model.

    TODO 5: Generate a passage of num_chars characters.
    1. Set model to eval mode: model.eval()
    2. Create a starting context of zeros: torch.zeros((1, 1), dtype=torch.long, device=device)
    3. Generate: model.generate(context, max_new_tokens=num_chars)
    4. Decode: decode(generated[0].tolist())
    5. Set model back to train mode: model.train()
    """
    # YOUR CODE HERE
    pass


# ==========================================================================
# Main
# ==========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 4: Scaling Up")
    print("=" * 60)
    print(f"Vocab size: {vocab_size}")
    print(f"Training data: {len(train_data):,} characters")
    print(f"Validation data: {len(val_data):,} characters")
    print()

    # Train the model
    model = train()

    # Generate text
    print("\n" + "=" * 60)
    print("Generated Shakespeare (1000 characters):")
    print("=" * 60)
    generated = generate_text(model, num_chars=1000)
    if generated:
        print(generated)
    else:
        print("(Complete TODO 5 to generate text)")
    print("=" * 60)
