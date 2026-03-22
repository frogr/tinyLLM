"""
Chapter 5 -- Text Generation Strategies (Solution)

Complete working implementation of all sampling strategies:
temperature scaling, top-k, top-p (nucleus), and combinations.

Try to work through exercises.py first before looking at this!

Run with: python solution.py
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# --- Device Setup ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# --- Hyperparameters ---
batch_size = 64
block_size = 256
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
learning_rate = 3e-4
max_iters = 3000  # Reduce for faster testing, increase for better quality
eval_interval = 500
eval_iters = 200

torch.manual_seed(1337)

# --- Data Loading ---
try:
    with open('../data/input.txt', 'r') as f:
        text = f.read()
except FileNotFoundError:
    print("Dataset not found! Run 'python ../data/download.py' first.")
    print("Using placeholder text...\n")
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

print(f"Dataset: {len(text):,} chars, vocab size: {vocab_size}")


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


# --- Model Definition (same as Chapter 4) ---

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
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
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


class GPTLanguageModel(nn.Module):
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

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=0, top_p=0.0):
        """
        Generate new tokens autoregressively.

        Args:
            idx:            (B, T) tensor of starting token indices
            max_new_tokens: how many new tokens to generate
            temperature:    >0. Higher = more random. 1.0 = default.
            top_k:          if >0, only sample from the top-k most likely tokens
            top_p:          if >0, use nucleus sampling (keep tokens until cumsum > p)
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx[:, -block_size:]
            # Forward pass
            logits, _ = self(idx_cond)
            # Focus on last time step
            logits = logits[:, -1, :]  # (B, C)

            # --- TODO 1: Temperature Scaling ---
            # Divide logits by temperature before softmax.
            # temperature < 1 = sharper (more conservative)
            # temperature > 1 = flatter (more creative)
            if temperature != 1.0:
                logits = logits / temperature

            # --- TODO 2: Top-k Sampling ---
            # Keep only the top-k highest logits, set the rest to -inf.
            if top_k > 0:
                # Clamp top_k to vocab size (can't keep more tokens than exist)
                k = min(top_k, logits.size(-1))
                top_k_values, _ = torch.topk(logits, k)
                # The k-th value (last in the sorted top-k) is our threshold
                threshold = top_k_values[:, -1].unsqueeze(-1)
                logits[logits < threshold] = float('-inf')

            # --- TODO 3: Top-p (Nucleus) Sampling ---
            # Keep the smallest set of tokens whose cumulative probability > top_p.
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # Mask tokens where cumulative prob (BEFORE this token) >= top_p
                # Subtracting sorted_probs ensures the token that crosses the
                # threshold is included, not excluded.
                sorted_mask = (cumulative_probs - sorted_probs) >= top_p
                sorted_logits[sorted_mask] = float('-inf')

                # Unsort back to original vocabulary order
                original_indices = sorted_indices.argsort(dim=-1)
                logits = sorted_logits.gather(1, original_indices)

            # Convert to probabilities and sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# --- Create and Train ---
print(f"\nCreating model...")
model = GPTLanguageModel().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model has {n_params:,} parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"\nTraining for {max_iters} steps...\n")
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

# Switch to eval mode
model.eval()


# --- Helper ---
def generate_text(prompt_text, max_tokens=200, temperature=1.0, top_k=0, top_p=0.0):
    """Generate text from a prompt string."""
    if prompt_text:
        context = torch.tensor([encode(prompt_text)], dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)

    with torch.no_grad():
        output = model.generate(context, max_tokens, temperature=temperature,
                                top_k=top_k, top_p=top_p)
    return decode(output[0].tolist())


# ============================================================================
# TODO 4: Combined Strategies
# ============================================================================
print("\n" + "=" * 70)
print("COMBINED: temperature=0.8, top_p=0.95 (good default)")
print("=" * 70)
print(generate_text("KING:", temperature=0.8, top_p=0.95))

print("\n" + "=" * 70)
print("COMBINED: temperature=0.5, top_k=40 (conservative)")
print("=" * 70)
print(generate_text("KING:", temperature=0.5, top_k=40))

print("\n" + "=" * 70)
print("COMBINED: temperature=1.2, top_p=0.9 (creative)")
print("=" * 70)
print(generate_text("KING:", temperature=1.2, top_p=0.9))


# ============================================================================
# TODO 5: Compare Outputs Side by Side
# ============================================================================
prompt = "ROMEO:"
settings = [
    ("Greedy (temp=0.01)",              dict(temperature=0.01)),
    ("Default (temp=1.0)",              dict(temperature=1.0)),
    ("Conservative (temp=0.5)",         dict(temperature=0.5)),
    ("Creative (temp=1.5)",             dict(temperature=1.5)),
    ("Top-k=10",                        dict(top_k=10)),
    ("Top-p=0.9",                       dict(top_p=0.9)),
    ("Balanced (temp=0.8, top_p=0.95)", dict(temperature=0.8, top_p=0.95)),
]

print("\n")
print("#" * 70)
print("#  SIDE-BY-SIDE COMPARISON")
print(f"#  Prompt: \"{prompt}\"")
print("#" * 70)

for label, kwargs in settings:
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    print(generate_text(prompt, max_tokens=200, **kwargs))

print("\n" + "#" * 70)
print("# OBSERVATIONS")
print("#" * 70)
print("""
Things to notice:
  - Greedy (temp=0.01) is deterministic and likely repetitive
  - Default (temp=1.0) has variety but may include odd choices
  - Conservative (temp=0.5) reads more coherently
  - Creative (temp=1.5) has interesting words but may lose coherence
  - Top-k=10 filters to just 10 options at each step
  - Top-p=0.9 adapts the filter to the distribution
  - Balanced combines temperature shaping with distribution filtering

In production LLMs:
  - Factual Q&A:       temperature=0.3, top_p=0.9
  - Creative writing:  temperature=1.0, top_p=0.95
  - Code generation:   temperature=0.2, top_p=0.95
  - Brainstorming:     temperature=1.2, top_k=100
""")
