"""
Chapter 5 -- Sandbox

Your playground for experimenting with generation strategies.
This file trains a smaller, faster model so you can iterate quickly.

Ideas to try:
  - What happens at extreme temperatures (0.01 vs 3.0)?
  - How does top-k=2 compare to top-k=100?
  - Can you find the "sweet spot" for your favorite output quality?
  - Generate the same prompt 5 times with temperature=1.0 -- how much does it vary?
  - Try generating from different prompts: "ROMEO:", "JULIET:", "First Citizen:"
  - What happens if you set top_p=0.1? (Very aggressive filtering)
  - Plot the probability distribution at a single step with different temperatures
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

# --- Smaller model for fast iteration ---
batch_size = 32
block_size = 128
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.2
learning_rate = 3e-4
max_iters = 2000  # Fast training -- bump to 5000 for better quality
eval_interval = 500
eval_iters = 100

torch.manual_seed(1337)

# --- Data Loading ---
try:
    with open('../data/input.txt', 'r') as f:
        text = f.read()
except FileNotFoundError:
    print("Run 'python ../data/download.py' first.")
    exit(1)

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


# --- Model (compact version) ---

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
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


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
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            # Temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k
            if top_k > 0:
                k = min(top_k, logits.size(-1))
                top_k_values, _ = torch.topk(logits, k)
                threshold = top_k_values[:, -1].unsqueeze(-1)
                logits[logits < threshold] = float('-inf')

            # Top-p
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_mask = (cumulative_probs - sorted_probs) >= top_p
                sorted_logits[sorted_mask] = float('-inf')
                original_indices = sorted_indices.argsort(dim=-1)
                logits = sorted_logits.gather(1, original_indices)

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# --- Train ---
print(f"Training smaller model ({max_iters} steps)...")
model = GPTLanguageModel().to(device)
print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {iter:5d}: train {losses['train']:.4f}, val {losses['val']:.4f}")
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

losses = estimate_loss(model)
print(f"Final:      train {losses['train']:.4f}, val {losses['val']:.4f}\n")
model.eval()


def gen(prompt="", max_tokens=200, **kwargs):
    if prompt:
        ctx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    else:
        ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(ctx, max_tokens, **kwargs)
    return decode(out[0].tolist())


# ============================================================================
# Experiment 1: Temperature sweep
# ============================================================================
print("=" * 70)
print("EXPERIMENT 1: Temperature Sweep")
print("=" * 70)
for temp in [0.01, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
    print(f"\n--- temperature={temp} ---")
    print(gen("KING:", max_tokens=150, temperature=temp))

# ============================================================================
# Experiment 2: Top-k sweep
# ============================================================================
print("\n" + "=" * 70)
print("EXPERIMENT 2: Top-k Sweep")
print("=" * 70)
for k in [1, 5, 10, 40, 100]:
    print(f"\n--- top_k={k} ---")
    print(gen("KING:", max_tokens=150, top_k=k))

# ============================================================================
# Experiment 3: Top-p sweep
# ============================================================================
print("\n" + "=" * 70)
print("EXPERIMENT 3: Top-p Sweep")
print("=" * 70)
for p in [0.1, 0.5, 0.8, 0.9, 0.95, 0.99]:
    print(f"\n--- top_p={p} ---")
    print(gen("KING:", max_tokens=150, top_p=p))

# ============================================================================
# Experiment 4: Reproducibility check
# ============================================================================
print("\n" + "=" * 70)
print("EXPERIMENT 4: Same prompt, same settings, 3 runs (temp=1.0)")
print("Notice how each run is different -- that's sampling randomness.")
print("=" * 70)
for i in range(3):
    print(f"\n--- Run {i+1} ---")
    print(gen("ROMEO:", max_tokens=100, temperature=1.0))

# ============================================================================
# Experiment 5: Visualize the probability distribution
# ============================================================================
print("\n" + "=" * 70)
print("EXPERIMENT 5: Probability distribution at a single step")
print("=" * 70)

prompt = "ROMEO:"
ctx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
with torch.no_grad():
    logits, _ = model(ctx)
    logits = logits[:, -1, :]  # Last position

for temp in [0.3, 1.0, 2.0]:
    scaled = logits / temp
    probs = F.softmax(scaled, dim=-1).squeeze()
    top_probs, top_idx = torch.topk(probs, 10)

    print(f"\ntemperature={temp} -- Top 10 next-token probabilities after \"{prompt}\":")
    for prob, idx in zip(top_probs, top_idx):
        char = itos[idx.item()]
        char_display = repr(char) if char in ['\n', ' '] else f" {char}"
        bar = "#" * int(prob.item() * 50)
        print(f"  {char_display:5s}  {prob.item():.3f}  {bar}")

print("\n\nDone! Edit this file to try your own experiments.")
