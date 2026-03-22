"""
Chapter 6 — Sandbox

Your playground for experimentation. The full GPT model and both tokenizers
are set up below — just scroll to the bottom and start hacking.

Experiment ideas (pick any that interest you):

  1. CORPUS EXPERIMENTS
     - Train on Python source code: does it learn indentation? def/class?
     - Train on song lyrics: does it pick up verse/chorus structure?
     - Concatenate two very different texts (Shakespeare + Python code).
       What comes out?

  2. BPE EXPERIMENTS
     - Try different numbers of merges (50, 100, 500, 1000).
       How does vocab size affect output quality?
     - Print out the BPE vocabulary — what tokens did it learn?
     - Compare compression ratios on different types of text.
     - Time encoding/decoding speed: BPE vs character-level.

  3. ARCHITECTURE EXPERIMENTS
     - Implement sinusoidal positional encodings (replace learned embeddings).
       Does it change output quality for our model size?
     - Try weight tying: share the token_embedding_table weights with lm_head.
       This is what GPT-2 does. Does it help?
     - Swap ReLU for GELU (what GPT-2 uses) or SiLU/SwiGLU (what LLaMA uses).
     - Add a learning rate warmup schedule.

  4. TRAINING EXPERIMENTS
     - Track train vs val loss over time. Plot the curves.
     - Try gradient accumulation to simulate larger batch sizes.
     - Implement early stopping (stop when val loss starts increasing).
     - Save and load model checkpoints so you don't have to retrain.

  5. GENERATION EXPERIMENTS
     - Implement beam search (keep top-k candidates at each step).
     - Implement repetition penalty (penalize tokens that appeared recently).
     - Try prompting: start generation with a specific string instead of zeros.
     - Generate long passages (5000+ tokens) and see where coherence breaks down.

Run with: python sandbox.py
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import time
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
# Hyperparameters — tweak these freely
# ============================================================================
batch_size = 64
block_size = 256
max_iters = 3000       # Lower for quick experiments
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

# ============================================================================
# Dataset — change the path to experiment with different corpora
# ============================================================================
data_path = '../data/input.txt'

def prepare_text(filepath, max_chars=None):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text)
    if max_chars:
        text = text[:max_chars]
    return text

text = prepare_text(data_path)
print(f"Loaded {len(text):,} characters\n")


# ============================================================================
# Tokenizers — both available, pick one
# ============================================================================
class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        return ''.join([self.itos[i] for i in ids])


class SimpleBPE:
    def __init__(self, text, num_merges=256):
        self.chars = sorted(list(set(text)))
        self.base_vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        token_ids = [self.stoi[c] for c in text]
        self.merges = {}

        print(f"BPE: {self.base_vocab_size} base chars, {num_merges} merges")
        for i in range(num_merges):
            pair_counts = Counter()
            for j in range(len(token_ids) - 1):
                pair_counts[(token_ids[j], token_ids[j + 1])] += 1
            if not pair_counts:
                break
            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < 2:
                break
            new_id = self.base_vocab_size + i
            self.merges[best_pair] = new_id
            self.itos[new_id] = self.itos[best_pair[0]] + self.itos[best_pair[1]]
            self.stoi[self.itos[new_id]] = new_id
            new_ids = []
            j = 0
            while j < len(token_ids):
                if (j < len(token_ids) - 1
                        and token_ids[j] == best_pair[0]
                        and token_ids[j + 1] == best_pair[1]):
                    new_ids.append(new_id)
                    j += 2
                else:
                    new_ids.append(token_ids[j])
                    j += 1
            token_ids = new_ids
            if (i + 1) % 100 == 0:
                print(f"  ... {i+1} merges done ({len(token_ids):,} tokens)")

        self.vocab_size = self.base_vocab_size + len(self.merges)
        print(f"BPE done: vocab={self.vocab_size}, "
              f"compression={len(text)/len(token_ids):.2f}x\n")

    def encode(self, text):
        ids = [self.stoi[c] for c in text if c in self.stoi]
        for pair, new_id in self.merges.items():
            new_ids = []
            j = 0
            while j < len(ids):
                if j < len(ids) - 1 and ids[j] == pair[0] and ids[j + 1] == pair[1]:
                    new_ids.append(new_id)
                    j += 2
                else:
                    new_ids.append(ids[j])
                    j += 1
            ids = new_ids
        return ids

    def decode(self, ids):
        return ''.join([self.itos[i] for i in ids])


# Pick your tokenizer:
USE_BPE = False
tokenizer = SimpleBPE(text, num_merges=256) if USE_BPE else CharTokenizer(text)
vocab_size = tokenizer.vocab_size
print(f"Vocab size: {vocab_size}")

# ============================================================================
# Data
# ============================================================================
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print(f"Train: {len(train_data):,} | Val: {len(val_data):,} tokens\n")

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
# Model — complete GPT from Chapter 4
# ============================================================================
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
        k, q = self.key(x), self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
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
        x = self.ln_f(self.blocks(tok_emb + pos_emb))
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        return logits, F.cross_entropy(logits.view(B*T, C), targets.view(B*T))

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
# Helper: Train and Generate (reusable for experiments)
# ============================================================================
def train_and_generate(model, num_iters=max_iters, sample_len=500, temperature=0.8):
    """Train the model and generate a sample. Returns the trained model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"Training for {num_iters} steps...")
    start = time.time()
    for iter in range(num_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss(model)
            elapsed = time.time() - start
            print(f"  step {iter:5d}: train {losses['train']:.4f}, "
                  f"val {losses['val']:.4f} ({elapsed:.1f}s)")
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    losses = estimate_loss(model)
    elapsed = time.time() - start
    print(f"  Final:      train {losses['train']:.4f}, val {losses['val']:.4f} ({elapsed:.1f}s)\n")

    # Generate
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=sample_len, temperature=temperature)
    print(tokenizer.decode(generated[0].tolist()))
    model.train()
    return model


# ============================================================================
# Helper: Prompt-based generation
# ============================================================================
def generate_from_prompt(model, prompt_text, max_new_tokens=200, temperature=0.8):
    """Generate text starting from a specific prompt."""
    model.eval()
    prompt_ids = tokenizer.encode(prompt_text)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)
    result = tokenizer.decode(generated[0].tolist())
    model.train()
    return result


# ============================================================================
# Helper: Save and load checkpoints
# ============================================================================
def save_checkpoint(model, optimizer, path='checkpoint.pt'):
    """Save model and optimizer state."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path='checkpoint.pt'):
    """Load model and optimizer state."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Checkpoint loaded from {path}")
    return model, optimizer


# ============================================================================
# YOUR EXPERIMENTS BELOW
# ============================================================================
# Uncomment any section to try it, or write your own.

# --- Quick training run ---
print("=" * 60)
print("Training the model...")
print("=" * 60)
model = GPT().to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
model = train_and_generate(model, num_iters=max_iters)


# --- Prompt-based generation ---
# Uncomment after training:
# print("\n--- Prompted generation ---")
# print(generate_from_prompt(model, "ROMEO:\n"))
# print(generate_from_prompt(model, "To be or not"))


# --- Compare tokenizers ---
# print("\n--- Tokenizer comparison ---")
# sample = text[:500]
# char_tok = CharTokenizer(text)
# bpe_tok = SimpleBPE(text, num_merges=256)
# print(f"Character tokens: {len(char_tok.encode(sample))}")
# print(f"BPE tokens:       {len(bpe_tok.encode(sample))}")
# print(f"Compression:      {len(char_tok.encode(sample)) / len(bpe_tok.encode(sample)):.2f}x")


# --- Sinusoidal positional encoding experiment ---
# def get_sinusoidal_encoding(block_size, n_embd):
#     """Create fixed sinusoidal positional encodings (from the original Transformer paper)."""
#     pe = torch.zeros(block_size, n_embd)
#     position = torch.arange(0, block_size).unsqueeze(1).float()
#     div_term = torch.exp(torch.arange(0, n_embd, 2).float() * -(math.log(10000.0) / n_embd))
#     pe[:, 0::2] = torch.sin(position * div_term)
#     pe[:, 1::2] = torch.cos(position * div_term)
#     return pe
#
# # To use sinusoidal encodings, modify the GPT class:
# # Replace self.position_embedding_table = nn.Embedding(block_size, n_embd)
# # With:
# #   self.register_buffer('position_embedding', get_sinusoidal_encoding(block_size, n_embd))
# # And in forward(), replace:
# #   pos_emb = self.position_embedding_table(torch.arange(T, device=device))
# # With:
# #   pos_emb = self.position_embedding[:T, :]


print("\nDone! Edit this file to try more experiments.")
