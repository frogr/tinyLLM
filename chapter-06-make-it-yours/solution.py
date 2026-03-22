"""
Chapter 6 Solution — Make It Yours (Complete BPE Implementation)
=================================================================

This is the complete working version with a BPE tokenizer implemented from
scratch. It can train on any text file passed as an argument.

Usage:
    python solution.py                         # Shakespeare, character-level
    python solution.py data/input.txt          # Shakespeare, character-level
    python solution.py data/input.txt --bpe    # Shakespeare, BPE
    python solution.py data/input.txt --bpe --merges 1000
    python solution.py data/input.txt --compare
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import sys
import time
import argparse

# ============================================================================
# Device Setup
# ============================================================================

device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# ============================================================================
# Hyperparameters
# ============================================================================

batch_size = 64
block_size = 256
max_iters = 3000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.2

# ============================================================================
# Data Loading
# ============================================================================

def load_text(filepath: str) -> str:
    """Load text data from a file."""
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        print("Did you run 'python data/download.py' first?")
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Loaded {len(text):,} characters from {filepath}")
    return text


# ============================================================================
# Character-Level Tokenizer
# ============================================================================

class CharTokenizer:
    """Character-level tokenizer — each character is a token."""

    def __init__(self, text: str):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        print(f"CharTokenizer: vocab_size = {self.vocab_size}")

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


# ============================================================================
# BPE Tokenizer — Complete Implementation
# ============================================================================

class BPETokenizer:
    """Byte Pair Encoding tokenizer, implemented from scratch.

    This tokenizer learns subword units by repeatedly merging the most
    frequent adjacent pair of tokens. The result is a vocabulary that
    balances between character-level (small vocab, many tokens) and
    word-level (huge vocab, few tokens) tokenization.
    """

    def __init__(self, text: str, num_merges: int = 256):
        """Train BPE on the given text.

        Args:
            text: The training text.
            num_merges: How many merge operations to learn.
        """
        self.num_merges = num_merges
        self.merges = {}   # (pair) -> new_token_id, ordered by training order
        self.vocab = {}    # token_id -> string for that token

        # Initialize vocabulary with individual characters
        chars = sorted(list(set(text)))
        self.base_vocab_size = len(chars)
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(chars)}

        for i, ch in enumerate(chars):
            self.vocab[i] = ch

        # Convert text to initial token IDs (one per character)
        tokens = [self.char_to_id[ch] for ch in text]

        print(f"BPE Training: {len(tokens):,} initial tokens, "
              f"base vocab = {self.base_vocab_size}")
        print(f"Learning {num_merges} merges...")

        # The merge loop: the core of BPE
        start_time = time.time()
        for i in range(num_merges):
            # Step 1: Count all adjacent pairs
            counts = self._count_pairs(tokens)

            if not counts:
                print(f"  No more pairs to merge after {i} merges.")
                break

            # Step 2: Find the most frequent pair
            best_pair = max(counts, key=counts.get)
            best_count = counts[best_pair]

            # If the best pair only appears once, further merging isn't useful
            if best_count < 2:
                print(f"  Best pair only appears {best_count} time(s). "
                      f"Stopping after {i} merges.")
                break

            # Step 3: Create a new token ID
            new_id = self.base_vocab_size + i

            # Step 4: Replace all occurrences of the pair with the new token
            tokens = self._merge(tokens, best_pair, new_id)

            # Step 5: Record the merge
            self.merges[best_pair] = new_id
            # The new token's string is the concatenation of its parts
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # Progress reporting
            if (i + 1) % 50 == 0 or i < 5:
                elapsed = time.time() - start_time
                merged_str = self.vocab[new_id]
                print(
                    f"  merge {i+1:4d}/{num_merges}: "
                    f"({self.vocab[best_pair[0]]!r}, {self.vocab[best_pair[1]]!r}) "
                    f"-> {merged_str!r} (count={best_count:,}, "
                    f"tokens={len(tokens):,}, time={elapsed:.1f}s)"
                )

        self.vocab_size = len(self.vocab)
        elapsed = time.time() - start_time
        compression = len(text) / len(tokens) if tokens else 0
        print(f"BPE Training complete in {elapsed:.1f}s")
        print(f"  Vocab size: {self.vocab_size}")
        print(f"  Compression ratio: {compression:.2f}x "
              f"({len(text):,} chars -> {len(tokens):,} tokens)")

    @staticmethod
    def _count_pairs(tokens: list[int]) -> dict[tuple[int, int], int]:
        """Count all adjacent pairs of tokens."""
        counts = {}
        for a, b in zip(tokens, tokens[1:]):
            pair = (a, b)
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def _merge(tokens: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """Replace all occurrences of `pair` in `tokens` with `new_id`."""
        new_tokens = []
        i = 0
        while i < len(tokens):
            # If we see the pair and we're not at the last token, merge
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(new_id)
                i += 2  # Skip both tokens in the pair
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def encode(self, text: str) -> list[int]:
        """Encode text into BPE token IDs.

        Apply merges in training order — this is critical because later
        merges may depend on tokens created by earlier merges.
        """
        # Start with character-level IDs
        tokens = [self.char_to_id.get(ch, 0) for ch in text]

        # Apply each merge rule in the order it was learned
        for pair, new_id in self.merges.items():
            tokens = self._merge(tokens, pair, new_id)

        return tokens

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        return "".join(self.vocab.get(i, "?") for i in ids)


# ============================================================================
# Full GPT Model
# ============================================================================

class Head(nn.Module):
    """Single head of self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

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
    """Position-wise feed-forward network."""

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
    """Transformer block."""

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


class GPT(nn.Module):
    """GPT language model."""

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
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

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate tokens with temperature and optional top-k sampling."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ============================================================================
# Training
# ============================================================================

def get_batch(split, split_data):
    """Get a random batch of training data."""
    data_split = split_data[0] if split == "train" else split_data[1]
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i : i + block_size] for i in ix])
    y = torch.stack([data_split[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, split_data):
    """Estimate loss on train and val splits."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, split_data)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train_model(text: str, tokenizer, label: str = "", num_iters: int = None):
    """Train the GPT model using the given tokenizer.

    Args:
        text: Raw text data.
        tokenizer: An object with encode(), decode(), and vocab_size attributes.
        label: A label for print statements.
        num_iters: Override max_iters if set.

    Returns:
        Tuple of (model, tokenizer, final_val_loss).
    """
    iters = num_iters or max_iters

    print(f"\n{'='*60}")
    print(f"Training with {label} tokenizer (vocab_size={tokenizer.vocab_size})")
    print(f"{'='*60}")

    # Encode
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"Encoded text length: {len(data):,} tokens")
    print(f"Compression: {len(text)/len(data):.2f} chars per token")

    # Train/val split
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    split_data = (train_data, val_data)

    if len(val_data) < block_size + 1:
        print("Warning: validation set is very small. Results may be noisy.")

    print(f"Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")

    # Build model
    model = GPT(tokenizer.vocab_size).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Train
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    final_val_loss = float("inf")

    start_time = time.time()
    for iter_num in range(iters):
        if iter_num % eval_interval == 0 or iter_num == iters - 1:
            losses = estimate_loss(model, split_data)
            elapsed = time.time() - start_time
            final_val_loss = losses["val"].item()
            print(
                f"  [{label}] step {iter_num:5d}/{iters} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | "
                f"time {elapsed:.1f}s"
            )

        xb, yb = get_batch("train", split_data)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Generate samples at different temperatures
    print(f"\n--- Sample outputs ({label}) ---")
    for temp in [0.5, 0.8, 1.0]:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated = model.generate(
            context, max_new_tokens=300, temperature=temp, top_k=40
        )
        text_out = tokenizer.decode(generated[0].tolist())
        print(f"\n[temperature={temp}]:")
        print(text_out[:300])
    print("--- End samples ---\n")

    return model, tokenizer, final_val_loss


def compare_tokenizers(text: str, num_bpe_merges: int = 500, num_iters: int = 2000):
    """Train with both char-level and BPE tokenizers, then compare."""

    print("\n" + "=" * 60)
    print("COMPARISON: Character-level vs BPE Tokenization")
    print(f"  BPE merges: {num_bpe_merges}")
    print(f"  Training iterations: {num_iters}")
    print("=" * 60)

    # Show tokenization comparison on a sample
    sample = text[:200]
    print(f"\nSample text ({len(sample)} chars):")
    print(f"  '{sample[:80]}...'")

    # Character tokenizer
    char_tok = CharTokenizer(text)
    char_encoded = char_tok.encode(sample)
    print(f"\nChar tokenizer:")
    print(f"  Vocab size: {char_tok.vocab_size}")
    print(f"  Sample -> {len(char_encoded)} tokens")
    print(f"  Ratio: {len(char_encoded)/len(sample):.2f} tokens per char")

    # BPE tokenizer
    bpe_tok = BPETokenizer(text, num_merges=num_bpe_merges)
    bpe_encoded = bpe_tok.encode(sample)
    print(f"\nBPE tokenizer:")
    print(f"  Vocab size: {bpe_tok.vocab_size}")
    print(f"  Sample -> {len(bpe_encoded)} tokens")
    print(f"  Ratio: {len(bpe_encoded)/len(sample):.2f} tokens per char")
    print(f"  Compression vs char: {len(char_encoded)/len(bpe_encoded):.2f}x")

    # Show what some BPE tokens look like
    print("\nTop 20 BPE tokens (by merge order, last = most complex):")
    merge_items = list(bpe_tok.merges.items())
    for pair, new_id in merge_items[-20:]:
        token_str = bpe_tok.vocab[new_id]
        print(f"  {new_id}: {token_str!r}")

    # Roundtrip test
    roundtrip = bpe_tok.decode(bpe_tok.encode(sample))
    assert roundtrip == sample, "BPE encode/decode roundtrip failed!"
    print("\nRoundtrip test passed (encode then decode = original text)")

    # Train both
    print("\n" + "-" * 40)
    print("Training character-level model...")
    _, _, char_loss = train_model(text, char_tok, label="char", num_iters=num_iters)

    print("\n" + "-" * 40)
    print("Training BPE model...")
    _, _, bpe_loss = train_model(text, bpe_tok, label="BPE", num_iters=num_iters)

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  {'Metric':<30} {'Char':>10} {'BPE':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10}")
    print(f"  {'Vocab size':<30} {char_tok.vocab_size:>10} {bpe_tok.vocab_size:>10}")
    print(f"  {'Tokens for sample':<30} {len(char_encoded):>10} {len(bpe_encoded):>10}")
    print(f"  {'Chars per token':<30} {len(sample)/len(char_encoded):>10.2f} "
          f"{len(sample)/len(bpe_encoded):>10.2f}")
    print(f"  {'Final val loss':<30} {char_loss:>10.4f} {bpe_loss:>10.4f}")
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Chapter 6: Train GPT with character-level or BPE tokenization"
    )
    parser.add_argument(
        "data_path",
        nargs="?",
        default=None,
        help="Path to text file (default: data/input.txt)",
    )
    parser.add_argument(
        "--bpe",
        action="store_true",
        help="Use BPE tokenization instead of character-level",
    )
    parser.add_argument(
        "--merges",
        type=int,
        default=500,
        help="Number of BPE merges (default: 500)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Train with both tokenizers and compare",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=None,
        help="Override number of training iterations",
    )

    args = parser.parse_args()

    # Determine data file
    if args.data_path:
        data_path = args.data_path
    else:
        data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "data", "input.txt"
        )

    text = load_text(data_path)

    # Show basic stats
    print(f"\nDataset statistics:")
    print(f"  Total characters: {len(text):,}")
    print(f"  Unique characters: {len(set(text))}")
    print(f"  Total words: {len(text.split()):,}")
    print(f"  First 100 chars: {repr(text[:100])}")

    if args.compare:
        compare_tokenizers(
            text,
            num_bpe_merges=args.merges,
            num_iters=args.iters or 2000,
        )
    elif args.bpe:
        bpe_tok = BPETokenizer(text, num_merges=args.merges)
        train_model(text, bpe_tok, label="BPE", num_iters=args.iters)
    else:
        char_tok = CharTokenizer(text)
        train_model(text, char_tok, label="char", num_iters=args.iters)


if __name__ == "__main__":
    main()
