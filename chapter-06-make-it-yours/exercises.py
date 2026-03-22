"""
Chapter 6 Exercises — Make It Yours
====================================

This chapter is open-ended. You'll:
  1. Implement a BPE (Byte Pair Encoding) tokenizer from scratch
  2. Modify the model to use BPE tokens instead of characters
  3. Train on a custom dataset
  4. Compare character-level vs BPE tokenization

The full GPT model from chapters 4-5 is included so this file is self-contained.
By default it trains on Shakespeare, but you can swap to any .txt file.

Usage:
    python exercises.py                    # Train on Shakespeare (character-level)
    python exercises.py data/myfile.txt    # Train on custom data (character-level)
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import sys
import time

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
    """Load text data from a file.

    This is the simplest possible data loading: read the entire file into a
    string. For our small models, this works fine. Production models use
    streaming data loaders that don't load everything into memory.

    Args:
        filepath: Path to a UTF-8 text file.

    Returns:
        The entire file contents as a string.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        print("Did you run 'python data/download.py' first?")
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Loaded {len(text):,} characters from {filepath}")
    return text


# ============================================================================
# Character-Level Tokenizer (what we've used so far)
# ============================================================================

class CharTokenizer:
    """Character-level tokenizer — each character is a token.

    This is what we've been using in chapters 1-5. Simple but inefficient:
    the word "the" takes 3 tokens, and block_size=256 only sees ~40 words.
    """

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
# TODO 1: Implement a BPE Tokenizer from Scratch
# ============================================================================

class BPETokenizer:
    """Byte Pair Encoding tokenizer.

    BPE learns common subword patterns by repeatedly merging the most frequent
    adjacent pair of tokens. This is how real LLMs tokenize text.

    The algorithm:
        1. Start with individual characters as tokens
        2. Count all adjacent pairs in the token sequence
        3. Find the most frequent pair
        4. Merge that pair into a new single token
        5. Repeat steps 2-4 for `num_merges` iterations

    After training, you have a list of merge rules. To encode new text:
        1. Start with characters
        2. Apply merge rules in order

    Example:
        Training text: "aaabdaaabac"

        Initial tokens: ['a', 'a', 'a', 'b', 'd', 'a', 'a', 'a', 'b', 'a', 'c']

        Iteration 1: Most frequent pair is ('a', 'a') -> merge into 'aa'
          Tokens: ['aa', 'a', 'b', 'd', 'aa', 'a', 'b', 'a', 'c']

        Iteration 2: Most frequent pair is ('aa', 'a') -> merge into 'aaa'
          Tokens: ['aaa', 'b', 'd', 'aaa', 'b', 'a', 'c']

        Iteration 3: Most frequent pair is ('aaa', 'b') -> merge into 'aaab'
          Tokens: ['aaab', 'd', 'aaab', 'a', 'c']

        And so on...
    """

    def __init__(self, text: str, num_merges: int = 256):
        """Train BPE on the given text.

        Args:
            text: The training text.
            num_merges: How many merge operations to learn. More merges =
                        larger vocabulary = fewer tokens per text.
                        GPT-2 uses ~50,000 merges. We'll start small.
        """
        self.num_merges = num_merges
        self.merges = {}   # (pair) -> new_token_id
        self.vocab = {}    # token_id -> bytes/string for that token

        # Step 1: Initialize vocabulary with individual characters
        # We'll use integers as token IDs. Characters 0-255 map to themselves
        # (or we use the unique characters in the text).
        chars = sorted(list(set(text)))
        self.base_vocab_size = len(chars)
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(chars)}

        # Initialize vocab: each character gets its own ID
        for i, ch in enumerate(chars):
            self.vocab[i] = ch

        # Step 2: Convert text to initial token IDs (one per character)
        tokens = [self.char_to_id[ch] for ch in text]

        print(f"BPE Training: {len(tokens):,} initial tokens, "
              f"base vocab = {self.base_vocab_size}")
        print(f"Learning {num_merges} merges...")

        # Step 3: The merge loop
        # TODO: Implement the BPE training loop.
        #
        # For each merge iteration:
        #   a) Count all adjacent pairs in `tokens`
        #      Hint: iterate through tokens with zip(tokens, tokens[1:])
        #      and use a dictionary to count occurrences.
        #
        #   b) Find the most frequent pair
        #      Hint: use max() with a key function, or sort the counts.
        #
        #   c) Create a new token ID for this pair
        #      The new ID should be: self.base_vocab_size + merge_number
        #      (so the first merge gets ID = base_vocab_size + 0,
        #       the second gets base_vocab_size + 1, etc.)
        #
        #   d) Replace all occurrences of the pair in `tokens` with the new ID
        #      Hint: iterate through the list and build a new list. When you
        #      see the pair, append the new ID and skip the next token.
        #
        #   e) Record the merge: self.merges[(pair)] = new_id
        #      Record the vocab entry: self.vocab[new_id] = merged string
        #
        #   f) If no pairs are found (list too short), break early.
        #
        # This loop should print progress every 50 merges or so.
        #
        # ---- YOUR CODE HERE ----
        # for i in range(num_merges):
        #     ... count pairs ...
        #     ... find best pair ...
        #     ... merge ...
        #     ... record ...
        pass  # Remove this when you implement the loop
        # ---- END YOUR CODE ----

        self.vocab_size = len(self.vocab)
        print(f"BPE Training complete. Vocab size: {self.vocab_size}")

    @staticmethod
    def _count_pairs(tokens: list[int]) -> dict[tuple[int, int], int]:
        """Count all adjacent pairs of tokens.

        Args:
            tokens: List of token IDs.

        Returns:
            Dictionary mapping (token_a, token_b) -> count.

        Example:
            tokens = [1, 2, 3, 1, 2]
            returns: {(1,2): 2, (2,3): 1, (3,1): 1}
        """
        # TODO: Implement this helper.
        # Hint: one line with a loop, or use collections.Counter
        #
        # ---- YOUR CODE HERE ----
        counts = {}
        return counts
        # ---- END YOUR CODE ----

    @staticmethod
    def _merge(tokens: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """Replace all occurrences of `pair` in `tokens` with `new_id`.

        Args:
            tokens: List of token IDs.
            pair: The (token_a, token_b) pair to merge.
            new_id: The new token ID to replace the pair with.

        Returns:
            New list of token IDs with merges applied.

        Example:
            tokens = [1, 2, 3, 1, 2, 4]
            pair = (1, 2)
            new_id = 99
            returns: [99, 3, 99, 4]
        """
        # TODO: Implement this helper.
        # Walk through the token list. When you see pair[0] followed by pair[1],
        # append new_id and skip the next token. Otherwise, append the current token.
        #
        # ---- YOUR CODE HERE ----
        new_tokens = []
        return new_tokens
        # ---- END YOUR CODE ----

    def encode(self, text: str) -> list[int]:
        """Encode text into BPE token IDs.

        To encode new text with a trained BPE:
        1. Start with character-level token IDs
        2. Apply each merge rule in the order it was learned

        This is important: merges must be applied in training order because
        later merges may depend on earlier ones.

        Args:
            text: String to encode.

        Returns:
            List of token IDs.
        """
        # TODO: Implement encoding.
        #
        # Step 1: Convert text to character-level IDs
        # tokens = [self.char_to_id[ch] for ch in text]
        #
        # Step 2: Apply each merge rule in order
        # for pair, new_id in self.merges.items():
        #     tokens = self._merge(tokens, pair, new_id)
        #
        # ---- YOUR CODE HERE ----
        tokens = [self.char_to_id.get(ch, 0) for ch in text]
        return tokens
        # ---- END YOUR CODE ----

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text.

        Args:
            ids: List of token IDs.

        Returns:
            Decoded string.
        """
        # TODO: Implement decoding.
        # Use self.vocab to look up the string for each token ID.
        #
        # ---- YOUR CODE HERE ----
        return "".join(self.vocab.get(i, "?") for i in ids)
        # ---- END YOUR CODE ----


# ============================================================================
# Full GPT Model (from chapters 4-5, self-contained)
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
    """Transformer block: attention + feed-forward with residual connections."""

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
# Training Utilities
# ============================================================================

def get_batch(data, split_data):
    """Get a random batch of training data."""
    data_split = split_data[0] if data == "train" else split_data[1]
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


# ============================================================================
# TODO 2: Modify Training to Use BPE
# ============================================================================

def train_with_tokenizer(text: str, tokenizer, label: str = ""):
    """Train the GPT model using the given tokenizer.

    This function encodes the text, splits into train/val, builds a model,
    and trains it.

    Args:
        text: Raw text data.
        tokenizer: An object with encode(), decode(), and vocab_size attributes.
        label: A label for print statements (e.g., "char" or "BPE").
    """
    print(f"\n{'='*60}")
    print(f"Training with {label} tokenizer (vocab_size={tokenizer.vocab_size})")
    print(f"{'='*60}")

    # Encode the full text
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"Encoded text length: {len(data):,} tokens")

    # TODO: Think about this — with BPE, the same text produces fewer tokens.
    # That means each training example (block_size tokens) covers more text.
    # This is why BPE helps: more context in each window!

    # Train/val split (90/10)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    split_data = (train_data, val_data)

    print(f"Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")

    # Build model
    model = GPT(tokenizer.vocab_size).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Train
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    start_time = time.time()
    for iter_num in range(max_iters):
        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            losses = estimate_loss(model, split_data)
            elapsed = time.time() - start_time
            print(
                f"  [{label}] step {iter_num:5d} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | "
                f"time {elapsed:.1f}s"
            )

        xb, yb = get_batch("train", split_data)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Generate a sample
    print(f"\n--- Sample output ({label}) ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=500, temperature=0.8, top_k=40)
    print(tokenizer.decode(generated[0].tolist()))
    print("--- End sample ---\n")

    return model, tokenizer


# ============================================================================
# TODO 3: Train on a Custom Dataset
# ============================================================================

# To train on your own data:
#
# 1. Find or create a text file (UTF-8, at least 100KB recommended)
#
# 2. Some ideas for where to get data:
#    - Project Gutenberg (gutenberg.org): free books
#      Example: download "Pride and Prejudice" or "Moby Dick"
#
#    - Your own writing: export emails, blog posts, notes
#
#    - Code: concatenate source files from a project
#      find myproject -name "*.py" -exec cat {} \; > python_code.txt
#
#    - Recipes: scrape a recipe site or find a dataset online
#
# 3. Place the file in the data/ directory
#
# 4. Run: python exercises.py data/your_file.txt
#
# Tips:
#   - More data = better results (aim for 500KB+)
#   - Clean data = better results (remove HTML, headers, etc.)
#   - Consistent formatting helps the model learn structure
#   - The model will reproduce whatever patterns are in the data,
#     including typos, formatting quirks, etc.


# ============================================================================
# TODO 4: Compare Character-Level vs BPE
# ============================================================================

def compare_tokenizers(text: str, num_bpe_merges: int = 500):
    """Train with both char-level and BPE tokenizers and compare.

    TODO: Implement this function to:
    1. Create a CharTokenizer
    2. Create a BPETokenizer with num_bpe_merges merges
    3. Train a model with each (you may want to reduce max_iters for speed)
    4. Compare:
       - Final validation loss
       - Tokens per text (how much "compression" BPE gives)
       - Quality of generated samples
       - Training time

    Try different values of num_bpe_merges: 100, 500, 1000, 2000
    What's the sweet spot?
    """
    print("\n" + "=" * 60)
    print("COMPARISON: Character-level vs BPE Tokenization")
    print("=" * 60)

    # ---- YOUR CODE HERE ----

    # Step 1: Show tokenization comparison on a sample
    sample = text[:200]
    print(f"\nSample text ({len(sample)} chars):")
    print(f"  '{sample[:80]}...'")

    char_tok = CharTokenizer(text)
    char_encoded = char_tok.encode(sample)
    print(f"\nChar tokenizer: {len(char_encoded)} tokens for {len(sample)} chars")
    print(f"  Ratio: {len(char_encoded) / len(sample):.2f} tokens per char")

    # TODO: Create BPE tokenizer and compare
    # bpe_tok = BPETokenizer(text, num_merges=num_bpe_merges)
    # bpe_encoded = bpe_tok.encode(sample)
    # print(f"\nBPE tokenizer: {len(bpe_encoded)} tokens for {len(sample)} chars")
    # print(f"  Ratio: {len(bpe_encoded) / len(sample):.2f} tokens per char")
    # print(f"  Compression: {len(char_encoded) / len(bpe_encoded):.2f}x fewer tokens")

    # TODO: Train with both and compare
    # (You may want to reduce max_iters for faster comparison)

    # ---- END YOUR CODE ----


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Determine data file
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        # Default to Shakespeare
        data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "data", "input.txt"
        )

    text = load_text(data_path)

    # Show basic stats about the data
    print(f"\nDataset statistics:")
    print(f"  Total characters: {len(text):,}")
    print(f"  Unique characters: {len(set(text))}")
    print(f"  First 100 chars: {repr(text[:100])}")

    # ---- Choose what to run ----
    # Uncomment the section you want to try:

    # Option A: Train with character-level tokenizer (default, what we know)
    print("\n" + "=" * 60)
    print("Training with character-level tokenizer")
    print("=" * 60)
    char_tokenizer = CharTokenizer(text)
    train_with_tokenizer(text, char_tokenizer, label="char")

    # Option B: Train with BPE tokenizer (complete TODO 1 first!)
    # print("\n" + "=" * 60)
    # print("Training with BPE tokenizer")
    # print("=" * 60)
    # bpe_tokenizer = BPETokenizer(text, num_merges=500)
    # train_with_tokenizer(text, bpe_tokenizer, label="BPE")

    # Option C: Compare both tokenizers (complete TODOs 1 and 4)
    # compare_tokenizers(text, num_bpe_merges=500)
