# Chapter 2 — Self-Attention

## Context

In Chapter 1, we built a bigram model. It looks at exactly ONE character and predicts the
next one. The result? Shakespeare-flavored gibberish. The model learned that 'q' is often
followed by 'u' and that newlines come after certain patterns, but it can never learn that
"thou" is a word, because it only sees one character at a time.

The fundamental problem: **each character is processed in total isolation**. When the model
sees 't', it doesn't know whether the previous characters were "wha" (suggesting the next
is probably 'h') or "no" (suggesting a space). Every 't' produces the exact same prediction.

In this chapter, we fix that. We teach the model to **look at multiple previous characters**
before making a prediction. The mechanism that does this is called **self-attention**, and
it's the key innovation behind transformers (GPT, BERT, Claude, and everything else).

By the end of this chapter, your model will still produce mostly gibberish — but noticeably
better gibberish. We're adding one attention head with 32-dimensional embeddings. The real
power comes when we stack many heads and layers in Chapter 3.

## Concepts

### What Attention Solves

Imagine you're a character in the sequence trying to predict what comes next. In the bigram
model, you're blindfolded — you only know your own identity. With attention, you can look
back at all the previous characters and ask: "What's relevant to my prediction?"

The key insight: **not all previous characters are equally useful**. If you're trying to
predict what comes after "To be or not to b", the most useful character is the recent "b",
not the "T" from the start. Attention learns which characters to pay attention to.

### Position Embeddings — Where Am I?

In Chapter 1, we only had a **token embedding**: each character gets a learned vector.
But now we need the model to know WHERE each character sits in the sequence.

Why? Because "ab" and "ba" have the same characters but mean different things. Without
position information, the model can't tell them apart.

The fix is simple: we add a second embedding table that maps each POSITION (0, 1, 2, ...,
block_size-1) to a learned vector, then add it to the token embedding:

```python
self.token_embedding_table = nn.Embedding(vocab_size, n_embd)      # what am I?
self.position_embedding_table = nn.Embedding(block_size, n_embd)   # where am I?

# In the forward pass:
tok_emb = self.token_embedding_table(idx)         # shape: (B, T, C)
pos_emb = self.position_embedding_table(positions) # shape: (T, C) → broadcast to (B, T, C)
x = tok_emb + pos_emb                             # shape: (B, T, C)
```

The model now gets TWO signals: **what** this character is, and **where** it sits.

### Queries, Keys, Values — The Database Analogy

If you've ever written a SQL query, you already have the right mental model:

```sql
SELECT value FROM table WHERE key MATCHES query
```

Self-attention works the same way, except "matches" is fuzzy (a similarity score) rather
than an exact match, and you get a weighted blend of ALL values, not just one.

Each character in the sequence produces three things:
- **Query (Q):** "What am I looking for?" — what kind of information would be useful
- **Key (K):** "What do I contain?" — what I advertise to other characters
- **Value (V):** "What do I actually provide?" — the information to share if matched

These are computed by multiplying the input by three learned weight matrices:

```python
self.key   = nn.Linear(n_embd, head_size, bias=False)
self.query = nn.Linear(n_embd, head_size, bias=False)
self.value = nn.Linear(n_embd, head_size, bias=False)

k = self.key(x)    # shape: (B, T, head_size)
q = self.query(x)  # shape: (B, T, head_size)
v = self.value(x)  # shape: (B, T, head_size)
```

Think of `nn.Linear` as a matrix multiply: it transforms a vector from one size to another.
In code terms: `k = x @ W_k` where `W_k` is a learned `(n_embd, head_size)` matrix.

### The Dot Product — Why It Measures Similarity

Here's the core of attention: how does one character decide to "pay attention" to another?

It takes its query and computes the **dot product** with every other character's key:

```python
# q shape: (B, T, head_size)
# k shape: (B, T, head_size)
# k.transpose(-2, -1) shape: (B, head_size, T)
weights = q @ k.transpose(-2, -1)  # shape: (B, T, T)
```

The result is a `(T, T)` matrix. Entry `[i][j]` tells you: "how much should character `i`
pay attention to character `j`?"

**Why does the dot product measure similarity?** Think of two vectors. If they point in the
same direction (similar), their dot product is large and positive. If they're perpendicular
(unrelated), it's near zero. If they point in opposite directions, it's large and negative.

```
Vector similarity via dot product:
  [1, 0] . [1, 0] = 1    (identical direction — high match)
  [1, 0] . [0, 1] = 0    (perpendicular — no match)
  [1, 0] . [-1, 0] = -1  (opposite — anti-match)
```

So when a query and a key point in similar directions, the attention weight is high.
The model learns Q and K matrices that make this alignment happen for useful pairs.

We also scale by `1/sqrt(head_size)` to keep the values from getting too extreme:

```python
weights = q @ k.transpose(-2, -1) * (head_size ** -0.5)  # shape: (B, T, T)
```

Without scaling, the dot products can be huge, which makes softmax output near-0-or-1 values,
killing gradients and making learning impossible. This is called the **scaled dot-product
attention** from the original "Attention Is All You Need" paper.

### Softmax — Converting Scores to Probabilities (Revisited)

You met softmax in Chapter 1 for text generation: converting logits to probabilities.
Now it appears again to convert attention scores to attention weights:

```python
weights = F.softmax(weights, dim=-1)  # shape: (B, T, T)
```

Each row in the `(T, T)` matrix now sums to 1.0. Row `i` is a probability distribution:
"for character at position i, here's how much to attend to each other position."

```
Before softmax: [2.1,  0.5, -1.0, -inf, -inf]
After softmax:  [0.72, 0.15, 0.03, 0.00, 0.00]  — sums to ~1.0
```

Softmax does two things:
1. Makes all values positive (via exponentiation)
2. Normalizes so they sum to 1 (dividing by the total)

Formula: `softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)`

### Masked Attention — No Spoilers

Here's a critical constraint: when predicting the next character at position `t`, the model
should only look at characters at positions `0, 1, ..., t`. It should NOT look at future
characters — that would be cheating (like reading ahead in a book during a quiz).

We enforce this with a **mask**: before softmax, we set all "future" entries to negative
infinity. Softmax turns `-inf` into 0, so those positions get zero attention weight.

```python
# Create a lower-triangular mask
tril = torch.tril(torch.ones(T, T))
# tril for T=5 looks like:
# [[1, 0, 0, 0, 0],
#  [1, 1, 0, 0, 0],
#  [1, 1, 1, 0, 0],
#  [1, 1, 1, 1, 0],
#  [1, 1, 1, 1, 1]]

weights = weights.masked_fill(tril == 0, float('-inf'))
# Position 0 can only look at position 0
# Position 1 can look at positions 0 and 1
# Position 4 can look at all positions 0-4
weights = F.softmax(weights, dim=-1)
```

Without masking: the model has access to the answer while training (data leakage).
With masking: the model must predict each character using only what came before.

This is why GPT-style models are called "autoregressive" — each prediction depends only
on prior predictions.

### Putting It All Together — A Single Attention Head

After computing the attention weights, we use them to create a weighted combination of values:

```python
out = weights @ v  # shape: (B, T, head_size)
```

Each position gets a weighted average of all the value vectors it's allowed to see.
Position 0 gets only its own value. Position 4 gets a blend of values from positions 0-4,
with the blend weights determined by query-key similarity.

## The Math

Here's the complete attention formula, then the code that implements it:

```
Attention(Q, K, V) = softmax( (Q @ K^T) / sqrt(d_k) ) @ V

Where:
  Q = X @ W_Q     query projection    (B, T, d_k)
  K = X @ W_K     key projection      (B, T, d_k)
  V = X @ W_V     value projection    (B, T, d_k)
  d_k = head_size (dimension of keys)
```

In PyTorch:

```python
class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # register_buffer = "this tensor is part of the model but not a parameter"
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape                                        # (B, T, n_embd)
        k = self.key(x)                                           # (B, T, head_size)
        q = self.query(x)                                         # (B, T, head_size)

        # Compute attention scores
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)              # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # mask future
        wei = F.softmax(wei, dim=-1)                              # (B, T, T)

        # Weighted aggregation of values
        v = self.value(x)                                         # (B, T, head_size)
        out = wei @ v                                             # (B, T, head_size)
        return out
```

## Architecture Diagram

```
Input: "To be" as indices [20, 53, 1, 40, 43]

  idx (B, T)
    │
    ├──► Token Embedding Table (vocab_size, n_embd=32)
    │         │
    │         ▼
    │    tok_emb (B, T, 32)        "what each character IS"
    │         │
    │    + ◄──┘
    │    │
    ├──► Position Embedding Table (block_size, n_embd=32)
    │         │
    │         ▼
    │    pos_emb (T, 32)           "where each character SITS"
    │         │
    └────► + ◄┘
           │
           ▼
      x (B, T, 32)                combined embedding
           │
     ┌─────┼──────┐
     ▼     ▼      ▼
   Query  Key   Value
   W_Q    W_K    W_V              three nn.Linear(32, 32)
     │     │      │
     ▼     ▼      │
  Q (B,T,32) K (B,T,32)           ▼
     │     │               V (B,T,32)
     ▼     ▼                      │
  Q @ K^T / sqrt(32)              │
     │                            │
     ▼                            │
  (B, T, T) raw scores            │
     │                            │
     ▼                            │
  mask future (tril)              │
     │                            │
     ▼                            │
  softmax → (B, T, T) weights     │
     │                            │
     └──────── @ ─────────────────┘
               │
               ▼
         (B, T, 32) context-aware representation
               │
               ▼
        nn.Linear(32, vocab_size)
               │
               ▼
         logits (B, T, vocab_size=65)
               │
               ▼
         cross_entropy(logits, targets) → loss
```

## Exercises

Open `exercises.py` and work through the TODOs in order. Each TODO builds on the previous
one. The file is runnable at every stage — incomplete TODOs use placeholder values.

1. **TODO 1** — Token + position embeddings
2. **TODO 2** — Implement a single self-attention head (the `Head` class)
3. **TODO 3** — Build the full model class using the attention head
4. **TODO 4** — Training loop
5. **TODO 5** — Generate text and compare to bigram output

## Stretch Goals

- [ ] Visualize the attention weights as a heatmap (use matplotlib's `imshow`)
- [ ] Change `head_size` to 16 vs 64 — how does it affect loss and generation?
- [ ] What happens without position embeddings? Remove them and compare output
- [ ] What happens without the mask? (Hint: loss should be suspiciously low during training
      but generation will be terrible — why?)
- [ ] Print the attention weights for a specific input and interpret them — which characters
      attend to which?
- [ ] Compare parameter counts: bigram (4,225) vs this model. How many parameters does
      self-attention add?

## Further Reading

- [Karpathy "Let's build GPT" video](https://www.youtube.com/watch?v=kCc8FmEb1nY) — ~27:00 to ~1:00:00 covers self-attention
- [3Blue1Brown "But what is attention?"](https://www.youtube.com/watch?v=eMlx5fFNoYc) — Beautiful visual explanation
- [The original "Attention Is All You Need" paper](https://arxiv.org/abs/1706.03762) — Section 3.2 is the math above
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar's visual walkthrough

## Think About It

1. **Shape detective.** Before running the code, predict the shapes at each step for a
   single batch with `batch_size=4, block_size=8, n_embd=32, head_size=32`. What shape is
   `q @ k.transpose(-2, -1)`? What about after masking and softmax? Write your predictions,
   then add print statements to verify. Where did you get it wrong?

2. **Remove the mask.** Comment out the `masked_fill` line and train the model. What happens
   to the training loss? Now generate text — is it better or worse than the masked version?
   Why does the loss look good but generation falls apart? (Hint: think about what information
   the model has access to during training vs. generation.)

3. **The scaling factor.** The attention scores are divided by `sqrt(head_size)`. Try
   removing this scaling (just use `q @ k.transpose(-2, -1)` without the `* (C ** -0.5)`).
   What happens to the attention weights after softmax? Print them — are they "sharp" (close
   to one-hot) or "diffuse" (spread out)? Why does this matter for learning?

4. **Position embeddings carry more weight than you think.** What happens if you ONLY use
   position embeddings (remove the token embedding)? The model would know WHERE it is but
   not WHAT character it's looking at. Can it still learn anything? Reason first, then try it.

5. **Attention is a communication mechanism.** In our model, we have one attention head with
   head_size=32. Each character can only "ask one type of question" (one query vector). Why
   might this be limiting? What kinds of patterns might need multiple different questions
   asked simultaneously? (Preview: Chapter 3 introduces multi-head attention.)

## Common Struggles

### Tensor Shape Mismatches

This is the #1 source of bugs in this chapter. Here's every shape, tracked through the model:

```
Input:
  idx                              (B, T)         e.g. (32, 8)

Embeddings:
  tok_emb                          (B, T, n_embd)      (32, 8, 32)
  pos = torch.arange(T)            (T,)                (8,)
  pos_emb                          (T, n_embd)         (8, 32)    ← broadcasts over B
  x = tok_emb + pos_emb           (B, T, n_embd)      (32, 8, 32)

Inside the Head:
  q = self.query(x)               (B, T, head_size)    (32, 8, 32)
  k = self.key(x)                 (B, T, head_size)    (32, 8, 32)
  k.transpose(-2, -1)             (B, head_size, T)    (32, 32, 8)
  wei = q @ k^T                   (B, T, T)            (32, 8, 8)   ← the T x T attention matrix!
  wei (after mask + softmax)       (B, T, T)            (32, 8, 8)
  v = self.value(x)               (B, T, head_size)    (32, 8, 32)
  out = wei @ v                   (B, T, head_size)    (32, 8, 32)

Output projection:
  logits = lm_head(out)           (B, T, vocab_size)   (32, 8, 65)

For loss:
  logits reshaped                  (B*T, vocab_size)    (256, 65)
  targets reshaped                 (B*T,)               (256,)
```

If you get a shape error, add `print(tensor.shape)` at every step and compare to this table.

### Q/K/V Dimensions Confusion

All three projections (query, key, value) go from `n_embd` to `head_size`. In our case both
are 32, which can mask the distinction. The important thing to remember:

- **n_embd** is the model's internal representation size (the embedding dimension)
- **head_size** is the attention head's working dimension
- They happen to be equal here, but in Chapter 3 with multi-head attention, they won't be

The `nn.Linear(n_embd, head_size, bias=False)` is just a matrix multiply: input vector of
size `n_embd` goes in, output vector of size `head_size` comes out. Three separate learned
matrices, three separate outputs (Q, K, V), all from the same input `x`.

### Why Masking Matters

Without the mask, the model cheats during training: when predicting position 3, it can see
positions 4, 5, 6, 7 (the future). The training loss looks great because the model has the
answers. But at generation time, there ARE no future tokens — the model has never learned to
predict without them. Result: garbage output despite low training loss.

This is a form of **data leakage** — the model has access to information during training that
it won't have at inference time. The mask forces the model to learn the same way it will be
used: predicting the next character using only what came before.
