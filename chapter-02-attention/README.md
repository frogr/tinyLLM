# Chapter 2: Self-Attention from Scratch

## Where We Are

In Chapter 1, we built a bigram model — it predicts the next character by looking at **only the current character**. That's like autocomplete that only sees the last letter you typed. If you see "h", you guess "e" (because "the" is common). But you have no idea if the previous character was "t" (making "the" likely) or "s" (making "she" likely).

The bigram model can't learn that "th" is usually followed by "e" because it only ever sees "h" — it has no memory of "t".

**What we're adding**: A mechanism called **self-attention** that lets each token "look at" the tokens before it and decide which ones are relevant for making its prediction.

This is the key innovation behind transformers and GPT. Everything else is optimization on top of this idea.

## Concepts

### Why Bigram Isn't Enough

Consider predicting the next character after seeing this sequence: `"The quick brown f"`

A bigram model only sees `"f"` and has to guess what comes next. It might reasonably guess "o" (for "for", "found", "four") or "a" (for "far", "fast") or "r" (for "from", "free"). It has no way to use the context "brown" to guess that "fox" is likely.

**We need a way for the model to look backward at previous tokens.** That's attention.

### Self-Attention: The Cocktail Party Analogy

Imagine you're at a cocktail party. When someone asks you a question, you don't listen to everyone in the room equally — you focus on the people who are relevant to the question. The person next to you might be very relevant, the person across the room might not be.

That's attention. Each token gets to "look at" the tokens before it and decide which ones matter for its prediction. The word "decides" is key — the model **learns** which tokens to pay attention to.

### Queries, Keys, and Values (Q, K, V)

This is the part where most people's eyes glaze over, so let's use a concrete analogy.

Think of a **search engine**:
- **Query (Q)**: Your search query — "What am I looking for?"
- **Key (K)**: The page title / metadata — "What do I have to offer?"
- **Value (V)**: The actual page content — "Here's the information if you choose me"

Every token produces all three:
- Its **query** says "I'm looking for tokens that are [something]"
- Its **key** says "I am a token that offers [something]"
- Its **value** says "If you attend to me, here's the information I'll give you"

The attention mechanism compares each query against all keys to find the best matches, then returns a weighted combination of the corresponding values.

In code, Q, K, and V are just **linear projections** — matrix multiplications that transform each token's embedding into three different vectors:

```python
self.query = nn.Linear(n_embd, head_size, bias=False)
self.key   = nn.Linear(n_embd, head_size, bias=False)
self.value = nn.Linear(n_embd, head_size, bias=False)

q = self.query(x)  # shape: (B, T, head_size)
k = self.key(x)    # shape: (B, T, head_size)
v = self.value(x)  # shape: (B, T, head_size)
```

### The Dot Product: How Q and K Get Compared

How do we measure how well a query matches a key? The **dot product**.

The dot product of two vectors is a similarity score. If Q and K point in the same direction (similar), the dot product is large. If they're perpendicular (unrelated), it's near zero.

```python
# For each pair of positions (i, j), compute the similarity
# between query at position i and key at position j
wei = q @ k.transpose(-2, -1)  # shape: (B, T, T)
```

The result is a T×T matrix where entry (i, j) says "how much should position i attend to position j?"

### Why sqrt(d_k)? (Scaled Dot-Product Attention)

We divide the dot products by `sqrt(head_size)`:

```python
wei = q @ k.transpose(-2, -1) * head_size**-0.5
```

Why? Without this, when head_size is large, the dot products become large numbers. Large numbers going into softmax produce very peaked distributions (almost one-hot — one position gets ~100% of the attention). Peaked distributions have near-zero gradients, so the model stops learning.

Dividing by `sqrt(head_size)` keeps the values in a reasonable range regardless of the dimension. It's a normalization trick.

### Softmax: Turning Scores into Probabilities

After computing attention scores, we need to convert them to probabilities (positive numbers that sum to 1). That's what **softmax** does:

```
softmax([2.0, 1.0, 0.1]) → [0.66, 0.24, 0.10]
```

Why not just divide by the sum? Because softmax amplifies differences — the biggest score gets a **disproportionately** larger share. This helps the model make clear decisions about what to attend to.

```python
wei = F.softmax(wei, dim=-1)  # shape: (B, T, T), each row sums to 1
```

### Masked Attention: No Peeking at the Future

This is a **language model** — it predicts the next token. If it could see the future, that'd be cheating. Token at position 5 should only be able to attend to positions 0, 1, 2, 3, 4.

We implement this with a **lower triangular mask**:

```
Position:  0  1  2  3
       0 [ 1  0  0  0 ]   ← position 0 can only see itself
       1 [ 1  1  0  0 ]   ← position 1 can see 0 and 1
       2 [ 1  1  1  0 ]   ← position 2 can see 0, 1, and 2
       3 [ 1  1  1  1 ]   ← position 3 can see everything before it
```

We set the zeros to negative infinity **before** softmax. Since softmax(−∞) = 0, those positions get zero attention weight:

```python
tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
```

### Position Embeddings: Where Am I?

In Chapter 1, the model had no sense of position — it treated each character the same regardless of where it appeared. Now we add **position embeddings**: a learnable vector for each position (0, 1, 2, ..., block_size-1).

```python
self.position_embedding_table = nn.Embedding(block_size, n_embd)
pos_emb = self.position_embedding_table(torch.arange(T, device=device))
x = tok_emb + pos_emb  # Token identity + position information
```

We **add** (not concatenate) the position embedding to the token embedding. This works because the model learns to use different dimensions for "what token is this" vs. "where is this token."

### head_size: A Tiny Numerical Example

Let's say `n_embd = 4` (each token is represented by 4 numbers) and `head_size = 2` (Q, K, V are 2-dimensional).

```
Token embedding for "H": [0.5, -0.3, 0.8, 0.1]

Query projection (2x4 matrix):     Key projection (2x4 matrix):
  [0.2, 0.1, -0.3, 0.4]             [0.1, -0.2, 0.5, 0.3]
  [0.5, -0.1, 0.2, 0.3]             [-0.3, 0.4, 0.1, -0.2]

Q for "H": [0.5*0.2 + (-0.3)*0.1 + 0.8*(-0.3) + 0.1*0.4,   = [-0.13,
             0.5*0.5 + (-0.3)*(-0.1) + 0.8*0.2 + 0.1*0.3]     0.47]

"H" is asking: "I'm looking for something in the direction [-0.13, 0.47]"
```

The head_size controls how rich this "search space" is. Bigger head_size = more nuanced matching between queries and keys.

## The Math

The full attention formula:

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

In code, step by step:

```python
# Step 1: Compute Q, K, V from input x (shape: B, T, n_embd)
q = self.query(x)   # (B, T, head_size)
k = self.key(x)     # (B, T, head_size)
v = self.value(x)   # (B, T, head_size)

# Step 2: Compute attention scores
wei = q @ k.transpose(-2, -1)  # (B, T, head_size) @ (B, head_size, T) → (B, T, T)
wei = wei * head_size**-0.5    # scale to prevent softmax saturation

# Step 3: Mask future positions
wei = wei.masked_fill(tril[:T, :T] == 0, float('-inf'))

# Step 4: Convert to probabilities
wei = F.softmax(wei, dim=-1)   # (B, T, T), each row sums to 1

# Step 5: Weighted sum of values
out = wei @ v                  # (B, T, T) @ (B, T, head_size) → (B, T, head_size)
```

## Architecture Diagram

```
Input Characters
      │
      ▼
┌─────────────┐
│ Token IDs   │  "H" → 45, "e" → 42, "l" → 49, ...
└─────┬───────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  Token Embedding  +  Position Embedding │
│  nn.Embedding          nn.Embedding     │
│  (vocab, n_embd)       (block, n_embd)  │
│                                         │
│  "what token"   +   "where it is"       │
└─────────────┬───────────────────────────┘
              │ shape: (B, T, n_embd)
              ▼
┌─────────────────────────────────────┐
│         Self-Attention Head         │
│                                     │
│  x ──┬──► Q = query(x)             │
│      ├──► K = key(x)               │
│      └──► V = value(x)             │
│                                     │
│  scores = Q @ K^T / √head_size     │
│  scores = mask(scores)             │
│  weights = softmax(scores)          │
│  output = weights @ V               │
└─────────────┬───────────────────────┘
              │ shape: (B, T, head_size)
              ▼
┌─────────────────────────┐
│  Linear (lm_head)       │
│  head_size → vocab_size │
└─────────────┬───────────┘
              │ shape: (B, T, vocab_size)
              ▼
         Predictions
```

## Common Struggle Points

### Tensor Shape Mismatches

This will be your #1 frustration. Here's the complete shape flow:

```
Input idx:           (B, T)           B=batch_size, T=block_size
Token embedding:     (B, T, n_embd)   Each token gets an n_embd-dim vector
Position embedding:  (T, n_embd)      Broadcasts across batch
Combined:            (B, T, n_embd)   Token + position info

Q, K, V:             (B, T, head_size) Projected to attention dimension
Attention scores:    (B, T, T)         Every position vs every position
Attention weights:   (B, T, T)         After mask + softmax
Attention output:    (B, T, head_size) Weighted sum of values

Final logits:        (B, T, vocab_size) One prediction per position
```

### Why Position Embeddings?

Without position embeddings, the attention mechanism is **permutation invariant** — it treats `"cat"` and `"tac"` identically. Position embeddings break this symmetry by telling the model "this token is at position 3."

## Exercises

Work through `exercises.py` in order. The TODOs build on each other:

1. **TODO 1-2**: Create and use Q, K, V projections (the search engine components)
2. **TODO 3**: Compute attention scores (the similarity matching)
3. **TODO 4**: Apply the causal mask (no peeking at the future)
4. **TODO 5**: Weighted sum of values (the actual information gathering)
5. **TODO 6-7**: Wire it all into the model with position embeddings

## Stretch Goals

- Print the attention weights for a sample input — which positions attend to which?
- Remove position embeddings and retrain. What happens?
- Try head_size = 4, 16, 32, 64. How does it affect loss and generation quality?
- Visualize attention patterns using matplotlib (create a heatmap)

## Think About It

1. **"Remove the mask (let the model see future tokens). What happens to the training loss? Why is this bad even if the loss is lower?"**
   Try it — the loss drops dramatically. But the model is cheating: it can see the answer. At generation time, there ARE no future tokens, so the model falls apart. It learned to copy rather than predict.

2. **"What happens if you increase head_size from 16 to 64? More parameters = better?"**
   More parameters doesn't always mean better. With a small dataset and small model, a bigger head_size might overfit. Also, head_size determines the "search space" — too small and queries can't express nuanced searches, too large and the model might not learn efficient patterns. Try it!

3. **"Why do we add position embeddings instead of concatenating them?"**
   Concatenation would double the embedding dimension and require all downstream layers to be larger. Addition works because the model can learn to use orthogonal (non-overlapping) dimensions for token identity vs. position. It's more parameter-efficient.

4. **"If you print the attention weights for a sample, which positions get the most attention? Does it make sense?"**
   Early in training, attention is roughly uniform. After training, you'll often see: recent positions get more attention (locality bias), and the first position gets extra attention (it acts as a "default" or "null" attention target). This mirrors how language works — nearby context matters most.

## Further Reading

- [Karpathy "Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) — timestamps 47:00 to 1:11:00 cover self-attention
- [3Blue1Brown: Attention in transformers, visually explained](https://www.youtube.com/watch?v=eMlx5fFNoYc)
- ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) — Section 3.2 (Scaled Dot-Product Attention)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Excellent visual walkthrough
