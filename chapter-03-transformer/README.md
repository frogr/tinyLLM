# Chapter 3 — The Transformer Block

## Context

In Chapter 2, we added a single self-attention head. The model could finally look at
previous characters before making a prediction — a huge leap from the bigram model. But
one attention head with 32 dimensions can only learn one "type" of pattern. It's like having
a single reviewer on a pull request — they'll catch some issues, but miss others.

The result? Slightly better gibberish. The model learned some basic character dependencies,
but it lacks the capacity to capture the richness of language. It can ask one question of
the context ("which characters are relevant?") but can't simultaneously ask "what vowels
precede me?" AND "am I inside a word or between words?" AND "does this look like the start
of a name?"

In this chapter, we build the **full transformer block** — the fundamental building block of
GPT, BERT, Claude, and every other modern language model. We add:

1. **Multi-head attention** — multiple attention heads running in parallel
2. **Feedforward network** — a small neural net that processes what attention gathered
3. **Residual connections** — skip connections that let information (and gradients) flow
4. **Layer normalization** — keeping activations stable so training works

By the end of this chapter, your model will still be small, but it will have the same
*architecture* as GPT. Everything from here on is just making it bigger.

## Concepts

### Multi-Head Attention — Multiple Reviewers on the PR

In Chapter 2, we built a single attention head. It computes one set of queries, keys, and
values, and produces one "view" of the context. But language has many simultaneous patterns:

- Syntactic: "is this a noun or verb position?"
- Positional: "what's the character right before me?"
- Semantic: "am I inside a character name?"

A single attention head can only capture one pattern (or a muddled combination). **Multi-head
attention** runs multiple heads in parallel — each with its own Q, K, V projections — then
concatenates their outputs.

Think of it like code review with multiple reviewers:
- Reviewer 1 checks for logic bugs
- Reviewer 2 checks for style issues
- Reviewer 3 checks for security vulnerabilities
- The final review = concatenation of all their findings

Each head has a smaller dimension. If `n_embd=32` and `n_head=4`, each head works with
`head_size = 32 // 4 = 8` dimensions. After concatenating 4 heads of size 8, we get back to
32 dimensions. Same total capacity, but specialized across heads.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)       # projection back into residual stream
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Run all heads in parallel, concatenate along the last dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_embd)
        out = self.dropout(self.proj(out))                     # (B, T, n_embd)
        return out
```

The `proj` linear layer at the end is the "output projection" — it lets the model learn how
to combine information from all the heads. Think of it as the tech lead summarizing all
reviewer feedback into a single coherent action plan.

### Feedforward Network — The Thinking Step

Attention is a *communication* mechanism — it lets tokens gather information from other
tokens. But after gathering that information, the model needs to actually *process* it.
That's the feedforward network (FFN).

It's just two linear layers with a ReLU activation in between:

```python
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),    # expand: think of more things
            nn.ReLU(),                          # non-linearity: make decisions
            nn.Linear(4 * n_embd, n_embd),     # compress: back to original size
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)                      # (B, T, n_embd) → (B, T, n_embd)
```

**The middleware analogy.** Attention is like gathering context from a database query.
The feedforward network is like the middleware that processes the results — it transforms,
filters, and enriches the information before passing it along. Each token goes through this
independently (no cross-token communication here; that already happened in attention).

**Why 4x expansion?** The inner layer is `4 * n_embd` — four times wider than the input.
This gives the model more "thinking room." It's like having a wider scratch pad for
intermediate computation. The factor of 4 comes from the original "Attention Is All You
Need" paper and has become standard.

**Why ReLU?** ReLU (Rectified Linear Unit) is the simplest useful non-linearity:
`ReLU(x) = max(0, x)`. Without a non-linearity, stacking linear layers would just be one
big linear layer (matrix multiply is associative). ReLU lets the network learn non-linear
patterns — it can "turn off" certain features by clamping them to zero.

### Residual Connections — The Highway Bypass

This is the simplest concept that has the biggest impact. A residual connection is just:

```python
x = x + sublayer(x)
```

That's it. Addition. But it's crucial for two reasons:

**1. The highway analogy.** Imagine a highway with multiple processing stations along it.
Without residual connections, information MUST pass through every station — if any station
corrupts the signal, it's lost forever. With residual connections, there's a bypass lane.
The original information flows directly through, and each station only needs to learn what
to *add* to it.

```
Without residual:    x → [Station A] → [Station B] → [Station C] → output
                     Information must survive every transformation

With residual:       x ──────────────────────────────────────────── + → output
                     │                                              ↑
                     └→ [Station A] → add ──→ [Station B] → add ──┘
                     Original signal always available
```

**2. Gradient flow.** During backpropagation, gradients need to flow from the loss all the
way back to the early layers. Without residual connections, gradients pass through many
multiplications and can shrink to near-zero ("vanishing gradients") or explode. The residual
connection provides a "gradient highway" — since the derivative of `x + f(x)` with respect
to `x` is `1 + f'(x)`, there's always a gradient of at least 1 flowing straight back. This
is why you can train networks with 100+ layers — without residuals, you'd struggle past 10.

For software engineers: think of it like error propagation. If you have a chain of 50
function calls and each one slightly dampens the error signal, by the end you can't tell
what went wrong. Residual connections are like having a direct log from every function to
the error reporter.

### Layer Normalization — Normalizing the Inputs

As values flow through the network, they can drift — some dimensions might be huge while
others are tiny. This makes training unstable because the optimizer has to deal with wildly
different scales.

**Layer normalization** standardizes each vector to have zero mean and unit variance, then
applies learned scale and shift parameters:

```python
# For each vector x of dimension n_embd:
mean = x.mean(dim=-1, keepdim=True)
std = x.std(dim=-1, keepdim=True)
x_norm = (x - mean) / (std + epsilon)       # normalize
x_out = gamma * x_norm + beta               # learned scale + shift
```

**The database analogy.** When you normalize values before storing them in a database (e.g.,
converting all timestamps to UTC, all prices to cents), downstream queries become much more
reliable. Layer norm does the same thing for neural network activations — it ensures all
dimensions are on a comparable scale, making optimization smoother.

PyTorch gives us this for free:

```python
self.ln = nn.LayerNorm(n_embd)
x = self.ln(x)  # normalizes the last dimension
```

**Pre-norm vs post-norm.** The original transformer paper puts layer norm *after* the
sublayer (post-norm). Modern practice puts it *before* (pre-norm) because it's more stable
during training. We use pre-norm:

```python
# Pre-norm (what we use — more stable):
x = x + attention(self.ln1(x))
x = x + ffn(self.ln2(x))

# Post-norm (original paper — can be unstable):
x = self.ln1(x + attention(x))
x = self.ln2(x + ffn(x))
```

The difference is subtle but matters: pre-norm normalizes the input to each sublayer,
preventing any single sublayer from receiving wildly scaled inputs.

### Assembling the Transformer Block

Now we put it all together. A single transformer block is:

1. Layer norm → Multi-head attention → Add (residual)
2. Layer norm → Feedforward → Add (residual)

```python
class Block(nn.Module):
    """Transformer block: communication (attention) followed by computation (ffn)."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)   # communication
        self.ffn = FeedForward(n_embd)                     # computation
        self.ln1 = nn.LayerNorm(n_embd)                    # normalize before attention
        self.ln2 = nn.LayerNorm(n_embd)                    # normalize before ffn

    def forward(self, x):
        x = x + self.sa(self.ln1(x))     # attend, then add back (residual)
        x = x + self.ffn(self.ln2(x))    # think, then add back (residual)
        return x
```

That's the entire transformer block. GPT-3 stacks 96 of these. We'll start with 1.

## The Math

### Multi-Head Attention

```
MultiHead(X) = Concat(head_1, head_2, ..., head_h) @ W_O

Where each head_i = Attention(X @ W_Qi, X @ W_Ki, X @ W_Vi)
      Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
      d_k = n_embd / n_head  (head size)
      W_O is the output projection matrix (n_embd, n_embd)
```

### Feedforward Network

```
FFN(x) = ReLU(x @ W_1 + b_1) @ W_2 + b_2

Where:
  W_1: (n_embd, 4 * n_embd)     — expand
  W_2: (4 * n_embd, n_embd)     — compress
```

### Full Transformer Block (pre-norm)

```
x = x + MultiHead(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

### Layer Normalization

```
LayerNorm(x) = gamma * (x - mean) / sqrt(variance + epsilon) + beta

Where mean and variance are computed over the last dimension (n_embd),
and gamma, beta are learned parameters of shape (n_embd,).
```

## Architecture Diagram

```
Input: "To be" as indices [20, 53, 1, 40, 43]

  idx (B, T)
    │
    ├──► Token Embedding (vocab_size, n_embd=32)     "what am I?"
    │         │
    │         ▼
    │    tok_emb (B, T, 32)
    │         │
    └──► Position Embedding (block_size, n_embd=32)  "where am I?"
              │
              ▼
         pos_emb (T, 32)
              │
    tok_emb + pos_emb
              │
              ▼
         x (B, T, 32)
              │
    ┌─────────┼─────────────────────────────────────────────┐
    │         │              TRANSFORMER BLOCK               │
    │         │                                              │
    │    ┌────┴────┐                                         │
    │    │         │                                         │
    │    │    LayerNorm1                                     │
    │    │         │                                         │
    │    │    Multi-Head Attention (4 heads, head_size=8)    │
    │    │    ┌────┼────┬────┐                               │
    │    │    │    │    │    │                                │
    │    │   H1   H2   H3   H4    each: (B, T, 8)           │
    │    │    │    │    │    │                                │
    │    │    └────┴────┴────┘                               │
    │    │         │                                         │
    │    │    Concat → (B, T, 32)                            │
    │    │         │                                         │
    │    │    Output Projection → (B, T, 32)                 │
    │    │         │                                         │
    │    └────► + ◄┘  (residual connection)                  │
    │              │                                         │
    │         ┌────┴────┐                                    │
    │         │         │                                    │
    │         │    LayerNorm2                                │
    │         │         │                                    │
    │         │    FeedForward                               │
    │         │    Linear(32, 128) → ReLU → Linear(128, 32) │
    │         │         │                                    │
    │         └────► + ◄┘  (residual connection)             │
    │                  │                                     │
    └──────────────────┼─────────────────────────────────────┘
                       │
                  LayerNorm (final)
                       │
                  Linear(32, vocab_size=65)
                       │
                       ▼
                  logits (B, T, 65)
                       │
                  cross_entropy → loss
```

## Exercises

Open `exercises.py` and work through the TODOs in order. Each TODO builds on the previous
one. The file is runnable at every stage — incomplete TODOs use placeholder code.

1. **TODO 1** — Multi-head attention: run multiple heads in parallel, concatenate outputs
2. **TODO 2** — Feedforward network: Linear → ReLU → Linear
3. **TODO 3** — Transformer Block: combine attention + feedforward with residual connections and layer norm
4. **TODO 4** — Full model: embeddings → transformer block(s) → layer norm → linear head
5. **TODO 5** — Training loop and text generation

## Stretch Goals

- [ ] Add a second transformer block (`n_layer=2`) — does loss improve? How much slower is training?
- [ ] Change `n_head` from 4 to 1 and to 8 — how does it affect loss? (Keep `n_embd=32`)
- [ ] Add `dropout=0.1` — does it help or hurt at this small scale?
- [ ] Visualize attention weights from different heads — do they specialize?
- [ ] Try the post-norm formulation instead of pre-norm — what happens to training stability?
- [ ] Compare parameter counts: bigram (4,225) vs chapter 2 vs this model
- [ ] Train for 10,000 iterations — does the loss plateau? How does generation quality change?

## Further Reading

- [Karpathy "Let's build GPT" video](https://www.youtube.com/watch?v=kCc8FmEb1nY) — ~1:00:00 to ~1:30:00 covers multi-head attention, feedforward, and the transformer block
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar's visual walkthrough (see the multi-head section)
- [The original "Attention Is All You Need" paper](https://arxiv.org/abs/1706.03762) — Sections 3.1 (multi-head), 3.3 (feedforward), 5.1 (residual + norm)
- [3Blue1Brown "But what is attention?"](https://www.youtube.com/watch?v=eMlx5fFNoYc) — Revisit with your new multi-head understanding

## Think About It

1. **Head specialization.** With 4 attention heads each of size 8, could one head learn to
   always attend to "the previous character" while another learns to attend to "the most
   recent space"? How would you verify this? What would the attention weight matrices look
   like for each pattern?

2. **Why concatenate instead of average?** Multi-head attention concatenates head outputs
   rather than averaging them. What information would be lost if you averaged instead?
   (Hint: think about what the output projection can do with concatenated vs averaged inputs.)

3. **Residual connections and initialization.** At the start of training, all weights are
   random and the sublayers produce near-random outputs. What does the residual connection
   `x = x + sublayer(x)` do in this case? Why is this better than `x = sublayer(x)` for the
   early stages of training?

4. **The 4x expansion in the feedforward layer.** We expand from 32 to 128 dimensions then
   back to 32. What if we used 1x (no expansion) or 16x? What's the tradeoff between
   expressiveness and parameter count? How many parameters does the feedforward layer add?

5. **Pre-norm vs post-norm.** We normalize *before* each sublayer. The original paper
   normalizes *after*. Draw the computation graph for both. In the post-norm version, what
   does the residual connection add to — normalized or unnormalized values? Why might this
   cause training instability for deep networks?

## Anticipating Struggles

### "Why do residual connections help? It's just addition."

Yes, and that's the beauty of it. The key insight is what it does to *gradients*. Without
residuals, the gradient of the loss with respect to an early layer has to pass through every
subsequent layer (lots of multiplications). Each multiplication can shrink the gradient.
With residuals, `d(x + f(x))/dx = 1 + f'(x)` — there's always a +1 term providing a clean
gradient path. Think of it as the difference between a serial pipeline (each stage can
corrupt the signal) and a system where every stage's output is *added* to the original
(the original is always preserved).

### "Where does layer norm go, and does it matter?"

Yes, it matters. We use pre-norm: normalize *before* each sublayer. This means each sublayer
receives well-conditioned inputs regardless of what happened earlier. The alternative
(post-norm) normalizes *after* the residual addition, which can be unstable because the
unnormalized residual values might have wildly different scales. In practice, pre-norm lets
you train deeper networks more reliably.

### "I'm confused about tensor shapes through the block."

Let's trace them with `batch_size=4, block_size=8, n_embd=32, n_head=4`:

```
Input idx:                        (4, 8)        — 4 sequences of 8 characters
After token + position embedding: (4, 8, 32)    — each character now a 32-dim vector

Inside the transformer block:
  After LayerNorm1:               (4, 8, 32)    — same shape, just normalized
  Inside each attention head:
    Q, K, V:                      (4, 8, 8)     — head_size = 32/4 = 8
    attention weights:            (4, 8, 8)     — each position attends to 8 positions
    head output:                  (4, 8, 8)     — weighted values
  After concat 4 heads:          (4, 8, 32)    — 4 heads × 8 dims = 32
  After output projection:       (4, 8, 32)    — linear layer, same shape
  After residual add:            (4, 8, 32)    — addition, same shape

  After LayerNorm2:               (4, 8, 32)    — same shape
  Inside feedforward:
    After first Linear:           (4, 8, 128)   — expansion: 32 → 4*32=128
    After ReLU:                   (4, 8, 128)   — same shape, just zeroed negatives
    After second Linear:          (4, 8, 32)    — compression: 128 → 32
  After residual add:            (4, 8, 32)    — same shape

After final LayerNorm:            (4, 8, 32)    — same shape
After linear head:                (4, 8, 65)    — project to vocab_size
```

The key insight: **the residual stream is always `(B, T, n_embd)`**. Every sublayer receives
and returns this shape. The feedforward layer temporarily expands to `4 * n_embd` internally,
but that's hidden from the rest of the network.
