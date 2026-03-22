# Chapter 3: The Full Transformer Block

## Where We Are

In Chapter 1, we built a bigram model — it looked at one character and predicted the next.
In Chapter 2, we added self-attention — a single attention head that let each token look at
previous tokens to gather context. The output improved, but a single attention head is like
having one person in a meeting — they can only focus on one thing at a time.

Now we're building the **full transformer block**: multiple attention heads running in
parallel, a feedforward network, residual connections, and layer normalization. This is the
architecture behind GPT, BERT, and every modern large language model.

By the end of this chapter, you'll understand every component in the diagram below — and
you'll have built each one from scratch.

---

## Concepts

### Multi-Head Attention

Instead of one attention head looking for one kind of pattern, we run **multiple heads in
parallel**. One head might learn to look for verb-noun relationships, another for nearby
characters, another for punctuation patterns. Then we **concatenate** their outputs and
project them back to the embedding dimension.

Think of it as a **committee** where each member pays attention to different things:

- Head 1: "I'm watching for grammatical structure"
- Head 2: "I'm tracking character names"
- Head 3: "I'm looking at recent context"
- Head 4: "I'm watching for rhyme and meter"

After the meeting, they combine their notes into one summary.

**The key insight**: if you have 4 heads of size 16 vs 1 head of size 64, the total
parameter count is the same. But the multi-head version works dramatically better because
it can attend to information from **different representation subspaces** at different
positions simultaneously.

```python
# Single head: one 64-dim attention
head = Head(head_size=64)

# Multi-head: four 16-dim attentions, concatenated back to 64
heads = [Head(head_size=16) for _ in range(4)]
out = torch.cat([h(x) for h in heads], dim=-1)  # (B, T, 64)
out = projection(out)  # (B, T, 64) — mix the heads' outputs
```

### Feedforward Network

Two linear layers with a ReLU activation in between:

```python
nn.Sequential(
    nn.Linear(n_embd, 4 * n_embd),    # expand
    nn.ReLU(),                          # nonlinearity
    nn.Linear(4 * n_embd, n_embd),     # project back
)
```

If attention is about **"gathering information from other tokens"**, the feedforward network
is about **"thinking about what I gathered."** It processes each position independently —
like each token going into a private office to think after the meeting.

Why `4 * n_embd`? The original "Attention Is All You Need" paper used this ratio. The idea
is that the feedforward layer needs a larger hidden space to do its processing. Think of it
as "expanding" the representation into a bigger workspace, doing some computation, and then
"compressing" back. The 4x multiplier is a convention that works well in practice.

### ReLU Activation

```
ReLU(x) = max(0, x)
```

That's it. It just zeros out negative values. Graphically:

```
output
  │      /
  │     /
  │    /
  │   /
──┼──/──────── input
  │
  │
```

**Why do we need it?** Without nonlinearities, stacking layers would be pointless — multiple
linear transformations collapse into a single linear transformation. If `y = Ax` and
`z = By`, then `z = BAx = Cx` where `C = BA`. You could replace two layers with one.

ReLU breaks this linearity and lets the network learn complex, nonlinear patterns. It's
the simplest possible nonlinearity that works well. (GPT-2 actually uses GELU, a smoother
variant, but ReLU is easier to understand and works fine for our purposes.)

### Residual Connections (Skip Connections)

Instead of:
```
x → layer → output
```

We do:
```
x → layer → output + x
```

We **ADD the input back to the output**. The `+ x` part is the residual connection.

**Why?** Imagine training a 50-layer network. Gradients need to flow backward through ALL
50 layers during backpropagation. That's like playing telephone — the message (gradient)
degrades with each step. By the time it reaches the first layers, it might be nearly zero
("vanishing gradient problem") or enormous ("exploding gradient problem").

Residual connections provide a **highway for gradients** to flow directly from later layers
to earlier layers. During backpropagation, the gradient through `output + x` is
`d(output)/dx + 1`. That `+ 1` means there's always a direct gradient path, no matter how
many layers you have.

In software terms: it's like having a `main()` that catches errors — even if middleware
breaks, the original request still gets through. Or think of it as Git: even if your
feature branch goes wrong, the main branch (the residual) preserves the original state.

```python
# Without residual connection:
x = self.attention(x)       # if attention learns poorly, x is corrupted

# With residual connection:
x = x + self.attention(x)   # attention output is ADDED to original
                              # worst case: attention outputs zeros, x unchanged
```

### Layer Normalization

Normalize the values across features so they don't grow too large or too small:

```python
# For each token position, across its embedding dimensions:
mean = x.mean(dim=-1, keepdim=True)
std = x.std(dim=-1, keepdim=True)
x_norm = (x - mean) / (std + 1e-5)  # epsilon for numerical stability
x_norm = gamma * x_norm + beta       # learnable scale and shift
```

Without LayerNorm, deep networks become **numerically unstable** — like a feedback loop
where small errors compound. If one layer's output is slightly too large, the next layer
amplifies it, and so on. LayerNorm keeps things in a reasonable range after every sub-layer.

PyTorch provides `nn.LayerNorm(n_embd)` which handles this for us, including the learnable
`gamma` and `beta` parameters.

### Pre-Norm vs Post-Norm

The original transformer paper applies LayerNorm **after** the residual connection
("post-norm"):

```
x = LayerNorm(x + sublayer(x))     # post-norm (original paper)
```

Modern implementations (including GPT-2) use **pre-norm** — applying LayerNorm **before**
each sub-layer:

```
x = x + sublayer(LayerNorm(x))     # pre-norm (what we use)
```

Pre-norm is easier to train and more stable. That's what we'll implement.

### The Full Transformer Block

All the pieces fit together like this:

```
Input
  │
  ├──────────────────┐
  ▼                  │ (residual connection)
LayerNorm            │
  │                  │
Multi-Head           │
Attention            │
  │                  │
  + ◄────────────────┘
  │
  ├──────────────────┐
  ▼                  │ (residual connection)
LayerNorm            │
  │                  │
FeedForward          │
  │                  │
  + ◄────────────────┘
  │
Output
```

In code, the entire block is remarkably simple:

```python
class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_head, n_embd // n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))    # attention + residual
        x = x + self.ffwd(self.ln2(x))    # feedforward + residual
        return x
```

That's it. Two sub-layers, each wrapped with LayerNorm and a residual connection.

### Stacking Blocks

One transformer block can learn some patterns. But GPT-2 uses **12 blocks** (for the small
version) and GPT-3 uses **96 blocks**. Why?

Each block refines the representation. Think of it as an assembly line:
- Block 1: Learns basic patterns ("this character often follows that one")
- Block 2: Builds on Block 1's output to learn higher-level patterns
- Block 3: Combines those patterns into even more abstract representations
- ...and so on

More blocks = more capacity to learn complex patterns. But more blocks also means more
parameters and more computation. It's a tradeoff.

In this chapter, we'll use **just 1 block** to keep things simple. In Chapter 4, we scale
up to multiple blocks.

---

## The Math

### Multi-Head Attention

Each head computes attention independently with its own Q, K, V projections:

```
head_i = Attention(Q_i, K_i, V_i)
```

where Q_i, K_i, V_i are projected from the input using separate weight matrices, and each
head has `head_size = n_embd // n_head`.

Then all heads are concatenated and projected:

```
MultiHead(x) = Concat(head_1, head_2, ..., head_h) @ W_o

where:
  head_i has shape (B, T, head_size)
  Concat gives shape  (B, T, h * head_size) = (B, T, n_embd)
  W_o has shape        (n_embd, n_embd)
  Output has shape     (B, T, n_embd)
```

```python
# In code:
heads = [head(x) for head in self.heads]            # list of (B, T, head_size)
out = torch.cat(heads, dim=-1)                       # (B, T, n_embd)
out = self.proj(out)                                  # (B, T, n_embd)
```

### Feedforward Network

```
FFN(x) = W_2 · ReLU(W_1 · x + b_1) + b_2

where:
  x has shape    (B, T, n_embd)
  W_1 has shape  (n_embd, 4 * n_embd)      — expand
  W_2 has shape  (4 * n_embd, n_embd)      — compress back
```

```python
# In code:
x = self.net(x)  # Sequential(Linear, ReLU, Linear)
```

### Residual + LayerNorm Pattern (Pre-Norm)

```
# Sub-layer 1: Multi-head attention
x = x + MultiHead(LayerNorm(x))

# Sub-layer 2: Feedforward
x = x + FFN(LayerNorm(x))
```

The LayerNorm normalizes across the last dimension (the embedding dimension):

```
LayerNorm(x) = gamma * (x - mean) / sqrt(var + eps) + beta

where mean and var are computed over the last dimension (n_embd)
gamma and beta are learnable parameters of shape (n_embd,)
```

---

## Dimension Tracking

This is where things get tricky. Let's trace every shape through the full model.

**Hyperparameters for this chapter:**
- `n_embd = 64` (embedding dimension)
- `n_head = 4` (number of attention heads)
- `head_size = n_embd // n_head = 16` (each head's dimension)
- `block_size = 32` (context length)
- `batch_size = 32`
- `vocab_size = 65` (Shakespeare characters)

```
Input:    (B, T)            = (32, 32)         — batch of token indices
                  ▼ Token embedding
Token:    (B, T, n_embd)    = (32, 32, 64)     — look up embeddings
                  ▼ + Position embedding
Embed:    (B, T, n_embd)    = (32, 32, 64)     — add positional info
                  ▼ TransformerBlock
                  │
                  ├─ LayerNorm:      (32, 32, 64)    — normalize
                  ├─ MultiHeadAttn:
                  │   ├─ Head 1:     (32, 32, 16)    — each head is small
                  │   ├─ Head 2:     (32, 32, 16)
                  │   ├─ Head 3:     (32, 32, 16)
                  │   ├─ Head 4:     (32, 32, 16)
                  │   ├─ Concat:     (32, 32, 64)    — back to full size
                  │   └─ Project:    (32, 32, 64)    — mix head outputs
                  ├─ + Residual:     (32, 32, 64)    — add input back
                  │
                  ├─ LayerNorm:      (32, 32, 64)    — normalize again
                  ├─ FeedForward:
                  │   ├─ Linear 1:   (32, 32, 256)   — expand (4x)
                  │   ├─ ReLU:       (32, 32, 256)   — nonlinearity
                  │   └─ Linear 2:   (32, 32, 64)    — compress back
                  ├─ + Residual:     (32, 32, 64)    — add input back
                  │
Final LN: (B, T, n_embd)    = (32, 32, 64)     — final layer norm
                  ▼ Linear head
Logits:   (B, T, vocab_size) = (32, 32, 65)     — scores for each char
```

---

## Common Struggle Points

### 1. Dimension Management with Multiple Heads

The most common bug: getting the shapes wrong when splitting into heads and concatenating
back. Remember:
- `n_embd` must be divisible by `n_head`
- Each head gets `head_size = n_embd // n_head` dimensions
- After concatenation, you're back to `n_embd`

### 2. Understanding Why Residual Connections Help

It's not just "the gradient flows better." Think about what happens during **initialization**.
At the start of training, the attention and feedforward layers output near-random values.
Without residual connections, the input signal is immediately corrupted. With residual
connections, the network starts as something close to the **identity function** — the input
passes through mostly unchanged, and each layer only needs to learn a small **delta**
(correction) to the representation.

### 3. Pre-Norm vs Post-Norm Order of Operations

We use pre-norm (LayerNorm before each sub-layer). This is important:

```python
# CORRECT (pre-norm):
x = x + self.attn(self.ln1(x))

# WRONG ORDER (post-norm — harder to train):
x = self.ln1(x + self.attn(x))

# ALSO WRONG (norm but no residual):
x = self.attn(self.ln1(x))
```

### 4. The Projection After Concatenation

After concatenating all heads, we apply a linear projection `W_o`. This isn't just
reshaping — it's a **learned** transformation that lets the model combine information
from different heads. Without it, the heads can't interact with each other.

---

## Think About It

1. **Head count vs head size**: If you have 4 heads of size 16 vs 1 head of size 64, the
   parameter count is the same. But the multi-head version works better. Why might that be?

   *Hint: Think about what each head can specialize in. A single large head must use the
   same attention pattern for everything. Multiple small heads can each learn different
   patterns.*

2. **Removing residual connections**: What happens if you remove the residual connections?
   Try it in `sandbox.py`. Why does training collapse?

   *Hint: Think about gradient flow AND initialization. What does the network compute
   when all weights are near-random?*

3. **The 4x feedforward expansion**: Why is the feedforward hidden dimension typically 4x
   the embedding dimension?

   *Hint: The feedforward network needs enough capacity to "process" the information
   gathered by attention. 4x is a convention from the original paper — you can try 2x
   or 8x and see what happens to loss.*

4. **Removing LayerNorm**: What does the model look like if you remove LayerNorm? Does it
   still train?

   *Hint: With just 1 block, it might still work. But try stacking 4+ blocks without
   LayerNorm. Watch the loss — it'll either explode or refuse to decrease.*

5. **Why add, not concatenate?** Residual connections use addition, not concatenation. Why?

   *Hint: If we concatenated, the dimension would double at every layer. With 12 layers,
   a 64-dim embedding would become 64 * 2^12 = 262,144 dimensions. Addition keeps the
   dimension constant.*

---

## Further Reading

- **Karpathy's video**: 1:11:00 - 1:28:00 covers multi-head attention, feedforward,
  residual connections, and layer norm. Watch this segment after completing the exercises.

- **"Attention Is All You Need"** (Vaswani et al., 2017): Sections 3.1 (Encoder-Decoder
  Attention), 3.2 (Multi-Head Attention), and 3.3 (Position-wise Feed-Forward Networks).
  The original paper is surprisingly readable.

- **"Deep Residual Learning for Image Recognition"** (He et al., 2015): The paper that
  introduced residual connections. Originally for CNNs, but the same principle applies to
  transformers.

- **"Layer Normalization"** (Ba et al., 2016): The paper that introduced LayerNorm. If
  you're curious about the difference between BatchNorm and LayerNorm, this explains it.

---

## What's Next

In Chapter 4, we scale up: multiple transformer blocks (6 layers), dropout for
regularization, and learning rate schedules. We'll go from a toy model to something that
generates surprisingly convincing Shakespeare.
