# Chapter 4: Scaling Up

## Where We Are

We have a working transformer block. But it's small -- one block, small embeddings,
no regularization. Now we scale up: 6 layers, bigger embeddings, dropout, and learning
rate schedules. This is where we go from "toy model" to "small but real model" that
generates recognizably Shakespeare-like text.

In previous chapters, we could see the model *trying* to be Shakespeare. In this
chapter, it will actually start producing text that sounds plausibly Elizabethan --
complete with thee's, thou's, and dramatic declarations about honor and death.

---

## What Changes From Chapter 3

| Aspect          | Chapter 3         | Chapter 4             |
|-----------------|-------------------|-----------------------|
| Layers          | 1 block           | 6 blocks              |
| Embedding dim   | 32                | 384                   |
| Heads           | 4 heads of 8      | 6 heads of 64         |
| Context window  | 8 tokens          | 256 tokens            |
| Dropout         | None              | 0.2                   |
| LR schedule     | Fixed             | Warmup + cosine decay |
| Parameters      | ~10K              | ~10M                  |
| Training        | ~1 min            | ~5-10 min             |
| Output quality  | Babbling          | Shakespeare-ish       |

---

## Concepts

### 1. Stacking Blocks (Depth)

Each transformer block refines the representation. The first block might learn simple
patterns ("q is followed by u"), later blocks learn more abstract patterns ("this
sounds like a soliloquy"). It's like a pipeline of code reviews -- each reviewer
catches different things.

Think of it as an assembly line:

```
Input tokens
    |
    v
[Block 1] -- learns basic character patterns and common bigrams
    |
[Block 2] -- learns word-level patterns, common spellings
    |
[Block 3] -- learns phrases and short idioms
    |
[Block 4] -- learns grammatical structure
    |
[Block 5] -- learns stylistic patterns, meter
    |
[Block 6] -- learns high-level coherence, theme consistency
    |
    v
Output predictions
```

In reality, each block doesn't neatly handle one "level" -- they all contribute
to all levels simultaneously. But the general principle holds: **depth lets the
model build increasingly abstract representations**.

Why not 100 layers? Diminishing returns plus training difficulty. Gradients have
to flow backward through every layer, and deep networks can suffer from vanishing
gradients. Our residual connections (from Chapter 3) help enormously here -- they
give gradients a "highway" to flow through.


### 2. Embedding Dimension (Width)

Bigger embeddings = richer representations. A 32-dim embedding can encode "this is
the letter e". A 384-dim embedding can encode "this is the letter e, at the start
of a word, in a question, spoken by a king." More dimensions = more nuance.

Think of it like describing a person:
- **2 dimensions**: tall, friendly
- **32 dimensions**: tall, friendly, brown hair, likes jazz, from Chicago...
- **384 dimensions**: an incredibly detailed profile capturing subtle aspects of
  personality, background, context, and relationships

Each dimension in the embedding is a feature the model can use to encode information.
More dimensions means more features, which means the model can represent finer
distinctions between tokens in different contexts.

**Rule of thumb**: `n_embd` should be divisible by `n_head` (since each head gets
`n_embd // n_head` dimensions). Common ratios in real models: head_size of 64 or 128.


### 3. Dropout

During training, randomly zero out some values. Why would breaking things help? It
prevents co-adaptation -- neurons can't rely on specific other neurons, so each one
has to be independently useful. It's like cross-training your team: if one person is
sick, the team still functions. Set to 0 during inference (generation).

```
Training mode (dropout=0.2):
    [0.5, 0.3, 0.0, 0.8, 0.1, 0.0, 0.4, 0.6, 0.0, 0.2]
     ^              ^         ^
     kept           zeroed    zeroed
     (scaled up     out       out
      by 1/0.8)

Inference mode (dropout=0):
    [0.5, 0.3, 0.7, 0.8, 0.1, 0.9, 0.4, 0.6, 0.3, 0.2]
     all values kept as-is
```

Where to apply dropout in a transformer:
1. **After attention weights** (after softmax) -- prevents over-reliance on
   specific positions
2. **After feed-forward layers** -- prevents co-adaptation between neurons
3. **After residual connections** -- regularizes the full sub-layer output

**Important**: PyTorch handles the train/eval switching for you. Call `model.train()`
before training and `model.eval()` before generation. Forgetting `model.eval()` is a
classic bug that makes generation worse.


### 4. Learning Rate Schedules

Start with a warmup (small LR -> big LR) then decay. Why? Early in training, the
model knows nothing -- big steps would be chaotic. After warmup, we take big steps to
learn fast. Then we slow down to fine-tune. It's like driving: slow in the parking lot,
fast on the highway, slow again to park.

```
Learning Rate Over Time:

LR
 ^
 |        ___________
 |       /           \
 |      /             \
 |     /               \
 |    /                  \
 |   /                    \
 |  /                      \___
 | /
 |/
 +---------------------------------> Step
 0   500  1000        4000  5000
     warmup    constant     decay

 Cosine decay (what we use):

LR
 ^
 |     .......
 |    .       .
 |   .         ..
 |  .            ..
 | .               ...
 |.                   ....
 |                        ......
 +---------------------------------> Step
```

**Why cosine decay?** It's smooth, well-studied, and works well in practice. The
gradual slowdown lets the model settle into a good minimum rather than bouncing
around it.


### 5. Hyperparameter Tuning

How to think about each hyperparameter:

| Parameter       | Too small                     | Too big                        | Sweet spot               |
|----------------|-------------------------------|--------------------------------|--------------------------|
| `batch_size`   | Noisy gradients, slow         | Memory errors, less stochastic | 32-64 for our scale      |
| `learning_rate`| Never converges               | Diverges (loss -> NaN)         | 1e-4 to 1e-3             |
| `n_layer`      | Can't learn abstractions      | Slow, diminishing returns      | 4-8 for small models     |
| `n_embd`       | Can't represent nuance        | Slow, overfitting              | 128-512 for small models |
| `n_head`       | Limited attention patterns    | Each head too small            | head_size 32-128         |
| `dropout`      | Overfitting                   | Underfitting (too much noise)  | 0.1-0.3                  |

**Rules of thumb**:
- Start with a known-good configuration, change one thing at a time
- If loss is NaN, your learning rate is too high
- If loss plateaus quickly, try a higher learning rate or bigger model
- Watch the gap between train and val loss -- that tells you about overfitting
- More parameters need lower learning rates (hence 3e-4 instead of 1e-3)


### 6. Overfitting vs Underfitting

If train loss is much lower than val loss, you're memorizing (overfitting). If both
are high, your model isn't powerful enough (underfitting). Dropout and smaller models
help overfitting. Bigger models help underfitting.

```
Underfitting:                    Overfitting:
  Loss                            Loss
   ^                               ^
   |  -------- train               |          ---- val
   |  -------- val                 |   ------
   |                               |  /
   |                               | /------- train
   +---------> step                +---------> step
   Both stay high.                 Gap grows over time.

Good fit:
  Loss
   ^
   |
   |  \
   |   \------- val
   |    \------ train
   |       (small gap)
   +---------> step
   Both decrease, gap stays small.
```

**Diagnosing and fixing**:
- Overfitting? Increase dropout, reduce model size, get more data, early stopping
- Underfitting? Bigger model, train longer, higher learning rate, less dropout

---

## Full Architecture

```
Input: "To be or not to be, that is the question"
  |
  v
+------------------------------------------------------------------+
|  Token Embedding (vocab_size=65 -> n_embd=384)                    |
|  Position Embedding (block_size=256 -> n_embd=384)                |
|  + Add token + position embeddings                                |
|  Dropout(0.2)                                                     |
+------------------------------------------------------------------+
  |
  v
+------------------------------------------------------------------+
|  Transformer Block 1                                              |
|  +------------------------------------------------------------+  |
|  | LayerNorm -> Multi-Head Attention (6 heads, head_size=64)   |  |
|  |              + Dropout on attention weights                 |  |
|  |              + Dropout on projection                        |  |
|  | + Residual connection                                       |  |
|  +------------------------------------------------------------+  |
|  +------------------------------------------------------------+  |
|  | LayerNorm -> FeedForward(384 -> 1536 -> 384)                |  |
|  |              + Dropout                                      |  |
|  | + Residual connection                                       |  |
|  +------------------------------------------------------------+  |
+------------------------------------------------------------------+
  |
  v
+------------------------------------------------------------------+
|  Transformer Block 2  (same structure)                            |
+------------------------------------------------------------------+
  |
  v
+------------------------------------------------------------------+
|  Transformer Block 3  (same structure)                            |
+------------------------------------------------------------------+
  |
  v
+------------------------------------------------------------------+
|  Transformer Block 4  (same structure)                            |
+------------------------------------------------------------------+
  |
  v
+------------------------------------------------------------------+
|  Transformer Block 5  (same structure)                            |
+------------------------------------------------------------------+
  |
  v
+------------------------------------------------------------------+
|  Transformer Block 6  (same structure)                            |
+------------------------------------------------------------------+
  |
  v
+------------------------------------------------------------------+
|  Final LayerNorm                                                  |
|  Linear (n_embd=384 -> vocab_size=65)                             |
+------------------------------------------------------------------+
  |
  v
Output: probability distribution over next character
```

**Parameter count breakdown** (approximate):
- Token embeddings: 65 x 384 = 24,960
- Position embeddings: 256 x 384 = 98,304
- Per block (x6):
  - Attention: 4 x (384 x 384) = 589,824
  - FeedForward: 384 x 1536 + 1536 x 384 = 1,179,648
  - LayerNorms: 2 x 384 x 2 = 1,536
  - Block subtotal: ~1,771,008
- All 6 blocks: ~10,626,048
- Final LayerNorm: 768
- Output linear: 384 x 65 = 24,960
- **Total: ~10.8M parameters**

---

## Hyperparameters for This Chapter

```python
batch_size = 64        # sequences per training step
block_size = 256       # context window (characters)
max_iters = 5000       # training steps
learning_rate = 3e-4   # peak learning rate
n_embd = 384           # embedding dimension
n_head = 6             # number of attention heads
n_layer = 6            # number of transformer blocks
dropout = 0.2          # dropout rate
eval_interval = 500    # how often to check val loss
eval_iters = 200       # batches to average for loss estimate
warmup_iters = 500     # learning rate warmup steps
```

---

## Think About It

1. **Training loss is 0.8 but validation loss is 1.5. What's happening? What would
   you try?**
   The model is overfitting -- it has memorized training data but doesn't generalize.
   Try: increase dropout, reduce model size (fewer layers or smaller embeddings),
   train for fewer steps (early stopping), or use more data.

2. **Why 6 heads of size 64 (384/6) instead of 64 heads of size 6? What's the
   trade-off?**
   Each head needs enough dimensions to compute meaningful attention patterns. A head
   with only 6 dimensions can barely represent anything useful. 64 dimensions per head
   lets each head learn rich, distinct patterns (one for syntax, one for word
   boundaries, etc.). The trade-off: fewer heads means fewer independent attention
   patterns, but each pattern is more expressive.

3. **What happens if you set dropout to 0.8? To 0.0? When is each appropriate?**
   At 0.8, you destroy 80% of information at each layer -- the model can barely learn
   anything (severe underfitting). At 0.0, there's no regularization -- fine for
   small models or large datasets, but risky for overfitting. Use 0.0 when you have
   tons of data relative to model size, or during inference. Use 0.1-0.3 for most
   training scenarios.

4. **Why is our learning rate (3e-4) smaller than chapter 1's (1e-3)? Hint: think
   about the model size.**
   Bigger models need smaller learning rates. With 10M parameters (vs ~10K in ch1),
   each gradient update affects many more interacting parameters. A large step in one
   parameter can cascade through millions of connections. The "loss landscape" of a
   larger model is more complex and requires more careful navigation.

---

## MPS-Specific Notes (Apple Silicon)

This chapter uses a larger model that can benefit from GPU acceleration. MPS
(Metal Performance Shaders) on Apple Silicon generally works well, but keep these
things in mind:

- **Training time**: Expect ~5-10 minutes on M1/M2/M3 Macs. On CPU, it might take
  20-30 minutes.
- **Memory**: The model is ~10M parameters. With batch_size=64 and block_size=256,
  you might need 2-4 GB of GPU memory. If you hit memory issues, reduce batch_size
  to 32.
- **Potential MPS issues**: Some operations (like certain attention implementations)
  may produce slightly different results on MPS vs CPU. If you get NaN losses or
  weird behavior, try:
  ```python
  device = "cpu"  # fall back to CPU
  ```
- **Monitoring**: Watch Activity Monitor -> GPU to confirm MPS is being utilized.
  If GPU usage is 0%, your tensors might not be on the right device.

---

## Further Reading

- **Andrej Karpathy's "Let's build GPT"** (1:28:00 to end): Covers exactly this
  scaling step. Watch how he goes from the single-block model to the full model
  and discusses each hyperparameter choice.
  https://www.youtube.com/watch?v=kCc8FmEb1nY

- **Chinchilla Scaling Laws** (Hoffmann et al., 2022): "Training Compute-Optimal
  Large Language Models." Establishes how to optimally trade off model size vs
  training data vs compute. Key insight: most models are too big and trained on too
  little data. For our tiny model, we're data-limited (only 1MB of Shakespeare), so
  we can't follow Chinchilla exactly, but the principles apply.
  https://arxiv.org/abs/2203.15556

- **Scaling Laws for Neural Language Models** (Kaplan et al., 2020): The predecessor
  to Chinchilla. Shows smooth power-law relationships between model size, dataset
  size, compute, and loss.
  https://arxiv.org/abs/2001.08361

---

## What's Next

After this chapter, you'll have a model that generates recognizable Shakespeare.
In Chapter 5, we'll explore different generation strategies -- temperature, top-k,
top-p sampling -- to control the quality and creativity of the output.
