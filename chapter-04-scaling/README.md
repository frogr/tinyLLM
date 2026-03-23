# Chapter 4 -- Scaling Up

## Context

In Chapters 1-3, we built every component of a GPT-style transformer: embeddings, self-attention,
multi-head attention, feedforward networks, residual connections, and layer norm. We assembled them
into a single transformer block and trained it. The output was... okay. You could see it was trying
to be Shakespeare, but it was more "fever dream Shakespeare" than "actual Shakespeare."

Now we make it actually work.

The architecture is done. What remains is *scaling* -- making the model bigger, training it
longer, and adding the tricks that prevent it from falling apart when we do. By the end of this
chapter, you'll have a model that produces semi-coherent Shakespeare. Real words, real character
names, something resembling iambic pentameter. Not perfect, but unmistakably Shakespeare.

This is the chapter where it goes from "toy" to "wow, this actually works."

## Concepts

### Stacking Multiple Transformer Blocks (n_layer=6)

In Chapter 3, we had one transformer block. Now we stack six.

Think of it like layers in a deep neural network (because that's literally what it is). Each
layer processes the same sequence, but at a higher level of abstraction:

```
Layer 1: learns character-level patterns     ("th" is common, "q" follows "u")
Layer 2: learns word-level patterns          ("the", "and", common words)
Layer 3: learns phrase patterns              ("my lord", "thou art")
Layer 4: learns syntactic structure          (sentence patterns, line breaks)
Layer 5: learns style patterns               (meter, dramatic structure)
Layer 6: learns high-level composition       (character voices, scene flow)
```

This is a simplification -- in practice, layers don't divide so neatly. But the principle is
real: deeper networks learn hierarchical representations. Early layers learn low-level features,
later layers learn high-level features. This same principle is why deep CNNs work for images
(edges -> textures -> shapes -> objects).

```python
# In Chapter 3, we had:
self.blocks = nn.Sequential(Block(n_embd, n_head=4))

# Now we stack 6:
self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
```

The `*` unpacks the list into separate arguments to `nn.Sequential`. It's the Python equivalent
of spread syntax in JavaScript.

### Increasing Embedding Dimensions (n_embd=384)

In earlier chapters, our embedding dimension was small (maybe 32 or 64). Now we use 384.

The embedding dimension controls how much information each token position can carry. Think of it
as the width of the "data bus" inside your model:

```
n_embd=32:  Each token is represented by 32 numbers
            Like describing a person with 32 attributes -- pretty limited

n_embd=384: Each token is represented by 384 numbers
            Like describing a person with 384 attributes -- much more nuanced
```

More capacity means the model can represent more complex patterns. But more capacity also means
more parameters, more memory, and slower training. It's a tradeoff.

Why 384 specifically? It's the value Karpathy uses in nanoGPT for Tiny Shakespeare. It's large
enough to learn meaningful patterns but small enough to train in a few minutes on a laptop.
It also needs to be divisible by `n_head` (6 heads, 384/6 = 64 dimensions per head).

### Dropout for Regularization

**Dropout** is a technique that randomly sets a fraction of activations to zero during training.

```python
self.dropout = nn.Dropout(0.2)  # Randomly zero out 20% of values during training
```

**The analogy:** Imagine a software team where, each day, a random 20% of the team doesn't show
up. The remaining team members have to cover for them. Over time, this forces every team member
to be broadly capable rather than hyper-specialized. No single person becomes a bottleneck.
The team becomes resilient.

That's exactly what dropout does to a neural network. It prevents the model from becoming
over-reliant on any single neuron or pathway. Each neuron has to be useful on its own because
it can't count on its neighbors being there.

**Why do we need this?** Without dropout, a model with 10M+ parameters will *memorize* the
training data rather than learning general patterns. It'll get near-zero training loss but
terrible validation loss. Dropout is a form of **regularization** -- it constrains the model
to learn more general, robust representations.

**Important:** Dropout is only active during training (`model.train()`). During evaluation and
generation (`model.eval()`), all neurons are active and their outputs are scaled accordingly.
PyTorch handles this automatically.

We add dropout in three places:
1. **After attention weights** -- prevents over-reliance on specific attention patterns
2. **After the feedforward layer** -- prevents over-reliance on specific features
3. **After embedding** -- adds noise to input representations

### Learning Rate Schedules

The learning rate controls how big each parameter update is. But a fixed learning rate isn't
optimal:

- **Too high at the start:** The model jumps around wildly and never converges
- **Too high near the end:** The model keeps overshooting good solutions
- **Too low at the start:** Training takes forever

**The analogy:** Writing an essay. You start with a rough draft (big changes, restructuring whole
paragraphs). Then you refine (smaller changes, rewriting sentences). Then you polish (tiny
changes, fixing individual words). You wouldn't start by polishing, and you wouldn't end by
restructuring.

For our model, we use a constant learning rate of 3e-4 (0.0003) with AdamW. This is
the "safe default" for transformers. AdamW has its own internal adaptive learning rate per
parameter, which provides some of the benefits of a schedule automatically.

For larger models, you'd use a proper schedule:

```
Learning rate
    ^
    |   /\
    |  /  \___________
    | /               \
    |/                 \____
    +-------------------------> Training step
    warmup    constant    decay
```

We skip this for our tiny model -- it trains well enough without it.

### Hyperparameter Tuning

Hyperparameters are the knobs you turn *before* training. The model can't learn them -- you
have to choose them. Here are the main ones and what they do:

| Hyperparameter | Our Value | What It Controls | Turn It Up | Turn It Down |
|----------------|-----------|-----------------|------------|--------------|
| `n_layer`      | 6         | Depth of model (number of transformer blocks) | More abstract reasoning | Faster training |
| `n_embd`       | 384       | Width of model (representation size) | More capacity per layer | Less memory |
| `n_head`       | 6         | Number of attention heads | More diverse attention patterns | Fewer parameters |
| `batch_size`   | 64        | Sequences per training step | More stable gradients | Less memory |
| `block_size`   | 256       | Context length (how far back the model looks) | Longer-range patterns | Faster, less memory |
| `dropout`      | 0.2       | Fraction of neurons dropped | More regularization | Better training loss |
| `learning_rate`| 3e-4      | Step size for optimization | Faster learning (risky) | Slower, more stable |
| `max_iters`    | 5000      | Total training steps | Better convergence | Faster experiments |

**The constraint:** These are not independent. Doubling `n_layer` and `n_embd` quadruples the
parameters and memory usage. `n_embd` must be divisible by `n_head`. Bigger `block_size` means
quadratically more attention computation (because attention is O(T^2)).

**How to tune:** Start with known-good values (like ours), train, look at the loss curves, and
adjust. There's no formula -- it's empirical. The values in this chapter are well-tested for
Tiny Shakespeare.

### Overfitting vs. Underfitting

Two failure modes when training:

**Underfitting** -- the model isn't learning enough:
```
Training loss:   2.5    (high)
Validation loss: 2.5    (high)
Diagnosis: Model is too small, not enough training, or learning rate too low
Fix: Bigger model, more training steps, higher learning rate
```

**Overfitting** -- the model memorized the training data instead of learning patterns:
```
Training loss:   0.1    (very low -- suspiciously low)
Validation loss: 3.0    (much higher than training)
Diagnosis: Gap between train and val loss = overfitting
Fix: More dropout, smaller model, more data, early stopping
```

**The sweet spot:**
```
Training loss:   1.0
Validation loss: 1.1    (slightly higher -- this is normal and healthy)
```

A small gap between training and validation loss is expected and fine. The model will always
perform slightly better on data it trained on. You're looking for a gap that's small and stable,
not one that grows over time.

## Architecture Diagram

```
Input: "To be or not to be" (as character indices)
       [20, 53, 1, 40, 43, 1, 53, 56, ...]
              |
              v
    +---------+---------+
    |  Token Embedding  |  (vocab_size, 384)
    +---------+---------+
              |
              +-----> (+) <--- Position Embedding (256, 384)
              |
              v
         [Dropout 0.2]
              |
    +---------+---------+     --+
    |  Transformer Block |       |
    |  +- Layer Norm     |       |
    |  +- Multi-Head     |       |
    |     Attention (6h) |       |
    |     + Dropout      |       |
    |  +- Residual Add   |       |
    |  +- Layer Norm     |       |     x6 blocks
    |  +- FeedForward    |       |     (stacked)
    |     (384 -> 1536   |       |
    |      -> 384)       |       |
    |     + Dropout      |       |
    |  +- Residual Add   |       |
    +---------+---------+     --+
              |
    +---------+---------+
    |     Layer Norm     |
    +---------+---------+
              |
    +---------+---------+
    |   Linear Head      |  (384, vocab_size)
    +---------+---------+
              |
              v
    logits: scores for each possible next character
    shape: (batch_size, block_size, vocab_size)
```

**Parameter count:**
- Token embedding: 65 x 384 = 24,960
- Position embedding: 256 x 384 = 98,304
- Each transformer block: ~1.2M parameters
- 6 blocks: ~7.2M parameters
- Final layer norm + linear head: ~25K
- **Total: ~10.8M parameters** (vs. 4,225 in Chapter 1!)

## Exercises

Open `exercises.py` and work through the TODOs in order.

1. **TODO 1** -- Build the full transformer model with all components from Chapter 3
2. **TODO 2** -- Add dropout to attention, feedforward, and after embeddings
3. **TODO 3** -- Implement the training loop with loss estimation
4. **TODO 4** -- Generate 1000 tokens and evaluate output quality

## Think About It

1. **We went from 4,225 parameters (Chapter 1) to ~10.8 million (Chapter 4). That's a
   2,500x increase.** Does the output quality improve by 2,500x? Why is the relationship
   between parameter count and output quality not linear? At what point do you think you'd
   see diminishing returns for Tiny Shakespeare?

2. **Dropout randomly removes 20% of activations during training.** Doesn't this make the
   model worse at learning? Why does making the model "forget" some of its neurons actually
   improve performance on new data? What would happen if you set dropout to 0.8 (removing
   80% of activations)?

3. **Our model's context length is 256 characters -- roughly 50 words.** That means when
   generating the 51st word, it's already forgotten the first word. How does this affect the
   quality of generated text? What patterns can a 256-character window capture, and what
   can't it capture? (Think about plays: character names, scene structure, plot.)

4. **Training loss goes down but validation loss starts going up.** You've seen this pattern
   if you've ever overfit a model. But what exactly is the model *doing* when it overfits?
   Is it memorizing specific Shakespeare passages? Specific character sequences? How would
   you test what it's memorized vs. what it's generalized?

5. **We use 6 attention heads, each with 64 dimensions (384/6).** What if we used 1 head
   with 384 dimensions instead? Or 384 heads with 1 dimension each? Why do multiple heads
   help? (Hint: think about what different heads might specialize in -- some might track
   word boundaries, others might track rhyme patterns, others might track character names.)

## Common Struggles

### "Training is taking forever"
On CPU, this model will take 10-20 minutes. On Apple Silicon (MPS), 2-5 minutes. On an
NVIDIA GPU (CUDA), under 2 minutes. If it's taking longer than expected:
- Make sure you're using the right device (check the "Using device:" print at the top)
- Reduce `max_iters` to 2000 for a faster test run (quality will be worse but you can verify
  things work)
- Reduce `n_embd` to 128 and `n_layer` to 4 for a much faster but lower-quality model

### "MPS out of memory" or MPS errors
Apple's MPS backend is good but not as mature as CUDA. If you get errors:
- Reduce `block_size` from 256 to 128
- Reduce `batch_size` from 64 to 32
- If all else fails, use `device = "cpu"` -- it's slower but always works

### "How do I know if my hyperparameters are good?"
Look at the validation loss. For Tiny Shakespeare with our architecture:
- Loss ~4.2 at the start (random, expected)
- Loss ~2.5 after a few hundred steps (learning basic character frequencies)
- Loss ~1.5-1.8 after 5000 steps (good -- the model has learned real patterns)
- If loss plateaus above 2.0, something is wrong (check learning rate, model size)
- If train loss is much lower than val loss (gap > 0.3), you're overfitting

### "The generated text is still garbage"
Make sure you're generating in eval mode (`model.eval()`). Dropout during generation
will make output much worse. Also, 5000 steps is the minimum for decent output -- try
10000 if you have the patience.

## Further Reading

- [Karpathy "Let's build GPT" video](https://www.youtube.com/watch?v=kCc8FmEb1nY) -- 1:28:00 to end covers scaling
- [Dropout paper (Srivastava et al., 2014)](https://jmlr.org/papers/v15/srivastava14a.html) -- the original dropout paper
- [Scaling Laws for Neural Language Models (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361) -- how model performance scales with size
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/) -- great visual reference
