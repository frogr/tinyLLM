# Chapter 1: The Bigram Language Model

## Context

Welcome to the starting point. There are no prior chapters, no prerequisites beyond basic Python, and no complex architectures to worry about. We are building the simplest possible language model: one that predicts the next character using only the current character.

Why start here? Because the entire training pipeline -- data loading, tokenization, model definition, loss computation, backpropagation, optimization, and text generation -- is exactly the same whether you are training a 100-parameter bigram model or a 100-billion-parameter transformer. The only difference is what happens in the middle. By starting simple, we learn the scaffolding without getting lost in the architecture.

By the end of this chapter, you will have a model that generates vaguely Shakespeare-like text. It will not be good. That is the point. Understanding *why* it is not good is what motivates every concept in the chapters that follow.

---

## Concepts

### Character-Level Tokenization

Before a neural network can process text, we need to convert characters to numbers. Neural networks only understand numbers.

Think of it as building a lookup table -- like a hash map from `char` to `int`. Every unique character in the Shakespeare dataset gets an ID. A newline might be `0`, a space might be `1`, `'A'` might be `13`, and so on.

```python
# We build two dictionaries:
stoi = {'a': 0, 'b': 1, 'c': 2, ...}   # string to integer
itos = {0: 'a', 1: 'b', 2: 'c', ...}   # integer to string

# Encoding: "abc" -> [0, 1, 2]
# Decoding: [0, 1, 2] -> "abc"
```

This is the simplest tokenization scheme. Real LLMs like GPT use more sophisticated tokenizers (BPE, SentencePiece) that work on sub-word chunks, but character-level tokenization is perfect for learning.

### Embeddings

An embedding is a learnable lookup table. If tokenization is the hash map that gives each character a number, the embedding is like a database row for each token -- except the database learns its own values during training.

```python
# nn.Embedding(vocab_size, embedding_dim)
# For our bigram model: nn.Embedding(65, 65)
# Each row is a vector of 65 numbers (one for each possible next character)
```

Before training, the embedding table is filled with random numbers. After training, each row encodes what the model has learned about that character. In the bigram model, each row directly represents "given this character, here is the probability distribution over what comes next."

### The Bigram Model

A bigram model predicts the next character using ONLY the current character. No memory of what came before. No lookahead. Just: "I see an 'H', so the next character is probably 'e'."

It is like autocomplete that only looks at the last letter you typed.

This is an extreme limitation. The model cannot learn that "th" is often followed by "e" because when it sees "h", it has already forgotten the "t". But this constraint is what makes the model simple enough to understand completely, and it motivates the attention mechanism we build in Chapter 2.

### Tensors

Tensors are PyTorch's version of arrays. They are the fundamental data structure for all deep learning.

- A **0D tensor** (scalar) is a single number: `torch.tensor(42)`
- A **1D tensor** is a list: `torch.tensor([1, 2, 3])`
- A **2D tensor** is a spreadsheet: `torch.tensor([[1, 2], [3, 4]])`
- A **3D tensor** is a stack of spreadsheets: shape `(batch, time, channels)`

In this chapter, our key tensor shapes are:

| Tensor | Shape | What it means |
|--------|-------|---------------|
| Input batch `x` | `(B, T)` | B sequences, each T characters long |
| Target batch `y` | `(B, T)` | Same shape, shifted by one position |
| Logits | `(B, T, C)` | For each position, a score for each possible next character |

Where `B` = batch size, `T` = block size (time), `C` = vocab size (channels).

### Loss Function (Cross-Entropy)

The loss function measures how wrong the model is. Lower loss = better predictions.

Imagine the model says "the next character is 40% likely to be 'e', 30% likely to be 't', 10% likely to be 'a', ..." Cross-entropy measures how far off those percentages are from reality.

If the correct answer is 'e' and the model assigned 40% probability to 'e', the loss is `-log(0.4) = 0.916`. If the model had assigned 99% to 'e', the loss would be `-log(0.99) = 0.01`. Perfect confidence = zero loss.

The formula:

```
Loss = -log(p(correct class))
```

Or more formally, for a batch:

```
L = -(1/N) * sum(log(p_i(y_i)))
```

Where `p_i(y_i)` is the probability the model assigned to the correct character at position `i`.

In PyTorch, this is one line:

```python
loss = F.cross_entropy(logits, targets)
```

PyTorch's `cross_entropy` handles the softmax internally, so you pass in raw logits (unnormalized scores), not probabilities.

### Backpropagation / `.backward()`

Backpropagation is how the model learns from being wrong. It is like a chain of blame: the loss tells each parameter "here is how much YOU contributed to the error, and which direction you should nudge yourself."

When you call `loss.backward()`, PyTorch walks backward through every operation that produced the loss, computing the gradient (partial derivative) of the loss with respect to every learnable parameter. These gradients are stored in each parameter's `.grad` attribute.

The gradient tells you two things:
1. **Direction**: Should this parameter go up or down to reduce the loss?
2. **Magnitude**: How much does this parameter affect the loss?

### The Training Loop

The training loop is the heartbeat of machine learning. It is a feedback loop:

1. **Forward pass**: Feed data through the model to get predictions
2. **Compute loss**: Measure how wrong the predictions are
3. **Backward pass**: Compute gradients (how to fix the error)
4. **Optimizer step**: Actually update the parameters

Think of it as: guess, check answer, learn from mistakes, repeat.

```python
for step in range(num_steps):
    # 1. Get a batch of data
    x, y = get_batch('train')

    # 2. Forward pass: predict
    logits, loss = model(x, y)

    # 3. Backward pass: compute gradients
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # 4. Update parameters
    optimizer.step()
```

The `optimizer.zero_grad()` call is necessary because PyTorch accumulates gradients by default. Without zeroing, gradients from previous steps would pile up.

### Optimizer (AdamW)

The optimizer is the thing that actually updates the model's parameters. It takes the gradients computed by `.backward()` and uses them to nudge each parameter in the right direction.

**SGD** (Stochastic Gradient Descent) is the simplest optimizer: `param = param - lr * grad`. But it treats every parameter the same.

**AdamW** is smarter. It maintains a running average of gradients and squared gradients for each parameter, effectively adapting the learning rate per-parameter. Parameters that have been getting consistent gradients get bigger updates. Parameters with noisy gradients get smaller updates.

The **learning rate** is the step size. Too large and the model overshoots (loss explodes). Too small and the model barely learns (loss decreases painfully slowly). `1e-3` (0.001) is a solid default.

### Batch Loading

We train on random chunks of text, not the whole dataset at once. Why?

1. **Memory**: The whole dataset might not fit in GPU memory.
2. **Speed**: Computing gradients on a small batch is much faster than on the full dataset.
3. **Noise is good**: The randomness in mini-batch sampling actually helps the model escape bad local minima. It is like studying with flashcards instead of re-reading the whole textbook -- the shuffled order helps you learn the patterns rather than the sequence.

Each batch contains `batch_size` independent sequences, each `block_size` characters long, sampled from random positions in the text.

---

## The Math

### Cross-Entropy Loss

The cross-entropy loss for a single prediction:

```
L = -log(softmax(logits)[correct_class])
```

Where softmax converts raw logits to probabilities:

```
softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)
```

In PyTorch:

```python
# These two are equivalent:
loss_manual = -torch.log(F.softmax(logits, dim=-1)[correct_class])
loss_pytorch = F.cross_entropy(logits, target)  # more numerically stable
```

### The "Shifted By One" Concept

This is the key insight for language modeling. The target is always the input shifted by one position:

```
Input:  "Hello" -> [H, e, l, l, o]
Target: "ello!" -> [e, l, l, o, !]

The model sees 'H' and should predict 'e'
The model sees 'e' and should predict 'l'
The model sees 'l' and should predict 'l'
The model sees 'l' and should predict 'o'
The model sees 'o' and should predict '!'
```

In code:

```python
data = [18, 47, 56, 57, 58, 1, 15, 47, 58]
x = data[0:8]   # [18, 47, 56, 57, 58,  1, 15, 47]  (input)
y = data[1:9]   # [47, 56, 57, 58,  1, 15, 47, 58]  (target)
```

Position 0: input=18, target=47 (model sees token 18, should predict token 47)
Position 1: input=47, target=56 (model sees token 47, should predict token 56)
And so on.

### What Loss to Expect

For a random model with vocabulary size 65, each character has equal probability `1/65`. The expected loss is:

```
-log(1/65) = log(65) ≈ 4.17
```

After training, the bigram model should get down to about 2.47. This is much better than random, but still far from optimal -- the model has no context beyond the current character.

---

## Architecture Diagram

```
Character -> Token ID -> Embedding Table -> Logits -> Softmax -> Next Character
   'H'    ->    45     -> [0.2, -0.1, ...] -> [1.2, 0.3, ...] -> [0.15, 0.08, ...] -> 'e'
```

More detailed view:

```
                    ┌──────────────────────────────────────────────┐
                    │           Embedding Table (65 x 65)          │
                    │                                              │
                    │   Row 0:  [ 0.12, -0.34,  0.56, ... ]       │
                    │   Row 1:  [-0.21,  0.45, -0.12, ... ]       │
                    │   Row 2:  [ 0.33,  0.11,  0.78, ... ]       │
                    │   ...                                        │
                    │   Row 45: [ 1.20,  0.30, -0.50, ... ]  <--- │
                    │   ...                                        │
                    │   Row 64: [-0.15,  0.62,  0.04, ... ]       │
                    └──────────────────────────────────────────────┘
                                        │
   Input: 'H'                          │
      │                                │
      ▼                                ▼
   Token ID: 45  ──────────────>  Look up row 45
                                        │
                                        ▼
                              Logits: [1.2, 0.3, -0.5, ...]
                                        │
                                        ▼
                              Softmax: [0.15, 0.08, 0.02, ...]
                                        │
                                        ▼
                              Sample from distribution
                                        │
                                        ▼
                              Next token: 'e' (index 50)
```

The training flow:

```
   ┌─────────────┐
   │  Get Batch   │ ◄──── Random chunks from training data
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │ Forward Pass │ ◄──── Input through model, get logits
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │ Compute Loss │ ◄──── Cross-entropy between logits and targets
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │   Backward   │ ◄──── Compute gradients for all parameters
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │  Optimizer   │ ◄──── Update parameters using gradients
   │    Step      │
   └──────┬──────┘
          │
          └──────────────► Repeat for max_iters steps
```

---

## Exercises

The exercises in `exercises.py` walk you through building the complete bigram model. Here is what each TODO asks you to do:

### TODO 1: Read the Data File
Standard Python file I/O. Read the Shakespeare text file into a string. This is not ML-specific -- just getting the raw data.

### TODO 2: Build the Vocabulary
Create the character-to-integer and integer-to-character mappings. You need `stoi` (string to int) and `itos` (int to string). These are your encoder and decoder.

### TODO 3: Encode the Dataset
This is already mostly done for you. The key line converts the string to a tensor of integers. The 90/10 train/val split is provided.

### TODO 4: Implement the Batch Loader
This is the most conceptually important TODO. You need to:
1. Pick random starting positions
2. Extract input chunks (x) and target chunks (y), where y is x shifted by one
3. Stack them into tensors and move to the GPU

### TODO 5: Create the Embedding Table
One line of code. `nn.Embedding(vocab_size, vocab_size)` creates a learnable lookup table where each row represents "given this character, here are the scores for what comes next."

### TODO 6: Implement the Forward Pass
Look up embeddings and optionally compute the loss. The reshaping from `(B, T, C)` to `(B*T, C)` is the trickiest part -- PyTorch's cross-entropy expects specific shapes.

### TODO 7: Implement Text Generation
The generation loop: forward pass, take last position's logits, softmax, sample, append. This is autoregressive generation -- each new token becomes part of the input for the next prediction.

### TODO 8: Create the Optimizer
One line: `torch.optim.AdamW(model.parameters(), lr=learning_rate)`.

### TODO 9: Implement the Training Loop
The four-step cycle: forward, zero_grad, backward, step. This pattern is identical in every PyTorch project.

---

## Stretch Goals

Once you have the basic model working, try these experiments:

### 1. Learning Rate Experiments
Try different learning rates (0.1, 0.01, 0.001, 0.0001) and plot the loss curves. You should see:
- `lr=0.1`: Loss might explode or oscillate wildly
- `lr=0.01`: Faster convergence but possibly unstable
- `lr=0.001`: Smooth, steady decrease (the default)
- `lr=0.0001`: Very slow but very stable convergence

```python
import matplotlib.pyplot as plt
# Track losses for each learning rate and plot them
```

### 2. Block Size Experiments
What happens with `block_size = 1`? What about `block_size = 64`? For a bigram model, the block size should not matter much (since it only looks at the current character), but it affects how many training examples you get per batch.

### 3. Compare to Raw Bigram Frequencies
Count bigram frequencies directly from the text (how often does 'e' follow 'H'?) and compare to what the model learned. They should be similar -- the bigram model is essentially learning these frequencies.

```python
# Count bigrams manually
bigram_counts = {}
for i in range(len(text) - 1):
    bigram = (text[i], text[i+1])
    bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1
```

---

## Think About It

These questions are meant to deepen your understanding. Try to answer them before looking at the hints.

1. **"What happens to the loss if you double the vocabulary size but keep everything else the same? Why? Try it."**
   Think about the formula: `-log(1/vocab_size)`. If you double the vocab, random chance gives each token half the probability, which means higher loss. The initial random loss would go from `log(65) ≈ 4.17` to `log(130) ≈ 4.87`.

2. **"The model is predicting the next character, but how does it know where it is in a word? Does it?"**
   It does not. That is the point of Chapter 2. The bigram model has no concept of position. It treats the 'e' at the start of "every" exactly the same as the 'e' in the middle of "the". Positional embeddings and attention fix this.

3. **"If you set the learning rate to 1.0 instead of 1e-3, what do you think happens? Run it and see."**
   The loss will likely explode. Large learning rates cause the optimizer to overshoot the minimum, bouncing around chaotically. You might see `nan` (not a number) values.

4. **"Why does the generated text have realistic word lengths even though the model has no concept of 'words'?"**
   Because the model learns bigram statistics, including how often spaces follow each character. In English, certain characters (like vowels after consonants) are commonly followed by spaces, which naturally creates word-like breaks.

5. **"What's the theoretical minimum loss for this model?"**
   The theoretical minimum is the entropy of the true bigram distribution. Even a perfect bigram model cannot get loss to zero because the next character is not deterministic -- 'h' might be followed by 'e', 'a', 'i', or 'o'. The best possible loss is around 2.45 for this dataset.

---

## Further Reading

- **Andrej Karpathy's "Let's build GPT from scratch"** (YouTube): Timestamps 0:00 - 47:00 cover the bigram model. This chapter follows that section closely.
  https://www.youtube.com/watch?v=kCc8FmEb1nY

- **3Blue1Brown: Neural Networks** (YouTube series): Outstanding visual explanations of what neural networks are, how gradient descent works, and what backpropagation is doing.
  https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

- **PyTorch Tutorials**: The official "60-minute blitz" is a solid intro to tensors and autograd.
  https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html

- **The Unreasonable Effectiveness of Recurrent Neural Networks** (Andrej Karpathy's blog): The original inspiration for character-level language models. Shows what happens when you scale up from bigrams.
  https://karpathy.github.io/2015/05/21/rnn-effectiveness/
