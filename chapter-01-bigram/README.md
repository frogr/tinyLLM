# Chapter 1 — The Bigram Model

## Context

This is where we start. No prior chapters, no existing model. We're going to build the
simplest possible language model: one that looks at a single character and predicts what
comes next. It's dumb, but it works well enough to teach us the entire training pipeline.

By the end of this chapter, you'll have a model that generates text. It'll be gibberish,
but it'll be *Shakespeare-flavored* gibberish. And you'll understand every line of code
that produces it.

## Concepts

### What Is a Language Model?

A language model is a program that predicts the next token (for us, a character) given
some context. That's it. ChatGPT, Claude, GPT-4 — at their core, they're doing this same
thing, just with much bigger models and word-pieces instead of characters.

Think of autocomplete on your phone. It looks at what you've typed and predicts the next
word. We're building the simplest version of that.

### Character-Level Tokenization

Before a model can process text, we need to convert characters to numbers. Computers
don't understand letters — they understand numbers.

**Tokenization** is the process of converting text to numbers and back. For us, it's simple:

```python
# Our "vocabulary" — every unique character in Shakespeare
chars = sorted(set(text))  # e.g., ['\n', ' ', '!', ..., 'a', 'b', ..., 'z']
vocab_size = len(chars)     # typically 65 characters

# Two lookup tables (like a bidirectional hash map)
stoi = {ch: i for i, ch in enumerate(chars)}  # string-to-integer: {'a': 0, 'b': 1, ...}
itos = {i: ch for i, ch in enumerate(chars)}  # integer-to-string: {0: 'a', 1: 'b', ...}

# Encode and decode
encode = lambda s: [stoi[c] for c in s]       # "hello" -> [46, 43, 50, 50, 53]
decode = lambda l: ''.join([itos[i] for i in l])  # [46, 43, 50, 50, 53] -> "hello"
```

Real LLMs use fancier tokenization (BPE, which we'll see in Chapter 6), but character-level
is perfect for learning because there's zero magic.

### Tensors

A **tensor** is just a multi-dimensional array. If you know NumPy arrays, tensors are the
PyTorch equivalent. They're the fundamental data structure — everything flows through
the model as tensors.

```
Scalar (0D tensor):  5
Vector (1D tensor):  [1, 2, 3]
Matrix (2D tensor):  [[1, 2], [3, 4]]
3D tensor:           [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
```

The key difference from NumPy arrays: tensors track their own computation history so
PyTorch can automatically compute gradients (more on this below).

### What Is an Embedding?

An **embedding** is a lookup table that converts an integer ID into a vector of numbers.

Think of it like a database table:

```
Character ID | Embedding Vector (learned)
-------------|---------------------------
0  ('a')     | [0.12, -0.34, 0.56, ...]
1  ('b')     | [-0.23, 0.45, 0.78, ...]
2  ('c')     | [0.67, -0.89, 0.11, ...]
```

In our bigram model, the embedding IS the model. We have a `vocab_size × vocab_size`
table. When we look up character `i`, we get a vector of `vocab_size` numbers — and those
numbers represent the model's belief about which character comes next.

```python
# This is literally the entire model:
self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

# When we feed in character 'h' (say, index 46):
# We get back a vector of 65 numbers (one per possible next character)
# Higher numbers = model thinks that character is more likely to come next
```

It's like a hash map where the values are learned during training.

### The Training Loop

Training a neural network follows the same pattern every time. It's like a game loop
in game development:

```
while not done:
    1. Forward pass  — Feed data through the model, get predictions
    2. Compute loss  — How wrong were the predictions? (a single number)
    3. Backward pass — Compute gradients (which direction to adjust each parameter)
    4. Update        — Nudge each parameter a tiny bit to reduce the loss
```

Let's break each step down.

### Forward Pass

Feed input through the model, get output. In our case:

```python
# Input: a batch of character sequences     shape: (batch_size, block_size)
# Output: predictions for each next char     shape: (batch_size, block_size, vocab_size)
logits = model(x)
```

**Logits** are the raw output numbers before they become probabilities. Think of them as
"scores" — higher score means the model thinks that option is more likely.

### Cross-Entropy Loss

**Loss** is a single number that measures how wrong the model is. Lower is better.

**Cross-entropy loss** is the standard loss function for classification tasks (and predicting
the next character IS a classification task — we're classifying which of the 65 characters
comes next).

Here's the intuition: if the model is 100% confident in the right answer, loss is 0.
If it gives equal probability to everything, loss is `ln(vocab_size)` ≈ 4.17 for 65 chars.
If it's confident in the WRONG answer, loss is very high.

```python
loss = F.cross_entropy(logits, targets)
# logits shape:  (B*T, vocab_size) — model's scores for each character
# targets shape: (B*T,)            — the actual next characters
# loss:          single number      — how wrong we are, on average
```

**Why cross-entropy specifically?** It's the mathematically natural way to measure
"how surprised am I by the correct answer given the probabilities I assigned?" If you
assigned high probability to the right answer, you're not surprised (low loss). If you
assigned low probability, you're very surprised (high loss).

At the start of training, the model is random, so loss should be approximately
`-ln(1/65)` ≈ 4.17 (the model gives roughly equal probability to all 65 characters).
If your initial loss is way higher than this, something is wrong.

### What .backward() Actually Does

This is where people often just cargo-cult the code. Let's actually understand it.

When you do computations with tensors, PyTorch secretly builds a **computation graph** —
a record of every operation you performed. When you call `loss.backward()`, PyTorch walks
this graph backwards and computes, for every parameter in the model: "if I increase this
parameter by a tiny amount, how much does the loss change?"

That rate of change is the **gradient**. It's stored in `parameter.grad`.

```python
loss.backward()

# Now every parameter has a .grad attribute
# param.grad tells you: "increase this param → loss goes up/down by this much"
```

This is **backpropagation** — it's just the chain rule from calculus, applied automatically.
You don't need to understand the math to use it, but here's the key insight: gradients
point "uphill" (toward higher loss). We want to go downhill (lower loss), so we subtract.

### The Optimizer Step

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# In the training loop:
optimizer.zero_grad()  # Clear old gradients (they accumulate by default!)
loss.backward()        # Compute new gradients
optimizer.step()       # Update parameters: param -= lr * grad (approximately)
```

**Learning rate** (`lr`) controls how big each step is. Too big and you overshoot.
Too small and training takes forever. `1e-3` (0.001) is a sensible default.

**AdamW** is a specific optimizer algorithm. Think of it like different pathfinding algorithms —
they all get you there, but some are faster. AdamW is the standard choice in 2024.

### Batching — Why We Don't Train on Everything at Once

We don't feed the entire Shakespeare text through the model at once. Instead, we grab
random chunks called **batches**.

```python
batch_size = 32    # How many sequences to process in parallel
block_size = 8     # How many characters per sequence (context length)

# A batch looks like this:
# x (inputs):  32 sequences, each 8 characters long  → shape (32, 8)
# y (targets): 32 sequences, each 8 characters long  → shape (32, 8)
# y is just x shifted right by one position
```

**Why batching?**
1. **Memory** — The whole dataset might not fit in GPU memory
2. **Speed** — GPUs are massively parallel; processing 32 sequences is barely slower than 1
3. **Better learning** — Random batches add noise that actually helps training (this is
   counterintuitive but well-established)

### The "Shifted by One" Thing

This is a common point of confusion. Let's use a tiny example:

```
Text: "hello"
As indices: [46, 43, 50, 50, 53]

Input (x):  [46, 43, 50, 50]  → "hell"
Target (y): [43, 50, 50, 53]  → "ello"
```

Each position in `x` is trying to predict the corresponding position in `y`:
- Given 'h', predict 'e'
- Given 'e', predict 'l'
- Given 'l', predict 'l'
- Given 'l', predict 'o'

The target is always "the next character." That's why `y` is `x` shifted right by one.

### Text Generation

Once trained, generating text is simple:

```python
# Start with a single character (or newline)
# Repeat:
#   1. Feed current sequence into model → get logits for next character
#   2. Convert logits to probabilities (softmax)
#   3. Sample from the probability distribution
#   4. Append sampled character to sequence
```

**Why softmax?** The model outputs raw scores (logits) which can be any number — negative,
positive, huge, tiny. We need probabilities (numbers between 0 and 1 that sum to 1).
Softmax does this conversion:

```python
probs = torch.softmax(logits, dim=-1)
# Input:  [2.0, 1.0, 0.1]
# Output: [0.659, 0.242, 0.099]  — sums to 1.0
```

It's called "softmax" because it's a smooth ("soft") version of taking the maximum.
The highest value gets the most probability, but other options still have a chance.

## Architecture Diagram

```
┌─────────────────────────────────────────────────┐
│                 BIGRAM MODEL                     │
│                                                  │
│  Input: "h" (index 46)                          │
│       │                                          │
│       ▼                                          │
│  ┌──────────────────────┐                        │
│  │   Embedding Table     │  shape: (65, 65)      │
│  │   (the whole model!)  │                       │
│  └──────────┬───────────┘                        │
│             │                                    │
│             ▼                                    │
│  logits: [0.2, -1.1, ..., 3.4]  (65 numbers)   │
│             │                                    │
│             ▼                                    │
│  softmax → probs: [0.01, 0.003, ..., 0.28]      │
│             │                                    │
│             ▼                                    │
│  sample → next character: "e" (index 43)         │
│                                                  │
└─────────────────────────────────────────────────┘
```

This is the simplest possible language model. It only looks at ONE character to predict
the next one. It can't learn that "th" is common or that "q" is usually followed by "u"
— because it only sees one character at a time. We'll fix this in Chapter 2.

## Exercises

Open `exercises.py` and work through the TODOs in order. Each TODO builds on the previous
one. The file is runnable at every stage — incomplete TODOs use placeholder values so
the script won't crash.

1. **TODO 1** — Load and explore the dataset
2. **TODO 2** — Build the character tokenizer (encode/decode)
3. **TODO 3** — Create training/validation splits
4. **TODO 4** — Build the batch loader
5. **TODO 5** — Define the Bigram model
6. **TODO 6** — Implement the forward pass
7. **TODO 7** — Write the training loop
8. **TODO 8** — Generate text from the trained model

## Stretch Goals

- [ ] Plot the training loss over time (use matplotlib)
- [ ] Try different learning rates: 1e-2, 1e-3, 1e-4 — what happens?
- [ ] What's the smallest `batch_size` that still trains? The largest?
- [ ] Count the parameters in your model — how many are there?
- [ ] Try training for 10x more steps — does loss keep going down?

## Further Reading

- [Karpathy "Let's build GPT" video](https://www.youtube.com/watch?v=kCc8FmEb1nY) — 0:00 to ~27:00 covers this chapter
- [PyTorch nn.Embedding docs](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
- [Visual explanation of cross-entropy loss](https://www.youtube.com/watch?v=ErfnhcEV1O8) (StatQuest)

## Think About It

1. **The initial loss sanity check.** Before any training, your model is random. What loss
   value do you expect? (Hint: if the model assigns equal probability to all 65 characters,
   what's `-ln(1/65)`?) Run it and check — if you get something wildly different, something
   is wrong. Why would it be different?

2. **What happens if you set the learning rate to 1.0 instead of 1e-3?** Don't just guess —
   form a hypothesis, then run it. What happens to the loss? Why? (Hint: imagine you're
   walking to the lowest point in a valley, and each step moves you 1000 feet instead of 1.)

3. **The model is predicting the next character, but how does it "know" where it is in a
   word?** Does it? What are the actual limitations of only looking at one character at a
   time? Come up with a concrete example where the bigram model will always fail.

4. **Why does the generated text have realistic-looking word lengths even though the model
   has no concept of "words"?** (Hint: think about what the space character looks like
   to the model and how often it appears in the training data.)

5. **The embedding table is `vocab_size × vocab_size` — that's 65×65 = 4,225 parameters.
   GPT-3 has 175 billion. Our model is ~41 million times smaller.** But it still learns
   something! What exactly *can* a 4,225-parameter model learn about Shakespeare?
   What can't it learn?
