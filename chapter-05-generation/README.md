# Chapter 5: Generation and Sampling Strategies

## Where We Are

We have a trained transformer model that produces decent Shakespeare. In Chapter 4, we scaled
it up and got text that actually looks like verse. But we've been using **vanilla sampling** — the
model outputs probabilities and we sample directly from that distribution.

Now we learn to **control** the generation: temperature, top-k, and top-p sampling. These are the
knobs behind the "creativity slider" in ChatGPT and every other LLM product you've used. When
you move that slider from "More Precise" to "More Creative," you're adjusting exactly the
parameters we'll implement in this chapter.

This chapter is different from the previous ones. We won't be changing the model architecture.
Instead, we'll focus entirely on what happens **after** the model produces its output but
**before** we pick the next token.

---

## Core Concepts

### Logits vs Probabilities

The model's final layer outputs a vector of raw scores — one number per token in the vocabulary.
These raw scores are called **logits**. They can be any real number: positive, negative, large,
small.

To turn logits into probabilities, we apply **softmax**:

```
probability_i = exp(logit_i) / sum(exp(logit_j) for all j)
```

This gives us a proper probability distribution: all values between 0 and 1, summing to 1.

The key insight for this chapter: **between the logits and the softmax, we can manipulate the
scores to change the output behavior.** The model's "opinion" is fixed — it's already done its
forward pass. But we get to decide how to interpret that opinion.

```
Model Output (Logits) → [ WE MANIPULATE HERE ] → Softmax → Probabilities → Sample
```

### Temperature

Temperature is the simplest and most intuitive control. Before applying softmax, we **divide
all logits by a temperature value T**:

```python
scaled_logits = logits / temperature
probs = softmax(scaled_logits)
```

What does this do?

- **T = 1.0**: Normal behavior. The probabilities are exactly what the model learned.
- **T < 1.0** (e.g., 0.5): Makes the distribution **sharper**. High-probability tokens get
  even higher probability, low-probability tokens get even lower. The model becomes more
  confident, more repetitive, more "safe." It doubles down on its best guesses.
- **T > 1.0** (e.g., 1.5): Makes the distribution **flatter**. Probabilities spread out more
  evenly. The model becomes more random, more creative, but also more prone to errors and
  nonsense.
- **T → 0**: Approaches **greedy decoding** — always pick the single most likely token.
  Deterministic output. Often leads to repetitive loops.
- **T → ∞**: Approaches **uniform random** — every token is equally likely. Pure noise.

Think of temperature as a **confidence dial**. Low temperature means "be very sure before you
commit." High temperature means "take risks, surprise me."

#### Intuition: Why Division?

When you divide logits by a small number (T=0.1), the differences between them get amplified.
If the top logit was 5.0 and the second was 4.0, after dividing by 0.1 they become 50.0 and
40.0 — a much bigger gap in softmax land.

When you divide by a large number (T=5.0), the differences shrink. 5.0 and 4.0 become 1.0
and 0.8 — much more similar after softmax.

### Top-k Sampling

Instead of sampling from **all** tokens in the vocabulary, only consider the **top k most
likely** tokens. Set everything else to probability zero.

```python
# Only keep the top k logits, set the rest to -infinity
top_k_logits = keep_only_top_k(logits, k)
probs = softmax(top_k_logits)
next_token = sample(probs)
```

- **k = 1**: Greedy decoding (only one choice).
- **k = vocab_size**: Unrestricted (all tokens are candidates).
- **k = 10**: Only the 10 most likely tokens can be chosen.

This prevents the model from ever picking a very unlikely token. It's like restricting a
spell-checker to only suggest the top 10 candidates — you cut off the long tail of weird
possibilities.

The downside: k is fixed regardless of context. Sometimes the model is very confident (one
token has 95% probability), and k=50 still allows 49 unlikely tokens. Other times the model
is uncertain (top 100 tokens each have ~1%), and k=10 cuts off perfectly reasonable options.

### Top-p (Nucleus) Sampling

Top-p fixes the rigidity problem of top-k. Instead of a fixed count, we pick the **smallest
set of tokens whose cumulative probability adds up to p**.

```python
# Sort tokens by probability (descending)
# Keep adding tokens until their probabilities sum to >= p
# Set everything else to -infinity
```

- **p = 0.9**: Keep tokens that account for 90% of the probability mass.
- **p = 1.0**: Keep everything (unrestricted).
- **p = 0.1**: Very restrictive — only the very top tokens.

The magic of top-p is that it **adapts to the model's confidence**:

- If the model is very confident (one token has 90% probability), the nucleus might be just
  1-2 tokens. We don't waste time on unlikely alternatives.
- If the model is uncertain (many tokens each have a few percent), the nucleus might be 20+
  tokens. We allow the model to explore.

This adaptivity is why top-p is generally considered better than top-k, and why it's the
default in most production LLM systems.

The name "nucleus sampling" comes from the Holtzman et al. paper — the "nucleus" is the
core set of high-probability tokens.

### How These Combine

In practice, these strategies are applied in sequence:

```
Logits
  → Divide by temperature          (reshape the distribution)
  → Apply top-k filter             (cut off unlikely tokens)
  → Apply top-p filter             (adapt to model confidence)
  → Softmax                        (get final probabilities)
  → Sample from distribution       (pick a token)
```

Temperature is always applied first because it changes the relative probabilities, which
affects what top-k and top-p will select.

You can use just temperature, just top-k, just top-p, or any combination. Common production
settings:

| Use Case | Temperature | Top-k | Top-p | Why |
|----------|-------------|-------|-------|-----|
| Code completion | 0.2 | — | 0.95 | Precision matters, few correct answers |
| Chat | 0.7 | — | 0.9 | Balance of coherence and variety |
| Creative writing | 1.0 | — | 0.95 | Allow surprising word choices |
| Brainstorming | 1.2 | 50 | — | Want diverse, unexpected ideas |
| Translation | 0.3 | — | 0.9 | Very few correct translations |

### Mapping to Products

When ChatGPT has a "temperature" slider, this is literally what it's doing — dividing logits
by a number before softmax. When you see "More Creative" vs "More Precise," that's temperature.

Top-p is typically set behind the scenes. OpenAI's API exposes both `temperature` and `top_p`
as parameters. Anthropic's Claude API exposes `temperature` and `top_k`. These are the exact
same algorithms you'll implement in this chapter.

The reason these exist as user-facing controls: different tasks genuinely need different
settings. You don't want your code assistant to be "creative" with syntax, and you don't
want your story generator to be boringly predictable.

---

## The Generation Pipeline

Here's the complete picture of what happens when a model generates one token:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Token Generation Pipeline                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input Tokens ──→ [ Transformer Model ] ──→ Raw Logits              │
│  (context)         (frozen, already trained)  (one per vocab token)  │
│                                                                     │
│  Raw Logits                                                         │
│      │                                                              │
│      ▼                                                              │
│  ┌──────────────────┐                                               │
│  │  ÷ Temperature   │  Reshape distribution (sharper or flatter)    │
│  └──────┬───────────┘                                               │
│         ▼                                                           │
│  ┌──────────────────┐                                               │
│  │  Top-k Filter    │  Keep only top k tokens (optional)            │
│  └──────┬───────────┘                                               │
│         ▼                                                           │
│  ┌──────────────────┐                                               │
│  │  Top-p Filter    │  Keep nucleus of tokens summing to p          │
│  └──────┬───────────┘                                               │
│         ▼                                                           │
│  ┌──────────────────┐                                               │
│  │  Softmax         │  Convert to probabilities                     │
│  └──────┬───────────┘                                               │
│         ▼                                                           │
│  ┌──────────────────┐                                               │
│  │  Sample          │  Randomly pick one token                      │
│  └──────┬───────────┘                                               │
│         ▼                                                           │
│  Output Token ──→ Append to input ──→ Repeat                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

Each generated token is appended to the context, and the whole process repeats. This is
**autoregressive generation** — the model generates one token at a time, each conditioned on
all previous tokens.

---

## Think About It

Before you start coding, consider these questions. Come back to them after implementing
each sampling strategy.

1. **Temperature extremes**: Set temperature to 0.01 and generate. Then set it to 2.0.
   What's the trade-off? At what point does the output become "too safe" or "too wild"?
   Is there a sweet spot for Shakespeare?

2. **Top-k vs top-p**: Top-k=5 and top-p=0.9 — which is more restrictive? Does it depend
   on the context? Think about a case where the model is very confident vs. very uncertain.
   Which strategy handles both cases better?

3. **Why not greedy?**: Why not just always use greedy decoding (pick the most likely token)?
   Try it and see what happens. Generate 500 characters with temperature=0.01 and look for
   repetition. Why does this happen?

4. **Task-specific settings**: If you were building a code completion tool vs. a creative
   writing tool, what settings would you use? What about a chatbot that needs to be factual?
   What about a game that generates NPC dialogue?

5. **Combining strategies**: What happens if you use temperature=0.5 with top-p=0.9?
   Is that different from temperature=1.0 with top-p=0.5? Both restrict the output, but
   in different ways. Which feels better?

---

## What You'll Implement

In `exercises.py`, you'll:

1. **TODO 1**: Implement temperature scaling — divide logits by temperature before sampling
2. **TODO 2**: Implement top-k sampling — zero out everything except the top k logits
3. **TODO 3**: Implement top-p (nucleus) sampling — keep the smallest set summing to p
4. **TODO 4**: Build a comparison function that generates text with different settings

The model code from Chapter 4 is provided complete (no TODOs) with smaller hyperparameters
so training is fast. The focus of this chapter is entirely on the generation strategies.

---

## Further Reading

- **Holtzman et al., "The Curious Case of Neural Text Degeneration" (2020)**
  https://arxiv.org/abs/1904.09751
  The paper that introduced nucleus (top-p) sampling. Shows why vanilla sampling and top-k
  both produce degenerate text, and why top-p fixes it. Highly readable.

- **Karpathy's nanoGPT generate.py**
  https://github.com/karpathy/nanoGPT/blob/master/generate.py
  Clean reference implementation of temperature + top-k sampling.

- **How to generate text: using different decoding methods (Hugging Face blog)**
  https://huggingface.co/blog/how-to-generate
  Excellent visual explanations of greedy, beam search, top-k, and top-p sampling.

- **OpenAI API documentation — Temperature and top_p**
  https://platform.openai.com/docs/api-reference/chat/create
  See how these exact parameters are exposed in production APIs.

---

## Running This Chapter

```bash
# Make sure you've downloaded the data
python data/download.py

# Work through the exercises (trains a small model first, then you implement sampling)
python chapter-05-generation/exercises.py

# If you get stuck, check the solution
python chapter-05-generation/solution.py

# Experiment freely
python chapter-05-generation/sandbox.py
```

Training the small model should take 2-5 minutes depending on your hardware. After that,
generation is nearly instant — you can experiment with different settings rapidly.
