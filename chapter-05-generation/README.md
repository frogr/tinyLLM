# Chapter 5 -- Text Generation Strategies

## Context

In Chapter 4, we trained a real GPT model on Shakespeare. It learned real patterns --
character names, iambic-ish rhythms, dialogue structure. But if you ran generation a
few times, you probably noticed something: the output quality varies wildly between
runs, and sometimes the model gets stuck repeating itself.

That's because how you **sample** from the model matters just as much as how you
**train** it. The model outputs a probability distribution over the next token.
What you do with that distribution -- whether you always pick the most likely token,
or add some randomness, or filter out unlikely options -- completely changes the
output quality.

This is the chapter where we connect to how real LLM products work. When you move
the temperature slider in ChatGPT, you're doing exactly what we'll implement here.

## Concepts

### Greedy Decoding -- Always Pick the Most Likely Token

The simplest strategy: at each step, pick the token with the highest probability.

```python
# Greedy: always take the argmax
idx_next = torch.argmax(probs, dim=-1, keepdim=True)
```

**The analogy:** Greedy decoding is like always taking the highway. It's safe and
predictable, but you'll never discover an interesting side street. And you'll often
end up in loops -- the model says "the", then "the" is likely again, so it says
"the" again, forever.

**When it's useful:** When you want deterministic, factual output. Think code
completion or structured data extraction -- you want the *most likely* answer,
not a creative one.

**The problem:** Repetition. Greedy decoding is notorious for producing text like
"I think I think I think I think..." because once the model enters a high-probability
loop, there's no randomness to break out of it.

### Temperature Scaling -- The Creativity Dial

Temperature is the single most important generation parameter. It controls how
"creative" vs "conservative" the model is.

**How it works:** Before converting logits to probabilities with softmax, divide
them by a temperature value `T`:

```python
# Original:  probs = softmax(logits)
# With temp: probs = softmax(logits / T)
```

**The math, concretely:**

Say the model outputs logits `[2.0, 1.0, 0.5]` for three tokens.

```
Temperature = 1.0 (default, no change):
  softmax([2.0, 1.0, 0.5]) = [0.593, 0.218, 0.189]
  -> Token 0 is ~3x more likely than token 2

Temperature = 0.5 (more conservative):
  softmax([2.0/0.5, 1.0/0.5, 0.5/0.5]) = softmax([4.0, 2.0, 1.0])
  = [0.844, 0.114, 0.042]
  -> Token 0 is ~20x more likely than token 2. Distribution is "sharper."

Temperature = 2.0 (more creative):
  softmax([2.0/2.0, 1.0/2.0, 0.5/2.0]) = softmax([1.0, 0.5, 0.25])
  = [0.420, 0.255, 0.325]  (approximately)
  -> Much more uniform. All tokens have a real chance.
```

**Why this works mathematically:** Dividing by a small number makes big logits
even bigger relative to small ones (amplifying differences). Dividing by a large
number compresses everything toward zero (reducing differences). Softmax then
exaggerates or smooths these differences into probabilities.

**The extremes:**
- `T -> 0`: Becomes greedy decoding (all probability on the top token)
- `T = 1.0`: The model's natural distribution (unchanged)
- `T -> infinity`: Uniform random (every token equally likely)

**This is literally what the temperature slider does in ChatGPT.** When you set
temperature to 0 in the API, you get near-deterministic output. At 2.0, you get
wild creativity. The default is usually around 0.7-1.0.

### Top-k Sampling -- Filter the Long Tail

Even with temperature, there's a problem: the model assigns *some* probability to
every token, including nonsensical ones. "The king rode his qxz..." should never
happen, but with enough randomness, it can.

**Top-k sampling** fixes this by only considering the `k` most likely tokens:

```python
# 1. Find the k tokens with the highest logits
# 2. Set all other logits to -infinity (so they get 0 probability after softmax)
# 3. Sample from the remaining k tokens
```

**The analogy:** You're at a restaurant with 200 items on the menu. Top-k is like
saying "just show me the top 10 options." You still get variety, but you won't
accidentally order something bizarre from page 7.

**Common values:** `k=50` is a popular choice. GPT-2 used `k=40` by default.

**The problem with top-k:** It's not adaptive. Sometimes the model is very confident
(one token has 90% probability) and `k=50` includes way too many unlikely options.
Other times the model is uncertain (probability spread across 100 tokens) and `k=50`
cuts off reasonable options. The right `k` depends on the distribution, which changes
at every step.

### Top-p (Nucleus) Sampling -- Adaptive Filtering

Top-p sampling solves top-k's rigidity by adapting to the actual distribution.

**How it works:** Instead of keeping a fixed number of tokens, keep the smallest
set of tokens whose cumulative probability exceeds `p`:

```python
# 1. Sort tokens by probability (highest first)
# 2. Compute cumulative sum: [0.40, 0.65, 0.80, 0.88, 0.93, ...]
# 3. Keep tokens until cumsum exceeds p (e.g., p=0.9)
# 4. Zero out everything else
# 5. Sample from what remains
```

**Why this is better than top-k:**
- When the model is confident: maybe only 3 tokens are kept (they already cover 90%
  of the probability mass)
- When the model is uncertain: maybe 50 tokens are kept (you need that many to
  reach 90%)

It automatically adjusts to the situation.

**Common values:** `p=0.9` or `p=0.95` are standard. The original nucleus sampling
paper used `p=0.95`.

**The analogy:** Top-k is like "give me exactly 10 options." Top-p is like "give me
enough options to cover 90% of the reasonable choices." The second adapts to whether
the decision is easy or hard.

### How These Map to Real LLM Products

These aren't academic concepts -- they're in every LLM API you'll ever use:

| Product | Parameter | What It Does |
|---------|-----------|-------------|
| OpenAI API | `temperature` | Exactly our temperature scaling |
| OpenAI API | `top_p` | Exactly our nucleus sampling |
| ChatGPT UI | Temperature slider | Temperature scaling |
| Copilot "creative" mode | Higher temperature + top-p | More diverse completions |
| Copilot "precise" mode | Lower temperature | More deterministic output |
| Claude API | `temperature` | Same concept |

When a product says "creative mode" vs "precise mode," they're typically just
adjusting temperature (and sometimes top-p). Now you know exactly what's
happening under the hood.

### Combining Strategies

In practice, you almost always combine these:

```
temperature + top-k:  Apply temperature first, then filter to top-k
temperature + top-p:  Apply temperature first, then apply nucleus sampling
```

Temperature changes the shape of the distribution. Top-k/top-p then trims the
tails. This gives you fine-grained control:

- **Factual Q&A:** `temperature=0.3, top_p=0.9` -- conservative but not rigid
- **Creative writing:** `temperature=1.0, top_p=0.95` -- natural creativity
- **Brainstorming:** `temperature=1.2, top_k=100` -- push for unusual ideas
- **Code generation:** `temperature=0.2, top_p=0.95` -- precise but not brittle

## The Math, In Code

### Temperature Scaling

```python
# logits shape: (B, vocab_size)
# temperature: float > 0

logits = logits / temperature
probs = F.softmax(logits, dim=-1)
idx_next = torch.multinomial(probs, num_samples=1)
```

That's it. One line of actual logic: divide logits by temperature.

### Top-k Sampling

```python
# logits shape: (B, vocab_size)
# top_k: int > 0

# Find the top-k values and the threshold
top_k_values, _ = torch.topk(logits, top_k)         # shape: (B, k)
threshold = top_k_values[:, -1].unsqueeze(-1)         # shape: (B, 1) -- the k-th value
logits[logits < threshold] = float('-inf')             # zero out everything below top-k

probs = F.softmax(logits, dim=-1)
idx_next = torch.multinomial(probs, num_samples=1)
```

### Top-p (Nucleus) Sampling

```python
# logits shape: (B, vocab_size)
# top_p: float in (0, 1]

sorted_logits, sorted_indices = torch.sort(logits, descending=True)
cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

# Find where cumulative probability exceeds p
sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
# Key detail: we subtract the current token's probability so the token
# that CROSSES the threshold is still included

sorted_logits[sorted_mask] = float('-inf')

# Unsort back to original order
logits = sorted_logits.scatter(1, sorted_indices.argsort(1), sorted_logits)
# (or equivalently, scatter back using the sorted indices)

probs = F.softmax(logits, dim=-1)
idx_next = torch.multinomial(probs, num_samples=1)
```

The top-p implementation is the trickiest. The key subtlety: when computing
the mask, you subtract the current token's own probability before comparing
to `p`. This ensures the token that *crosses* the threshold is still included.
Without this, you'd sometimes filter out all tokens when one token already
exceeds `p` by itself.

## Exercises

Open `exercises.py` and work through the TODOs in order:

1. **TODO 1** -- Implement temperature scaling in the generate function
2. **TODO 2** -- Implement top-k sampling
3. **TODO 3** -- Implement top-p (nucleus) sampling
4. **TODO 4** -- Implement combined sampling (temperature + top-k or top-p)
5. **TODO 5** -- Compare outputs across different settings

Each TODO produces visible output so you can see the effect of each strategy.

## Think About It

1. **Temperature = 0.** What happens mathematically when you divide logits by a
   temperature approaching 0? What does the softmax output look like? Why do most
   APIs implement `temperature=0` as argmax rather than actually dividing by 0?

2. **Top-k vs top-p on confident vs uncertain predictions.** Imagine the model is
   very confident: one token has logit 10.0, everything else is below 1.0. Now
   imagine it's uncertain: the top 50 tokens all have logits around 2.0. How does
   top-k=10 behave in each case? How does top-p=0.9 behave? Which adapts better?

3. **Order of operations matters.** If you apply temperature=2.0 and THEN top-k=10,
   you get a different result than top-k=10 and THEN temperature=2.0. Why? Which
   order makes more sense? (Hint: think about what temperature does to the ranking
   of tokens vs. the distribution shape.)

4. **The repetition problem.** Why does greedy decoding cause repetition loops but
   sampling with temperature=1.0 usually doesn't? At what temperature would you
   start seeing repetition again? (Hint: think about what happens as temperature
   approaches 0.)

5. **Real-world trade-offs.** You're building a customer service chatbot. A user
   asks "What's your return policy?" What generation settings would you use and why?
   Now imagine you're building a creative writing assistant. How would your settings
   change? Be specific about temperature, top-k, and top-p values.

## Anticipating Struggles

**"Why does dividing by temperature work?"** Think about what dividing does to the
*differences* between logits. If logits are `[5.0, 3.0, 1.0]`, the differences are
2.0 between adjacent values. Dividing by `T=0.5` gives `[10.0, 6.0, 2.0]` --
differences of 4.0. Dividing by `T=2.0` gives `[2.5, 1.5, 0.5]` -- differences of
1.0. Softmax exponentiates these, so bigger differences = sharper distribution.
It's not magic -- it's just scaling differences before the exponential function
amplifies them.

**"Top-p implementation is confusing."** The trick is the `cumulative_probs - current_prob >= top_p`
line. Walk through it with a concrete example:

```
Sorted probs:     [0.40, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02]
Cumulative:       [0.40, 0.65, 0.80, 0.90, 0.95, 0.98, 1.00]
Minus current:    [0.00, 0.40, 0.65, 0.80, 0.90, 0.95, 0.98]
>= 0.9?           [  F,    F,    F,    F,    T,    T,    T  ]

So we keep the first 4 tokens (they cover 0.90 cumulative probability).
The 5th token is where cumsum first hits 0.90, but we include the token
that crosses the threshold, so we keep 4 tokens covering 90%.
```

**"What values should I use in practice?"** Start with `temperature=0.8, top_p=0.95`.
This is a solid default for most text generation tasks. Adjust from there:
lower temperature for more factual tasks, higher for more creative ones.

## Further Reading

- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) -- The original nucleus sampling paper. Very readable.
- [How to Generate Text (Hugging Face blog)](https://huggingface.co/blog/how-to-generate) -- Great visual walkthrough of all these strategies.
- [OpenAI API docs on sampling parameters](https://platform.openai.com/docs/api-reference/chat/create) -- See temperature, top_p, etc. in production.
