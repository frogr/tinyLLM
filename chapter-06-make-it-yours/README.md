# Chapter 6: Make It Yours

## The Big Picture

You've built a GPT from scratch. You understand tokenization, attention, transformers,
training, and generation. Now it's time to experiment freely: swap the training data,
try different tokenization, and explore ideas.

This chapter is open-ended by design. There are no "right answers" — just ideas to
try, concepts to explore, and knobs to turn. The exercises give you a starting point,
but the real learning comes from your own experiments.

---

## Concepts

### Custom Datasets

You can train on anything — your own writing, song lyrics, code, recipes, legal
documents. The model will learn the patterns of whatever you feed it. The key
constraint is size: you need at least a few hundred KB of text for the model to
learn anything interesting.

Think about what makes each kind of text different:

| Dataset | Patterns the Model Learns |
|---------|--------------------------|
| Shakespeare | Iambic pentameter, "thee/thou", dramatic structure |
| Python code | Indentation, `def`/`class`/`if`, variable naming |
| Cooking recipes | Ingredient lists, measurements, "preheat", "stir" |
| Legal documents | "Whereas", "hereby", long nested clauses |
| Song lyrics | Rhyme, repetition, verse/chorus structure |
| Chat messages | Short sentences, slang, emoji patterns |

The model doesn't "understand" any of these domains. It just learns statistical
patterns: which tokens tend to follow which other tokens. But the results can be
surprisingly convincing.

**How to prepare a custom dataset:**

1. Gather your text into a single `.txt` file (UTF-8 encoding)
2. Aim for at least 100KB — more is better
3. Clean it up: remove HTML tags, fix encoding issues, strip metadata you don't want
4. Place it in the `data/` directory
5. Point the training script at it

Some sources for text data:
- [Project Gutenberg](https://www.gutenberg.org/) — free books
- Your own notes, emails, or blog posts (export and concatenate)
- GitHub repos (concatenate source files)
- Wikipedia dumps (filtered by topic)
- Song lyrics sites (be mindful of copyright)

---

### BPE Tokenization (Byte Pair Encoding)

So far we've used character-level tokenization — each character is a token. Real
LLMs like GPT use BPE, which learns common subword patterns. `ing` becomes one
token, `tion` becomes one token, common words become single tokens. This is more
efficient — the model can see more context in the same `block_size`.

Think of it as compression: instead of spelling out every word letter by letter,
you use common abbreviations.

**Why does this matter?**

With character-level tokenization and `block_size=64`, the model sees about 10-12
words of context. With BPE, it might see 40-50 words in the same window. That's a
massive difference for understanding patterns that span sentences.

Here's a concrete comparison:

```
Input text: "The transformer architecture revolutionized natural language processing"

Character-level tokens (68 tokens):
  ['T', 'h', 'e', ' ', 't', 'r', 'a', 'n', 's', 'f', 'o', 'r', 'm', 'e', 'r', ...]

BPE tokens (maybe 7-8 tokens):
  ['The', ' transform', 'er', ' architecture', ' revolution', 'ized', ' natural', ...]

GPT-2 tokens (6 tokens):
  ['The', ' transformer', ' architecture', ' revolutionized', ' natural', ' language', ' processing']
```

Fewer tokens = more context in the same window = better predictions.

---

### How BPE Works

BPE starts with individual characters (or bytes) and repeatedly merges the most
common adjacent pair. Here's a step-by-step example:

**Starting vocabulary:** all individual characters in the text

**Training corpus (simplified):** `"low low low lowest lowest"

**Step 1: Count all adjacent pairs**

```
('l', 'o'): 5 times
('o', 'w'): 5 times
('w', ' '): 3 times
('w', 'e'): 2 times
('e', 's'): 2 times
('s', 't'): 2 times
(' ', 'l'): 4 times
...
```

**Step 2: Merge the most frequent pair**

`('l', 'o')` appears 5 times. Merge them into a new token `lo`.

Now the text looks like: `"lo w  lo w  lo w  lo w est  lo w est"`

**Step 3: Count pairs again with the new token**

```
('lo', 'w'): 5 times    <-- most frequent
('w', ' '): 3 times
...
```

**Step 4: Merge `('lo', 'w')` into `low`**

Now: `"low  low  low  low est  low est"`

**Step 5: Continue until you reach your desired vocabulary size**

Each merge creates a new token and reduces the total number of tokens needed to
represent the text. You keep going for a fixed number of merges (e.g., 256, 1000,
or 50,000 for production models).

**The key insight:** BPE learns a vocabulary that's adapted to your specific data.
If you train on Python code, tokens like `def`, `self`, `return` might become single
tokens. If you train on medical text, `patient`, `diagnosis` might merge.

**The algorithm in pseudocode:**

```
function train_bpe(text, num_merges):
    tokens = list(text)          # start with characters
    merges = {}                  # record of what we merged

    for i in 1..num_merges:
        pairs = count_adjacent_pairs(tokens)
        best_pair = most_frequent(pairs)
        tokens = merge_pair(tokens, best_pair)
        merges[best_pair] = new_token_id

    return merges

function encode(text, merges):
    tokens = list(text)
    for (pair, new_id) in merges:
        tokens = merge_pair(tokens, pair)
    return tokens

function decode(token_ids, vocab):
    return "".join(vocab[id] for id in token_ids)
```

---

### Positional Encoding Variations

In our model, we used **learned positional embeddings**:

```python
self.position_embedding_table = nn.Embedding(block_size, n_embd)
```

This creates a lookup table where PyTorch learns the best embedding for each
position. Position 0 gets one vector, position 1 gets another, etc. The model
figures out what these should be during training.

There are other approaches:

**Sinusoidal Positional Encoding (Original Transformer)**

The original "Attention Is All You Need" paper used fixed mathematical functions:

```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Each position gets a unique pattern of sine and cosine waves at different
frequencies. The advantage: it can theoretically generalize to sequence lengths
not seen during training. The disadvantage: it's not learned, so it might not
capture the position patterns that matter most for your data.

**RoPE (Rotary Position Embedding) — What Modern Models Use**

Models like LLaMA, Mistral, and GPT-NeoX use RoPE. Instead of adding position
information to the token embeddings, RoPE rotates the query and key vectors in
attention based on their position. Think of it as encoding position through
rotation angles rather than additive vectors.

The key advantage of RoPE: it encodes *relative* position naturally. The model
learns that "3 tokens apart" has a specific meaning, regardless of where in the
sequence those tokens are. This helps with generalization to longer sequences.

For our small model, learned embeddings work great. But if you wanted to experiment:

```python
# Sinusoidal encoding (drop-in replacement)
import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, block_size, n_embd):
        super().__init__()
        pe = torch.zeros(block_size, n_embd)
        position = torch.arange(0, block_size).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, n_embd, 2).float() * -(math.log(10000.0) / n_embd)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:x.size(1)]  # (T, n_embd)
```

---

### Ideas for Experimentation

Here are some things worth trying. Pick whatever sounds fun:

1. **Train on Python code** — Does the model learn indentation? Does it produce
   syntactically valid Python? Try `temperature=0.1` for "real" code vs `1.5` for
   creative gibberish.

2. **Train on song lyrics** — Scrape lyrics from one artist. Does it capture their
   style? Their vocabulary? Try mixing two very different artists.

3. **Train on cooking recipes** — The model should learn the format: title,
   ingredients list, numbered steps. Does it invent plausible recipes?

4. **Train on legal documents** — Legal text is highly formulaic. A small model
   can learn the patterns surprisingly well.

5. **Train on a foreign language** — Does the model learn the character patterns
   of French, German, or Japanese? What about mixing languages?

6. **Train on your own chat messages** — Export your messages from a messaging app.
   What does your "style" look like to a language model?

7. **Compare model sizes** — Train with 2 heads vs 8 heads, 2 layers vs 8 layers.
   How does output quality change? How does training time change?

8. **Implement beam search** — Instead of sampling one token at a time, maintain
   the top-k partial sequences and pick the best complete one.

9. **Try different learning rates** — Plot loss curves for `lr=1e-2`, `1e-3`, `1e-4`.
   Which converges fastest? Which gets the lowest final loss?

10. **Build a simple web interface** — Use Flask to serve a page where you type a
    prompt and the model completes it. Add sliders for temperature and top-k.

11. **Train on musical notation** — ABC notation or MusicXML. Can the model learn
    valid musical structures?

12. **Implement a simple RLHF approximation** — Generate two completions, manually
    pick the better one, use that signal to fine-tune.

13. **Try different context lengths** — Train the same model with `block_size=32`,
    `64`, `128`, `256`. How does coherence change?

14. **Dataset mixing** — Train on 50% Shakespeare + 50% Python code. What comes out?

---

## Think About It

These questions don't have single right answers. They're meant to deepen your
intuition. Try to answer them before reading on, then test your hypotheses.

### 1. Temperature and Code

> If you train on Python code, what would `temperature=0.1` vs `temperature=1.5`
> produce? Which is more useful?

`temperature=0.1` sharpens the probability distribution — the model almost always
picks the most likely next token. For code, this means it produces more
syntactically correct, conventional code. It'll write `def __init__(self):` because
that's overwhelmingly the most common pattern after `def __init__`.

`temperature=1.5` flattens the distribution — rare tokens get a fighting chance.
For code, this means creative variable names, unusual function combinations, and
probably lots of syntax errors. Not useful for writing real code, but interesting
for seeing what the model "knows" about rare patterns.

For code generation, low temperature is almost always more useful. For creative
text, higher temperature adds variety.

### 2. BPE and Rare Words

> Why does BPE help with rare words? What happens to a character-level model with
> the word "onomatopoeia"?

A character-level model sees `onomatopoeia` as 12 separate tokens:
`o-n-o-m-a-t-o-p-o-e-i-a`. It needs to learn this exact sequence to reproduce
the word, and if it's rare in the training data, it probably won't learn it well.

BPE might tokenize it as `ono-mat-opo-eia` or similar subword chunks. Even if the
full word is rare, the subword pieces appear in other words (`mat` in "material",
`eia` in other words). The model can compose the rare word from familiar pieces.

This is why BPE handles rare words, technical jargon, and even made-up words much
better than character-level or word-level tokenization.

### 3. Multi-Author Style

> Could you train this model on two different authors and get it to generate in the
> "style" of either? What would you need to change?

Yes, but you'd need to give the model a way to know which author's style to use.
Options:

- **Prefix tokens**: Start each training example with `<AUTHOR_A>` or `<AUTHOR_B>`,
  then at generation time, start with the desired prefix
- **Separate fine-tuning**: Train a base model, then fine-tune separate copies on
  each author
- **Conditional generation**: Add an author embedding that gets added to all token
  embeddings (similar to how we add position embeddings)

The simplest approach is prefix tokens — it requires no architecture changes.

### 4. Minimum Dataset Size

> What's the minimum dataset size needed to get coherent output? Try with 10KB,
> 100KB, and 1MB.

This is best answered by experiment! But as a rough guide:

- **10KB** (~10,000 characters): The model will learn basic character patterns and
  common short words, but output will be mostly gibberish with occasional
  recognizable fragments.
- **100KB** (~100,000 characters): The model starts producing word-like sequences
  and short phrases that sound like the source material. Grammar is hit-or-miss.
- **1MB** (~1,000,000 characters): The model produces surprisingly coherent
  sentences, maintains consistent style, and occasionally produces multi-sentence
  passages that make some sense.

The Shakespeare dataset is about 1MB, which is why it works well for our model.

### 5. Music Notation

> How would you adapt this to generate music notation? What would the vocabulary
> look like?

ABC notation is a good fit because it's plain text:

```
X:1
T:Example Tune
M:4/4
K:G
|: G2 A B | c2 B A | G2 A B | d4 :|
```

Your vocabulary would include: note letters (A-G), accidentals (#, b), octave
markers (', ,), bar lines (|), duration numbers (2, 4, 8), rhythm markers,
key/time signature tokens, and structural tokens (|:, :|).

You could train character-level (each character is a token) or create a custom
tokenizer that treats each note+duration as a single token (`G2`, `c4`, etc.).

The model would learn musical structures: scales stay within keys, phrases
tend to be 4 or 8 bars, melodies often return to the tonic note, etc.

---

## Where to Go from Here

Congratulations! You've built a language model from scratch and understand every
piece of it. Here's what real GPT models add on top of what you've built:

### Scale

GPT-2 has 1.5 billion parameters (ours has ~10,000-100,000). GPT-3 has 175 billion.
GPT-4 is estimated to be much larger still. More parameters = more capacity to
store and combine patterns.

### Training Data

We trained on ~1MB of Shakespeare. GPT-3 trained on ~570GB of text from the
internet — books, websites, Wikipedia, code. The diversity and volume of training
data is a huge factor in model capability.

### RLHF (Reinforcement Learning from Human Feedback)

The base GPT model just predicts the next token. ChatGPT adds a step: humans rate
different completions, and the model is fine-tuned to produce responses that humans
prefer. This is what makes it helpful and conversational rather than just
autocompleting text.

### Instruction Tuning

Before RLHF, models are fine-tuned on (instruction, response) pairs so they learn
to follow directions. This is why ChatGPT answers questions instead of just
continuing your text.

### Longer Context

Our model sees 64-256 tokens of context. GPT-4 can see 128,000+ tokens. This
requires architectural innovations (efficient attention, RoPE for position
encoding, etc.) but the core transformer is the same.

### Key Takeaway

Everything you've built in this repo is the real foundation. The jump from our
tiny model to GPT-4 is mostly:
- More data
- More parameters
- More compute
- Better training techniques (RLHF, instruction tuning)
- Engineering optimizations (efficient attention, quantization, etc.)

The core ideas — attention, transformers, autoregressive generation — are exactly
what you've implemented.

---

## Further Reading

Now that you understand the fundamentals, these resources will make much more sense:

### Videos
- **Andrej Karpathy's "Let's build GPT"** — The video that inspired this repo.
  Now you can follow every detail.
  [youtube.com/watch?v=kCc8FmEb1nY](https://www.youtube.com/watch?v=kCc8FmEb1nY)

- **Karpathy's makemore series** — Builds up from bigrams to MLPs to transformers,
  with great intuition-building.
  [youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)

### Blog Posts
- **"The Illustrated Transformer"** by Jay Alammar — Beautiful visual explanations
  of every component. Now that you've built one, these diagrams will click.
  [jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)

- **"The Illustrated GPT-2"** by Jay Alammar — Same style, focused on GPT
  specifically.
  [jalammar.github.io/illustrated-gpt2/](https://jalammar.github.io/illustrated-gpt2/)

### Papers
- **"Attention Is All You Need" (2017)** — The original transformer paper. Dense
  but rewarding now that you know the architecture.
  [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

- **"Language Models are Unsupervised Multitask Learners" (GPT-2, 2019)** — Showed
  that scaling up transformers leads to emergent capabilities.
  [cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

- **"Language Models are Few-Shot Learners" (GPT-3, 2020)** — The paper that
  convinced the world that scale matters.
  [arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)

### Courses
- **Hugging Face NLP Course** — Practical course on using and fine-tuning
  transformers with the Hugging Face library. A natural next step after this repo.
  [huggingface.co/learn/nlp-course](https://huggingface.co/learn/nlp-course)

- **Stanford CS224N** — The academic deep-dive into NLP with transformers.
  Lecture videos and assignments are free.
  [web.stanford.edu/class/cs224n/](https://web.stanford.edu/class/cs224n/)

### Code
- **nanoGPT** by Karpathy — The repo that inspired this one. A clean, minimal
  GPT implementation that's production-quality.
  [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)

- **minbpe** by Karpathy — Minimal BPE tokenizer implementation. Perfect companion
  to the BPE exercises in this chapter.
  [github.com/karpathy/minbpe](https://github.com/karpathy/minbpe)

---

## What's Next for You?

You've gone from zero ML knowledge to understanding and implementing a transformer
language model. That puts you ahead of most people who use LLMs daily without
understanding how they work.

From here, you can:
1. **Fine-tune real models** — Use Hugging Face to fine-tune GPT-2 or LLaMA on
   your own data. You'll understand every layer.
2. **Build applications** — Use APIs (OpenAI, Anthropic, etc.) with deep
   understanding of what's happening behind the scenes.
3. **Go deeper into ML** — CNNs, diffusion models, reinforcement learning — the
   training loop and gradient concepts transfer everywhere.
4. **Contribute to open source** — Projects like LLaMA.cpp, vLLM, and others need
   contributors who understand the fundamentals.

Whatever you do next — you've built a GPT from scratch. That's not nothing.
