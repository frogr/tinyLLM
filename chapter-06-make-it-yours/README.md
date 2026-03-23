# Chapter 6 -- Make It Yours

## Context

You've built a working GPT from scratch. You understand embeddings, self-attention,
multi-head attention, residual connections, layer norm, feedforward networks, dropout,
and generation strategies. You've watched a model go from random gibberish to something
that sounds like Shakespeare.

That's the whole architecture behind GPT-2, GPT-3, and (roughly) GPT-4.

Now it's time to make it your own. This chapter is deliberately open-ended. There are
no strict TODOs -- instead, there are guided explorations. Pick what interests you.
Break things. Try weird ideas. The best way to solidify your understanding is to push
the model beyond what we've done so far.

## Guided Explorations

### 1. Swap the Corpus -- Train on Your Own Text

Our model learned Shakespeare because that's what we fed it. Feed it something else,
and it'll learn that instead. The architecture doesn't care -- it just learns patterns
in whatever text you give it.

**Ideas for training data:**
- Your own emails or Slack messages (export from Gmail/Slack)
- A programming language (Python source code, shell scripts)
- Song lyrics from your favorite artist
- Recipes from a cookbook
- Legal contracts, medical notes, poetry, tweets, anything

**How to prepare your text file:**

1. Get your text into a single `.txt` file. UTF-8 encoding, no binary.
2. The bigger the better, but even 100KB can produce interesting results.
   Our Shakespeare file is about 1MB -- that's a sweet spot for this model size.
3. Remove anything the model shouldn't learn (headers, footers, HTML tags, metadata).
   The model will faithfully reproduce whatever patterns exist in the data, including
   junk.
4. Save it somewhere accessible, e.g., `../data/my_corpus.txt`.

```bash
# Example: concatenate all Python files in a project into one training file
find ~/my-project -name "*.py" -exec cat {} \; > ../data/python_code.txt

# Example: download Project Gutenberg books
curl -o ../data/moby_dick.txt https://www.gutenberg.org/files/2701/2701-0.txt
```

**What to expect:** The model will pick up the *style* of your data surprisingly
quickly. Train on Python code and it'll generate syntactically plausible (but
semantically nonsensical) Python. Train on recipes and you'll get ingredients lists
that almost make sense. The character-level model won't understand *meaning*, but
it'll nail the *form*.

### 2. BPE Tokenization -- How Real LLMs Actually Tokenize

Our model uses character-level tokenization: each character is one token. This is
simple but inefficient. The word "the" requires three tokens (`t`, `h`, `e`) and three
forward passes through the model. Real LLMs use **Byte Pair Encoding (BPE)** to create
a smarter vocabulary.

**What BPE does:** It's like creating shorthand. Instead of spelling out every word
letter by letter, you notice that `th` appears constantly, so you create a single
token for it. Then you notice `the` appears constantly, so you merge `th` + `e` into
one token. Then `the ` (with a space) becomes one token. And so on.

**The algorithm, step by step:**

1. Start with a vocabulary of individual bytes (256 tokens for raw bytes, or your
   character set).
2. Scan the entire training corpus. Find the most frequent pair of adjacent tokens.
3. Merge that pair into a single new token. Add it to your vocabulary.
4. Replace all occurrences of that pair in the corpus with the new token.
5. Repeat steps 2-4 until you reach your desired vocabulary size.

**Concrete example:**

```
Starting text: "aabaabaab"
Vocabulary: {a, b}

Step 1: Most common pair is (a, a). Merge into new token Z.
  Text becomes: "ZbZbZb"    (but we track that Z = "aa")
  Vocabulary: {a, b, Z="aa"}

Step 2: Most common pair is (Z, b). Merge into new token Y.
  Text becomes: "YYYb"... wait, let's be more careful.
  Actually: "Zb" "Zb" "Zb" -> "Y" "Y" "Zb"...
```

The real algorithm handles edge cases, but this is the core idea: greedily merge
the most frequent pairs.

**Why this matters:**

| Tokenization | "the cat sat" | Tokens |
|-------------|---------------|--------|
| Character   | t,h,e, ,c,a,t, ,s,a,t | 11 |
| BPE (GPT-2) | the, cat, sat | 4 |

Fewer tokens means:
- The model sees more context in the same window (256 BPE tokens covers much more
  text than 256 characters)
- Training is more efficient (fewer forward passes per document)
- The model can learn word-level and subword-level patterns directly

**In practice:** You don't need to implement BPE yourself for production use.
OpenAI's `tiktoken` library provides GPT-2/GPT-4 tokenizers:

```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("the cat sat")  # [1169, 3797, 3332]
text = enc.decode(tokens)            # "the cat sat"
```

But implementing it from scratch is an excellent learning exercise, which is why
we do it in `exercises.py`.

### 3. Positional Encoding Variations

Our model uses **learned positional embeddings**: a trainable lookup table that maps
each position (0, 1, 2, ..., block_size-1) to a vector. The model learns what
"position 0" and "position 47" mean during training.

```python
# What we use (learned):
self.position_embedding_table = nn.Embedding(block_size, n_embd)
```

The original "Attention Is All You Need" paper used **sinusoidal positional encodings**
instead: fixed mathematical functions (sines and cosines at different frequencies)
that encode position without any learned parameters.

```python
# Sinusoidal (fixed):
pe = torch.zeros(block_size, n_embd)
position = torch.arange(0, block_size).unsqueeze(1).float()
div_term = torch.exp(torch.arange(0, n_embd, 2).float() * -(math.log(10000.0) / n_embd))
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
```

**The tradeoff:**

| Approach | Pros | Cons |
|----------|------|------|
| Learned | Can learn task-specific position patterns | Can't extrapolate beyond training length |
| Sinusoidal | Works at any sequence length, no extra params | Fixed -- can't adapt to the task |

Modern models mostly use learned embeddings (or more advanced schemes like RoPE --
Rotary Position Embedding -- which is what LLaMA and many current models use).
For our toy model, learned embeddings work fine.

### 4. Scaling Considerations -- What Would It Take to Make This Actually Good?

Our model has ~10M parameters and trains on ~1MB of text. GPT-3 has 175 billion
parameters and trained on ~570GB of text. That's a factor of roughly 17,000x in
parameters and 570,000x in data.

**The scaling laws story:** Researchers at OpenAI discovered that model performance
improves predictably as you scale up three things:

1. **Model size** (number of parameters)
2. **Dataset size** (amount of training data)
3. **Compute** (training time / hardware)

And the relationship follows a power law -- you need to scale all three together.
A bigger model trained on the same data will overfit. More data with a too-small
model will underfit. You need to increase them in proportion.

**What you'd need to make our model "actually good":**

| Level | Parameters | Data | Training Time | Output Quality |
|-------|-----------|------|---------------|---------------|
| Our model | ~10M | 1MB | Minutes | Shakespeare-flavored babble |
| Decent | ~100M | 10GB | Hours (1 GPU) | Coherent paragraphs |
| Good | ~1B | 100GB | Days (8 GPUs) | Fluent text, follows prompts |
| GPT-3 | 175B | 570GB | Months (thousands of GPUs) | Impressive general knowledge |
| GPT-4 | (undisclosed) | (undisclosed) | (undisclosed) | You've used it |

The architecture we built is essentially the same at every level. The difference
is scale.

### 5. Connection to Real LLMs -- How Our Toy Relates to GPT-4 and Claude

**What's the same:**
- The core transformer architecture (attention, feedforward, residual, layer norm)
- The autoregressive training objective (predict the next token)
- The generation strategies (temperature, top-k, top-p)
- The fundamental idea: learn patterns in text by predicting what comes next

**What's different:**

**Tokenization:** Real models use BPE (or SentencePiece) with vocabularies of
30,000-100,000+ tokens, not 65 characters. This is why they can process text
efficiently.

**Pre-training data:** Real models train on vast, curated datasets: web pages,
books, code, Wikipedia, scientific papers. Data quality and curation matter
enormously.

**Instruction tuning (SFT):** After pre-training, models are fine-tuned on
examples of following instructions. This is what turns a "text completion engine"
into a "helpful assistant." The base model would just continue your text; the
instruction-tuned model tries to *answer* your question.

**RLHF / RLAIF (Reinforcement Learning from Human/AI Feedback):** Models are
further refined using human preferences. Humans compare two model outputs and
pick the better one. This signal trains a "reward model," which then guides
further training. This is a huge part of what makes modern models feel "helpful"
rather than just "fluent."

**Safety training:** Significant effort goes into making models refuse harmful
requests, avoid generating dangerous content, and behave appropriately. This
involves specialized training data, red-teaming, and evaluation.

**Architecture tweaks:** Real models use various improvements over the vanilla
transformer: RoPE or ALiBi for positional encoding, grouped query attention for
efficiency, SwiGLU activation functions, RMSNorm instead of LayerNorm, etc.
These are optimizations, not fundamental changes to the architecture.

**The remarkable thing:** Despite all these additions, the core of what you built
in Chapters 1-5 is genuinely the same mechanism that powers the most capable
AI systems in the world. The rest is engineering, scale, and data.

## Ideas for Experimentation

Here are some projects you could try. Pick one or two that interest you:

- **Train on Python code** and see if the model learns indentation, def/class
  structure, and common patterns like `if __name__ == "__main__":`
- **Train on song lyrics** and generate new verses in the style of your favorite
  artist
- **Implement BPE from scratch** (guided in exercises.py) and compare the output
  quality to character-level tokenization
- **Try sinusoidal positional encodings** instead of learned ones -- does it make
  a difference for our model size?
- **Increase context length** from 256 to 512 or 1024 and observe the effect on
  training speed and output coherence
- **Add a learning rate schedule** (warmup + cosine decay) and compare training
  curves to the constant learning rate
- **Visualize attention patterns** -- which positions attend to which? Do different
  heads specialize?
- **Train on two different corpora** (e.g., Shakespeare + Python) and see what the
  model produces -- does it mix styles?
- **Implement beam search** as an alternative to sampling-based generation
- **Try weight tying** -- share weights between the token embedding and the output
  projection layer (this is what GPT-2 does and it reduces parameters significantly)

## Think About It

These are reflective questions about the whole journey, not just this chapter.

1. **The model predicts one token at a time, left to right.** It has no explicit
   concept of words, grammar, sentences, or meaning. Yet it produces text that
   exhibits all of these properties. How? Where does the "understanding" live?
   Is it in the attention patterns? The embeddings? Somewhere else?

2. **We trained on Shakespeare and got Shakespeare-like output.** If you trained
   the exact same architecture on every book ever written, every webpage, every
   conversation -- would it "understand" language? What would be missing compared
   to how humans understand language? (This is the core question of the AI debate.)

3. **Real LLMs are instruction-tuned after pre-training.** Our model is a pure
   text completion engine -- it continues text. ChatGPT answers questions. What
   changes during instruction tuning? Is the model learning something fundamentally
   new, or is it learning to access capabilities it already had in a different way?

4. **The same architecture works for text, code, music, protein sequences, and more.**
   What does this tell us about the transformer architecture? Is there something
   universal about "predict the next token" as a learning objective, or are we just
   applying a hammer to everything because it works well enough?

5. **You now understand the core of how LLMs work -- attention, transformers,
   autoregressive generation.** What questions about AI feel *differently* to you
   now compared to before you started this course? What's less mysterious? What's
   *more* mysterious now that you've seen the internals?

## Resources for Going Deeper

### Papers
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) -- The original transformer paper. You can read it now.
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) -- The GPT-2 paper. Surprisingly readable.
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) -- Scaling up and in-context learning.
- [Scaling Laws for Neural Language Models (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361) -- The math behind "bigger is better."
- [Training Language Models to Follow Instructions (InstructGPT)](https://arxiv.org/abs/2203.02155) -- How RLHF works.
- [Neural Machine Translation of Rare Words with Subword Units (BPE paper)](https://arxiv.org/abs/1508.07909) -- The original BPE for NLP paper.

### Courses and Videos
- [Andrej Karpathy: "Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) -- The 2-hour video that inspired this repo. Watch it now -- you'll understand everything.
- [Andrej Karpathy: "Let's build the GPT Tokenizer"](https://www.youtube.com/watch?v=zduSFxRajkE) -- Deep dive into BPE tokenization.
- [3Blue1Brown: "But what is a GPT?"](https://www.youtube.com/watch?v=wjZofJX0v4M) -- Beautiful visual explanation.
- [Stanford CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) -- Full university course, free online.
- [Fast.ai Practical Deep Learning](https://course.fast.ai/) -- Excellent hands-on ML course.

### Code and Tools
- [nanoGPT by Karpathy](https://github.com/karpathy/nanoGPT) -- The repo our project is based on. Production-quality version of what you built.
- [minbpe by Karpathy](https://github.com/karpathy/minbpe) -- Minimal BPE implementations in Python.
- [tiktoken by OpenAI](https://github.com/openai/tiktoken) -- Production BPE tokenizer used by GPT models.
- [Hugging Face Transformers](https://github.com/huggingface/transformers) -- The industry-standard library for working with pre-trained models.

### Books
- *Build a Large Language Model (From Scratch)* by Sebastian Raschka -- A full book version of this learning path.
- *Dive into Deep Learning* (d2l.ai) -- Free online textbook with runnable code.
