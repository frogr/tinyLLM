# tinyLLM — Build a GPT from Scratch

A structured, hands-on learning repo that walks you through building a GPT-style language model
from scratch. Inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and
his ["Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) video.

**Target audience:** Experienced software engineers with zero ML/PyTorch background.

**Training corpus:** Tiny Shakespeare (~1MB of Shakespeare's complete works).

**Hardware:** Apple Silicon Mac (MPS backend) or CPU. Training runs complete in minutes, not hours.

## Prerequisites

- Python 3.11+
- Basic comfort with Python (classes, functions, list comprehensions)
- No ML experience needed — every concept is explained from scratch

## Setup

```bash
# 1. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the training data
python data/download.py

# 4. Start with Chapter 1
cd chapter-01-bigram
python exercises.py
```

## Chapters

| Chapter | What You Build | Key Concept |
|---------|---------------|-------------|
| [01 — Bigram Model](chapter-01-bigram/) | A simple next-character predictor | Training loops, embeddings, loss functions |
| [02 — Self-Attention](chapter-02-attention/) | Single attention head from scratch | Queries, keys, values, masked attention |
| [03 — Transformer Block](chapter-03-transformer/) | Full transformer with multi-head attention | Residual connections, layer norm, feedforward |
| [04 — Scaling Up](chapter-04-scaling/) | A real model that generates Shakespeare | Hyperparameters, dropout, learning rate schedules |
| [05 — Generation](chapter-05-generation/) | Temperature, top-k, top-p sampling | How "creativity" works in LLM products |
| [06 — Make It Yours](chapter-06-make-it-yours/) | Custom data, BPE tokenization | Open-ended experimentation |

## How to Use This Repo

Each chapter has:

- **`README.md`** — Concepts explained in plain English with software engineering analogies
- **`exercises.py`** — Guided implementation with TODOs (runnable at every stage!)
- **`solution.py`** — Complete working code (no peeking until you've tried)
- **`sandbox.py`** — Empty playground for your own experiments
- **`notes.md`** — Template to record your observations

**The exercises are designed to run even with TODOs incomplete.** You'll see placeholder
output that changes to real results as you fill in each TODO.

## What You'll Build

By the end, you'll have a character-level language model that generates text like this:

```
ROMEO:
O, she doth teach the torches to burn bright!
It seems she hangs upon the cheek of night
Like a rich jewel in an Ethiope's ear;
```

(Well, approximately. It won't be *that* good, but you'll recognize the style.)

## Why This Exists

Understanding how GPT works by reading papers is like understanding web development by
reading the HTTP spec. You *could* do it, but building something and watching it work
teaches you 10x faster. This repo is the "build a CRUD app" of language models.
