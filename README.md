# tinyLLM — Build a GPT from Scratch

A hands-on learning repo that walks you through building a GPT-style language model from scratch, one concept at a time. Inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and his ["Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) video.

## Who This Is For

You're a software engineer who knows how to code but has zero ML/PyTorch experience. You know what a function is, what a class is, and how to debug — but "tensor", "gradient descent", and "attention" are new territory.

## What You'll Build

By the end of this repo, you'll have built a character-level language model trained on Shakespeare that generates text like this:

```
ROMEO:
What is the city of the fair that doth
Make the most sovereign parts of my heart,
To be a rogue, and stand upon the state...
```

It won't be Shakespeare, but it'll *feel* like Shakespeare. And you'll understand exactly how it works.

## Prerequisites

- Python 3.11+
- A machine with a GPU (Apple Silicon with MPS, or NVIDIA with CUDA) — CPU works too, just slower
- Basic comfort with Python classes, loops, and data structures
- No ML experience needed — that's the whole point

## Setup

```bash
# Clone the repo
git clone <this-repo-url>
cd tinyLLM

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download the training data (tiny Shakespeare)
python data/download.py
```

## How to Use This Repo

Work through the chapters in order. Each chapter has:

| File | Purpose |
|------|---------|
| `README.md` | Concepts, explanations, architecture diagrams |
| `exercises.py` | Guided implementation with TODOs — this is where you learn |
| `solution.py` | Complete working code — peek when stuck, not before |
| `sandbox.py` | Empty playground for your experiments |
| `notes.md` | Template for your observations |

**The workflow for each chapter:**

1. Read the README to understand the concepts
2. Open `exercises.py` and work through the TODOs in order
3. Run the file after each TODO — it's designed to work at every stage
4. If stuck for more than 15 minutes, peek at `solution.py` for just that TODO
5. Use `sandbox.py` to try "what if" experiments
6. Jot observations in `notes.md`

## Chapter Map

| Chapter | What You Build | Key Concept |
|---------|---------------|-------------|
| [1 — Bigram Model](chapter-01-bigram/) | A model that predicts the next character based only on the current one | The training loop: forward → loss → backward → update |
| [2 — Self-Attention](chapter-02-attention/) | Attention mechanism from scratch | How tokens learn to "look at" other tokens |
| [3 — Transformer Block](chapter-03-transformer/) | Full transformer with multi-head attention, feedforward, residual connections | The architecture behind GPT |
| [4 — Scaling Up](chapter-04-scaling/) | A real model with 6 layers, dropout, learning rate schedules | Going from toy to (small) real |
| [5 — Generation](chapter-05-generation/) | Temperature, top-k, top-p sampling | The "creativity dial" in LLM products |
| [6 — Make It Yours](chapter-06-make-it-yours/) | Custom data, BPE tokenization, experiments | Open-ended exploration |

## Target Hardware

This repo is optimized for Apple Silicon Macs (M-series chips) using PyTorch's MPS backend. Every exercise includes automatic device detection:

```python
device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
```

Training runs are designed to complete in minutes, not hours.

## Acknowledgments

This repo follows the arc of Andrej Karpathy's excellent [nanoGPT](https://github.com/karpathy/nanoGPT) project and his ["Let's build GPT from scratch"](https://www.youtube.com/watch?v=kCc8FmEb1nY) video. If you learn well from video, watch that first — this repo is a companion for hands-on practice.
