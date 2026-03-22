# Chapter 2 — My Notes

## Key Concepts I Learned


## Things That Surprised Me


## Things I'm Still Confused About


## Experiment Results

| Experiment | What I Changed | What Happened |
|-----------|---------------|---------------|
|           |               |               |

## Shape Tracking

Fill this in as you work through the code. Shapes are the #1 source of bugs.

| Variable | Expected Shape | Actual Shape | Notes |
|----------|---------------|-------------|-------|
| idx (input) | (B, T) | | B=32, T=8 |
| tok_emb | (B, T, n_embd) | | n_embd=32 |
| pos_emb | (T, n_embd) | | broadcasts over B |
| x (combined) | (B, T, n_embd) | | |
| q (query) | (B, T, head_size) | | head_size=32 |
| k (key) | (B, T, head_size) | | |
| v (value) | (B, T, head_size) | | |
| q @ k^T | (B, T, T) | | the attention scores |
| wei (after softmax) | (B, T, T) | | each row sums to 1 |
| wei @ v | (B, T, head_size) | | context-aware output |
| logits | (B, T, vocab_size) | | vocab_size=65 |

## My Answers to "Think About It"

1. Shape detective (predicted vs actual shapes):


2. Remove the mask — what happened?


3. The scaling factor — sharp vs diffuse weights:


4. Position-only embeddings — can the model learn anything?


5. Why one attention head might be limiting:


## Comparison: Bigram vs Attention Model

| Metric | Bigram (Ch1) | Attention (Ch2) |
|--------|-------------|----------------|
| Parameters | 4,225 | |
| Final train loss | | |
| Final val loss | | |
| Generated text quality | | |

## Questions for Next Chapter
