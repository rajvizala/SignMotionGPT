# Exp 3: LLM Training with Masked Denoising + RAG + SWA

## Hypothesis
After improving the VQ-VAE (Exp 1 or 2), the LLM becomes the next bottleneck.
This experiment combines the best LLM-side interventions from all three debate
plans: masked denoising (Plan 2), RAG context injection (Plan 1), SWA (Plan 3),
and word-level replay (Plan 2).

## What to Measure
- Validation loss (worst-group and average)
- Token edit distance on held-out test sentences
- Per-bucket metrics (length, coverage, novelty)

## Success Criterion
- Worst-group val loss improves over vanilla LLM baseline
- Token edit distance improves on held-out test set

## Kill Criterion
- Val loss does not improve for 10 epochs: stop
- Train loss < 0.5 but val loss > 3.0: severe overfitting, stop
