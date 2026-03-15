# Experiment 3

## Hypothesis

If `exp_1` and `exp_2` fail, the bottleneck is deeper than downstream prompting and regularization can compensate for. This experiment tests a narrower tokenizer rescue than a full multi-stream redesign: keep the flat single-stream interface, but strengthen hand-priority reconstruction, temporal smoothness, and codebook-usage pressure.

## What to measure

- shared worst-group reconstruction metrics from `eval_harness.py`,
- codebook coverage,
- hand MSE vs body MSE,
- VQ loss and perplexity during training.

## Success criterion

`best_worst_group.pt` improves worst-group reconstruction and hand MSE over the incoming tokenizer checkpoint without collapsing codebook coverage.

## Kill criterion

- early stop on worst-group validation,
- if VQ loss exceeds `3.0`, print a stop warning,
- if codebook coverage drops below `60%`, treat the run as collapse,
- if hand MSE stagnates while body MSE improves, assume the rescue is missing the true failure mode.
