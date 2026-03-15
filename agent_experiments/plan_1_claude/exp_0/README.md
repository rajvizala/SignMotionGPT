# Exp 0: Baseline Evaluation with Domain-Aware Bucketing

## Hypothesis
The current VQ-VAE checkpoint (1180 epochs) has hit its generalization ceiling.
This experiment establishes baseline metrics per domain bucket so all subsequent
experiments can be measured against these numbers.

## What to Measure
- Per-bucket VQ-VAE reconstruction MSE (length, coverage, novelty)
- Codebook coverage percentage on validation data
- Hand MSE vs body MSE ratio
- Worst-group score across all buckets

## Success Criterion
This is a measurement experiment, not a training experiment. Success = having
reliable baseline numbers to compare against.

## Kill Criterion
N/A -- this runs once and produces a JSON.
