# Experiment 0

## Hypothesis

A sentence-level Qwen baseline with worst-group checkpoint selection, replay, denoising, and SWA provides a fair control for the later lexicon-aware runs.

## What to measure

- worst-group validation token loss,
- average validation token loss,
- bucketed losses every 5 epochs,
- optional tokenizer diagnostics from the shared VQ evaluation harness if `--val-motion-dir` is provided.

## Success criterion

This run is successful if it establishes stable baseline checkpoints (`best_worst_group.pt` and `best_avg.pt`) and a reproducible worst-group reference.

## Kill criterion

- early stop when worst-group validation fails to improve for the configured patience,
- if tokenizer diagnostics are enabled and codebook coverage drops below 60%, treat the downstream run as untrustworthy.
