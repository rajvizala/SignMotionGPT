# Experiment 2

## Hypothesis

If residual failure remains after lexicon prompting, the next likely causes are brittle autoregressive recovery and weak text-motion binding. Contrastive alignment and randomized positions test that hypothesis while keeping the tokenizer fixed.

## What to measure

- worst-group validation token loss,
- lexical and novelty bucket behavior relative to `exp_1`,
- optional tokenizer diagnostics via the shared evaluation harness.

## Success criterion

This run is successful if worst-group validation improves beyond `exp_1`, especially on novelty buckets, without losing the lexical-coverage gains from lexicon prompting.

## Kill criterion

- early stop on worst-group validation,
- if `exp_2` improves train loss but not worst-group validation, treat the extra regularization as unnecessary under the current tokenization.
