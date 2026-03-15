# Experiment 1

## Hypothesis

If the current token stream still has usable compositional structure, explicit lexicon-memory prompting from word-level ASL examples should improve worst-group generalization on unseen sentences with high lexical coverage.

## What to measure

- worst-group validation token loss,
- bucketed validation losses,
- whether lexical-coverage buckets improve relative to `exp_0`,
- optional tokenizer diagnostics from the shared evaluation harness.

## Success criterion

`coverage_high` and `coverage_medium` worst-group slices improve over `exp_0` without degrading novelty buckets catastrophically.

## Kill criterion

- early stop on worst-group validation,
- if lexical buckets do not improve at all and qualitative hand failures remain unchanged, treat this as evidence that the bottleneck is upstream of the LLM.
