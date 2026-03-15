# Results Template

Fill this after each run and send it back.

## Run metadata

- Experiment config:
- Commit:
- GPU:
- Total training hours:
- Dataset JSONs used:

## Aggregate metrics

- Train loss:
- Dev loss:
- Test loss:
- FID:
- Diversity:
- Multimodality:
- DTW-JPE:
- DTW-PA-JPE:

## Bucketed metrics

### By lexical coverage

- 0.00 to 0.25:
- 0.25 to 0.50:
- 0.50 to 0.75:
- 0.75 to 1.00:

### By sentence motion length

- short:
- medium:
- long:

## Qualitative examples

### Success cases

1.
2.
3.

### Failure cases

1.
2.
3.

## Interpretation

- Did the model improve on unseen compositions where words were individually known?
- Did longer sentences improve or degrade?
- Did hand motion look better, worse, or unchanged?
- Did replay help preserve word-level behavior?

## Next step recommendation

- continue scaling current setup,
- reduce auxiliary loss weights,
- strengthen lexicon memory,
- revisit VQ-VAE adaptation,
- or stop this direction.
