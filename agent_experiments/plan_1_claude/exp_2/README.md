# Exp 2: Residual VQ-VAE (RQ-VAE)

## Hypothesis
The flat 512-code codebook over-specialises to training data. 30% of codes
never appear on test data. Multi-level residual quantisation distributes
information across levels, with level 0 capturing coarse structure and higher
levels capturing fine detail. This should improve test-time coverage because
each level needs only represent residual information.

Ref: MOGO (arXiv:2506.05952), SoundStream (IEEE TASLP 2022)

## What to Measure
- Per-level perplexity (all levels should be > 100)
- Per-level codebook utilisation
- Reconstruction MSE (train + val) vs baseline
- Worst-group score across length buckets

## Success Criterion
- Overall val MSE improves over baseline Exp 0
- All levels have perplexity > 50
- Codebook coverage on val > 80% (vs 70.1% baseline)

## Kill Criterion
- Total loss exceeds 3.0: stop
- Worst-group does not improve for 30 epochs: stop
- Any level perplexity drops below 10: collapse, stop
- hand_mse not improving while body_mse is: stop
