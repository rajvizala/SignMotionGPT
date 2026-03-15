# Exp 1: Part-Aware VQ-VAE with Feature Augmentation

## Hypothesis
The flat 512-code VQ-VAE entangles body, hand, and face features in a single
codebook, preventing novel body+hand combinations on unseen test data. Splitting
into 4 per-part codebooks with cross-part fusion enables compositional
recombination.

Ref: MotionGPT-2 (arXiv:2410.21747), SOKE (ICCV 2025)

## What to Measure
- Per-part reconstruction MSE (hands vs body vs face)
- Per-part codebook perplexity and utilization
- Worst-group reconstruction on length/coverage buckets
- Compositional test: encode body from sample A, hands from sample B

## Success Criterion
- Hand MSE improves over baseline Exp 0
- Worst-group score improves over baseline Exp 0
- All 4 codebook perplexities > 50 (no collapse)

## Kill Criterion
- If total loss exceeds 3.0 at any epoch: stop, reduce LR
- If worst-group val does not improve for 30 epochs: stop
- If any codebook perplexity drops below 10: codebook collapse, stop
- If hand_mse not improving while body_mse is: architecture issue, stop
