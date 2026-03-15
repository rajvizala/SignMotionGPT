# Plan GPT-5.4

This plan implements a gated experiment ladder for ASL OOD generalization under the fixed `VQ-VAE + autoregressive LLM` constraint.

## Position on the main disagreements

- `LLM first vs VQ first`: LLM-first, because the current sentence stage still does not explicitly reuse word-level supervision and that residual headroom is cheaper to test before rewriting the tokenizer interface.
- `Single-stream repair vs part-aware multi-stream`: single-stream repair first, because it preserves comparability and avoids forcing an immediate LLM retokenization reset.
- `Lexicon injection risk`: controlled lexicon injection is worth testing, but only with stochastic inclusion and worst-group checkpoint selection so shortcut copying does not look like real generalization.

## Experiment order

1. `exp_0`
   - Sentence-level LLM baseline with worst-group checkpointing, replay, denoising, and SWA.
   - Purpose: establish a fair LLM baseline under the new evaluation discipline.

2. `exp_1`
   - Adds lexicon-memory prompting to `exp_0`.
   - Main bet: if unseen high-coverage sentences improve, the token stream still has compositional headroom.

3. `exp_2`
   - Adds controlled lexicon prompting plus contrastive alignment and randomized positions.
   - Purpose: test whether residual brittleness is due to exposure mismatch and weak text-motion binding.

4. `exp_3`
   - Narrow single-stream tokenizer repair fallback.
   - Purpose: if `exp_1` and `exp_2` do not improve worst-group metrics, move to the tokenizer layer without immediately adopting a full multi-stream redesign.

## Kill criteria

- Stop any run early if worst-group validation does not improve for the configured patience.
- Print a collapse warning if codebook coverage on validation drops below 60%.
- Print a diagnostic if hand MSE stalls while body MSE improves.
- Print a warning if VQ loss exceeds `3.0` in the tokenizer repair run.

## Shared outputs

Every training script saves:

- `best_worst_group.pt`
- `best_avg.pt`
- `resume_state.pt`

The shared evaluator writes a machine-readable JSON with the common schema required for cross-plan comparison.
