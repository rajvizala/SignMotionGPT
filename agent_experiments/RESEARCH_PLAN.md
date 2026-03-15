# Research Plan: Achieving OOD Generalization in ASL Motion Generation

## Scope constraints (fixed)

- Keep architecture family: `VQ-VAE tokenizer + autoregressive LLM`
- Keep data sources: `ASL Citizen` and `How2Sign`
- No external dataset addition

## Core failure to solve

Current model performs strongly on train but weakly on unseen `How2Sign` test sentences, indicating brittle generalization under composition and distribution shift.

## High-level approach

Treat this as a **generalization-under-shift** problem, not an architecture replacement problem. The strategy is to improve:

1. Optimization geometry (flat minima)
2. Stochastic consistency (dropout invariance)
3. Continual transfer retention from word-stage to sentence-stage
4. Validation protocol for realistic checkpoint selection

## Experiment matrix

### Stage 0: Reproduce baseline

- Config: `baseline_repro.toml`
- Purpose: establish trusted baseline with this framework.

### Stage 1: Consistency + flatness

- Config: `exp_a_rdrop_sam.toml`
- Adds:
  - R-Drop
  - SAM
- Expected: lower worst-group validation loss and reduced overfit slope.

### Stage 2: Weight-space stabilization

- Config: `exp_b_rdrop_sam_swa.toml`
- Adds:
  - SWA late averaging
- Expected: lower variance across seeds and slightly stronger OOD metrics.

### Stage 3: Retain word-level priors

- Config: `exp_c_add_replay_ewc.toml`
- Adds:
  - Word-level replay during sentence fine-tuning
  - EWC regularization
- Expected: better compositional behavior on rare sentence combinations.

### Stage 4: Full recipe

- Config: `exp_d_full_recipe.toml`
- Stronger regularization and replay settings.
- Expected: best worst-group score if tuned correctly.

## Decision rules

Use these rules to avoid false-positive gains:

1. Promote only if at least 2/3 seeds improve worst-group metrics.
2. Reject if average improves but worst-group worsens.
3. Reject if seed variance increases materially.
4. Prefer stable gains over single-run peaks.

## Common failure patterns and actions

1. **Val improves early then degrades**
   - Action: earlier stopping, stronger SAM or SWA start earlier.

2. **Worse short-sequence groups but better long-sequence groups**
   - Action: reduce replay ratio, reduce EWC lambda.

3. **Mode collapse / repetitive token loops**
   - Action: increase R-Drop alpha slightly, increase label smoothing modestly.

4. **Underfitting everywhere**
   - Action: lower EWC lambda, reduce smoothing, lower SAM rho.

## Publication-oriented framing

If successful, paper contribution can be:

- "Architecture-preserving OOD gains for sign motion generation"
- "Pseudo-domain validation as a reliable selector for sign generation checkpoints"
- "Continual-transfer stabilization from isolated-word to sentence-level sign generation"

This contribution is incremental but valuable if gains are repeatable and analyzed rigorously.

## References

1. Duarte et al., CVPR 2021, How2Sign.
2. Desai et al., NeurIPS 2023, ASL Citizen.
3. van den Oord et al., NeurIPS 2017, VQ-VAE.
4. Jiang et al., 2023, MotionGPT.
5. Liang et al., NeurIPS 2021, R-Drop.
6. Foret et al., 2020, SAM.
7. Izmailov et al., UAI 2018, SWA.
8. Kirkpatrick et al., 2016, EWC.
9. Gulrajani and Lopez-Paz, ICLR 2021, DomainBed perspective.
10. Saunders et al., 2020, Progressive Transformers for SLP.

