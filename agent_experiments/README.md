# ASL Generalization Experiment Workspace

This folder contains a research-oriented experiment package for improving out-of-domain generalization in the existing `VQ-VAE + LLM` ASL motion pipeline without changing the underlying data sources. The design assumption is that isolated-word knowledge from Microsoft ASL Citizen should help sentence-level How2Sign generation, but that this knowledge must be made available to the sentence generator in a compositional and robust way instead of only through sequential fine-tuning.

The code here is not a paper reproduction. It is a hypothesis-driven extension of the current repository that combines:

- lexicon-memory prompting from isolated-word data,
- continued word replay during sentence training,
- masked motion-token denoising,
- contrastive text-motion alignment,
- randomized positional offsets for better length extrapolation, and
- stronger split analysis for measuring when the model fails.

These choices are inspired by prior work on discrete motion generation, motion-language pretraining, retrieval-augmented language modeling, and sign production, but adapted into a single research plan targeted at your exact failure mode: strong train-set fit, weak unseen How2Sign generalization.

## Why this workspace exists

Your current repo already includes sentence-level template augmentation, VQ-codebook initialization, and curriculum learning in `training/sentence_level/pipeline.py`. This workspace therefore focuses on techniques that go beyond the current sentence baseline rather than repeating it.

## Experiment ladder

1. `baseline_reproduction.json`
   - No new objectives.
   - Establishes a clean comparison inside this workspace.

2. `exp1_lexicon_memory.json`
   - Adds lexicon-memory prompting and word replay.
   - Tests whether isolated-word supervision can be made compositional at sentence time.

3. `exp2_alignment_denoise.json`
   - Adds masked denoising, contrastive alignment, and randomized positions.
   - Tests whether the unseen-set gap is mainly caused by exposure bias and weak text-motion binding.

4. `exp3_full_stack.json`
   - Combines the two directions.
   - This is the main candidate for a research-worthy result.

## Suggested research claim if this works

If the full stack improves unseen How2Sign metrics while preserving or improving word-level controllability, the claim is not merely "another sign generator". The stronger claim is:

> Isolated-word sign knowledge can be converted into sentence-level compositional guidance when the generator is trained with lexicon-aware conditioning, replay, and robustness objectives.

That is closer to a paper contribution than simply adding another augmentation heuristic.

## Files

- `research_plan.md`: detailed motivation, experiment rationale, expected outcomes, and interpretation guide.
- `references.md`: annotated paper list with URLs and what each paper contributes to these experiments.
- `results_template.md`: a reporting template for the metrics you should send back after running experiments.
- `configs/*.json`: runnable experiment configurations.
- `src/`: experiment package.
- `tests/smoke_test.py`: lightweight smoke test for the data and prompt pipeline.

## Primary commands

Analyze coverage and split difficulty before training:

`python -m agent_experiments.src.analyze_generalization --train-json /path/to/train.json --word-json /path/to/word_or_mixed.json --output-dir ./agent_experiments/out/analysis`

Dry-run one forward pass for an experiment:

`python -m agent_experiments.src.run_experiment --config agent_experiments/configs/exp3_full_stack.json --dry-run`

Run training:

`python -m agent_experiments.src.run_experiment --config agent_experiments/configs/exp3_full_stack.json`

## Minimum evaluation protocol

For every run, report:

- train loss,
- dev loss,
- held-out How2Sign test metrics,
- metrics bucketed by lexical coverage,
- metrics bucketed by length,
- at least 5 qualitative examples where all sentence words are seen individually but the sentence is unseen,
- at least 5 failure cases.

## Important note

The code is intentionally conservative about dependencies. It uses the existing PyTorch and Transformers stack and avoids introducing new libraries unless your later results justify it.
