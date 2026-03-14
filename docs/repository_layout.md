# Repository layout

This file explains the intended role of each top-level directory after the cleanup.

## `signmotion_gpt/`

Primary Python package for all maintained code.

- `common/config.py`
  - shared environment-driven configuration used by the word pipeline, evaluation, and visualization code.
- `word_pipeline/`
  - `data.py`: dataset loading, deduplication, and motion-vocabulary helpers.
  - `model.py`: tokenizer and Qwen model setup.
  - `train.py`: training loops and Hugging Face checkpoint helpers.
  - `pipeline.py`: end-to-end word-level training entrypoint.
  - `inference.py`: CLI inference entrypoint for word-level models.
- `sentence_pipeline/`
  - `pipeline.py`: sentence-level training pipeline with curriculum and VQ-VAE-based initialization.
- `evaluation/`
  - `metrics.py`: evaluation metrics and generation-time helpers.
  - `generation.py`: constrained decoding utilities.
  - `test_dataset_eval.py`: held-out SMPL-X test-set evaluation.
- `visualization/`
  - `visualize.py`: motion-token decoding and HTML visualization.
- `vqvae/`
  - `finetune_sentence_level.py`: sentence-level VQ-VAE finetuning.
  - `train_word_vqvae.py`: older word-level VQ-VAE training script kept for reference.
  - `train_mgpt_vqvae.py`: alternate VQ-VAE training script kept for reference.

## `scripts/`

Clean CLI entrypoints that import from the package instead of keeping the implementation at the repository root.

- `train_word_pipeline.py`
- `train_sentence_pipeline.py`
- `infer_motion.py`
- `visualize_motion.py`
- `evaluate_test_dataset.py`
- `finetune_sentence_vqvae.py`
- `train_word_vqvae.py`
- `train_mgpt_vqvae.py`

## `legacy/`

Archived files that do not belong in the main training path anymore.

- `test_overfit.py`: original monolithic reference script.
- `templates.py`: prompt-template helper from an older token scheme.
- `collators.py`: older collator utility not used by the main pipelines.
- `install_artifacts/`: stray installation or notebook-side artifact files moved out of the root.

## `mGPT/`

Vendored VQ-VAE architecture dependency used by the VQ-VAE and visualization code. This remains separate because the scripts already depend on the `mGPT.archs` import path.

## `docs/`

Human-facing documentation:

- `repository_layout.md`: structure and ownership guide.
- `cleanup_candidates.md`: files to review before deleting.
- `inference_and_visualization.md`: runtime examples for generation and rendering.
