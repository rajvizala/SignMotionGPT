# Cleanup candidates

These files are not part of the main organized training path anymore. I kept them in the repository so you can review them before deleting anything.

## High-confidence archival files

These are the safest candidates to remove after you confirm you do not need them for historical reference:

- `legacy/test_overfit.py`
  - original monolithic reference implementation that the modular word pipeline was derived from.
- `legacy/templates.py`
  - older prompt-template helper tied to an outdated token scheme.
- `legacy/collators.py`
  - older collator utility that is not used by the current word or sentence training pipelines.
- `legacy/install_artifacts/=0.20.0`
- `legacy/install_artifacts/=0.22.0`
- `legacy/install_artifacts/=0.4.0`
- `legacy/install_artifacts/=0.4.7`
- `legacy/install_artifacts/=0.41.0`
- `legacy/install_artifacts/=1.24.0`
- `legacy/install_artifacts/=2.0.0`
- `legacy/install_artifacts/=2.14.0`
- `legacy/install_artifacts/=3.0.0`
- `legacy/install_artifacts/=4.40.0`
- `legacy/install_artifacts/=4.65.0`
- `legacy/install_artifacts/=5.2.0`
  - these look like accidental pip or notebook artifact files rather than source code.

## Review-first candidates

These are not in the main documented training path, but I would review them before deleting because they may still contain useful experiments or recovery workflows:

- `signmotion_gpt/vqvae/train_word_vqvae.py`
  - retained as an older VQ-VAE training script.
- `signmotion_gpt/vqvae/train_mgpt_vqvae.py`
  - retained as an alternate VQ-VAE training script with environment-specific logic.

## Keep

These are still part of the organized repo and should not be treated as cleanup targets:

- `signmotion_gpt/word_pipeline/`
- `signmotion_gpt/sentence_pipeline/`
- `signmotion_gpt/evaluation/`
- `signmotion_gpt/visualization/`
- `signmotion_gpt/vqvae/finetune_sentence_level.py`
- `mGPT/`
- `scripts/`
