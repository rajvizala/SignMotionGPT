
# SignMotionGPT

SignMotionGPT now follows a cleaner research-repo layout with the code organized by pipeline and function instead of keeping most training logic as unrelated top-level files.

The repository contains two main training tracks:

1. Word-level pipeline:
   - train or reuse a VQ-VAE,
   - build a JSON dataset that maps each word to motion tokens,
   - train the Qwen-based LLM stages.
2. Sentence-level pipeline:
   - start from a VQ-VAE checkpoint,
   - optionally finetune that VQ-VAE for sentence data,
   - build a JSON dataset with sentence samples and motion tokens,
   - train the sentence-level Qwen pipeline.

## Repository layout

```text
.
|-- README.md
|-- docs/
|   |-- cleanup_candidates.md
|   |-- inference_and_visualization.md
|   `-- repository_layout.md
|-- legacy/
|   |-- README.md
|   |-- collators.py
|   |-- templates.py
|   |-- test_overfit.py
|   `-- install_artifacts/
|-- mGPT/
|-- scripts/
|   |-- evaluate_test_dataset.py
|   |-- finetune_sentence_vqvae.py
|   |-- infer_motion.py
|   |-- train_mgpt_vqvae.py
|   |-- train_sentence_pipeline.py
|   |-- train_word_pipeline.py
|   |-- train_word_vqvae.py
|   `-- visualize_motion.py
|-- signmotion_gpt/
|   |-- common/
|   |-- evaluation/
|   |-- sentence_pipeline/
|   |-- visualization/
|   |-- vqvae/
|   `-- word_pipeline/
|-- requirements.txt
`-- setup_env.sh
```

## Main code locations

- `signmotion_gpt/common/`: shared configuration.
- `signmotion_gpt/word_pipeline/`: word-level dataset prep, model setup, training, and inference.
- `signmotion_gpt/sentence_pipeline/`: sentence-level training pipeline.
- `signmotion_gpt/evaluation/`: metrics, constrained generation helpers, and held-out test evaluation.
- `signmotion_gpt/visualization/`: token-to-motion visualization.
- `signmotion_gpt/vqvae/`: VQ-VAE training and sentence-level VQ-VAE finetuning code.
- `legacy/`: archival or diagnostic code kept out of the main training path.

Detailed folder notes are in `docs/repository_layout.md`.

## Setup

```bash
bash setup_env.sh
```

Important environment variables:

```bash
export WORK_DIR=/path/to/repo
export DATA_JSON_PATH=/path/to/motion_llm_dataset.json
export HUGGINGFACE_HUB_TOKEN=your_token
```

## Word-level pipeline

Use this path when your dataset maps individual words to motion-token sequences.

### Expected dataset fields

Typical fields used by the word pipeline:

```json
[
  {
    "word": "library",
    "motion_tokens": "42 18 91 17",
    "participant_id": "P001"
  }
]
```

### Train

```bash
python scripts/train_word_pipeline.py
```

This pipeline handles:

1. dataset loading and deduplication,
2. motion vocabulary construction,
3. Stage 1 motion-language pretraining,
4. Stage 2 text-to-motion finetuning,
5. optional Stage 3 instruct finetuning,
6. evaluation and optional held-out test evaluation.

### Inference

```bash
python scripts/infer_motion.py --prompt "library" --stage 3
```

### Visualization

```bash
python scripts/visualize_motion.py --input generated_motion.txt
```

## Sentence-level pipeline

Use this path when your dataset contains sentence samples and motion-token sequences.

### Expected dataset fields

Sentence training expects items marked with `type: "sentence"` and a sentence text field such as `text` or `sentence`.

```json
[
  {
    "type": "sentence",
    "text": "the person walks to the door",
    "motion_tokens": "42 18 91 17"
  }
]
```

### Step 1: finetune the VQ-VAE for sentence data

```bash
python scripts/finetune_sentence_vqvae.py \
  --vqvae-ckpt /path/to/base_vqvae.pt \
  --word-data-dir /path/to/word_level_motion_data \
  --sentence-data-dir /path/to/sentence_level_motion_data \
  --output-dir /path/to/output_dir
```

### Step 2: train the sentence-level LLM

```bash
python scripts/train_sentence_pipeline.py \
  --dataset-path /path/to/sentence_dataset.json \
  --vqvae-ckpt /path/to/finetuned_vqvae.pt
```

## Evaluation and held-out testing

Held-out evaluation is available as a standalone entrypoint:

```bash
python scripts/evaluate_test_dataset.py --help
```

Additional inference and visualization examples live in `docs/inference_and_visualization.md`.

## Archived material and cleanup candidates

- Archival files moved out of the main path are documented in `legacy/README.md`.
- Likely non-core or removable files are listed in `docs/cleanup_candidates.md`.

## Notes

- `mGPT/` is preserved as the vendored VQ-VAE architecture dependency used by the VQ-VAE and visualization code.
- The main entrypoints are now under `scripts/`, while the reusable code lives under `signmotion_gpt/`.
