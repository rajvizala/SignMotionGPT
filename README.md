# SignMotionGPT

SignMotionGPT has two training tracks:
- Word-level pipeline: train VQ token language alignment from words.
- Sentence-level pipeline: train and evaluate sentence-conditioned motion generation.

This repository has been reorganized to follow a modular research-repo structure while preserving backward-compatible root entrypoints.

## Repository structure

```text
.
├── config.py                         # shared config used across pipelines
├── data.py                           # shared data loading and dataset utilities
├── model.py                          # shared model/tokenizer setup
├── train.py                          # shared stage training loops + HF checkpoint utilities
├── metrics.py                        # shared evaluation metrics and generation helpers
├── mGPT/                             # VQ-VAE architecture modules
│
├── pipelines/
│   ├── word/
│   │   └── train_pipeline.py         # word-level stage1/stage2/stage3 orchestration
│   └── sentence/
│       ├── train_sentence_pipeline_v2.py
│       └── finetune_vqvae_sentence_level.py
│
├── evaluation/
│   └── test_dataset_eval.py          # held-out SMPL-X test set evaluation
│
├── inference/
│   ├── inference.py                  # text -> motion token generation
│   ├── visualize.py                  # motion token -> SMPL-X visualization
│   └── generate.py                   # constrained decoding helpers
│
├── experiments/
│   └── legacy/                       # old/ad-hoc scripts retained for reference
│
├── docs/
│   └── inference_and_visualization.md
│
├── train_pipeline.py                 # compatibility wrapper
├── train_sentence_pipeline_v2.py     # compatibility wrapper
├── finetune_vqvae_sentence_level.py  # compatibility wrapper
├── test_dataset_eval.py              # compatibility wrapper
├── inference.py                      # compatibility wrapper
├── visualize.py                      # compatibility wrapper
└── setup_env.sh
```

## Setup

```bash
bash setup_env.sh
```

Default paths:
- `WORK_DIR`: current directory
- `DATA_JSON_PATH`: `./data/motion_llm_dataset.json`

Override if needed:

```bash
export WORK_DIR=/path/to/workdir
export DATA_JSON_PATH=/path/to/motion_llm_dataset.json
```

## Training workflows

### A) Word-level pipeline

Main command (backward-compatible):

```bash
python train_pipeline.py
```

Equivalent module command:

```bash
python -m pipelines.word.train_pipeline
```

Supported stages:
- `--stage 1`
- `--stage 2`
- `--stage 3`
- `--stage all` (default)

### B) Sentence-level pipeline

```bash
python train_sentence_pipeline_v2.py \
  --dataset-path /path/to/sentence_dataset.json \
  --vqvae-ckpt /path/to/vqvae_checkpoint.pt \
  --stage all
```

Equivalent module command:

```bash
python -m pipelines.sentence.train_sentence_pipeline_v2 \
  --dataset-path /path/to/sentence_dataset.json \
  --vqvae-ckpt /path/to/vqvae_checkpoint.pt
```

### C) Sentence-level VQ-VAE finetuning

```bash
python finetune_vqvae_sentence_level.py \
  --vqvae-ckpt /path/to/word_level_vqvae.pt \
  --word-data-dir /path/to/word_npz \
  --sentence-data-dir /path/to/how2sign \
  --stats-path /path/to/stats.pt \
  --output-dir /path/to/output
```

## Evaluation and inference

Held-out test set evaluation:

```bash
python test_dataset_eval.py --local-extracted-dir /path/to/extracted_test_data
```

Inference from trained model:

```bash
python inference.py --prompt "walking forward" --stage 3
```

Visualization:

```bash
python visualize.py --tokens "<M177> <M135> <M210>"
```

More examples are in `docs/inference_and_visualization.md`.

## Configuration

Most shared training/eval settings are in `config.py`, including:
- data and output paths
- model name
- stage hyperparameters
- Hugging Face checkpointing options
- test-evaluation defaults

## Legacy scripts and cleanup suggestions

The following files are now isolated under `experiments/legacy/` because they are not in the default word/sentence training path:
- `experiments/legacy/test_overfit.py`
- `experiments/legacy/train_vqvae.py`
- `experiments/legacy/train_mgpt_vqvae.py`
- `experiments/legacy/collators.py`
- `experiments/legacy/templates.py`

You can keep them for reference or remove them if you do not need historical experiments.

## Notes on compatibility

- Existing root commands still work via thin compatibility wrappers.
- New modular paths are preferred for future development.
