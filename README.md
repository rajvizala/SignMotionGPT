

### 1) Configure setup script (one time)

Run the setup:

```bash
bash setup_env.sh
```

After setup, defaults are:
- `WORK_DIR` = current directory
- `DATA_JSON_PATH` = `./data/motion_llm_dataset.json`

You can override via environment variables if needed:

```bash
export WORK_DIR=/path/to/workdir
export DATA_JSON_PATH=/path/to/motion_llm_dataset.json
```



## Overview

This repository implements a multi-stage training approach for motion generation:
- **Stage 1**: Motion-only Language Model (MLM) - Model learns motion token distributions
- **Stage 2**: Multi-task Pretraining - Text-to-Motion (T2M), Motion-to-Text (M2T), and Denoising
- **Stage 3**: Supervised Fine-Tuning (SFT) - Final T2M refinement

## Installation

```bash
# Clone the repository
git clone https://github.com/rajvizala/SignMotionGPT.git
cd SignMotionGPT

# Setup Everything
bash setup_env.sh
```

## Dataset Format

Your dataset should be a JSON file with the following structure:

```json
[
  {
    "text_query": "a person walks forward",
    "motion_tokens": "42 18 91 ...",
    "participant_id": "P001"  // Optional
  },
  ...
]
```

- `text_query`: Natural language description of the motion
- `motion_tokens`: Space-separated discrete motion token IDs
- `participant_id`: (Optional) Participant identifier for personalized generation

## Quick Start

### 1. Configure Training

Edit `config.py` to set your paths and hyperparameters:

```python
DATA_JSON_PATH = "path/to/your/motion_dataset.json"
MODEL_NAME = "unsloth/Qwen3-1.7B"  # or "unsloth/gemma-3-1b-it"
WORK_DIR = "./output"
```

### 2. Run Full-Data Training (test_overfit pipeline)

```bash
python train_pipeline.py \
  --dataset-path ./data/motion_llm_dataset.json \
  --output-dir ./motion_gpt_full_model \
  --hf-token "$HUGGINGFACE_HUB_TOKEN"
```

`train_pipeline.py` is now a thin wrapper that forwards all options to
`test_overfit.py`, guaranteeing the same two-stage training recipe that was
validated in Colab. Key flags:

- `--s1-epochs / --s2-epochs`, `--s1-lr / --s2-lr`, `--s1-batch-size / --s2-batch-size`
- `--hf-stage1-repo / --hf-stage2-repo` (defaults pulled from `config.py`)
- `--stage2-subdir` for the Hugging Face checkpoint subfolder (defaults to `stage2_v2`)
- `--evals-only` or `--full-train` to toggle the metrics-only mode inside `test_overfit.py`

Provide `HUGGINGFACE_HUB_TOKEN` (or pass `--hf-token`) for private repos.

### 3. Generate Motions

After training, use the inference script:

```bash
python inference.py \
  --model_path ./output/stage3_t2m_sft \
  --prompt "a person walking forward" \
  --output motion_output.txt
```


## Held-out Test Dataset Evaluation

Use `test_dataset_eval.py` to measure FID / Diversity / Multimodality on a
Drive-hosted or locally extracted SMPL-X test set that the model never sees
during training.

```bash
# Evaluate using a directory that already contains video_data.pkl files
python test_dataset_eval.py \
  --local-extracted-dir ./test_data/extracted/batch01 \
  --hf-repo-id rdz-falcon/SignMotionGPTfit-archive \
  --hf-subfolder stage2_v2/epoch-020
```

To download raw archives from Google Drive (requires `pip install gdown`):

```bash
python test_dataset_eval.py \
  --drive-id 1AbCdEfGhIjKlMnOp \
  --download-dir ./test_data/downloads \
  --extract-dir ./test_data/extracted \
  --sample-limit 300
```

Metrics are written to `metrics_test.json` inside `TEST_EVAL_OUTPUT_DIR`
(configurable via `config.py` or CLI flags). The script loads the Stage 2 model
from Hugging Face along with the VQ-VAE assets declared in `visualize.py`.


## Configuration

Key hyperparameters in `config.py`:

### Model Settings
- `MODEL_NAME`: Base model to fine-tune
- `MAX_SEQ_LEN`: Maximum sequence length
- `LORA_R`: LoRA rank (default: 16)

### Training Settings
- `BATCH_TRAIN`: Training batch size per device
- `GRAD_ACCUM`: Gradient accumulation steps
- `LR`: Learning rate
- `EPOCHS_S1/S2/S3`: Epochs per stage

### Generation Settings
- `GEN_TEMPERATURE`: Sampling temperature
- `GEN_TOP_P`: Nucleus sampling threshold
- `GEN_END_LOGIT_SLOPE`: Bias toward target length

### Pipeline / Evaluation Overrides
- `PIPELINE_*` keys control the defaults consumed by `train_pipeline.py` →
  `test_overfit.py` (model name, stage lengths, learning rates, HF subfolders, etc.).
- `TEST_EVAL_*` keys configure `test_dataset_eval.py` (download/extract directories,
  sample limit, HF repo/subfolder).

## Training Stages Explained

### Stage 1: Motion-only LM
- Trains model to generate coherent motion token sequences
- No text conditioning, learns motion token distributions
- Helps model understand motion vocabulary structure

### Stage 2: Multi-task Pretraining
- **T2M (50%)**: Generate motion from text description
- **M2T (30%)**: Generate text description from motion
- **Denoise (20%)**: Reconstruct masked motion tokens
- Builds cross-modal understanding

### Stage 3: T2M SFT
- Focused supervised fine-tuning on text-to-motion
- Refines generation quality
- Final model for deployment

## Inference & Generation

After training all 3 stages, generate motion tokens from any text prompt:

```bash
# Basic usage (uses Stage 3 model by default)
python inference.py --prompt "walking forward"

# Choose a specific stage
python inference.py --prompt "dancing" --stage 2

# Save output to file
python inference.py --prompt "jumping" --stage 3 --output motion_tokens.txt

# Generate with participant ID (if dataset has PIDs)
python inference.py --prompt "yoga" --pid P40 --stage 3
```

### Inference Options

- `--prompt`: Text description of desired motion (required)
- `--stage`: Model stage to use (1, 2, or 3; default: 3)
- `--pid`: Optional participant ID for personalized generation
- `--output`: Save generated tokens to file
- `--no-per-prompt-vocab`: Allow all motion tokens (not just those seen with prompt)
- `--device`: Device to run on (cpu, cuda, cuda:0, etc.)

---

## Visualization

Convert generated motion tokens to 3D SMPL-X animation:

### Setup Visualization Assets

Edit `setup_env.sh` and add Google Drive IDs for:
```bash
VQVAE_MODEL_ID="your_vqvae_checkpoint_id"
VQVAE_STATS_ID="your_stats_file_id"
SMPLX_MODELS_ID="your_smplx_models_zip_id"
```

Or manually set paths via environment variables:
```bash
export VQVAE_CHECKPOINT=/path/to/vqvae_model.pt
export VQVAE_STATS_PATH=/path/to/vqvae_stats.pt
export SMPLX_MODEL_DIR=/path/to/smplx_models
```

### Usage

```bash
# Visualize from token string
python visualize.py --tokens "<MOT_BEGIN><motion_177><motion_135>...<MOT_END>"

# Visualize from saved file
python visualize.py --input motion_tokens.txt

# Generate and visualize in one command
python visualize.py --prompt "walking" --stage 3

# Custom output path
python visualize.py --tokens "..." --output my_animation.html --fps 30
```
