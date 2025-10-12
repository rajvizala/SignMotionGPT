## SignMotionGPT — LLM Fine‑tuning (Colab & Kaggle)

This repo contains utilities to fine‑tune the LLM component in three stages with resilient checkpointing to the Hugging Face Hub. Training can start in Colab and resume in Kaggle (or vice‑versa) without losing progress.

### What’s included
- Two‑point Hub checkpointing (halfway and final) with auto‑resume
- Per‑stage Hub repos (defaults):
  - `rdz-falcon/signmotiongpt-stage1`
  - `rdz-falcon/signmotiongpt-stage2`
  - `rdz-falcon/signmotiongpt-stage3`
- Dynamic `WORK_DIR` and dataset path defaults to the current directory
- Setup script to install dependencies and (optionally) download a dataset from a public Google Drive link

---

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

---

### 2) Configuration
All key settings live in `config.py`:
- `WORK_DIR`, `DATA_DIR`, `DATA_JSON_PATH` (defaults to current working dir)
- Hub:
  - `HF_USER` (defaults to `rdz-falcon`)
  - `HF_TOKEN` (auto‑read from `HUGGINGFACE_HUB_TOKEN` env)
  - `HUB_REPO_S1`, `HUB_REPO_S2`, `HUB_REPO_S3`
- Training hyperparameters and stage output directories (`OUT_S1`, `OUT_S2`, `OUT_S3`)

Ensure Kaggle has Internet enabled when training. If you embed your token in the script, avoid committing secrets publicly.

---

## Overview

This repository implements a multi-stage training approach for motion generation:
- **Stage 1**: Motion-only Language Model (MLM) - Model learns motion token distributions
- **Stage 2**: Multi-task Pretraining - Text-to-Motion (T2M), Motion-to-Text (M2T), and Denoising
- **Stage 3**: Supervised Fine-Tuning (SFT) - Final T2M refinement

## Repository Structure

```
motion-llm/
├── config.py              # Configuration and hyperparameters
├── data.py                # Dataset loading and preprocessing
├── model.py               # Model initialization with custom tokens
├── templates.py           # Prompt templates for each stage
├── collators.py           # Data collators with label masking
├── generate.py            # Generation with constrained decoding
├── metrics.py             # Evaluation metrics
├── train.py               # Training utilities
├── train_pipeline.py      # Main training script
├── inference.py           # Standalone inference script
├── requirements.txt       # Dependencies
└── README.md             # This file
```

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

### 2. Run Training Pipeline

```bash
python train_pipeline.py
```

This will execute all 3 training stages sequentially.

### 3. Generate Motions

After training, use the inference script:

```bash
python inference.py \
  --model_path ./output/stage3_t2m_sft \
  --prompt "a person walking forward" \
  --output motion_output.txt
```


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

### Visualization Options

- `--tokens`: Motion token string (direct input)
- `--input`: Path to file with motion tokens
- `--prompt`: Generate tokens first, then visualize (requires `--stage`)
- `--stage`: Stage model for generation (if using `--prompt`)
- `--vqvae-ckpt`: Path to VQ-VAE checkpoint
- `--stats`: Path to normalization stats
- `--smplx-dir`: Path to SMPL-X model directory
- `--output`: HTML file to save animation
- `--title`: Animation title
- `--fps`: Frames per second (default: 20)

The visualization pipeline:
1. **Parse tokens** from string or file
2. **Decode via VQ-VAE** to SMPL-X parameters (182-dim per frame)
3. **Run SMPL-X model** to get 3D vertices
4. **Generate interactive HTML** with Plotly (rotate, zoom, play/pause)

---

## Advanced Usage

### Custom Participant ID

Generate motions for specific participants (if dataset has participant IDs):

```bash
python inference.py --prompt "walking forward" --pid P001 --stage 3
```

### Adjust Generation Parameters

Edit `config.py` to change generation behavior:
```python
GEN_TEMPERATURE = 0.7      # Sampling temperature (higher = more random)
GEN_TOP_P = 0.9            # Nucleus sampling threshold
GEN_REPETITION_PENALTY = 1.2  # Discourage repetition
GEN_END_LOGIT_SLOPE = 0.25    # Bias toward expected length
```
