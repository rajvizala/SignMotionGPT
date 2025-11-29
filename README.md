
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

This repository implements a robust 2-stage training pipeline for motion generation, replicating the high-performance "overfit" test setup:
- **Stage 1**: Motion-only Language Model (MLM) - Pre-training on motion token sequences to learn the "language of motion".
- **Stage 2**: Text-to-Motion Fine-Tuning (T2M) - Supervised fine-tuning to align text prompts with motion sequences.

Key features:
- **Integrated Evaluation**: Automatically computes FID, Diversity, and Multimodality (MIM) metrics.
- **Side-by-Side Visualization**: Generates HTML comparisons of Ground Truth vs Generated motions.
- **Test Set Evaluation**: Can optionally run evaluation on a held-out test set (SMPL-X data).
- **Hugging Face Integration**: Automatic checkpointing and resuming from the Hub.

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

## Quick Start

### 1. Configure Training

Edit `config.py` to set your paths and hyperparameters. Key settings include:
- `DATA_JSON_PATH`: Path to your dataset.
- `MODEL_NAME`: Base model (e.g., "Qwen/Qwen3-0.6B").
- `PIPELINE_OUTPUT_DIR`: Directory for checkpoints and results.
- `HF_TOKEN`: Your Hugging Face token (or set via env var).

### 2. Run Full Pipeline

```bash
python train_pipeline.py
```

This script orchestrates the entire process:
1.  **Data Loading & Cleaning**: Deduplicates samples and builds vocabulary.
2.  **Stage 1 Training**: Motion Language Modeling (Pre-training).
3.  **Stage 2 Training**: Text-to-Motion Fine-Tuning.
4.  **Evaluation**: Runs inference on specific words, computes metrics (FID, Diversity, MIM), and generates visualizations.
5.  **Test Set Evaluation**: (Optional) Runs evaluation on held-out test data if configured.

### 3. Environment Variables

You can control many aspects via environment variables without editing code:

```bash
# Training Config
export PIPELINE_S1_EPOCHS=20
export PIPELINE_S2_EPOCHS=20
export PIPELINE_S1_BATCH=8
export PIPELINE_S2_BATCH=8

# Hugging Face
export HUGGINGFACE_HUB_TOKEN="your_token"
export HF_UPLOAD_INTERVAL_EPOCHS=2

# Evaluation
export EVALUATION_WORDS="passport,send,library"
export TEST_EVAL_SAMPLE_LIMIT=100
```

## Held-out Test Dataset Evaluation

The pipeline includes integration with `test_dataset_eval.py` to measure performance on an unseen SMPL-X test dataset.

To enable this, ensure `TEST_EVAL_DOWNLOAD_DIR` or `TEST_EVAL_EXTRACT_DIR` are configured in `config.py` or via env vars. The pipeline will attempt to run this after training if data is available.

## Visualization

The pipeline automatically generates side-by-side HTML visualizations in the output directory (`html_visualizations` folder). You can open these in any browser to compare Ground Truth motions with the model's generations.

To manually visualize tokens:

```bash
python visualize.py --tokens "<MOT_BEGIN><motion_177>...<MOT_END>" --output my_anim.html
```
