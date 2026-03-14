# SignMotionGPT

A two-pipeline framework for sign language motion generation using VQ-VAE tokenization and LLM-based sequence modeling (Qwen).

## Overview

SignMotionGPT converts raw SMPL-X motion capture data into discrete motion tokens via a VQ-VAE, then trains a language model (Qwen3-0.6B) to generate those tokens conditioned on text input. The framework supports two levels of granularity:

- **Word-Level Pipeline**: Generates motion for individual sign language words (3-stage training).
- **Sentence-Level Pipeline**: Generates motion for full sentences with length conditioning and curriculum learning (2-stage training).

### Architecture

```
Raw Motion Data (NPZ / PKL)
        |
        v
   VQ-VAE Encoder  -->  Discrete Motion Tokens  -->  JSON Dataset
                                                         |
                                                         v
                                                  LLM Training (Qwen)
                                                         |
                                                         v
                                              Text-Conditioned Generation
                                                         |
                                                         v
                                               VQ-VAE Decoder  -->  SMPL-X Motion
```

## Project Structure

```
SignMotionGPT/
|-- configs/
|   +-- config.py                  # Central configuration (paths, hyperparams, HF settings)
|
|-- models/                        # Model architectures
|   |-- vqvae.py                   # VQ-VAE encoder/decoder (from MotionGPT)
|   |-- resnet.py                  # 1D ResNet blocks for VQ-VAE
|   |-- quantize_cnn.py            # Vector quantization modules (EMA, reset variants)
|   +-- llm.py                     # LLM (Qwen) setup, tokenizer augmentation, LoRA (legacy)
|
|-- datasets/                      # Data loading and processing
|   +-- motion_dataset.py          # Dataset classes (MotionDataset, TextMotionDataset, etc.)
|
|-- training/                      # Training pipelines
|   |-- word_level/
|   |   |-- stages.py              # Stage 1/2/3 training loops + HF checkpoint utils
|   |   +-- pipeline.py            # Full word-level orchestrator (data -> train -> eval)
|   +-- sentence_level/
|       +-- pipeline.py            # Full sentence-level pipeline (self-contained)
|
|-- evaluation/                    # Metrics and evaluation
|   |-- metrics.py                 # FID, Diversity, Multimodality (MIM), encoder-style eval
|   +-- test_eval.py              # Held-out test set evaluation (SMPL-X data)
|
|-- inference/                     # Inference and generation
|   |-- generate.py                # Length-aware constrained decoding (logits processor)
|   +-- predict.py                 # CLI inference script (text -> motion tokens)
|
|-- visualization/                 # Motion visualization
|   +-- visualize.py               # Tokens -> SMPL-X -> HTML/video/interactive 3D
|
|-- scripts/                       # Standalone training scripts
|   +-- vqvae/
|       |-- train_vqvae.py         # Train VQ-VAE on NPZ data (Colab-oriented)
|       |-- train_mgpt_vqvae.py    # Train VQ-VAE on SMPL-X PKL data (Kaggle-oriented)
|       +-- finetune_sentence_vqvae.py  # Fine-tune word-level VQ-VAE for sentence data
|
|-- docs/
|   +-- INFERENCE_AND_VIS.md       # Detailed inference and visualization guide
|
|-- mGPT/                          # Backward-compatible re-exports (legacy import paths)
|
|-- deprecated/                    # Unused/legacy files (candidates for deletion)
|   |-- templates.py               # Legacy prompt templates (not imported)
|   |-- collators.py               # Legacy data collator (not imported)
|   +-- test_overfit.py            # One-off Colab reference script
|
|-- requirements.txt
|-- setup_env.sh
+-- README.md
```

## Installation

```bash
git clone https://github.com/rajvizala/SignMotionGPT.git
cd SignMotionGPT
bash setup_env.sh
```

The setup script installs Python dependencies, downloads the training dataset from Google Drive, and optionally downloads VQ-VAE/SMPL-X assets for visualization.

## Dataset Format

The training dataset is a JSON file with entries in one of these formats:

**Word-level** (used by the word-level pipeline):
```json
[
  {
    "word": "hello",
    "participant_id": "P001",
    "motion_tokens": "42 18 91 205 ..."
  }
]
```

**Sentence-level** (used by the sentence-level pipeline):
```json
[
  {
    "type": "sentence",
    "text": "how are you doing today",
    "motion_tokens": "42 18 91 205 ..."
  }
]
```

Motion tokens are space-separated integer IDs produced by encoding raw SMPL-X sequences through the VQ-VAE.

## Training

### Step 1: Train the VQ-VAE

Before LLM training, encode raw motion data into discrete tokens:

```bash
# From NPZ data (Colab)
python scripts/vqvae/train_vqvae.py

# From SMPL-X PKL data (Kaggle)
python scripts/vqvae/train_mgpt_vqvae.py

# Fine-tune word-level VQ-VAE for sentence data
python scripts/vqvae/finetune_sentence_vqvae.py \
    --vqvae-ckpt path/to/word_level_vqvae.pt \
    --word-data-dir path/to/npz_data \
    --sentence-data-dir path/to/how2sign \
    --stats-path path/to/stats.pt \
    --output-dir path/to/output
```

### Step 2: Train the LLM

**Word-Level Pipeline** (3 stages):

```bash
# Run all stages (1 -> 2 -> 3) + evaluation
python -m training.word_level.pipeline

# Run a specific stage
python -m training.word_level.pipeline --stage 2

# Skip evaluation
python -m training.word_level.pipeline --skip-eval --skip-test-eval
```

Stages:
1. **Motion Language Pre-training**: Learn the "language of motion" from token sequences.
2. **Text-to-Motion Fine-tuning**: Align text prompts (word + participant ID) with motion.
3. **Instruct Tuning**: Word-only prompts without participant ID (1-to-many mapping).

**Sentence-Level Pipeline** (2 stages):

```bash
python -m training.sentence_level.pipeline \
    --data-json path/to/sentence_dataset.json \
    --output-dir ./sentence_model
```

Features: template augmentation, VQ-VAE embedding initialization, length conditioning (`<|LEN_SHORT|>`, `<|LEN_MEDIUM|>`, `<|LEN_LONG|>`), and curriculum learning.

## Configuration

Edit `configs/config.py` or use environment variables:

```bash
# Paths
export DATA_JSON_PATH=./data/motion_llm_dataset.json
export PIPELINE_OUTPUT_DIR=./motion_gpt_full_model

# Hugging Face
export HUGGINGFACE_HUB_TOKEN=hf_...
export HF_UPLOAD_INTERVAL_EPOCHS=2

# Training (word-level)
export S1_EPOCHS=20
export S2_EPOCHS=20
export S3_EPOCHS=2
```

## Inference

```bash
# Generate motion tokens from text
python -m inference.predict --prompt "hello" --stage 3

# With specific participant ID
python -m inference.predict --prompt "walking" --stage 2 --pid P40

# Save output
python -m inference.predict --prompt "jumping" --output motion_output.txt
```

## Visualization

```bash
# Interactive HTML animation
python -m visualization.visualize --tokens "42 18 91 205 ..."

# High-quality video
python -m visualization.visualize --tokens "42 18 91 ..." --render-mode video --output motion.mp4

# Generate + visualize in one step
python -m visualization.visualize --prompt "hello" --stage 3

# Different rendering styles
python -m visualization.visualize --tokens "..." --render-mode video --style hands-highlight
```

See `docs/INFERENCE_AND_VIS.md` for the full visualization guide.

## Evaluation

The pipeline automatically computes:
- **FID** (Frechet Inception Distance) between ground-truth and generated motion features
- **Diversity** of generated motion sequences
- **MIM** (Multimodality Index Metric) measuring generation variety per prompt

Results are saved to `metrics.json` in the output directory.

For held-out test evaluation on SMPL-X data:

```bash
python -m training.word_level.pipeline --test-local-extracted-dir /path/to/test_data
```

## Files Suggested for Removal

The following files in `deprecated/` are not used by any active training pipeline and can be safely deleted:

| File | Reason |
|------|--------|
| `deprecated/templates.py` | Uses legacy `<MOT_BEGIN>` format; imports non-existent function |
| `deprecated/collators.py` | Legacy chat-format collator; not imported anywhere |
| `deprecated/test_overfit.py` | One-off Colab reference script with hardcoded paths |

## Acknowledgments

VQ-VAE architecture adapted from [MotionGPT](https://github.com/OpenMotionLab/MotionGPT). LLM backbone: [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B).
