# Motion LLM: Text-to-Motion Generation with Language Models

A 3-stage training pipeline for fine-tuning language models on discrete motion tokens for text-to-motion generation tasks.

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
git clone https://github.com/yourusername/motion-llm.git
cd motion-llm

# Install dependencies
pip install -r requirements.txt
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

Or use in Python:

```python
from inference import MotionGenerator

generator = MotionGenerator(
    model_path="./output/stage3_t2m_sft",
    dataset_path="path/to/your/motion_dataset.json"
)

motion_tokens = generator.generate("a person walking forward")
print(motion_tokens)
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

## Constrained Decoding

The model uses a custom `LengthAwareMotionLogitsProcessor` that:
1. Enforces valid motion token vocabulary
2. Controls sequence length (min/soft_target/max)
3. Biases generation toward expected length per prompt
4. Prevents generating invalid tokens

## Evaluation Metrics

- **Edit Distance**: Token-level Levenshtein distance to references
- **Length Accuracy**: How well generated lengths match training distribution

## Advanced Usage

### Custom Participant ID

Generate motions for specific participants:

```python
motion_tokens = generator.generate(
    prompt="walking forward",
    participant_id="P001"
)
```

### Adjust Generation Parameters

```python
motion_tokens = generator.generate(
    prompt="dancing",
    max_new_tokens=512,
    temperature=0.8,
    per_prompt_vocab=True  # Restrict to tokens seen with this prompt
)
```

- Built with [Unsloth](https://github.com/unslothai/unsloth) for efficient LLM training
- Inspired by MotionGPT and related motion generation work
