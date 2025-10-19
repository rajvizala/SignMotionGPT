#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   bash setup_env.sh
#
# - Installs Python dependencies from requirements.txt
# - Downloads a public Google Drive dataset file into ./data/motion_llm_dataset.json
# - Exports env vars for this session (optional) and prints instructions

THIS_DIR="$(pwd)"
DATA_DIR="$THIS_DIR/data"
mkdir -p "$DATA_DIR"

# --- Explicit placeholders (replace these later) ---
# Training dataset
GDRIVE_ID="11711RgTmzauXpYVFoqLF8DZXiZlZovfn"

# Visualization assets (optional - only needed for visualize.py)
VQVAE_MODEL_ID="1JEMKVZWFG4Ue7k3Nm7q1o7-uBVsVricY"
VQVAE_STATS_ID="1WTwP5DdBl4c-X5Kj7jXtlEHofOX2BifZ"
SMPLX_MODELS_ID="1tZEfqw9zHgOaBEw5X_oazAEnesRtE9ky"

# Hugging Face token
HF_TOKEN_IN=""
# ---------------------------------------------------

echo "Installing Python dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt

if [[ -n "$GDRIVE_ID" ]] && [[ "$GDRIVE_ID" != "YOUR_GOOGLE_DRIVE_FILE_ID_HERE" ]]; then
  echo "Downloading training dataset from Google Drive (file id: $GDRIVE_ID)..."
  gdown --id "$GDRIVE_ID" -O "$DATA_DIR/motion_llm_dataset.json"
else
  echo "No training dataset Google Drive ID provided. Skipping dataset download."
fi

# Download visualization assets if IDs are provided
if [[ -n "$VQVAE_MODEL_ID" ]] && [[ "$VQVAE_MODEL_ID" != "YOUR_VQVAE_CHECKPOINT_GDRIVE_ID_HERE" ]]; then
  echo "Downloading VQ-VAE model from Google Drive (file id: $VQVAE_MODEL_ID)..."
  gdown --id "$VQVAE_MODEL_ID" -O "$DATA_DIR/vqvae_model.pt"
fi

if [[ -n "$VQVAE_STATS_ID" ]] && [[ "$VQVAE_STATS_ID" != "YOUR_VQVAE_STATS_GDRIVE_ID_HERE" ]]; then
  echo "Downloading VQ-VAE stats from Google Drive (file id: $VQVAE_STATS_ID)..."
  gdown --id "$VQVAE_STATS_ID" -O "$DATA_DIR/vqvae_stats.pt"
fi

if [[ -n "$SMPLX_MODELS_ID" ]] && [[ "$SMPLX_MODELS_ID" != "YOUR_SMPLX_MODELS_GDRIVE_ID_HERE" ]]; then
  echo "Downloading SMPL-X neutral model (.npz) from Google Drive (file id: $SMPLX_MODELS_ID)..."
  mkdir -p "$DATA_DIR/smplx_models"
  gdown --id "$SMPLX_MODELS_ID" -O "$DATA_DIR/smplx_models/SMPLX_NEUTRAL.npz"
  echo "Saved SMPLX_NEUTRAL.npz to $DATA_DIR/smplx_models"
fi

if [[ -n "$HF_TOKEN_IN" ]]; then
  echo "Exporting HUGGINGFACE_HUB_TOKEN for this shell session..."
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN_IN"
fi

echo
echo "Environment setup complete."
echo "- WORK_DIR defaults to: $THIS_DIR"
echo "- DATA_JSON_PATH defaults to: $DATA_DIR/motion_llm_dataset.json"
echo "- To persist HF token, set an environment variable before running:"
echo "    export HUGGINGFACE_HUB_TOKEN=hf_..."
echo
echo "You can now run your training scripts."