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
GDRIVE_ID="YOUR_GOOGLE_DRIVE_FILE_ID_HERE"
HF_TOKEN_IN="YOUR_HUGGINGFACE_TOKEN_HERE"
# ---------------------------------------------------

echo "Installing Python dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt

if [[ -n "$GDRIVE_ID" ]]; then
  echo "Downloading dataset from Google Drive (file id: $GDRIVE_ID)..."
  # Save to standard filename expected by config.py
  gdown --id "$GDRIVE_ID" -O "$DATA_DIR/motion_llm_dataset.json"
else
  echo "No Google Drive file id provided. Skipping dataset download."
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
