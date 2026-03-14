# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

SignMotionGPT is a Python ML pipeline that generates sign language motions from text using an LLM (Qwen3-0.6B). The primary pipeline is in `train_pipeline.py` with a 3-stage training flow (Motion LM, Text-to-Motion, Instruct tuning). See `README.md` for full documentation.

### Running the pipeline

- **Primary entry point**: `python3 train_pipeline.py` (orchestrates all 3 stages + evaluation)
- **Inference**: `python3 inference.py --prompt "walking" --stage 3`
- Training and inference work on CPU (slow) or GPU. Set `HF_USE_HUB=false` (env var) to skip Hugging Face Hub sync when no token is available.
- Use environment variables to control training epochs and batch sizes (see `config.py` and `README.md`).

### Key caveats

- The `unsloth` package in `requirements.txt` is a **legacy/optional** dependency only needed for the old LoRA pipeline. The primary pipeline uses standard `transformers` and does not need `unsloth`. Skip installing it if it fails (GPU/CUDA-only package).
- `templates.py` has a pre-existing import error (`ids_to_motion_specials` missing from `data.py`). This module is not used by the primary pipeline.
- No GPU is available in the Cloud Agent VM. Full training on all 23k samples is impractical on CPU. For testing, use a small subset of data (5-50 samples) with 1 epoch.
- The dataset (`data/motion_llm_dataset.json`, ~17MB) is downloaded from Google Drive via `gdown`. If the download fails due to rate limits, re-run or use `setup_env.sh`.
- There are no formal test suites (pytest, unittest), no linting configs (flake8, ruff, pylint), and no CI/CD. Validation is done by running the pipeline scripts and checking outputs.
- `python3` must be used explicitly (`python` is not aliased on this VM).
