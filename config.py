"""
Configuration file for Motion LLM training
"""
import os
import torch

# Random seed
SEED = 42

# Paths
# WORK_DIR defaults to current working directory if not explicitly set
WORK_DIR = os.environ.get("WORK_DIR", os.getcwd())
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(WORK_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

# Single-file JSON dataset path (can be overridden via env)
DATA_JSON_PATH = os.environ.get(
    "DATA_JSON_PATH",
    os.path.join(DATA_DIR, "motion_llm_dataset.json"),
)

# Directory Configuration
# PIPELINE_OUTPUT_DIR matches test_overfit's default "./motion_gpt_full_model"
PIPELINE_OUTPUT_DIR = os.environ.get("PIPELINE_OUTPUT_DIR", "./motion_gpt_full_model")
METRICS_JSON_PATH = os.path.join(PIPELINE_OUTPUT_DIR, "metrics.json")
CHECKPOINTS_DIR = os.path.join(PIPELINE_OUTPUT_DIR, "checkpoints")

# Model configuration
MODEL_NAME = "Qwen/Qwen3-0.6B"  # Matches test_overfit.py
MAX_SEQ_LEN = 512 # Kept from previous config, though test_overfit uses 256 in datasets
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16

# Evaluation Words (matches test_overfit.py)
EVALUATION_WORDS = ["passport", "send", "library", "push"]

# Training Hyperparameters (matches test_overfit.py)
# Stage 1
S1_EPOCHS = 20
S1_LR = 5e-5
S1_BATCH_SIZE = 8

# Stage 2
S2_EPOCHS = 20
S2_LR = 2e-5
S2_BATCH_SIZE = 8

# Inference Hyperparameters (matches test_overfit.py)
INFERENCE_REPETITION_PENALTY = 1.2
INFERENCE_TEMPERATURE = 0.7
INFERENCE_TOP_K = 50

# Special Tokens (matches test_overfit.py)
M_START = "<M_START>"
M_END = "<M_END>"
PAD_TOKEN = "<PAD>"

# Hugging Face Hub Configuration
HF_USE_HUB = True
HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("hf_auth_token")
HF_USER = os.environ.get("HF_USER", "rdz-falcon") # Derived from test_overfit.py repo ids
HF_STAGE1_REPO_ID = "rdz-falcon/SignMotionGPTfit-archive"
HF_STAGE2_REPO_ID = "rdz-falcon/SignMotionGPTfit-archive"
HF_PRIVATE_REPO = os.environ.get("HF_PRIVATE", "true").lower() != "false"
FORCE_STAGE2_FROM_STAGE1_RAW = os.environ.get("FORCE_STAGE2_FROM_STAGE1", "false")
FORCE_STAGE2_FROM_STAGE1 = str(FORCE_STAGE2_FROM_STAGE1_RAW).strip().lower() not in ("0", "false", "no", "off")
HF_STAGE2_SAVE_SUBDIR = os.environ.get("HF_STAGE2_SAVE_SUBDIR", "stage2_v2")
CHECKPOINT_UPLOAD_INTERVAL_EPOCHS = int(os.environ.get("HF_UPLOAD_INTERVAL_EPOCHS", "2"))
HF_DISABLE_PROGRESS = os.environ.get("HF_DISABLE_PROGRESS", "true").lower() != "false"

# Evaluation controls
RUN_EVALS_ONLY = False
EVAL_SAMPLE_LIMIT = 100

# Test Eval Config (from test_dataset_eval.py defaults)
TEST_EVAL_OUTPUT_DIR = os.environ.get("TEST_EVAL_OUTPUT_DIR", PIPELINE_OUTPUT_DIR)
TEST_EVAL_DOWNLOAD_DIR = os.environ.get(
    "TEST_EVAL_DOWNLOAD_DIR", os.path.join(WORK_DIR, "test_data", "downloads")
)
TEST_EVAL_EXTRACT_DIR = os.environ.get(
    "TEST_EVAL_EXTRACT_DIR", os.path.join(WORK_DIR, "test_data", "extracted")
)
TEST_EVAL_SAMPLE_LIMIT = int(os.environ.get("TEST_EVAL_SAMPLE_LIMIT", "300"))
TEST_EVAL_MAX_ZIPS = int(os.environ.get("TEST_EVAL_MAX_ZIPS", "500"))
TEST_EVAL_HF_REPO = os.environ.get("TEST_EVAL_HF_REPO", "rdz-falcon/SignMotionGPTfit-archive")
TEST_EVAL_HF_SUBFOLDER = os.environ.get(
    "TEST_EVAL_HF_SUBFOLDER", f"{HF_STAGE2_SAVE_SUBDIR}/latest"
)
