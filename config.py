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
OUT_S1 = f"{WORK_DIR}/stage1_mlm"
OUT_S2 = f"{WORK_DIR}/stage2_multitask"
OUT_S3 = f"{WORK_DIR}/stage3_t2m_sft"

# Model configuration
MODEL_NAME = "unsloth/Qwen3-1.7B"  # or "unsloth/gemma-3-1b-it"
MAX_SEQ_LEN = 512
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16

# Training hyperparameters
BATCH_TRAIN = 8
BATCH_EVAL = 8
GRAD_ACCUM = 8
LR = 1e-5
WARMUP = 0.1
LOG_STEPS = 20
EVAL_STEPS = 100
SAVE_STEPS = 500

# Epochs per stage
EPOCHS_S1 = 3
EPOCHS_S2 = 2
EPOCHS_S3 = 3

# Sampling limits (None = use all data)
MAX_TRAIN_SAMPLES_S1 = None
MAX_TRAIN_SAMPLES_S2 = None
MAX_TRAIN_SAMPLES_S3 = None
MAX_EVAL_SAMPLES = 1000

# LoRA configuration
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LORA_MODULES_TO_SAVE = ["embed_tokens", "lm_head"]

# Generation parameters
GEN_MAX_NEW_TOKENS = 256
GEN_TEMPERATURE = 0.7
GEN_TOP_P = 0.9
GEN_TOP_K = 0
GEN_NO_REPEAT_NGRAM_SIZE = 6
GEN_REPETITION_PENALTY = 1.2
GEN_END_LOGIT_SLOPE = 0.25

# System prompt
SYSTEM_MSG = (
    "You are a MotionGPT-style assistant. Use discrete motion tokens enclosed by MOT_BEGIN/MOT_END."
)

# Hugging Face Hub configuration
# Set HF_TOKEN via environment or here for convenience (prefer environment for security).
HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN", "")
HF_USER = os.environ.get("HF_USER", "rdz-falcon")

# Per-stage Hub repos (override via env if needed)
HUB_REPO_S1 = os.environ.get("HUB_REPO_S1", f"{HF_USER}/signmotiongpt-stage1")
HUB_REPO_S2 = os.environ.get("HUB_REPO_S2", f"{HF_USER}/signmotiongpt-stage2")
HUB_REPO_S3 = os.environ.get("HUB_REPO_S3", f"{HF_USER}/signmotiongpt-stage3")