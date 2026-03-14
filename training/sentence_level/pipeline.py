"""
train_sentence_pipeline_v2.py

Sentence-Level SignMotionGPT Training Pipeline V2

CHANGES FROM V1:
1. DEDUPLICATION WITH TEMPLATE AUGMENTATION - Keep unique sentences, multiply with 4 instruction templates
2. VQ-VAE WEIGHT INITIALIZATION - "Semantic Bridge" for motion tokens
3. IMPLICIT LENGTH CONDITIONING - <|LEN_SHORT|>, <|LEN_MEDIUM|>, <|LEN_LONG|>
4. CUMULATIVE CURRICULUM LEARNING - Gradual data introduction by length
5. BEAM SEARCH EVALUATION - Deterministic generation for smoother outputs
6. 4 INSTRUCTION TEMPLATES for Stage 2 (data augmentation)

Key Features:
1. SENTENCE-LEVEL ONLY - No word-level data
2. DYNAMIC PADDING - Pad to longest in batch, not fixed max_length
3. On-the-fly tokenization in collate_fn to reduce memory
4. Mixed precision training with torch.amp
5. Gradient accumulation support
6. Better memory management
7. COSINE/PLATEAU LR SCHEDULER with warmup
8. 4 INSTRUCTION TEMPLATES for Stage 2 (data augmentation)

Two-stage training approach:
- Stage 1: Motion Language Pre-training
- Stage 2: Instruction Tuning with multiple templates
"""

import os
import sys
import json
import math
import random
import re
import time
import shutil
import argparse
import warnings
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from huggingface_hub import HfApi, hf_hub_download, upload_folder

# Import evaluation functions from metrics module
try:
    from evaluation.metrics import (
        evaluate_metrics_encoder_style,
        run_inference_on_all_samples,
        evaluate_metrics_motiongpt_style,
        save_side_by_side_visualizations,
        evaluate_sentence_level_encoder_style,
        generate_motion,
    )
except ImportError:
    print("Warning: Could not import evaluation.metrics module. Evaluation will be skipped.")
    evaluate_metrics_encoder_style = None
    run_inference_on_all_samples = None
    evaluate_metrics_motiongpt_style = None
    save_side_by_side_visualizations = None
    evaluate_sentence_level_encoder_style = None
    generate_motion = None

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

SEED = 42
MODEL_NAME = "Qwen/Qwen3-0.6B"

# Max sequence length for sentence-level data
MAX_SEQ_LEN = 256

M_START = "<M_START>"
M_END = "<M_END>"
PAD_TOKEN = "<PAD>"

# Length conditioning tokens (NEW in V2)
LEN_SHORT = "<|LEN_SHORT|>"
LEN_MEDIUM = "<|LEN_MEDIUM|>"
LEN_LONG = "<|LEN_LONG|>"

# Length thresholds for curriculum learning
CURRICULUM_SHORT_MAX = 30
CURRICULUM_MEDIUM_MAX = 80

# Stage 2 Instruction Templates (for data augmentation)
# Each unique sentence will be paired with ALL templates to increase data diversity
TEMPLATES = [
    "Instruction: Generate sign language motion for the sentence: '{text}'",
    "Instruction: Translate this text to sign language: '{text}'",
    "Instruction: How would you sign '{text}'?",
    "Instruction: Create a sign language animation for: '{text}'"
]

# Training Hyperparameters
# Stage 1: Motion Language Pre-training
S1_EPOCHS = 20
S1_LR = 5e-5
S1_BATCH_SIZE = 32
S1_GRAD_ACCUM = 1

# Stage 2: Instruction Tuning
S2_EPOCHS = 40
S2_LR = 3e-5
S2_BATCH_SIZE = 32
S2_GRAD_ACCUM = 1

# Learning Rate Scheduler Configuration
LR_WARMUP_RATIO = 0.05   # 5% of total steps for warmup
LR_MIN_RATIO = 0.1       # Minimum LR will be 10% of initial

# LR SCHEDULE MODE
# "cosine"   - Warmup + cosine decay
# "constant" - Constant LR after warmup
# "plateau"  - Reduce LR when loss plateaus (recommended)
LR_SCHEDULE_MODE = "plateau"

# Plateau scheduler settings (only used if LR_SCHEDULE_MODE = "plateau")
LR_PLATEAU_FACTOR = 0.5
LR_PLATEAU_PATIENCE = 3
LR_PLATEAU_MIN = 1e-6

# Output
PIPELINE_OUTPUT_DIR = os.environ.get("PIPELINE_OUTPUT_DIR", "./sentence_motion_model_v2")
CHECKPOINTS_DIR = os.path.join(PIPELINE_OUTPUT_DIR, "checkpoints")

# HuggingFace Hub
HF_USE_HUB = True
HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
HF_SENTENCE_REPO_ID = os.environ.get("HF_SENTENCE_REPO_ID", "SignMotionGPT-Sentence-v2")
HF_PRIVATE_REPO = False
CHECKPOINT_UPLOAD_INTERVAL_EPOCHS = 2

# Evaluation Configuration
EVAL_SAMPLE_LIMIT = int(os.environ.get("EVAL_SAMPLE_LIMIT", "100"))
EVAL_INFERENCE_EXAMPLES = int(os.environ.get("EVAL_INFERENCE_EXAMPLES", "5"))
RUN_EVALS_ONLY = os.environ.get("RUN_EVALS_ONLY", "false").lower() == "true"

# =============================================================================
# Utility Functions
# =============================================================================

def set_seeds(seed: int = SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _format_seconds(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:d}:{m:02d}:{s:02d}" if h > 0 else f"{m:d}:{s:02d}"

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def print_banner(title: str, char: str = "=", width: int = 80):
    print("\n" + char * width)
    print(f"  {title}")
    print(char * width)

# =============================================================================
# Data Loading - Sentence-Level Only
# =============================================================================

def load_sentence_dataset(json_path: str) -> List[Dict[str, Any]]:
    """Load dataset and filter for sentence-level samples only."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset not found at: {json_path}")
    
    print(f"\n[Data] Loading dataset from: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        data = data.get("samples", data.get("data", []))
    
    # Filter for sentence-level data only
    sentence_data = []
    for item in data:
        item_type = item.get("type", "").lower()
        if item_type == "sentence":
            sentence_data.append(item)
    
    print(f"   Total samples in file: {len(data)}")
    print(f"   Sentence samples extracted: {len(sentence_data)}")
    
    if len(sentence_data) == 0:
        print("   WARNING: No sentence-level samples found!")
        print("   Make sure your dataset has items with 'type': 'sentence'")
    
    return sentence_data

def analyze_sentence_dataset(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze sentence-level dataset statistics."""
    stats = {
        "total_samples": len(data),
        "unique_sentences": set(),
        "all_motion_tokens": set(),
        "max_motion_length": 0,
        "min_motion_length": float('inf'),
        "avg_motion_length": 0,
        "motion_length_distribution": defaultdict(int),
    }
    
    motion_lengths = []
    
    for item in data:
        text = item.get("text") or item.get("sentence", "")
        stats["unique_sentences"].add(text[:100])
        
        motion_str = item.get("motion_tokens", "")
        motion_len = len(motion_str.split()) if motion_str.strip() else 0
        
        if motion_len > 0:
            motion_lengths.append(motion_len)
            stats["max_motion_length"] = max(stats["max_motion_length"], motion_len)
            stats["min_motion_length"] = min(stats["min_motion_length"], motion_len)
            
            # Bucket into ranges for distribution
            bucket = (motion_len // 10) * 10
            stats["motion_length_distribution"][bucket] += 1
        
        for token in motion_str.split():
            if token.strip():
                stats["all_motion_tokens"].add(token.strip())
    
    if motion_lengths:
        stats["avg_motion_length"] = sum(motion_lengths) / len(motion_lengths)
    else:
        stats["min_motion_length"] = 0
    
    stats["unique_sentences"] = len(stats["unique_sentences"])
    stats["num_unique_tokens"] = len(stats["all_motion_tokens"])
    stats["all_motion_tokens"] = sorted(list(stats["all_motion_tokens"]))
    stats["motion_length_distribution"] = dict(sorted(stats["motion_length_distribution"].items()))
    
    return stats

def prepare_motion_tokens(data: List[Dict[str, Any]]) -> List[str]:
    """Extract unique motion tokens from dataset."""
    all_tokens = set()
    for item in data:
        for token in item.get("motion_tokens", "").split():
            if token.strip():
                all_tokens.add(f"<M{token.strip()}>")
    return sorted(list(all_tokens))

def deduplicate_sentence_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate sentence-level data based on normalized text.
    Keeps only unique sentences (removing augmentation duplicates).
    
    Returns:
        List of unique samples (one per unique sentence text)
    """
    seen_texts = {}
    unique_samples = []
    
    for item in data:
        text = (item.get("text") or item.get("sentence", "")).strip().lower()
        
        # Skip empty text
        if not text:
            continue
        
        # Keep first occurrence of each unique text
        if text not in seen_texts:
            seen_texts[text] = item
            unique_samples.append(item)
    
    print(f"   Deduplication: {len(data)} -> {len(unique_samples)} unique sentences")
    return unique_samples

# =============================================================================
# VQ-VAE Weight Initialization (NEW in V2 - "Semantic Bridge")
# =============================================================================

def initialize_embeddings_from_vqvae(llm_model, tokenizer, vqvae_ckpt_path: str, device):
    """
    Initialize motion token embeddings from VQ-VAE codebook weights.
    
    This creates a "semantic bridge" by copying the learned VQ-VAE codebook
    vectors into the LLM's input embedding matrix for tokens <M0> through <M511>.
    
    Since LLM dimension (e.g., 1536 for Qwen3-0.6B) > VQ dimension (512),
    we copy VQ weights to the first N dimensions and pad the rest with zeros.
    """
    print(f"\n[Init] Loading VQ-VAE weights from {vqvae_ckpt_path}...")
    
    if not os.path.exists(vqvae_ckpt_path):
        print(f"  [WARNING] VQ-VAE checkpoint not found at {vqvae_ckpt_path}. Skipping init.")
        return
    
    ckpt = torch.load(vqvae_ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    
    # Extract codebook - try multiple possible key patterns
    # The VQ-VAE uses QuantizeEMAReset which stores codebook as a buffer (not .weight)
    codebook = None
    
    # Priority order of keys to try
    possible_keys = [
        'vqvae.quantizer.codebook',           # EMA quantizer buffer (most common for this project)
        'quantizer.codebook',                  # Without vqvae prefix
        'vqvae.quantizer.codebook.weight',    # If saved as parameter
        'quantizer.codebook.weight',           # Without vqvae prefix
        'vqvae.quantizer.embedding.weight',   # Standard Quantizer
        'quantizer.embedding.weight',          # Without vqvae prefix
    ]
    
    for key in possible_keys:
        if key in state_dict:
            codebook = state_dict[key]
            print(f"  Found codebook at key: {key}")
            break
    
    # Fallback: search for any key containing 'codebook'
    if codebook is None:
        for key in state_dict.keys():
            if 'codebook' in key.lower():
                codebook = state_dict[key]
                print(f"  Found codebook at key: {key}")
                break
    
    # Final fallback: search for embedding in quantizer
    if codebook is None:
        for key in state_dict.keys():
            if 'quantizer' in key.lower() and 'embedding' in key.lower():
                codebook = state_dict[key]
                print(f"  Found codebook embedding at key: {key}")
                break
    
    if codebook is None:
        print("  [WARNING] Could not find codebook in checkpoint. Skipping init.")
        print(f"  Available keys containing 'quant': {[k for k in state_dict.keys() if 'quant' in k.lower()]}")
        return
    
    input_embeddings = llm_model.get_input_embeddings().weight.data
    n_codes, vq_dim = codebook.shape
    llm_dim = input_embeddings.shape[1]
    
    print(f"  Injecting {n_codes} VQ codes (dim {vq_dim}) into LLM (dim {llm_dim})...")
    
    count = 0
    with torch.no_grad():
        for i in range(n_codes):
            token = f"<M{i}>"
            if token in tokenizer.get_vocab():
                idx = tokenizer.convert_tokens_to_ids(token)
                # Copy weights to first vq_dim columns, zero out the rest
                input_embeddings[idx, :vq_dim] = codebook[i]
                input_embeddings[idx, vq_dim:] = 0.0
                count += 1
    
    print(f"  Initialized {count} motion tokens from VQ-VAE codebook.")

# =============================================================================
# Model Setup
# =============================================================================

def setup_model_and_tokenizer(model_name: str, motion_tokens: List[str]):
    """Initialize model and tokenizer with motion tokens."""
    print(f"\n[Model] Loading base model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model in default dtype first, convert after resizing embeddings
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=None
    )
    
    # V2: Add length conditioning tokens alongside standard special tokens
    tokenizer.add_special_tokens({
        "pad_token": PAD_TOKEN,
        "additional_special_tokens": [M_START, M_END, LEN_SHORT, LEN_MEDIUM, LEN_LONG]
    })
    
    print(f"   Adding {len(motion_tokens)} motion tokens...")
    tokenizer.add_tokens(motion_tokens, special_tokens=True)
    
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Convert to bfloat16 after resizing embeddings
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        model = model.to(torch.bfloat16)
        print(f"   Model converted to bfloat16")
    else:
        model = model.to(torch.float16)
        print(f"   Model converted to float16")
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    
    print(f"   Final vocabulary size: {len(tokenizer)}")
    
    return model, tokenizer

# =============================================================================
# Dataset Classes (V2 with Length Conditioning)
# =============================================================================

class Stage1Dataset(Dataset):
    """
    Stage 1: Motion Language Pre-training
    Returns RAW motion sequences - tokenization happens in collate_fn.
    """
    
    def __init__(self, data: List[Dict[str, Any]]):
        self.sequences = []
        self.motion_lengths = []  # Track lengths for curriculum learning
        
        for item in data:
            tokens_str = item.get("motion_tokens", "")
            if not tokens_str.strip():
                continue
            
            motion_len = len(tokens_str.split())
            wrapped_tokens = " ".join([f"<M{t}>" for t in tokens_str.split()])
            full_sequence = f"{M_START} {wrapped_tokens} {M_END}"
            self.sequences.append(full_sequence)
            self.motion_lengths.append(motion_len)
        
        print(f"   Stage1Dataset: {len(self.sequences)} motion sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]
    
    def get_motion_length(self, idx):
        """Get motion length for curriculum filtering."""
        return self.motion_lengths[idx]


class Stage2Dataset(Dataset):
    """
    Stage 2: Instruction Tuning (V2 with Length Conditioning + Template Augmentation)
    Returns (prompt, full_text) tuples for instruction-based motion generation.
    
    V2 Changes:
    - Adds length conditioning token to prompt based on GT motion length
    - Uses 4 different instruction templates for each sentence
    - This effectively creates 4x the training data for better generalization
    """
    
    def __init__(self, data: List[Dict[str, Any]]):
        self.items = []
        
        for item in data:
            text = (item.get("text") or item.get("sentence", "")).strip()
            tokens_str = item.get("motion_tokens", "")
            
            if not text or not tokens_str.strip():
                continue
            
            # V2: Calculate motion length and determine length token
            motion_len = len(tokens_str.split())
            if motion_len < CURRICULUM_SHORT_MAX:
                len_token = LEN_SHORT
            elif motion_len < CURRICULUM_MEDIUM_MAX:
                len_token = LEN_MEDIUM
            else:
                len_token = LEN_LONG
            
            wrapped_tokens = " ".join([f"<M{t}>" for t in tokens_str.split()])
            target_sequence = f"{M_START} {wrapped_tokens} {M_END}"
            
            # V2: Create one sample for EACH template (4x data augmentation)
            for template in TEMPLATES:
                # Format the template with the text and add length conditioning
                instruction = template.format(text=text)
                prompt = f"{instruction} (Length: {len_token})\nMotion: "
                
                self.items.append({
                    "prompt": prompt,
                    "full_text": prompt + target_sequence,
                    "sentence": text,
                    "motion_length": motion_len,  # Store for curriculum learning
                    "len_token": len_token,
                })
        
        print(f"   Stage2Dataset: {len(self.items)} items (from {len(data)} samples x {len(TEMPLATES)} templates)")
        
        # Print length distribution
        short_count = sum(1 for item in self.items if item["len_token"] == LEN_SHORT)
        medium_count = sum(1 for item in self.items if item["len_token"] == LEN_MEDIUM)
        long_count = sum(1 for item in self.items if item["len_token"] == LEN_LONG)
        print(f"      Short (<{CURRICULUM_SHORT_MAX}): {short_count}")
        print(f"      Medium ({CURRICULUM_SHORT_MAX}-{CURRICULUM_MEDIUM_MAX}): {medium_count}")
        print(f"      Long (>{CURRICULUM_MEDIUM_MAX}): {long_count}")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return self.items[idx]
    
    def get_motion_length(self, idx):
        """Get motion length for curriculum filtering."""
        return self.items[idx]["motion_length"]


# =============================================================================
# Dynamic Padding Collate Functions
# =============================================================================

def create_stage1_collate_fn(tokenizer, max_length=MAX_SEQ_LEN):
    """
    Collate function for Stage 1 - pads to LONGEST IN BATCH.
    """
    pad_token_id = tokenizer.pad_token_id
    
    def collate_fn(batch):
        tokenized = tokenizer(
            batch,
            truncation=True,
            max_length=max_length,
            padding=True,  # Dynamic padding to longest in batch
            return_tensors="pt"
        )
        
        labels = tokenized["input_ids"].clone()
        
        # Mask padding tokens with -100
        if pad_token_id is not None:
            pad_mask = (tokenized["input_ids"] == pad_token_id)
            labels[pad_mask] = -100
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels
        }
    
    return collate_fn


def create_stage2_collate_fn(tokenizer, max_length=MAX_SEQ_LEN):
    """
    Collate function for Stage 2 - dynamic padding with prompt masking.
    """
    pad_token_id = tokenizer.pad_token_id
    m_start_id = tokenizer.convert_tokens_to_ids(M_START)
    
    def collate_fn(batch):
        prompts = [item["prompt"] for item in batch]
        full_texts = [item["full_text"] for item in batch]
        
        tokenized = tokenizer(
            full_texts,
            truncation=True,
            max_length=max_length,
            padding=True,  # Dynamic padding
            return_tensors="pt"
        )
        
        labels = tokenized["input_ids"].clone()
        
        for i in range(len(full_texts)):
            full_ids = tokenized["input_ids"][i]
            
            # Find M_START token position to determine prompt length
            prompt_len = len(full_ids)
            if m_start_id is not None:
                m_start_positions = (full_ids == m_start_id).nonzero(as_tuple=True)[0]
                if len(m_start_positions) > 0:
                    prompt_len = m_start_positions[0].item()
            
            # Mask prompt tokens
            labels[i, :prompt_len] = -100
            
            # Mask padding tokens
            if pad_token_id is not None:
                pad_mask = (tokenized["input_ids"][i] == pad_token_id)
                labels[i, pad_mask] = -100
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels
        }
    
    return collate_fn


# =============================================================================
# HuggingFace Hub Utilities
# =============================================================================

def resolve_and_ensure_repo(repo_id: str, hf_auth_token: Optional[str] = None):
    """Create or verify HuggingFace repository."""
    if not HF_USE_HUB:
        return None
    
    token = hf_auth_token or HF_TOKEN
    if not token:
        print("[HF] Token not found.")
        return None
    
    api = HfApi()
    
    try:
        who = api.whoami(token=token)
        namespace = who.get("name")
    except Exception:
        namespace = None
    
    full_repo_id = f"{namespace}/{repo_id}" if "/" not in repo_id and namespace else repo_id
    
    try:
        api.create_repo(
            repo_id=full_repo_id,
            token=token,
            repo_type="model",
            private=HF_PRIVATE_REPO,
            exist_ok=True,
        )
        print(f"[HF] Repo ready: {full_repo_id}")
    except Exception as e:
        print(f"[HF] create_repo: {e}")
    
    return full_repo_id


def save_and_push_checkpoint(
    stage: str,
    epoch: int,
    model,
    tokenizer,
    optimizer,
    avg_loss: float,
    repo_id: Optional[str],
    total_epochs: int,
    scheduler=None,
    global_step: int = 0,
):
    """Save checkpoint locally and optionally push to HuggingFace."""
    token = HF_TOKEN
    epoch_number = epoch + 1
    stage_dir = os.path.join(CHECKPOINTS_DIR, stage)
    epoch_dir = os.path.join(stage_dir, f"epoch-{epoch_number:03d}")
    latest_dir = os.path.join(stage_dir, "latest")
    
    _ensure_dir(epoch_dir)
    
    model.save_pretrained(epoch_dir)
    tokenizer.save_pretrained(epoch_dir)
    torch.save(optimizer.state_dict(), os.path.join(epoch_dir, "optimizer.pt"))
    
    if scheduler is not None:
        torch.save(scheduler.state_dict(), os.path.join(epoch_dir, "scheduler.pt"))
    
    training_state = {
        "stage": stage,
        "epoch_completed": epoch_number,
        "total_epochs": total_epochs,
        "avg_loss": float(avg_loss),
        "global_step": global_step,
        "saved_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(epoch_dir, "training_state.json"), "w") as f:
        json.dump(training_state, f, indent=2)
    
    if os.path.exists(latest_dir):
        shutil.rmtree(latest_dir)
    shutil.copytree(epoch_dir, latest_dir)
    
    if HF_USE_HUB and repo_id and token:
        try:
            upload_folder(
                repo_id=repo_id,
                folder_path=latest_dir,
                path_in_repo=f"{stage}/latest",
                repo_type="model",
                token=token,
                commit_message=f"{stage}: epoch {epoch_number} (loss={avg_loss:.4f})",
            )
            print(f"[HF] Pushed to: {repo_id}/{stage}/latest")
        except Exception as e:
            print(f"[HF] Push failed: {e}")


# =============================================================================
# Curriculum Learning Helper (NEW in V2)
# =============================================================================

def get_curriculum_config(epoch: int) -> Dict[str, Any]:
    """
    Get curriculum configuration for the current epoch.
    
    Curriculum Schedule:
    - Epoch 0-5:   SHORT only (motion_length < 30), batch_size=32
    - Epoch 6-10:  SHORT + MEDIUM (motion_length < 80), batch_size=16
    - Epoch 11-15: LONG_FOCUS - weighted sampling (70% long, 15% medium, 15% short), batch_size=16
    - Epoch 16+:   ALL data uniformly, batch_size=16
    
    Returns dict with:
    - max_length: Maximum motion length to include (None = all)
    - batch_size: Recommended batch size for this phase
    - phase: Human-readable phase name
    - sampling_weights: (optional) dict with target proportions per length category
    """
    if epoch < 6:
        return {
            "max_length": CURRICULUM_SHORT_MAX,
            "batch_size": 32,
            "phase": "SHORT"
        }
    elif epoch < 11:
        return {
            "max_length": CURRICULUM_MEDIUM_MAX,
            "batch_size": 16,
            "phase": "SHORT+MEDIUM"
        }
    elif epoch < 16:
        return {
            "max_length": None,  # All data included
            "batch_size": 32,
            "phase": "LONG_FOCUS",
            "sampling_weights": {"long": 0.70, "medium": 0.15, "short": 0.15}
        }
    else:
        return {
            "max_length": None,  # All data
            "batch_size": 32,
            "phase": "ALL"
        }


def get_curriculum_indices(dataset, epoch: int, is_stage2: bool = True) -> Tuple[List[int], int, Optional[WeightedRandomSampler]]:
    """
    Get dataset indices for curriculum learning based on current epoch.
    
    Curriculum Schedule:
    - Epoch 0-5:   Train only on SHORT samples (motion_length < 30)
    - Epoch 6-10:  Train on SHORT + MEDIUM samples (motion_length < 80)
    - Epoch 11-15: LONG_FOCUS - all data with weighted sampling (70% long, 15% medium, 15% short)
    - Epoch 16+:   Train on ALL data uniformly
    
    Returns:
        Tuple of (list of indices, recommended batch size, optional WeightedRandomSampler)
    """
    all_indices = list(range(len(dataset)))
    config = get_curriculum_config(epoch)
    
    max_length = config["max_length"]
    batch_size = config["batch_size"]
    phase = config["phase"]
    sampling_weights = config.get("sampling_weights", None)
    
    # --- LONG_FOCUS phase: weighted sampling across all data ---
    if sampling_weights is not None:
        # Categorize every sample
        short_ids, medium_ids, long_ids = [], [], []
        for idx in all_indices:
            ml = dataset.get_motion_length(idx)
            if ml < CURRICULUM_SHORT_MAX:
                short_ids.append(idx)
            elif ml < CURRICULUM_MEDIUM_MAX:
                medium_ids.append(idx)
            else:
                long_ids.append(idx)
        
        n_short = len(short_ids)
        n_medium = len(medium_ids)
        n_long = len(long_ids)
        total = len(all_indices)
        
        # Compute per-sample weight so that the expected draw proportion
        # matches the target percentages (handle empty buckets gracefully)
        w_short  = (sampling_weights["short"]  / n_short)  if n_short  > 0 else 0.0
        w_medium = (sampling_weights["medium"] / n_medium) if n_medium > 0 else 0.0
        w_long   = (sampling_weights["long"]   / n_long)   if n_long   > 0 else 0.0
        
        # Build weight vector aligned with all_indices (0..N-1)
        weights = [0.0] * total
        for idx in short_ids:
            weights[idx] = w_short
        for idx in medium_ids:
            weights[idx] = w_medium
        for idx in long_ids:
            weights[idx] = w_long
        
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=total,        # draw same number of samples as full dataset
            replacement=True,         # required for weighted sampling
        )
        
        print(f"  [Curriculum] Epoch {epoch+1}: {phase} phase -> {total} samples with weighted sampling")
        print(f"      Target: {sampling_weights['long']:.0%} long ({n_long}), "
              f"{sampling_weights['medium']:.0%} medium ({n_medium}), "
              f"{sampling_weights['short']:.0%} short ({n_short})")
        print(f"      batch_size={batch_size}")
        
        return all_indices, batch_size, sampler
    
    # --- ALL phase: uniform sampling over everything ---
    if max_length is None:
        print(f"  [Curriculum] Epoch {epoch+1}: {phase} phase -> {len(all_indices)}/{len(all_indices)} samples, batch_size={batch_size}")
        return all_indices, batch_size, None
    
    # --- SHORT / SHORT+MEDIUM phases: filter by max length ---
    filtered_indices = []
    for idx in all_indices:
        motion_len = dataset.get_motion_length(idx)
        if motion_len < max_length:
            filtered_indices.append(idx)
    
    print(f"  [Curriculum] Epoch {epoch+1}: {phase} phase -> {len(filtered_indices)}/{len(all_indices)} samples (max_len={max_length}), batch_size={batch_size}")
    
    return filtered_indices, batch_size, None


# =============================================================================
# Training Function (V2 with Curriculum Learning)
# =============================================================================

def train_stage(
    stage_name: str,
    model,
    tokenizer,
    dataset,
    collate_fn,
    device,
    epochs: int,
    lr: float,
    batch_size: int,
    grad_accum: int = 1,
    hf_repo_id: Optional[str] = None,
    start_epoch: int = 0,
    use_curriculum: bool = False,  # V2: Enable curriculum learning
):
    """
    Training function for all stages.
    
    V2 Changes:
    - Added curriculum learning support for Stage 2
    - Dynamic data filtering based on epoch
    
    Includes LR scheduler with warmup + cosine/plateau decay.
    """
    
    print_banner(f"STAGE: {stage_name}")
    print(f"  Samples: {len(dataset)}")
    print(f"  Epochs: {start_epoch + 1} to {epochs}")
    print(f"  Initial Learning Rate: {lr}")
    print(f"  Batch Size: {batch_size} (effective: {batch_size * grad_accum})")
    print(f"  LR Schedule Mode: {LR_SCHEDULE_MODE}")
    print(f"  Curriculum Learning: {'ENABLED' if use_curriculum else 'disabled'}")
    
    prev_epoch_loss = None
    best_loss = float('inf')
    
    # Calculate total training steps for scheduler
    batches_per_epoch = len(dataset) // batch_size + 1
    total_training_steps = batches_per_epoch * (epochs - start_epoch)
    warmup_steps = int(total_training_steps * LR_WARMUP_RATIO)
    
    print(f"  Estimated total steps: {total_training_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=lr)
    
    # Initialize scheduler based on mode
    scheduler = None
    use_step_scheduler = False
    
    if LR_SCHEDULE_MODE == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )
        use_step_scheduler = True
        print(f"  Using Cosine schedule with {warmup_steps} warmup steps")
        
    elif LR_SCHEDULE_MODE == "constant":
        scheduler = None
        use_step_scheduler = False
        print(f"  Using Constant LR (manual warmup for {warmup_steps} steps)")
        
    elif LR_SCHEDULE_MODE == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=LR_PLATEAU_FACTOR,
            patience=LR_PLATEAU_PATIENCE,
            min_lr=LR_PLATEAU_MIN
           
        )
        use_step_scheduler = False
        print(f"  Using Plateau schedule (patience={LR_PLATEAU_PATIENCE}, factor={LR_PLATEAU_FACTOR})")
    
    global_step = start_epoch * batches_per_epoch
    
    model.to(device)
    model.train()
    
    for epoch in range(start_epoch, epochs):
        # V2: Apply curriculum learning if enabled (now with dynamic batch size + weighted sampling)
        epoch_sampler = None  # Will be set for weighted-sampling phases
        if use_curriculum:
            curriculum_indices, epoch_batch_size, epoch_sampler = get_curriculum_indices(dataset, epoch, is_stage2=True)
            if len(curriculum_indices) == 0:
                print(f"  [WARNING] No samples match curriculum criteria for epoch {epoch+1}. Using all data.")
                curriculum_indices = list(range(len(dataset)))
                epoch_batch_size = batch_size  # Fallback to default
                epoch_sampler = None
            
            if epoch_sampler is not None:
                # LONG_FOCUS phase: use all data with weighted sampler
                epoch_dataset = dataset
            else:
                # SHORT / SHORT+MEDIUM / ALL phases: subset or full dataset
                if len(curriculum_indices) < len(dataset):
                    epoch_dataset = Subset(dataset, curriculum_indices)
                else:
                    epoch_dataset = dataset
        else:
            epoch_dataset = dataset
            epoch_batch_size = batch_size  # Use default batch size
        
        # Create DataLoader for this epoch
        # When a weighted sampler is active, shuffle must be False
        dataloader = DataLoader(
            epoch_dataset,
            batch_size=epoch_batch_size,
            shuffle=(epoch_sampler is None),  # mutually exclusive with sampler
            sampler=epoch_sampler,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        
        total_loss = 0
        total_batches = len(dataloader)
        epoch_start = time.time()
        step_interval = max(1, total_batches // 20)
        
        optimizer.zero_grad()
        
        for i, batch in enumerate(dataloader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Mixed precision forward pass
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
            
            # Scale loss for gradient accumulation
            loss = loss / grad_accum
            loss.backward()
            
            # Gradient accumulation step
            if i % grad_accum == 0 or i == total_batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Handle LR scheduling
                if use_step_scheduler and scheduler is not None:
                    scheduler.step()
                    current_lr = scheduler.get_last_lr()[0]
                    min_lr = lr * LR_MIN_RATIO
                    if current_lr < min_lr:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = min_lr
                
                elif LR_SCHEDULE_MODE == "constant" and global_step < warmup_steps:
                    warmup_lr = lr * (global_step + 1) / warmup_steps
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = warmup_lr
                
                optimizer.zero_grad()
                global_step += 1
            
            total_loss += outputs.loss.item()
            
            # Progress update
            if i == 1 or i % step_interval == 0 or i == total_batches:
                elapsed = time.time() - epoch_start
                speed = i / elapsed if elapsed > 0 else 0
                eta = (total_batches - i) / speed if speed > 0 else 0
                
                avg_seq_len = input_ids.shape[1]
                current_lr = optimizer.param_groups[0]['lr']
                print(
                    f"\r  [{stage_name}] Epoch {epoch + 1}/{epochs} - "
                    f"{i}/{total_batches} ({100*i/total_batches:.0f}%) - "
                    f"seq_len={avg_seq_len} - "
                    f"lr={current_lr:.2e} - "
                    f"ETA {_format_seconds(eta)}",
                    end="", flush=True
                )
        
        print()
        avg_loss = total_loss / total_batches
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        # Loss change diagnostics
        if prev_epoch_loss is not None:
            loss_change = avg_loss - prev_epoch_loss
            change_pct = (loss_change / prev_epoch_loss * 100) if prev_epoch_loss > 0 else 0
            status = "v" if loss_change < 0 else "^" if loss_change > 0 else "="
            print(f"  {status} Loss change: {loss_change:+.6f} ({change_pct:+.2f}%)")
        prev_epoch_loss = avg_loss
        
        # Step plateau scheduler at epoch end
        if LR_SCHEDULE_MODE == "plateau" and scheduler is not None:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(avg_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < old_lr:
                print(f"  [Plateau] LR reduced: {old_lr:.2e} -> {new_lr:.2e}")
            current_lr = new_lr
        
        # Track best loss
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        print(f"  Epoch {epoch + 1} Complete: loss={avg_loss:.4f}, best={best_loss:.4f}, lr={current_lr:.2e}, time={_format_seconds(epoch_time)}")
        
        # Checkpoint
        push = ((epoch + 1) % CHECKPOINT_UPLOAD_INTERVAL_EPOCHS == 0) or ((epoch + 1) == epochs)
        save_and_push_checkpoint(
            stage=stage_name,
            epoch=epoch,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            avg_loss=avg_loss,
            repo_id=hf_repo_id if push else None,
            total_epochs=epochs,
            scheduler=scheduler,
            global_step=global_step,
        )
    
    print(f"\n[OK] {stage_name} Training Complete")
    return model


# =============================================================================
# Resume Utilities
# =============================================================================

def repo_has_stage_latest(repo_id: str, stage: str, hf_auth_token: Optional[str] = None) -> bool:
    """Check if a stage/latest checkpoint exists."""
    token = hf_auth_token or HF_TOKEN
    if not HF_USE_HUB or not token:
        return False
    
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="model", token=token)
        return any(path.startswith(f"{stage}/latest/") and path.endswith("config.json") for path in files)
    except Exception:
        return False


def load_model_and_tokenizer_from_hf(
    repo_id: str,
    stage: str,
    motion_tokens: List[str] = None,
    hf_auth_token: Optional[str] = None
) -> Optional[Tuple[AutoModelForCausalLM, AutoTokenizer]]:
    """
    Load model and tokenizer from HF checkpoint.
    
    If the saved tokenizer fails to load (e.g. Qwen3 serialization issues),
    falls back to reconstructing tokenizer from base model + motion tokens.
    """
    token = hf_auth_token or HF_TOKEN
    
    if not repo_has_stage_latest(repo_id, stage, token):
        return None
    
    print(f"\n[Resume] Loading checkpoint from HF: {repo_id}/{stage}/latest")
    
    try:
        # Try loading the saved tokenizer first
        tokenizer = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                repo_id, subfolder=f"{stage}/latest", 
                trust_remote_code=True, token=token
            )
            print(f"   [Resume] Tokenizer loaded from checkpoint")
        except Exception as tok_err:
            print(f"   [Resume] Could not load saved tokenizer: {tok_err}")
            print(f"   [Resume] Reconstructing tokenizer from base model...")
            
            # Reconstruct from base model
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
            
            # Re-add all special tokens
            tokenizer.add_special_tokens({
                "pad_token": PAD_TOKEN,
                "additional_special_tokens": [M_START, M_END, LEN_SHORT, LEN_MEDIUM, LEN_LONG]
            })
            
            # Re-add motion tokens
            if motion_tokens:
                tokenizer.add_tokens(motion_tokens, special_tokens=True)
            
            print(f"   [Resume] Tokenizer reconstructed. Vocab size: {len(tokenizer)}")
        
        # Load model weights
        model = AutoModelForCausalLM.from_pretrained(
            repo_id, 
            subfolder=f"{stage}/latest", 
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=None,  # Let it use saved dtype
            token=token
        )
        
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
        
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        
        # Enable gradient checkpointing for memory savings
        model.gradient_checkpointing_enable()
        
        # Convert dtype after loading (match fresh model setup)
        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
            model = model.to(torch.bfloat16)
            print(f"   [Resume] Model converted to bfloat16")
        else:
            model = model.to(torch.float16)
            print(f"   [Resume] Model converted to float16")
        
        return model, tokenizer
    except Exception as e:
        print(f"[Resume] Failed to load from HF: {e}")
        import traceback
        traceback.print_exc()
        return None


def download_training_state(repo_id: str, stage: str, hf_auth_token: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Download training state JSON from HF."""
    token = hf_auth_token or HF_TOKEN
    if not HF_USE_HUB or not token:
        return None
    
    try:
        state_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{stage}/latest/training_state.json",
            repo_type="model",
            token=token,
        )
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# =============================================================================
# Main Pipeline
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Sentence-Level SignMotionGPT Training V2")
    p.add_argument("--dataset-path", type=str, required=True, help="Path to dataset JSON")
    p.add_argument("--vqvae-ckpt", type=str, required=True, help="Path to VQ-VAE checkpoint for initialization")
    p.add_argument("--stage", type=str, default="all", choices=["1", "2", "all"])
    p.add_argument("--hf-repo", type=str, default=HF_SENTENCE_REPO_ID)
    p.add_argument("--output-dir", type=str, default=PIPELINE_OUTPUT_DIR)
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint on HF")
    p.add_argument("--skip-eval", action="store_true", help="Skip evaluation after training")
    p.add_argument("--eval-sample-limit", type=int, default=EVAL_SAMPLE_LIMIT, help="Max samples for evaluation")
    p.add_argument("--no-curriculum", action="store_true", help="Disable curriculum learning for Stage 2")
    p.add_argument("--dtw-only", action="store_true",
                   help="Only compute DTW-JPE / DTW-PA-JPE metrics (skips FID, edit-dist, etc.). "
                        "Useful for a fast DTW-only evaluation pass.")
    return p.parse_args()


def main():
    args = parse_args()
    
    global PIPELINE_OUTPUT_DIR, CHECKPOINTS_DIR
    PIPELINE_OUTPUT_DIR = args.output_dir
    CHECKPOINTS_DIR = os.path.join(PIPELINE_OUTPUT_DIR, "checkpoints")
    
    print_banner("SignMotionGPT Sentence-Level Training V2", char="=")
    print(f"  Dataset: {args.dataset_path}")
    print(f"  VQ-VAE Checkpoint: {args.vqvae_ckpt}")
    print(f"  Stage(s): {args.stage}")
    print(f"  Max Seq Length: {MAX_SEQ_LEN}")
    print(f"  Resume: {args.resume}")
    print(f"  Curriculum Learning: {'disabled' if args.no_curriculum else 'ENABLED'}")
    print(f"  Output Dir: {PIPELINE_OUTPUT_DIR}")
    
    set_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    
    # Load and prepare sentence-level data
    raw_data = load_sentence_dataset(args.dataset_path)
    
    if len(raw_data) == 0:
        print("\n[ERROR] No sentence-level data found. Exiting.")
        return
    
    stats = analyze_sentence_dataset(raw_data)
    
    print("\n[Stats] Sentence Dataset Statistics:")
    print(f"   Total samples: {stats['total_samples']}")
    print(f"   Unique sentences: {stats['unique_sentences']}")
    print(f"   Avg motion length: {stats['avg_motion_length']:.1f}")
    print(f"   Min motion length: {stats['min_motion_length']}")
    print(f"   Max motion length: {stats['max_motion_length']}")
    print(f"   Unique motion tokens: {stats['num_unique_tokens']}")
    
    print(f"\n   Motion length distribution:")
    for bucket, count in stats["motion_length_distribution"].items():
        print(f"     {bucket}-{bucket+9}: {count} samples")
    
    estimated_max_seq = stats['max_motion_length'] + 80
    print(f"\n   Estimated max sequence: ~{estimated_max_seq} tokens")
    print(f"   Using MAX_SEQ_LEN: {MAX_SEQ_LEN} (headroom: {MAX_SEQ_LEN - estimated_max_seq})")
    
    # V2: DEDUPLICATION - Keep only unique sentences, then augment with templates in Stage2Dataset
    print("\n[Data] V2: Deduplicating to unique sentences (template augmentation happens in Stage2Dataset)")
    cleaned_data = deduplicate_sentence_data(raw_data)
    
    all_motion_tokens = prepare_motion_tokens(cleaned_data)
    print(f"\n[Vocab] Motion token vocabulary: {len(all_motion_tokens)} tokens")
    
    # Setup HF repo
    hf_repo = resolve_and_ensure_repo(args.hf_repo, HF_TOKEN) if HF_USE_HUB else None
    
    # Stage names
    s1_name = "stage1"
    s2_name = "stage2"
    
    # Track start epochs for each stage
    start_epochs = {
        s1_name: 0,
        s2_name: 0
    }
    
    # Resume logic
    model = None
    tokenizer = None
    
    if args.resume and hf_repo:
        target_stages = []
        if args.stage == "all":
            target_stages = [s1_name, s2_name]
        elif args.stage == "1":
            target_stages = [s1_name]
        elif args.stage == "2":
            target_stages = [s2_name]
        
        # Check if requested stage has checkpoint
        for stage_check in reversed(target_stages):
            if repo_has_stage_latest(hf_repo, stage_check):
                res = load_model_and_tokenizer_from_hf(hf_repo, stage_check, all_motion_tokens)
                if res:
                    model, tokenizer = res
                    
                    state = download_training_state(hf_repo, stage_check)
                    if state:
                        completed_epoch = state.get("epoch_completed", 0)
                        start_epochs[stage_check] = completed_epoch
                        print(f"[Resume] Resuming {stage_check} from epoch {completed_epoch}")
                    
                    # Mark previous stages as complete
                    if stage_check == s2_name:
                        start_epochs[s1_name] = S1_EPOCHS
                        print(f"   Marking {s1_name} as complete.")
                    break
        
        # If no intra-stage checkpoint, try previous stage
        if model is None:
            if args.stage == "2":
                if repo_has_stage_latest(hf_repo, s1_name):
                    res = load_model_and_tokenizer_from_hf(hf_repo, s1_name, all_motion_tokens)
                    if res:
                        model, tokenizer = res
                        print(f"[Resume] Loaded {s1_name} checkpoint to start {s2_name} fresh")
    
    if model is None:
        if args.resume and args.stage != "1":
            print("[Resume] No checkpoint found. Loading base model.")
        model, tokenizer = setup_model_and_tokenizer(MODEL_NAME, all_motion_tokens)
        
        # V2: Initialize motion token embeddings from VQ-VAE
        initialize_embeddings_from_vqvae(model, tokenizer, args.vqvae_ckpt, device)
    
    # Ensure vocabulary is synced
    if len(tokenizer) != model.config.vocab_size:
        print(f"[Vocab] Size mismatch ({len(tokenizer)} vs {model.config.vocab_size}). Resizing...")
        model.resize_token_embeddings(len(tokenizer))
    
    do_s1 = args.stage in ("1", "all")
    do_s2 = args.stage in ("2", "all")
    
    # ==========================================================================
    # Stage 1: Motion Language Pre-training
    # ==========================================================================
    if do_s1:
        if start_epochs[s1_name] < S1_EPOCHS:
            print(f"\n{'='*60}")
            print(f"  STAGE 1: Motion Language Pre-training")
            print(f"  Learning motion token sequences")
            print(f"{'='*60}")
            
            dataset = Stage1Dataset(cleaned_data)
            collate_fn = create_stage1_collate_fn(tokenizer, MAX_SEQ_LEN)
            model = train_stage(
                s1_name, model, tokenizer, dataset, collate_fn,
                device, S1_EPOCHS, S1_LR, S1_BATCH_SIZE, S1_GRAD_ACCUM, hf_repo,
                start_epoch=start_epochs[s1_name],
                use_curriculum=False,  # No curriculum for Stage 1
            )
        else:
            print(f"[SKIP] {s1_name} (completed {start_epochs[s1_name]}/{S1_EPOCHS} epochs)")
    
    # ==========================================================================
    # Stage 2: Instruction Tuning (with Curriculum Learning)
    # ==========================================================================
    if do_s2:
        if start_epochs[s2_name] < S2_EPOCHS:
            print(f"\n{'='*60}")
            print(f"  STAGE 2: Instruction Tuning (V2 with Length Conditioning)")
            print(f"  Learning text-to-motion mapping with curriculum")
            print(f"{'='*60}")
            
            dataset = Stage2Dataset(cleaned_data)
            collate_fn = create_stage2_collate_fn(tokenizer, MAX_SEQ_LEN)
            model = train_stage(
                s2_name, model, tokenizer, dataset, collate_fn,
                device, S2_EPOCHS, S2_LR, S2_BATCH_SIZE, S2_GRAD_ACCUM, hf_repo,
                start_epoch=start_epochs[s2_name],
                use_curriculum=not args.no_curriculum,  # V2: Enable curriculum by default
            )
        else:
            print(f"[SKIP] {s2_name} (completed {start_epochs[s2_name]}/{S2_EPOCHS} epochs)")
    
    model.to(device)
    
    # Save final model
    _ensure_dir(PIPELINE_OUTPUT_DIR)
    model.save_pretrained(PIPELINE_OUTPUT_DIR)
    tokenizer.save_pretrained(PIPELINE_OUTPUT_DIR)
    
    # Determine last trained stage
    last_trained_stage = "base"
    if do_s2 and start_epochs[s2_name] < S2_EPOCHS:
        last_trained_stage = "stage2"
    elif do_s1 and start_epochs[s1_name] < S1_EPOCHS:
        last_trained_stage = "stage1"
    else:
        if do_s2:
            last_trained_stage = "stage2"
        elif do_s1:
            last_trained_stage = "stage1"
    
    print(f"\n[INFO] Last trained stage: {last_trained_stage}")
    
    # =============================================================================
    # Evaluation (V2 with Beam Search)
    # =============================================================================
    
    metrics_json_path = os.path.join(PIPELINE_OUTPUT_DIR, "metrics.json")
    
    if args.skip_eval:
        print("\n[Eval] Evaluation skipped (--skip-eval).")
    else:
        print("\n[Eval] Running sentence-level evaluation...")
        
        evaluation_data = cleaned_data
        
        if len(evaluation_data) == 0:
            print("[Eval] No evaluation data available.")
        elif evaluate_sentence_level_encoder_style is None and evaluate_metrics_encoder_style is None:
            print("[Eval] Metrics module not available. Skipping evaluation.")
        else:
            # Print inference examples with V2 BEAM SEARCH
            if generate_motion is not None:
                print(f"\n[Eval] Sentence-Level Inference Examples (V2: Beam Search):")
                
                sample_indices = list(range(len(evaluation_data)))
                random.shuffle(sample_indices)
                example_count = min(EVAL_INFERENCE_EXAMPLES, len(sample_indices))
                
                for n, idx in enumerate(sample_indices[:example_count], start=1):
                    sample = evaluation_data[idx]
                    text = (sample.get("text") or sample.get("sentence", "")).strip()
                    
                    gt_tokens_str = str(sample.get("motion_tokens", "")).strip()
                    gt_length = len(gt_tokens_str.split())
                    
                    # V2: Determine length token for prompt
                    if gt_length < CURRICULUM_SHORT_MAX:
                        len_token = LEN_SHORT
                    elif gt_length < CURRICULUM_MEDIUM_MAX:
                        len_token = LEN_MEDIUM
                    else:
                        len_token = LEN_LONG
                    
                    # V2: Use a random template from TEMPLATES for evaluation variety
                    template = random.choice(TEMPLATES)
                    instruction = template.format(text=text)
                    prompt = f"{instruction} (Length: {len_token})\nMotion: "
                    
                    gt_wrapped = " ".join([f"<M{t}>" for t in gt_tokens_str.split() if t.strip()])
                    gt_sequence = f"{M_START} {gt_wrapped} {M_END}"
                    
                    # V2: Generate with greedy decoding for consistent outputs
                    generated_sequence = generate_motion(
                        model, tokenizer, prompt, device,
                        max_new_tokens=180,
                        min_new_tokens=max(20, int(gt_length * 0.3)),
                        use_greedy=True,
                        temperature=0.3,
                    )
                    
                    gen_tokens = [t for t in generated_sequence.split() if t.startswith("<M") and t not in [M_START, M_END]]
                    
                    print(f"\nExample {n}/{example_count}")
                    print("Prompt:")
                    print(prompt)
                    print(f"LLM Output ({len(gen_tokens)} motion tokens):")
                    print(generated_sequence)
                    print(f"Ground Truth ({gt_length} motion tokens):")
                    print(gt_sequence)
                    print("-" * 80)
            
            # Run metrics evaluation
            if last_trained_stage == "stage2" and evaluate_sentence_level_encoder_style is not None:
                print("\n[Eval] Running sentence-level evaluation...")
                
                sentence_metrics = evaluate_sentence_level_encoder_style(
                    model,
                    tokenizer,
                    evaluation_data,
                    device,
                    sample_limit=min(args.eval_sample_limit, len(evaluation_data)),
                    dtw_only=getattr(args, 'dtw_only', False),
                )
                
                combined_metrics = {
                    "sentence_level": {k: v for k, v in sentence_metrics.items() if k != "pairs_closest"},
                    "pairs_closest": sentence_metrics.get("pairs_closest", []),
                    "num_samples": len(evaluation_data),
                }
                
                _ensure_dir(PIPELINE_OUTPUT_DIR)
                with open(metrics_json_path, "w", encoding="utf-8") as f:
                    json.dump(combined_metrics, f, ensure_ascii=False, indent=2)
                print(f"\n[Eval] Saved metrics to {metrics_json_path}")
                
                # Visualizations
                viz_dir = os.path.join(PIPELINE_OUTPUT_DIR, "visualizations")
                if save_side_by_side_visualizations is not None:
                    viz_pairs = sentence_metrics.get("pairs_closest", [])[:8]
                    save_side_by_side_visualizations(viz_pairs, viz_dir, limit=8, data_level="sentence", output_format="video")
                    print(f"[Eval] Saved visualizations to {viz_dir}")
            
            elif evaluate_metrics_encoder_style is not None:
                print("\n[Eval] Running encoder-style metrics...")
                
                metrics_enc = evaluate_metrics_encoder_style(
                    model, tokenizer, evaluation_data, device,
                    sample_limit=args.eval_sample_limit,
                    include_participant=False,
                )
                
                metrics_payload = {
                    "source": "vqvae_encoder",
                    "fid": metrics_enc.get("fid"),
                    "diversity": {
                        "ground_truth": metrics_enc.get("diversity_gt"),
                        "model": metrics_enc.get("diversity_gen"),
                    },
                    "multimodality": {
                        "ground_truth": metrics_enc.get("mim_gt"),
                        "model": metrics_enc.get("mim_gen"),
                    },
                    "num_pairs": len(metrics_enc.get("pairs", [])),
                    "num_samples": len(evaluation_data),
                }
                
                _ensure_dir(PIPELINE_OUTPUT_DIR)
                with open(metrics_json_path, "w", encoding="utf-8") as f:
                    json.dump(metrics_payload, f, ensure_ascii=False, indent=2)
                print(f"\n[Eval] Saved metrics to {metrics_json_path}")
    
    print_banner("TRAINING COMPLETE (V2)", char="=")
    print(f"  Output: {PIPELINE_OUTPUT_DIR}")
    if hf_repo:
        print(f"  HuggingFace: https://huggingface.co/{hf_repo}")
    if not args.skip_eval and len(cleaned_data) > 0:
        print(f"  Metrics: {metrics_json_path}")
        print(f"  Visualizations: {os.path.join(PIPELINE_OUTPUT_DIR, 'visualizations')}")


if __name__ == "__main__":
    main()
