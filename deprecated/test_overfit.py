
import os
import re
import json
import random
from typing import Dict, List, Tuple, Any, Optional
import shutil
from datetime import datetime
import time

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from huggingface_hub import HfApi, upload_folder, hf_hub_download

import numpy as np
import scipy.linalg
# ======================================================================================
# 0. Configuration
# ======================================================================================
# --- Paths and Words ---
DATASET_PATH = "/content/SignMotionGPT/data/motion_llm_dataset.json"
MODEL_NAME = "Qwen/Qwen3-0.6B"
# We will train on the full dataset, but use these words for our final evaluation
EVALUATION_WORDS = ["passport", "send", "library", "push"]
OUTPUT_DIR = "./motion_gpt_full_model"

# --- Evaluation controls ---
# If True: after training, only compute metrics (FID, Diversity, MIM) and save to JSON.
#          Skip per-sample inference logs and HTML visualizations.
# If False: run the existing flow and also compute these 3 metrics.
RUN_EVALS_ONLY = False
EVAL_SAMPLE_LIMIT = 100
METRICS_JSON_PATH = ""

# --- Training Hyperparameters ---
# NOTE: Training on the full dataset will take longer.
# These epochs are a starting point.
S1_EPOCHS = 20
S1_LR = 5e-5
S1_BATCH_SIZE = 8 # Kept small for Colab VRAM

S2_EPOCHS = 20
S2_LR = 2e-5
S2_BATCH_SIZE = 8

# --- Inference Hyperparameters ---
INFERENCE_REPETITION_PENALTY = 1.2
INFERENCE_TEMPERATURE = 0.7
INFERENCE_TOP_K = 50

# --- Special Tokens ---
M_START = "<M_START>"
M_END = "<M_END>"
PAD_TOKEN = "<PAD>"

# --- Hugging Face Hub Configuration ---
# Provide HUGGINGFACE_HUB_TOKEN or hf_auth_token in environment for private repos.
HF_USE_HUB = True
hf_auth_token = os.getenv("hf_auth_token")
if hf_auth_token is None:
    raise ValueError("hf_auth_token environment variable is not set")
HF_STAGE1_REPO_ID = "rdz-falcon/SignMotionGPTfit-archive"
HF_STAGE2_REPO_ID = "rdz-falcon/SignMotionGPTfit-archive"
HF_PRIVATE_REPO = os.environ.get("HF_PRIVATE", "true").lower() != "false"
FORCE_STAGE2_FROM_STAGE1_RAW = os.environ.get("FORCE_STAGE2_FROM_STAGE1", "false")
FORCE_STAGE2_FROM_STAGE1 = str(FORCE_STAGE2_FROM_STAGE1_RAW).strip().lower() not in ("0", "false", "no", "off")
# Save Stage 2 checkpoints to a new subfolder so old stage2 checkpoints remain intact
HF_STAGE2_SAVE_SUBDIR = os.environ.get("HF_STAGE2_SAVE_SUBDIR", "stage2_v2")

# --- Local Checkpoint Root ---
CHECKPOINTS_DIR = ""

# --- Upload frequency and progress control ---
# Push to Hugging Face only every N epochs (still save locally every epoch)
CHECKPOINT_UPLOAD_INTERVAL_EPOCHS = int(os.environ.get("HF_UPLOAD_INTERVAL_EPOCHS", "2"))
# Disable HF Hub progress bars to reduce noisy logs (set HF_DISABLE_PROGRESS=false to re-enable)
HF_DISABLE_PROGRESS = os.environ.get("HF_DISABLE_PROGRESS", "true").lower() != "false"


def _refresh_runtime_paths() -> None:
    """Refresh derived paths when OUTPUT_DIR changes."""
    global METRICS_JSON_PATH, CHECKPOINTS_DIR
    METRICS_JSON_PATH = os.path.join(OUTPUT_DIR, "metrics.json")
    CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, "checkpoints")


def _apply_progress_setting() -> None:
    """Apply huggingface_hub progress bar preference."""
    if HF_DISABLE_PROGRESS:
        try:
            # Also respected by huggingface_hub internal progress usage
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            from huggingface_hub.utils import disable_progress_bars  # type: ignore

            disable_progress_bars()
        except Exception:
            pass
    else:
        os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)


def apply_config_overrides(overrides: Optional[Dict[str, Any]] = None) -> None:
    """
    Allow external callers to override module-level configuration prior to running main().
    """
    global hf_auth_token, HF_DISABLE_PROGRESS, OUTPUT_DIR
    if not overrides:
        return

    updated_paths = False
    progress_flag_updated = False
    for key, value in overrides.items():
        if key == "hf_auth_token":
            hf_auth_token = value
            continue
        if key not in globals():
            print(f"[config] Unknown override ignored: {key}")
            continue
        globals()[key] = value
        if key == "OUTPUT_DIR":
            updated_paths = True
        if key == "HF_DISABLE_PROGRESS":
            progress_flag_updated = True
    if updated_paths:
        _refresh_runtime_paths()
    if progress_flag_updated:
        _apply_progress_setting()


_refresh_runtime_paths()
_apply_progress_setting()


# ======================================================================================
# 1. Data Loading and Preparation (NEW & IMPROVED)
# ======================================================================================
def read_json_data(json_path: str) -> List[Dict[str, Any]]:
    """Loads the dataset from the specified JSON file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset not found at: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def deduplicate_and_prepare_data(entries: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Cleans the entire dataset by ensuring each (word, participant_id) pair is unique.
    If a conflict is found (same pair, different motion), it keeps only the first one encountered.
    Then, it prepares the full list of motion tokens from the cleaned data.
    """
    print("\n---> Cleaning dataset by removing ambiguous (word, participant_id) pairs...")
    
    unique_samples = {}
    conflicts_found = 0
    
    for entry in entries:
        word = entry.get("word", "").lower()
        pid = entry.get("participant_id", "")
        key = (word, pid)
        
        if key not in unique_samples:
            unique_samples[key] = entry
        else:
            # A sample for this key already exists. We only care if it's a conflict.
            existing_tokens = unique_samples[key].get("motion_tokens")
            current_tokens = entry.get("motion_tokens")
            if existing_tokens != current_tokens:
                conflicts_found += 1
                # We do nothing, effectively discarding this new conflicting sample.
    
    cleaned_data = list(unique_samples.values())
    
    print(f"Original samples: {len(entries)}")
    print(f"Cleaned samples (unique (word, pid) pairs): {len(cleaned_data)}")
    print(f"Removed {len(entries) - len(cleaned_data)} total samples. ({conflicts_found} were direct conflicts).")

    print("\n---> Extracting motion tokens from the full cleaned dataset...")
    all_motion_tokens = set()
    for entry in cleaned_data:
        motion_tokens = entry.get("motion_tokens", "").strip().split()
        for token in motion_tokens:
            all_motion_tokens.add(f"<M{token}>")

    unique_tokens = sorted(list(all_motion_tokens))
    print(f"Found {len(unique_tokens)} unique motion tokens in the entire dataset.")
    
    return cleaned_data, unique_tokens

# ======================================================================================
# 2. Model and Tokenizer Setup
# ======================================================================================
def setup_model_and_tokenizer(model_name: str, motion_tokens: List[str]) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Loads the model and tokenizer, adding special and motion tokens."""
    print(f"\n---> Loading base model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    tokenizer.add_special_tokens({"pad_token": PAD_TOKEN, "additional_special_tokens": [M_START, M_END]})
    
    print(f"Adding {len(motion_tokens)} motion tokens to the tokenizer.")
    tokenizer.add_tokens(motion_tokens, special_tokens=True)
    
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer

# ======================================================================================
# 2b. Hugging Face Hub Utilities and Checkpointing
# ======================================================================================
def _format_seconds(seconds: float) -> str:
    """Formats seconds into H:MM:SS or M:SS."""
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _resolve_and_ensure_repo(repo_id: str) -> Optional[str]:
    """
    Ensures the HF repo exists. Returns the fully-qualified repo_id (namespace/repo)
    when token is available; otherwise returns the input repo_id.
    """
    if not HF_USE_HUB:
        return None
    if hf_auth_token is None:
        print("⚠️  HF token not found. Set HUGGINGFACE_HUB_TOKEN or hf_auth_token to enable Hub sync.")
        return None
    api = HfApi()
    try:
        who = api.whoami(token=hf_auth_token)
        namespace = who.get("name") or (who.get("orgs", [None])[0] if isinstance(who.get("orgs"), list) else None)
    except Exception as exc:
        print(f"⚠️  Unable to resolve HF namespace: {exc}")
        namespace = None
    if "/" not in repo_id and namespace:
        full_repo_id = f"{namespace}/{repo_id}"
    else:
        full_repo_id = repo_id
    try:
        api.create_repo(
            repo_id=full_repo_id,
            token=hf_auth_token,
            repo_type="model",
            private=HF_PRIVATE_REPO,
            exist_ok=True,
        )
    except Exception as exc:
        print(f"⚠️  create_repo failed (may already exist): {exc}")
    return full_repo_id

def _repo_has_stage_latest(repo_id: str, stage: str) -> bool:
    """Checks if a stage/latest checkpoint exists in the HF repo."""
    if not HF_USE_HUB or hf_auth_token is None:
        return False
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="model", token=hf_auth_token)
        return any(path.startswith(f"{stage}/latest/") and path.endswith("config.json") for path in files)
    except Exception as exc:
        print(f"⚠️  Could not list files for {repo_id}: {exc}")
        return False

def _repo_list_epoch_numbers(repo_id: str, stage: str) -> List[int]:
    """
    Returns sorted list of epoch numbers available under {stage}/epoch-XXX/ by scanning files.
    Works even if 'latest' does not exist.
    """
    if not HF_USE_HUB or hf_auth_token is None:
        return []
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="model", token=hf_auth_token)
    except Exception as exc:
        print(f"⚠️  Could not list files for {repo_id}: {exc}")
        return []
    epoch_numbers: List[int] = []
    pattern = re.compile(rf"^{re.escape(stage)}/epoch-(\d+)/config\.json$")
    for path in files:
        m = pattern.match(path)
        if m:
            try:
                epoch_numbers.append(int(m.group(1)))
            except ValueError:
                pass
    return sorted(set(epoch_numbers))

def _repo_get_latest_epoch_subfolder(repo_id: str, stage: str) -> Optional[str]:
    """
    Returns subfolder path like '{stage}/epoch-XXX' for the highest available epoch, or None.
    """
    epochs = _repo_list_epoch_numbers(repo_id, stage)
    if not epochs:
        return None
    latest = max(epochs)
    return f"{stage}/epoch-{latest:03d}"

def _load_model_and_tokenizer_from_hf_subfolder(repo_id: str, subfolder: str) -> Optional[Tuple[AutoModelForCausalLM, AutoTokenizer]]:
    """
    Loads model and tokenizer from HF under a specific subfolder (e.g., 'stage1/epoch-020').
    """
    if not HF_USE_HUB or hf_auth_token is None:
        return None
    print(f"\n---> Loading checkpoint from Hugging Face: {repo_id} (subfolder='{subfolder}')")
    try:
        tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder=subfolder, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(repo_id, subfolder=subfolder, trust_remote_code=True)
    except Exception as exc:
        print(f"⚠️  Failed to load model/tokenizer from subfolder '{subfolder}': {exc}")
        return None
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer

def _download_training_state_from_subfolder(repo_id: str, subfolder: str) -> Optional[Dict[str, Any]]:
    """
    Downloads training_state.json from a specific subfolder (e.g., 'stage1/epoch-020').
    """
    if not HF_USE_HUB or hf_auth_token is None:
        return None
    try:
        state_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{subfolder}/training_state.json",
            repo_type="model",
            token=hf_auth_token,
        )
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _download_training_state(repo_id: str, stage: str) -> Optional[Dict[str, Any]]:
    """Downloads training_state.json from HF if present."""
    if not HF_USE_HUB or hf_auth_token is None:
        return None
    try:
        state_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{stage}/latest/training_state.json",
            repo_type="model",
            token=hf_auth_token,
        )
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _download_optimizer_state(repo_id: str, stage: str) -> Optional[str]:
    """Downloads optimizer.pt for resuming optimizer state."""
    if not HF_USE_HUB or hf_auth_token is None:
        return None
    try:
        opt_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{stage}/latest/optimizer.pt",
            repo_type="model",
            token=hf_auth_token,
        )
        return opt_path
    except Exception:
        return None

def _load_model_and_tokenizer_from_hf(repo_id: str, stage: str) -> Optional[Tuple[AutoModelForCausalLM, AutoTokenizer]]:
    """
    Loads model and tokenizer from HF under subfolder {stage}/latest if available.
    """
    if not _repo_has_stage_latest(repo_id, stage):
        return None
    print(f"\n---> Loading checkpoint from Hugging Face: {repo_id} (subfolder='{stage}/latest')")
    tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder=f"{stage}/latest", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(repo_id, subfolder=f"{stage}/latest", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer

def _ensure_tokenizer_has_motion_tokens(tokenizer: AutoTokenizer, motion_tokens: List[str]) -> int:
    """
    Adds any missing motion tokens to the tokenizer. Returns number of tokens added.
    """
    tokenizer.add_special_tokens({"pad_token": PAD_TOKEN, "additional_special_tokens": [M_START, M_END]})
    added = tokenizer.add_tokens(motion_tokens, special_tokens=True)
    return added

def _save_and_push_checkpoint(
    stage: str,
    epoch_index_zero_based: int,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    optimizer: AdamW,
    avg_loss: float,
    dataloader_len: int,
    batch_size: int,
    total_epochs: int,
    repo_id: Optional[str],
) -> None:
    """
    Saves checkpoint locally (per-epoch and latest) and pushes to HF under:
      - {stage}/epoch-XXX
      - {stage}/latest
    Also saves optimizer state and training_state.json to preserve resume info.
    """
    epoch_number = epoch_index_zero_based + 1
    stage_dir = os.path.join(CHECKPOINTS_DIR, stage)
    epoch_dir_name = f"epoch-{epoch_number:03d}"
    epoch_dir = os.path.join(stage_dir, epoch_dir_name)
    latest_dir = os.path.join(stage_dir, "latest")
    _ensure_dir(epoch_dir)
    _ensure_dir(stage_dir)

    # Save model + tokenizer
    model.save_pretrained(epoch_dir)
    tokenizer.save_pretrained(epoch_dir)

    # Save optimizer state
    torch.save(optimizer.state_dict(), os.path.join(epoch_dir, "optimizer.pt"))

    # Save training state
    training_state = {
        "stage": stage,
        "epoch_completed": epoch_number,
        "total_epochs_for_stage": total_epochs,
        "global_step": epoch_number * dataloader_len,
        "avg_loss": float(avg_loss),
        "batch_size": batch_size,
        "saved_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(epoch_dir, "training_state.json"), "w", encoding="utf-8") as f:
        json.dump(training_state, f, ensure_ascii=False, indent=2)

    # Update "latest"
    if os.path.exists(latest_dir):
        shutil.rmtree(latest_dir)
    shutil.copytree(epoch_dir, latest_dir)

    # Push to Hugging Face
    if HF_USE_HUB and repo_id and hf_auth_token:
        try:
            upload_folder(
                repo_id=repo_id,
                folder_path=epoch_dir,
                path_in_repo=f"{stage}/{epoch_dir_name}",
                repo_type="model",
                token=hf_auth_token,
                commit_message=f"{stage}: save {epoch_dir_name}",
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=latest_dir,
                path_in_repo=f"{stage}/latest",
                repo_type="model",
                token=hf_auth_token,
                commit_message=f"{stage}: update latest -> {epoch_dir_name}",
            )
            print(f"☁️  Pushed checkpoint to HF: {repo_id} ({stage}/{epoch_dir_name} and {stage}/latest)")
        except Exception as exc:
            print(f"⚠️  Failed to push checkpoint to HF: {exc}")
    else:
        print("ℹ️  Skipped HF push (Hub disabled or token/repo missing).")

# ======================================================================================
# 3. Training Stage 1: Motion Language Modeling
# ======================================================================================
class MotionDataset(Dataset):
    """Dataset for Stage 1: Contains only motion token sequences."""
    def __init__(self, data: List[Dict[str, Any]], tokenizer: AutoTokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sequences = []

        for item in data:
            tokens_str = item.get("motion_tokens", "")
            wrapped_tokens = " ".join([f"<M{t}>" for t in tokens_str.split()])
            full_sequence = f"{M_START} {wrapped_tokens} {M_END}"
            self.sequences.append(full_sequence)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.tokenizer(
            self.sequences[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

def train_stage1(
    model,
    tokenizer,
    data,
    device,
    start_epoch: int = 0,
    hf_repo_id: Optional[str] = None,
):
    """Trains the model on motion sequences only to learn the 'language of motion'.
    Resumes from Hugging Face if available (model/tokenizer/optimizer)."""
    print("\n" + "="*80)
    print("      STAGE 1: MOTION LANGUAGE MODELING (PRE-TRAINING)")
    print(f"      Training on {len(data)} samples.")
    print("="*80)
    
    dataset = MotionDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=S1_BATCH_SIZE, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=S1_LR)
    model.to(device)
    model.train()

    # Try to resume optimizer if we resumed from HF
    if hf_repo_id and start_epoch > 0 and HF_USE_HUB and hf_auth_token:
        opt_path = _download_optimizer_state(hf_repo_id, "stage1")
        if opt_path is not None:
            try:
                optimizer.load_state_dict(torch.load(opt_path, map_location=device))
                print("↩️  Resumed optimizer state for Stage 1 from HF.")
            except Exception as exc:
                print(f"⚠️  Failed to load optimizer state for Stage 1: {exc}")

    for epoch in range(start_epoch, S1_EPOCHS):
        total_loss = 0
        total_batches = len(dataloader)
        epoch_start_time = time.time()
        step_interval = max(1, total_batches // 50)  # ~2% progress updates
        for i, batch in enumerate(dataloader, 1):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Progress with ETA
            if i == 1 or (i % step_interval == 0) or (i == total_batches):
                elapsed = time.time() - epoch_start_time
                est_total = (elapsed / i) * total_batches
                eta = est_total - elapsed
                pct = (i / total_batches) * 100.0
                print(
                    f"\r[Stage 1] Epoch {epoch+1}/{S1_EPOCHS} - "
                    f"{i}/{total_batches} ({pct:.1f}%) - ETA {_format_seconds(eta)}",
                    end="",
                    flush=True,
                )
        
        # Finish the progress line
        print()
        avg_loss = total_loss / len(dataloader)
        print(f"--- End of Epoch {epoch+1}/{S1_EPOCHS}, Average Loss: {avg_loss:.4f} ---")
        # Save checkpoint locally every epoch; push to HF only at interval or final epoch
        push_this_epoch = ((epoch + 1) % CHECKPOINT_UPLOAD_INTERVAL_EPOCHS == 0) or ((epoch + 1) == S1_EPOCHS)
        repo_for_epoch = hf_repo_id if push_this_epoch else None
        _save_and_push_checkpoint(
            stage="stage1",
            epoch_index_zero_based=epoch,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            avg_loss=avg_loss,
            dataloader_len=len(dataloader),
            batch_size=S1_BATCH_SIZE,
            total_epochs=S1_EPOCHS,
            repo_id=repo_for_epoch,
        )
    
    print("\n✅ Stage 1 Training Complete.")
    return model

# ======================================================================================
# 4. Training Stage 2: Text-to-Motion Fine-Tuning
# ======================================================================================
class TextMotionDataset(Dataset):
    """Dataset for Stage 2: Contains (prompt, motion_sequence) pairs."""
    def __init__(self, data: List[Dict[str, Any]], tokenizer: AutoTokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.items = []

        for item in data:
            prompt = f"Instruction: Generate motion for word '{item['word']}' with variant '{item['participant_id']}'.\nMotion: "
            
            tokens_str = item.get("motion_tokens", "")
            wrapped_tokens = " ".join([f"<M{t}>" for t in tokens_str.split()])
            target_sequence = f"{M_START} {wrapped_tokens} {M_END}"
            
            full_text = prompt + target_sequence
            
            tokenized = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            prompt_tokenized = self.tokenizer(prompt, return_tensors="pt")
            prompt_len = prompt_tokenized.input_ids.shape[1]
            
            labels = tokenized['input_ids'].clone()
            labels[0, :prompt_len] = -100
            
            self.items.append({
                "input_ids": tokenized['input_ids'].squeeze(0),
                "attention_mask": tokenized['attention_mask'].squeeze(0),
                "labels": labels.squeeze(0)
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

def train_stage2(
    model,
    tokenizer,
    data,
    device,
    start_epoch: int = 0,
    hf_repo_id: Optional[str] = None,
    hf_stage_subdir: str = "stage2",
):
    """Fine-tunes the motion-aware model to connect text prompts to motions.
    Resumes from Hugging Face if available (model/tokenizer/optimizer)."""
    print("\n" + "="*80)
    print("      STAGE 2: TEXT-TO-MOTION FINE-TUNING")
    print(f"      Training on {len(data)} samples.")
    print("="*80)
    
    dataset = TextMotionDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=S2_BATCH_SIZE, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=S2_LR)
    model.to(device)
    model.train()

    # Try to resume optimizer if we resumed from HF
    if hf_repo_id and start_epoch > 0 and HF_USE_HUB and hf_auth_token:
        opt_path = _download_optimizer_state(hf_repo_id, hf_stage_subdir)
        if opt_path is not None:
            try:
                optimizer.load_state_dict(torch.load(opt_path, map_location=device))
                print("↩️  Resumed optimizer state for Stage 2 from HF.")
            except Exception as exc:
                print(f"⚠️  Failed to load optimizer state for Stage 2: {exc}")

    for epoch in range(start_epoch, S2_EPOCHS):
        total_loss = 0
        total_batches = len(dataloader)
        epoch_start_time = time.time()
        step_interval = max(1, total_batches // 50)  # ~2% progress updates
        for i, batch in enumerate(dataloader, 1):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Progress with ETA
            if i == 1 or (i % step_interval == 0) or (i == total_batches):
                elapsed = time.time() - epoch_start_time
                est_total = (elapsed / i) * total_batches
                eta = est_total - elapsed
                pct = (i / total_batches) * 100.0
                print(
                    f"\r[Stage 2] Epoch {epoch+1}/{S2_EPOCHS} - "
                    f"{i}/{total_batches} ({pct:.1f}%) - ETA {_format_seconds(eta)}",
                    end="",
                    flush=True,
                )
        
        # Finish the progress line
        print()
        avg_loss = total_loss / len(dataloader)
        print(f"--- End of Epoch {epoch+1}/{S2_EPOCHS}, Average Loss: {avg_loss:.4f} ---")
        # Save checkpoint locally every epoch; push to HF only at interval or final epoch
        push_this_epoch = ((epoch + 1) % CHECKPOINT_UPLOAD_INTERVAL_EPOCHS == 0) or ((epoch + 1) == S2_EPOCHS)
        repo_for_epoch = hf_repo_id if push_this_epoch else None
        _save_and_push_checkpoint(
            stage=hf_stage_subdir,
            epoch_index_zero_based=epoch,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            avg_loss=avg_loss,
            dataloader_len=len(dataloader),
            batch_size=S2_BATCH_SIZE,
            total_epochs=S2_EPOCHS,
            repo_id=repo_for_epoch,
        )
        
    print("\n✅ Stage 2 Training Complete.")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")
    return model

# ======================================================================================
# 5. Inference and Comparison
# ======================================================================================
def generate_motion(model, tokenizer, prompt, device):
    """Generates a motion sequence from a prompt using sampling."""
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=INFERENCE_TEMPERATURE,
            top_k=INFERENCE_TOP_K,
            repetition_penalty=INFERENCE_REPETITION_PENALTY,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids(M_END),
            early_stopping=True
        )
    
    decoded = tokenizer.decode(output[0], skip_special_tokens=False)
    motion_part = decoded.split("Motion: ")[-1]
    return motion_part.strip()

def compare_sequences(gt: str, gen: str):
    """Provides a simple visual diff of two sequences without external libraries."""
    gt_tokens = gt.split()
    gen_tokens = gen.split()

    print("\nDetailed Comparison (✅ = Match, ❌ = Mismatch/Missing/Added):")
    
    gt_str =   "  GT:  "
    gen_str =  "  GEN: "
    diff_str = "       "
    
    max_len = max(len(gt_tokens), len(gen_tokens))
    
    for i in range(max_len):
        gt_tok = gt_tokens[i] if i < len(gt_tokens) else "___"
        gen_tok = gen_tokens[i] if i < len(gen_tokens) else "___"
        
        max_tok_len = max(len(gt_tok), len(gen_tok))
        gt_tok_padded = gt_tok.ljust(max_tok_len)
        gen_tok_padded = gen_tok.ljust(max_tok_len)
        
        gt_str += gt_tok_padded + " "
        gen_str += gen_tok_padded + " "
        
        if gt_tok == gen_tok:
            diff_str += "✅".ljust(max_tok_len) + " "
        else:
            diff_str += "❌".ljust(max_tok_len) + " "
            
    print(gt_str)
    print(gen_str)
    print(diff_str)

def run_inference_on_all_samples(model, tokenizer, data, device):
    """
    Runs inference on ALL available samples for the trained words and compares 
    each one to its specific ground truth.
    """
    print("\n" + "="*80)
    print("      INFERENCE AND EVALUATION (ALL SAMPLES)")
    print("      Goal: Test the model's performance on every variant.")
    print("="*80)

    data_by_word = {}
    for item in data:
        word = item['word']
        if word not in data_by_word:
            data_by_word[word] = []
        data_by_word[word].append(item)

    for word, samples in data_by_word.items():
        print(f"\n\n{'='*25} TESTING WORD: '{word}' {'='*25}")
        num_correct = 0
        
        for i, sample in enumerate(samples):
            print(f"\n--- Testing Variant {i+1}/{len(samples)}: '{sample['participant_id']}' ---")
            
            gt_tokens_str = sample.get("motion_tokens", "")
            gt_wrapped = " ".join([f"<M{t}>" for t in gt_tokens_str.split()])
            gt_sequence = f"{M_START} {gt_wrapped} {M_END}"
            print(f"Ground Truth:\n{gt_sequence}")

            prompt = f"Instruction: Generate motion for word '{sample['word']}' with variant '{sample['participant_id']}'.\nMotion: "
            generated_sequence = generate_motion(model, tokenizer, prompt, device)
            print(f"\nLLM Generated:\n{generated_sequence}")
            
            compare_sequences(gt_sequence, generated_sequence)

            if gt_sequence.strip() == generated_sequence.strip():
                num_correct += 1
            
            print("-" * 80)
        
        accuracy = (num_correct / len(samples)) * 100
        print(f"\nSUMMARY FOR '{word}': {num_correct}/{len(samples)} correct ({accuracy:.1f}%)")

# ======================================================================================
# 5b. Metrics: FID, Diversity, Multimodality (MIM) using MotionGPT-style utils
# ======================================================================================
def calculate_activation_statistics_np(activations: np.ndarray):
    """
    Params:
    -- activations: num_samples x dim_feat (numpy)
    Returns:
    -- mu: dim_feat
    -- sigma: dim_feat x dim_feat
    """
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov

def calculate_frechet_distance_np(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def calculate_diversity_np(activation: np.ndarray, diversity_times: int = 200) -> float:
    """Mean pairwise L2 distance across random pairs."""
    assert len(activation.shape) == 2
    assert activation.shape[0] > max(2, diversity_times)
    num_samples = activation.shape[0]
    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    diffs = activation[first_indices] - activation[second_indices]
    dist = np.linalg.norm(diffs, axis=1)
    return float(dist.mean())

def calculate_multimodality_np(activation: np.ndarray, multimodality_times: int = 20) -> float:
    """
    activation: [num_labels, num_per_label, D]
    Returns mean pairwise within-label diversity (higher = more multimodal).
    """
    assert len(activation.shape) == 3
    num_labels, num_per_label, _ = activation.shape
    assert num_per_label > multimodality_times
    first_dices = np.random.choice(num_per_label, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_label, multimodality_times, replace=False)
    diffs = activation[:, first_dices] - activation[:, second_dices]
    dist = np.linalg.norm(diffs, axis=2)
    return float(dist.mean())

# --------------------------------------------------------------------------------------
# Token sequence → activation (bag-of-motion-tokens) helpers
# --------------------------------------------------------------------------------------
def _extract_motion_tokens_from_sequence(seq: str) -> list[str]:
    # Expect tokens like <M123>, within M_START/M_END fences; keep only <M...>
    return [tok for tok in seq.split() if tok.startswith("<M") and tok.endswith(">")]

def _build_token_index(tokens_vocab: list[str]) -> Dict[str, int]:
    return {tok: idx for idx, tok in enumerate(tokens_vocab)}

def _sequence_to_activation(seq: str, token_to_index: Dict[str, int]) -> np.ndarray:
    vec = np.zeros((len(token_to_index),), dtype=np.float32)
    for tok in _extract_motion_tokens_from_sequence(seq):
        idx = token_to_index.get(tok)
        if idx is not None:
            vec[idx] += 1.0
    # Normalize to unit length to reduce length bias
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

def _collect_eval_pairs(model, tokenizer, data, device) -> list[Tuple[str, str, str]]:
    """
    Returns list of (word, participant_id, gt_sequence, generated_sequence) for each sample in data.
    """
    results = []
    for sample in data:
        gt_tokens_str = sample.get("motion_tokens", "")
        gt_wrapped = " ".join([f"<M{t}>" for t in gt_tokens_str.split()])
        gt_sequence = f"{M_START} {gt_wrapped} {M_END}"
        prompt = f"Instruction: Generate motion for word '{sample['word']}' with variant '{sample['participant_id']}'.\nMotion: "
        generated_sequence = generate_motion(model, tokenizer, prompt, device)
        pid = str(sample.get("participant_id", ""))
        results.append((sample["word"], pid, gt_sequence, generated_sequence))
    return results

def _activations_from_pairs(pairs: list[Tuple[str, str, str]], vocab_tokens: list[str]):
    """
    Build numpy activations and labels arrays from sequences.
    Returns:
      gt_acts: (N, D)
      gen_acts: (N, D)
      labels: list[str] length N (word labels)
    """
    token_to_index = _build_token_index(vocab_tokens)
    gt_vecs = []
    gen_vecs = []
    labels = []
    for pair in pairs:
        # Support both legacy 3-tuple (word, gt, gen) and new 4-tuple (word, pid, gt, gen)
        if len(pair) == 4:
            word, _pid, gt_seq, gen_seq = pair
        else:
            word, gt_seq, gen_seq = pair
        gt_vecs.append(_sequence_to_activation(gt_seq, token_to_index))
        gen_vecs.append(_sequence_to_activation(gen_seq, token_to_index))
        labels.append(word)
    return np.stack(gt_vecs, axis=0), np.stack(gen_vecs, axis=0), labels

def _to_label_tensor3(acts: np.ndarray, labels: list[str]) -> np.ndarray:
    """
    Convert N x D activations with string labels to [L, K, D] by truncating each label
    to the minimum count across labels.
    """
    label_to_indices: Dict[str, list[int]] = {}
    for i, lbl in enumerate(labels):
        label_to_indices.setdefault(lbl, []).append(i)
    per_label_counts = [len(idxs) for idxs in label_to_indices.values()]
    if len(per_label_counts) == 0:
        raise ValueError("No labels found for multimodality computation.")
    min_count = max(2, min(per_label_counts))
    label_names = sorted(label_to_indices.keys())
    stacked = []
    for lbl in label_names:
        idxs = label_to_indices[lbl][:min_count]
        stacked.append(acts[idxs])
    return np.stack(stacked, axis=0)  # [L, K, D]

def evaluate_metrics_motiongpt_style(model, tokenizer, eval_data, all_motion_tokens, device):
    """
    Computes:
      - Diversity: GT vs GEN (pair)
      - Multimodality (MIM): GT vs GEN (pair)
      - FID: between GT and GEN
    """
    print("\n" + "="*80)
    print("      METRICS EVALUATION (FID, Diversity, Multimodality)")
    print("="*80)
    pairs = _collect_eval_pairs(model, tokenizer, eval_data, device)
    gt_acts, gen_acts, labels = _activations_from_pairs(pairs, all_motion_tokens)
    # Diversity
    diversity_times = min(200, max(4, gt_acts.shape[0] - 1))
    diversity_gt = calculate_diversity_np(gt_acts, diversity_times=diversity_times)
    diversity_gen = calculate_diversity_np(gen_acts, diversity_times=diversity_times)
    # Multimodality (MIM)
    try:
        gt_lbl_tensor = _to_label_tensor3(gt_acts, labels)
        gen_lbl_tensor = _to_label_tensor3(gen_acts, labels)
        multimodality_times = min(20, max(3, gt_lbl_tensor.shape[1] - 1))
        mim_gt = calculate_multimodality_np(gt_lbl_tensor, multimodality_times=multimodality_times)
        mim_gen = calculate_multimodality_np(gen_lbl_tensor, multimodality_times=multimodality_times)
    except Exception as exc:
        print(f"⚠️  Multimodality could not be computed reliably: {exc}")
        mim_gt = float("nan")
        mim_gen = float("nan")
    # FID
    mu_gen, cov_gen = calculate_activation_statistics_np(gen_acts)
    mu_gt, cov_gt = calculate_activation_statistics_np(gt_acts)
    fid = calculate_frechet_distance_np(mu_gt, cov_gt, mu_gen, cov_gen)
    print(f"Diversity:    GT = {diversity_gt:.4f} | GEN = {diversity_gen:.4f}")
    print(f"Multimodality (MIM): GT = {mim_gt:.4f} | GEN = {mim_gen:.4f}")
    print(f"FID (GT vs GEN): {fid:.4f}")
    return {
        "diversity_gt": diversity_gt,
        "diversity_gen": diversity_gen,
        "mim_gt": mim_gt,
        "mim_gen": mim_gen,
        "fid": fid,
        "pairs": pairs,  # for visualization usage
    }

# ======================================================================================
# 5b-ALT. Metrics using VQ-VAE codebook embeddings (near-standard activations)
# ======================================================================================
def _sequence_to_codebook_feature(seq: str, vq_model) -> np.ndarray:
    """
    Build a single clip feature by mean-pooling VQ-VAE codebook embeddings
    corresponding to the token ids in the sequence. L2-normalized.
    """
    token_ids = _extract_ids_from_sequence(seq)
    # Resolve code dimension and codebook availability
    quantizer = getattr(vq_model.vqvae, "quantizer", None)
    if quantizer is None:
        raise RuntimeError("VQ-VAE quantizer missing; cannot extract codebook embeddings.")
    # Try dequantize -> mean over time (preferred)
    feat_vec = None
    if hasattr(quantizer, "dequantize") and token_ids:
        try:
            idx = torch.tensor(token_ids, dtype=torch.long, device=next(vq_model.parameters()).device).unsqueeze(0)
            with torch.no_grad():
                dq = quantizer.dequantize(idx)
            if dq is not None:
                # Expect shape [N, code_dim, T]; average over T
                if dq.ndim == 3:
                    if dq.shape[0] == 1:
                        x = dq.squeeze(0)  # [code_dim, T] or [T, code_dim]
                    else:
                        x = dq.mean(dim=0)
                    if x.shape[0] < x.shape[1]:
                        # [code_dim, T]
                        feat = x.mean(dim=1)
                    else:
                        # [T, code_dim]
                        feat = x.mean(dim=0)
                    feat_vec = feat.detach().cpu().numpy().astype(np.float32)
        except Exception:
            feat_vec = None
    # Fallback: direct codebook lookup -> mean over token ids
    if feat_vec is None:
        codebook = getattr(quantizer, "codebook", None)
        if codebook is None:
            raise RuntimeError("Quantizer has neither dequantize() nor codebook.")
        code_np = codebook.detach().cpu().numpy()  # [K, D]
        if not token_ids:
            feat_vec = np.zeros((code_np.shape[1],), dtype=np.float32)
        else:
            ids = np.asarray(token_ids, dtype=np.int64)
            ids = np.clip(ids, 0, code_np.shape[0] - 1)
            feat_vec = code_np[ids].mean(axis=0).astype(np.float32)
    # L2-normalize to reduce length/scale bias
    norm = np.linalg.norm(feat_vec)
    if norm > 0:
        feat_vec = feat_vec / norm
    return feat_vec


def _activations_from_pairs_codebook(pairs: list[Tuple[str, str, str]], vq_model):
    """
    Produce codebook-embedding features for GT and GEN sequences and their labels.
    Returns:
      gt_feats: (N, D)
      gen_feats: (N, D)
      labels: list[str] of length N (word labels)
    """
    gt_feats = []
    gen_feats = []
    labels = []
    for pair in pairs:
        if len(pair) == 4:
            word, _pid, gt_seq, gen_seq = pair
        else:
            word, gt_seq, gen_seq = pair
        gt_feats.append(_sequence_to_codebook_feature(gt_seq, vq_model))
        gen_feats.append(_sequence_to_codebook_feature(gen_seq, vq_model))
        labels.append(word)
    return np.stack(gt_feats, axis=0), np.stack(gen_feats, axis=0), labels


def evaluate_metrics_codebook_style(model, tokenizer, eval_data, device, vqvae_ckpt: Optional[str] = None):
    """
    Computes FID, Diversity, and MIM using features derived from the VQ-VAE codebook:
      - Feature per clip = mean-pooled codebook embeddings over token sequence, L2-normalized
      - Diversity/MIM computed exactly as in MotionGPT-style helpers but on these features
      - FID computed via full covariance Fréchet distance on these features
    Returns a dict mirroring evaluate_metrics_motiongpt_style.
    """
    print("\n" + "="*80)
    print("      METRICS EVALUATION (Codebook-Embedding Features)")
    print("="*80)
    # Lazy import to avoid hard dependency at module import time
    try:
        from visualize import load_vqvae, VQVAE_CHECKPOINT as DEFAULT_VQ
        vq_ckpt = vqvae_ckpt or os.getenv("VQVAE_CHECKPOINT", DEFAULT_VQ)
        vq_model = load_vqvae(vq_ckpt, device=device)
    except Exception as exc:
        print(f"⚠️  Could not load VQ-VAE for codebook metrics: {exc}")
        return {}
    # Collect pairs and build features
    pairs = _collect_eval_pairs(model, tokenizer, eval_data, device)
    gt_feats, gen_feats, labels = _activations_from_pairs_codebook(pairs, vq_model)
    # Diversity
    diversity_times = min(200, max(4, gt_feats.shape[0] - 1))
    diversity_gt = calculate_diversity_np(gt_feats, diversity_times=diversity_times)
    diversity_gen = calculate_diversity_np(gen_feats, diversity_times=diversity_times)
    # Multimodality (MIM)
    try:
        gt_lbl_tensor = _to_label_tensor3(gt_feats, labels)
        gen_lbl_tensor = _to_label_tensor3(gen_feats, labels)
        multimodality_times = min(20, max(3, gt_lbl_tensor.shape[1] - 1))
        mim_gt = calculate_multimodality_np(gt_lbl_tensor, multimodality_times=multimodality_times)
        mim_gen = calculate_multimodality_np(gen_lbl_tensor, multimodality_times=multimodality_times)
    except Exception as exc:
        print(f"⚠️  Multimodality could not be computed reliably: {exc}")
        mim_gt = float("nan")
        mim_gen = float("nan")
    # FID (on codebook features)
    mu_gen, cov_gen = calculate_activation_statistics_np(gen_feats)
    mu_gt, cov_gt = calculate_activation_statistics_np(gt_feats)
    fid = calculate_frechet_distance_np(mu_gt, cov_gt, mu_gen, cov_gen)
    print(f"Diversity (codebook feats):    GT = {diversity_gt:.4f} | GEN = {diversity_gen:.4f}")
    print(f"Multimodality (MIM, codebook): GT = {mim_gt:.4f} | GEN = {mim_gen:.4f}")
    print(f"FID (codebook feats, GT vs GEN): {fid:.4f}")
    return {
        "diversity_gt": diversity_gt,
        "diversity_gen": diversity_gen,
        "mim_gt": mim_gt,
        "mim_gen": mim_gen,
        "fid": fid,
        "pairs": pairs,
    }

# ======================================================================================
# 5b-ALT2. Metrics using VQ-VAE encoder pre-quantization features (as described)
# ======================================================================================
def _encode_params_to_feature(params: np.ndarray, vq_model, mean, std, device) -> np.ndarray:
    """
    Convert SMPL-X parameter sequence (T, D) into a single clip feature using
    the VQ-VAE encoder output BEFORE quantization. Average-pool over time to get (D_embed,).
    - Attempts to use vq_model.vqvae.preprocess; otherwise applies manual normalization with mean/std.
    - Handles encoder outputs shaped as [N, D, T] or [N, T, D_embed].
    """
    if params.size == 0:
        return np.zeros((getattr(vq_model.vqvae, "output_emb_width", 512),), dtype=np.float32)
    x = torch.from_numpy(params.astype(np.float32)).to(device)  # [T, D]
    x = x.unsqueeze(0)  # [1, T, D]
    with torch.no_grad():
        # Normalize / preprocess
        x_pre = None
        if hasattr(vq_model.vqvae, "preprocess"):
            try:
                x_pre = vq_model.vqvae.preprocess(x)  # expected to return tensor ready for encoder
            except Exception:
                x_pre = None
        if x_pre is None:
            # Manual normalization with provided mean/std
            if mean is not None and std is not None:
                mean_t = torch.from_numpy(np.array(mean, dtype=np.float32)).to(device).view(1, 1, -1)
                std_t = torch.from_numpy(np.array(std, dtype=np.float32)).to(device).view(1, 1, -1)
                x_norm = (x - mean_t) / (std_t + 1e-8)
            else:
                x_norm = x
            # Some encoders expect [N, D, T]
            x_pre = x_norm.transpose(1, 2).contiguous()  # [1, D, T]
        # Encode to get pre-quant latent
        z_e = vq_model.vqvae.encoder(x_pre)
        # z_e could be [N, D_embed, T_q] or [N, T_q, D_embed]
        if z_e.dim() == 3:
            # Determine which axis is time by comparing to known embed dim when available,
            # otherwise assume time is the smaller dimension (varies per clip).
            embed_dim_known = getattr(vq_model.vqvae, "output_emb_width", None)
            if embed_dim_known is not None:
                if z_e.shape[1] == embed_dim_known:
                    time_axis = 2  # [N, D_embed, T_q]
                elif z_e.shape[2] == embed_dim_known:
                    time_axis = 1  # [N, T_q, D_embed]
                else:
                    time_axis = 2 if z_e.shape[2] < z_e.shape[1] else 1
            else:
                time_axis = 2 if z_e.shape[2] < z_e.shape[1] else 1
            feat = z_e.mean(dim=time_axis).squeeze(0)
        elif z_e.dim() == 2:
            feat = z_e.squeeze(0)
        else:
            # Fallback: flatten then reduce
            feat = z_e.view(1, -1).mean(dim=0)
        feat_np = feat.detach().cpu().numpy().astype(np.float32)
        # L2 normalize
        norm = np.linalg.norm(feat_np)
        if norm > 0:
            feat_np = feat_np / norm
        return feat_np


def evaluate_metrics_encoder_style(
    model,
    tokenizer,
    eval_data,
    device,
    vqvae_ckpt: Optional[str] = None,
    stats_path: Optional[str] = None,
    sample_limit: int = 100,
):
    """
    Computes FID, Diversity, and MIM using VQ-VAE encoder pre-quantization features:
      - For each sample, decode tokens -> SMPL-X params, then run through VQ-VAE encoder,
        average-pool across time, L2-normalize to get a clip feature.
      - Diversity/MIM identical formulations but on these encoder features.
      - FID via full covariance Fréchet distance on these encoder features.
    Evaluates on up to 'sample_limit' samples for speed.
    """
    print("\n" + "="*80)
    print("      METRICS EVALUATION (VQ-VAE Encoder Features)")
    print("="*80)
    # Lazy import to reuse your visualization utilities and stats
    try:
        from visualize import load_vqvae, load_stats, VQVAE_CHECKPOINT as DEFAULT_VQ, STATS_PATH as DEFAULT_STATS
        vq_ckpt = vqvae_ckpt or os.getenv("VQVAE_CHECKPOINT", DEFAULT_VQ)
        stats_p = stats_path or os.getenv("VQVAE_STATS_PATH", DEFAULT_STATS)
        vq_model = load_vqvae(vq_ckpt, device=device)
        mean, std = load_stats(stats_p)
        from visualize import decode_tokens_to_params
    except Exception as exc:
        print(f"⚠️  Could not set up VQ-VAE encoder metrics: {exc}")
        return {}
    # Collect GT/GEN token sequences for pairs (limit to speed-up)
    pairs = _collect_eval_pairs(model, tokenizer, eval_data[:sample_limit], device)
    # Build features
    gt_feats = []
    gen_feats = []
    labels = []
    for pair in pairs:
        if len(pair) == 4:
            word, _pid, gt_seq, gen_seq = pair
        else:
            word, gt_seq, gen_seq = pair
        # Decode to SMPL-X
        tokens_gt = _extract_ids_from_sequence(gt_seq)
        tokens_gen = _extract_ids_from_sequence(gen_seq)
        try:
            params_gt = decode_tokens_to_params(tokens_gt, vq_model, mean, std, device=device)  # (T, D) denorm
        except Exception:
            params_gt = np.zeros((0, 182), dtype=np.float32)
        try:
            params_gen = decode_tokens_to_params(tokens_gen, vq_model, mean, std, device=device)  # (T, D) denorm
        except Exception:
            params_gen = np.zeros((0, 182), dtype=np.float32)
        # Encode (pre-quant) -> pooled feature
        feat_gt = _encode_params_to_feature(params_gt, vq_model, mean, std, device)
        feat_gen = _encode_params_to_feature(params_gen, vq_model, mean, std, device)
        gt_feats.append(feat_gt)
        gen_feats.append(feat_gen)
        labels.append(word)
    gt_feats = np.stack(gt_feats, axis=0)
    gen_feats = np.stack(gen_feats, axis=0)
    # Diversity
    diversity_times = min(200, max(4, gt_feats.shape[0] - 1))
    diversity_gt = calculate_diversity_np(gt_feats, diversity_times=diversity_times)
    diversity_gen = calculate_diversity_np(gen_feats, diversity_times=diversity_times)
    # Multimodality (MIM)
    try:
        gt_lbl_tensor = _to_label_tensor3(gt_feats, labels)
        gen_lbl_tensor = _to_label_tensor3(gen_feats, labels)
        multimodality_times = min(20, max(3, gt_lbl_tensor.shape[1] - 1))
        mim_gt = calculate_multimodality_np(gt_lbl_tensor, multimodality_times=multimodality_times)
        mim_gen = calculate_multimodality_np(gen_lbl_tensor, multimodality_times=multimodality_times)
    except Exception as exc:
        print(f"⚠️  Multimodality could not be computed reliably: {exc}")
        mim_gt = float("nan")
        mim_gen = float("nan")
    # FID (on encoder features)
    mu_gen, cov_gen = calculate_activation_statistics_np(gen_feats)
    mu_gt, cov_gt = calculate_activation_statistics_np(gt_feats)
    fid = calculate_frechet_distance_np(mu_gt, cov_gt, mu_gen, cov_gen)
    print(f"Diversity (encoder feats):    GT = {diversity_gt:.4f} | GEN = {diversity_gen:.4f}")
    print(f"Multimodality (MIM, encoder): GT = {mim_gt:.4f} | GEN = {mim_gen:.4f}")
    print(f"FID (encoder feats, GT vs GEN): {fid:.4f}")
    return {
        "diversity_gt": diversity_gt,
        "diversity_gen": diversity_gen,
        "mim_gt": mim_gt,
        "mim_gen": mim_gen,
        "fid": fid,
        "pairs": pairs,
    }

# ======================================================================================
# 5c. Side-by-side visualization (4 samples)
# ======================================================================================
def _extract_ids_from_sequence(seq: str) -> list[int]:
    return [int(t[2:-1]) for t in _extract_motion_tokens_from_sequence(seq) if t[2:-1].isdigit()]

def save_side_by_side_visualizations(pairs: list[Tuple[str, str, str]], output_dir: str, limit: int = 4):
    """
    Generate side-by-side 3D animations for GT vs GEN, saving one HTML per sample
    using filename scheme: word_PID_side_by_side.html.
    - Processes ALL samples for up to `limit` distinct words (if provided).
    - Requires visualize.py utilities and plotly.
    """
    try:
        from visualize import (
            load_vqvae, load_stats, load_smplx_model,
            decode_tokens_to_params, params_to_vertices,
            VQVAE_CHECKPOINT as DEFAULT_VQ, STATS_PATH as DEFAULT_STATS, SMPLX_MODEL_DIR as DEFAULT_SMPLX
        )
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as exc:
        print(f"⚠️  Visualization skipped (missing dependencies): {exc}")
        return

    os.makedirs(output_dir, exist_ok=True)
    vqvae_ckpt = os.getenv("VQVAE_CHECKPOINT", DEFAULT_VQ)
    stats_path = os.getenv("VQVAE_STATS_PATH", DEFAULT_STATS)
    smplx_dir = os.getenv("SMPLX_MODEL_DIR", DEFAULT_SMPLX)

    print("Loading VQ-VAE, stats, SMPL-X ...")
    vq_model = load_vqvae(vqvae_ckpt)
    mean, std = load_stats(stats_path)
    smplx_model = load_smplx_model(smplx_dir)

    def animate_side_by_side(verts_left, faces, verts_right, fps=20, titles=("Ground Truth", "LLM Generated"), output_html=None):
        T = min(verts_left.shape[0], verts_right.shape[0])
        verts_left, verts_right = verts_left[:T], verts_right[:T]
        i, j, k = faces.T.tolist()
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scene'}, {'type': 'scene'}]],
            horizontal_spacing=0.05,
            subplot_titles=list(titles)
        )
        left_mesh = go.Mesh3d(x=verts_left[0,:,0], y=verts_left[0,:,1], z=verts_left[0,:,2], i=i,j=j,k=k,opacity=0.7,showscale=False)
        right_mesh = go.Mesh3d(x=verts_right[0,:,0], y=verts_right[0,:,1], z=verts_right[0,:,2], i=i,j=j,k=k,opacity=0.7,showscale=False)
        fig.add_trace(left_mesh, row=1, col=1)
        fig.add_trace(right_mesh, row=1, col=2)
        frames = []
        for t in range(T):
            frames.append(go.Frame(
                name=str(t),
                data=[
                    go.Mesh3d(x=verts_left[t,:,0], y=verts_left[t,:,1], z=verts_left[t,:,2], i=i,j=j,k=k,opacity=0.7,showscale=False,scene="scene"),
                    go.Mesh3d(x=verts_right[t,:,0], y=verts_right[t,:,1], z=verts_right[t,:,2], i=i,j=j,k=k,opacity=0.7,showscale=False,scene="scene2")
                ]
            ))
        fig.frames = frames
        fig.update_layout(
            showlegend=False,
            margin=dict(l=10, r=10, t=50, b=10),
            scene=dict(aspectmode='data',xaxis=dict(visible=False),yaxis=dict(visible=False),zaxis=dict(visible=False),
                       camera=dict(eye=dict(x=0,y=-2,z=0.7))),
            scene2=dict(aspectmode='data',xaxis=dict(visible=False),yaxis=dict(visible=False),zaxis=dict(visible=False),
                        camera=dict(eye=dict(x=0,y=-2,z=0.7))),
            updatemenus=[dict(
                type="buttons", x=0.5, xanchor="center", y=1.15, yanchor="top",
                buttons=[
                    dict(label="Play", method="animate", args=[None, {"frame": {"duration": max(1,1000//fps), "redraw": True}, "fromcurrent": True}]),
                    dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}}])
                ]
            )]
        )
        if output_html:
            fig.write_html(output_html)
            print(f"✅ Saved: {output_html}")
        return fig

    # Determine which words to include (up to `limit` distinct words)
    allowed_words = None
    if isinstance(limit, int) and limit > 0:
        ordered_unique_words = []
        for pair in pairs:
            word = pair[0]
            if word not in ordered_unique_words:
                ordered_unique_words.append(word)
            if len(ordered_unique_words) >= limit:
                break
        allowed_words = set(ordered_unique_words)

    for pair in pairs:
        try:
            if len(pair) == 4:
                word, pid, gt_seq, gen_seq = pair
            else:
                word, gt_seq, gen_seq = pair
                pid = "unknown"
            if allowed_words is not None and word not in allowed_words:
                continue
            tokens_gt = _extract_ids_from_sequence(gt_seq)
            tokens_gen = _extract_ids_from_sequence(gen_seq)
            params_gt = decode_tokens_to_params(tokens_gt, vq_model, mean, std)
            params_gen = decode_tokens_to_params(tokens_gen, vq_model, mean, std)
            verts_gt, faces = params_to_vertices(params_gt, smplx_model)
            verts_gen, _ = params_to_vertices(params_gen, smplx_model)
            out_dir = os.path.join(output_dir)
            os.makedirs(out_dir, exist_ok=True)
            # Sanitize for filesystem safety
            safe_word = re.sub(r'[^A-Za-z0-9_-]+', '_', str(word))
            safe_pid = re.sub(r'[^A-Za-z0-9_-]+', '_', str(pid))
            output_html = os.path.join(out_dir, f"word_{safe_word}_{safe_pid}_side_by_side.html")
            animate_side_by_side(
                verts_left=verts_gt,
                faces=faces,
                verts_right=verts_gen,
                fps=20,
                titles=("Ground Truth", "LLM Generated"),
                output_html=output_html
            )
        except Exception as exc:
            print(f"⚠️  Error creating visualization for word '{pair[0]}': {exc}")

# ======================================================================================
# 6. Main Execution Block (UPDATED)
# ======================================================================================
def main(config_overrides: Optional[Dict[str, Any]] = None):
    """Main function to run the entire pipeline."""
    apply_config_overrides(config_overrides)
    if config_overrides:
        printable = {k: v for k, v in config_overrides.items() if "token" not in k.lower()}
        if printable:
            print("\nApplied config overrides:")
            for key, value in printable.items():
                print(f"  - {key} = {value}")
    random.seed(42)
    torch.manual_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load ALL data
    all_entries = read_json_data(DATASET_PATH)
    
    # 2. Clean the ENTIRE dataset and get all tokens
    cleaned_data, all_motion_tokens = deduplicate_and_prepare_data(all_entries)

    # 3. Stage 1: Initialize or resume from HF, then train
    resolved_stage1_repo = _resolve_and_ensure_repo(HF_STAGE1_REPO_ID) if HF_USE_HUB else None
    start_epoch_s1 = 0
    stage1_loaded = None
    if resolved_stage1_repo:
        if _repo_has_stage_latest(resolved_stage1_repo, "stage1"):
            stage1_loaded = _load_model_and_tokenizer_from_hf(resolved_stage1_repo, "stage1")
            state_s1 = _download_training_state(resolved_stage1_repo, "stage1")
            if state_s1 and isinstance(state_s1.get("epoch_completed"), int):
                start_epoch_s1 = state_s1["epoch_completed"]
        else:
            # Fallback: no 'latest' folder; select highest epoch-XXX
            latest_s1_sub = _repo_get_latest_epoch_subfolder(resolved_stage1_repo, "stage1")
            if latest_s1_sub:
                stage1_loaded = _load_model_and_tokenizer_from_hf_subfolder(resolved_stage1_repo, latest_s1_sub)
                state_s1 = _download_training_state_from_subfolder(resolved_stage1_repo, latest_s1_sub)
                if state_s1 and isinstance(state_s1.get("epoch_completed"), int):
                    start_epoch_s1 = state_s1["epoch_completed"]

    if stage1_loaded:
        base_model, tokenizer = stage1_loaded
        # Ensure tokenizer contains all motion tokens (add missing if dataset expanded)
        added = _ensure_tokenizer_has_motion_tokens(tokenizer, all_motion_tokens)
        if added > 0:
            base_model.resize_token_embeddings(len(tokenizer))
    else:
        base_model, tokenizer = setup_model_and_tokenizer(MODEL_NAME, all_motion_tokens)

    print(f"\nStarting Stage 1 training on {len(cleaned_data)} samples (resume from epoch {start_epoch_s1}).")
    motion_model = train_stage1(
        base_model,
        tokenizer,
        cleaned_data,
        device,
        start_epoch=start_epoch_s1,
        hf_repo_id=resolved_stage1_repo,
    )

    # 4. Stage 2: Initialize or resume from HF, then train
    resolved_stage2_repo = _resolve_and_ensure_repo(HF_STAGE2_REPO_ID) if HF_USE_HUB else None
    start_epoch_s2 = 0
    stage2_loaded = None
    print(f"\nStage 2 resume policy: FORCE_STAGE2_FROM_STAGE1={FORCE_STAGE2_FROM_STAGE1}, save_subdir='{HF_STAGE2_SAVE_SUBDIR}'")
    # For this run we want Stage 2 to start from Stage 1 epoch-20 even if an old stage2 exists.
    # Only resume Stage 2 if explicitly allowed and if there is a checkpoint under the save subdir.
    if not FORCE_STAGE2_FROM_STAGE1 and resolved_stage2_repo:
        # Prefer loading from the configured Stage 2 save subdir (e.g., 'stage2_v2')
        if _repo_has_stage_latest(resolved_stage2_repo, HF_STAGE2_SAVE_SUBDIR):
            stage2_loaded = _load_model_and_tokenizer_from_hf(resolved_stage2_repo, HF_STAGE2_SAVE_SUBDIR)
            state_s2 = _download_training_state(resolved_stage2_repo, HF_STAGE2_SAVE_SUBDIR)
            if state_s2 and isinstance(state_s2.get("epoch_completed"), int):
                start_epoch_s2 = state_s2["epoch_completed"]
            print(f"Resuming Stage 2 from HF subfolder: {HF_STAGE2_SAVE_SUBDIR}/latest (epoch_completed={start_epoch_s2})")
        else:
            latest_s2_sub = _repo_get_latest_epoch_subfolder(resolved_stage2_repo, HF_STAGE2_SAVE_SUBDIR)
            if latest_s2_sub:
                stage2_loaded = _load_model_and_tokenizer_from_hf_subfolder(resolved_stage2_repo, latest_s2_sub)
                state_s2 = _download_training_state_from_subfolder(resolved_stage2_repo, latest_s2_sub)
                if state_s2 and isinstance(state_s2.get("epoch_completed"), int):
                    start_epoch_s2 = state_s2["epoch_completed"]
                print(f"Resuming Stage 2 from HF subfolder: {latest_s2_sub} (epoch_completed={start_epoch_s2})")

    if stage2_loaded:
        stage2_model, tokenizer = stage2_loaded
        added2 = _ensure_tokenizer_has_motion_tokens(tokenizer, all_motion_tokens)
        if added2 > 0:
            stage2_model.resize_token_embeddings(len(tokenizer))
    else:
        stage2_model = motion_model  # Start Stage 2 from Stage 1 model

    print(f"\nStarting Stage 2 training on {len(cleaned_data)} samples (resume from epoch {start_epoch_s2}).")
    final_model = train_stage2(
        stage2_model,
        tokenizer,
        cleaned_data,
        device,
        start_epoch=start_epoch_s2,
        hf_repo_id=resolved_stage2_repo,
        hf_stage_subdir=HF_STAGE2_SAVE_SUBDIR,
    )
    
    # 5. Filter the cleaned data to get a smaller set for evaluation
    # This keeps the evaluation focused on our benchmark words and the logs readable
    print("\n--- Filtering data for evaluation on specific words ---")
    evaluation_data = [item for item in cleaned_data if item['word'].lower() in EVALUATION_WORDS]
    print(f"Found {len(evaluation_data)} samples for evaluation words: {EVALUATION_WORDS}")

    # 6. Metrics-only mode or full flow
    if RUN_EVALS_ONLY:
        # Compute the 3 metrics using VQ-VAE encoder features and save to JSON
        metrics_enc = evaluate_metrics_encoder_style(
            final_model, tokenizer, evaluation_data, device, sample_limit=EVAL_SAMPLE_LIMIT
        )
        os.makedirs(OUTPUT_DIR, exist_ok=True)
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
        }
        with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Saved metrics to {METRICS_JSON_PATH}")
        return

    # Full flow: inference logs + MotionGPT-style metrics + encoder metrics + visualizations
    run_inference_on_all_samples(final_model, tokenizer, evaluation_data, device)
    metrics_token = evaluate_metrics_motiongpt_style(final_model, tokenizer, evaluation_data, all_motion_tokens, device)
    # Also compute encoder-based 3 metrics
    metrics_enc = evaluate_metrics_encoder_style(
        final_model, tokenizer, evaluation_data, device, sample_limit=EVAL_SAMPLE_LIMIT
    )
    # Visualizations (skip if metrics-only)
    viz_dir = os.path.join(OUTPUT_DIR, "html_visualizations")
    save_side_by_side_visualizations(metrics_token["pairs"], viz_dir, limit=4)

if __name__ == "__main__":
    main()