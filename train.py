"""
Training utilities and functions
"""
import math
import os
import re
import time
import json
import shutil
import torch
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_callback import TrainerCallback
from huggingface_hub import HfApi, upload_folder, snapshot_download, hf_hub_download

from config import (
    BATCH_TRAIN, BATCH_EVAL, GRAD_ACCUM, LR, WARMUP,
    LOG_STEPS, EVAL_STEPS, SAVE_STEPS, SEED, DTYPE,
    HUB_REPO_S1, HUB_REPO_S2, HUB_REPO_S3, HF_TOKEN,
    CHECKPOINTS_DIR, HF_USE_HUB, CHECKPOINT_UPLOAD_INTERVAL_EPOCHS,
    S1_BATCH_SIZE, S1_LR, S1_EPOCHS, S2_BATCH_SIZE, S2_LR, S2_EPOCHS,
    PAD_TOKEN, M_START, M_END
)

# ======================================================================================
# Logic from test_overfit.py (Raw Training Loops & HF Utils)
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

def resolve_and_ensure_repo(repo_id: str, hf_auth_token: Optional[str] = None) -> Optional[str]:
    """
    Ensures the HF repo exists. Returns the fully-qualified repo_id (namespace/repo)
    when token is available; otherwise returns the input repo_id.
    """
    if not HF_USE_HUB:
        return None
    token = hf_auth_token or HF_TOKEN
    if not token:
        print("⚠️  HF token not found. Set HUGGINGFACE_HUB_TOKEN to enable Hub sync.")
        return None
    api = HfApi()
    try:
        who = api.whoami(token=token)
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
            token=token,
            repo_type="model",
            private=True, # Default to private as in test_overfit config if not specified
            exist_ok=True,
        )
    except Exception as exc:
        print(f"⚠️  create_repo failed (may already exist): {exc}")
    return full_repo_id

def repo_has_stage_latest(repo_id: str, stage: str, hf_auth_token: Optional[str] = None) -> bool:
    """Checks if a stage/latest checkpoint exists in the HF repo."""
    token = hf_auth_token or HF_TOKEN
    if not HF_USE_HUB or not token:
        return False
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="model", token=token)
        return any(path.startswith(f"{stage}/latest/") and path.endswith("config.json") for path in files)
    except Exception as exc:
        print(f"⚠️  Could not list files for {repo_id}: {exc}")
        return False

def repo_list_epoch_numbers(repo_id: str, stage: str, hf_auth_token: Optional[str] = None) -> List[int]:
    """
    Returns sorted list of epoch numbers available under {stage}/epoch-XXX/ by scanning files.
    """
    token = hf_auth_token or HF_TOKEN
    if not HF_USE_HUB or not token:
        return []
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="model", token=token)
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

def repo_get_latest_epoch_subfolder(repo_id: str, stage: str, hf_auth_token: Optional[str] = None) -> Optional[str]:
    """
    Returns subfolder path like '{stage}/epoch-XXX' for the highest available epoch, or None.
    """
    epochs = repo_list_epoch_numbers(repo_id, stage, hf_auth_token)
    if not epochs:
        return None
    latest = max(epochs)
    return f"{stage}/epoch-{latest:03d}"

def load_model_and_tokenizer_from_hf_subfolder(repo_id: str, subfolder: str, hf_auth_token: Optional[str] = None) -> Optional[Tuple[AutoModelForCausalLM, AutoTokenizer]]:
    """
    Loads model and tokenizer from HF under a specific subfolder.
    """
    if not HF_USE_HUB or (not hf_auth_token and not HF_TOKEN):
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

def download_training_state_from_subfolder(repo_id: str, subfolder: str, hf_auth_token: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Downloads training_state.json from a specific subfolder.
    """
    token = hf_auth_token or HF_TOKEN
    if not HF_USE_HUB or not token:
        return None
    try:
        state_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{subfolder}/training_state.json",
            repo_type="model",
            token=token,
        )
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def download_training_state(repo_id: str, stage: str, hf_auth_token: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Downloads training_state.json from HF if present."""
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

def download_optimizer_state(repo_id: str, stage: str, hf_auth_token: Optional[str] = None) -> Optional[str]:
    """Downloads optimizer.pt for resuming optimizer state."""
    token = hf_auth_token or HF_TOKEN
    if not HF_USE_HUB or not token:
        return None
    try:
        opt_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{stage}/latest/optimizer.pt",
            repo_type="model",
            token=token,
        )
        return opt_path
    except Exception:
        return None

def load_model_and_tokenizer_from_hf(repo_id: str, stage: str, hf_auth_token: Optional[str] = None) -> Optional[Tuple[AutoModelForCausalLM, AutoTokenizer]]:
    """
    Loads model and tokenizer from HF under subfolder {stage}/latest if available.
    """
    if not repo_has_stage_latest(repo_id, stage, hf_auth_token):
        return None
    print(f"\n---> Loading checkpoint from Hugging Face: {repo_id} (subfolder='{stage}/latest')")
    tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder=f"{stage}/latest", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(repo_id, subfolder=f"{stage}/latest", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer

def save_and_push_checkpoint(
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
    hf_auth_token: Optional[str] = None
) -> None:
    """
    Saves checkpoint locally and pushes to HF.
    """
    token = hf_auth_token or HF_TOKEN
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
    if HF_USE_HUB and repo_id and token:
        try:
            upload_folder(
                repo_id=repo_id,
                folder_path=epoch_dir,
                path_in_repo=f"{stage}/{epoch_dir_name}",
                repo_type="model",
                token=token,
                commit_message=f"{stage}: save {epoch_dir_name}",
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=latest_dir,
                path_in_repo=f"{stage}/latest",
                repo_type="model",
                token=token,
                commit_message=f"{stage}: update latest -> {epoch_dir_name}",
            )
            print(f"☁️  Pushed checkpoint to HF: {repo_id} ({stage}/{epoch_dir_name} and {stage}/latest)")
        except Exception as exc:
            print(f"⚠️  Failed to push checkpoint to HF: {exc}")
    else:
        print("ℹ️  Skipped HF push (Hub disabled or token/repo missing).")

def train_stage1_raw(
    model,
    tokenizer,
    data: List[Dict[str, Any]],
    device,
    start_epoch: int = 0,
    hf_repo_id: Optional[str] = None,
):
    """Trains the model on motion sequences only to learn the 'language of motion'."""
    from data import MotionDataset # Import here to avoid circular imports
    
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
    token = HF_TOKEN
    if hf_repo_id and start_epoch > 0 and HF_USE_HUB and token:
        opt_path = download_optimizer_state(hf_repo_id, "stage1", token)
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
        save_and_push_checkpoint(
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
            hf_auth_token=token
        )
    
    print("\n✅ Stage 1 Training Complete.")
    return model

def train_stage2_raw(
    model,
    tokenizer,
    data: List[Dict[str, Any]],
    device,
    start_epoch: int = 0,
    hf_repo_id: Optional[str] = None,
    hf_stage_subdir: str = "stage2",
):
    """Fine-tunes the motion-aware model to connect text prompts to motions."""
    from data import TextMotionDataset # Import here to avoid circular imports

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
    token = HF_TOKEN
    if hf_repo_id and start_epoch > 0 and HF_USE_HUB and token:
        opt_path = download_optimizer_state(hf_repo_id, hf_stage_subdir, token)
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
        save_and_push_checkpoint(
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
            hf_auth_token=token
        )
        
    print("\n✅ Stage 2 Training Complete.")
    return model

# ======================================================================================
# Existing Utilities
# ======================================================================================

def make_training_args(out_dir: str, epochs: int, two_point_hub: bool = False) -> TrainingArguments:
    """
    Create TrainingArguments for a training stage
    """
    return TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=BATCH_TRAIN,
        per_device_eval_batch_size=BATCH_EVAL,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_train_epochs=epochs,
        logging_steps=LOG_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        # When using two-point hub checkpointing, disable periodic local saves and rely on forced saves
        save_steps=(10**12 if two_point_hub else SAVE_STEPS),
        save_total_limit=2,
        warmup_ratio=WARMUP,
        bf16=(DTYPE == torch.bfloat16),
        fp16=(DTYPE == torch.float16),
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        report_to="none",
        seed=SEED,
        remove_unused_columns=False,
    )


def latest_hub_checkpoint(repo_id: str) -> Optional[str]:
    """
    Download and return the local path to the latest checkpoint folder from the Hub.
    Returns None if no checkpoint exists or on failure.
    """
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    except Exception as e:
        print(f"Hub list failed for {repo_id}: {e}")
        return None

    def _step_key(dirname: str) -> int:
        nums = re.findall(r"\d+", dirname)
        return int(nums[-1]) if nums else -1

    ckpt_dirs = sorted(
        {p.split('/')[0] for p in files if p.startswith("checkpoint-")},
        key=_step_key,
    )
    if not ckpt_dirs:
        return None
    latest = ckpt_dirs[-1]
    local_root = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=[f"{latest}/**", "trainer_state.json"],
        local_dir_use_symlinks=False,
    )
    return os.path.join(local_root, latest)


class TwoPointHubCheckpointCallback(TrainerCallback):
    """
    Save to Hugging Face Hub exactly twice per training: halfway and at final step.
    Keeps only the most recent N checkpoints on Hub.
    """

    def __init__(self, repo_id: str, keep_last: int = 2, token: Optional[str] = None):
        self.repo_id = repo_id
        self.keep_last = keep_last
        self.api = HfApi()
        self.token = token or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        self._half_step: Optional[int] = None
        self._final_step: Optional[int] = None
        self._saved_steps = set()
        self._pending_push_for_step: Optional[int] = None
        try:
            self.api.create_repo(repo_id=self.repo_id, private=True, exist_ok=True, token=self.token)
        except Exception as e:
            print(f"Could not ensure repo exists: {e}")

    def _enforce_keep_last(self) -> None:
        try:
            files = self.api.list_repo_files(repo_id=self.repo_id, repo_type="model", token=self.token)

            def _step_key(dirname: str) -> int:
                nums = re.findall(r"\d+", dirname)
                return int(nums[-1]) if nums else -1

            ckpt_dirs = sorted(
                {p.split('/')[0] for p in files if p.startswith("checkpoint-")},
                key=_step_key,
            )
            if len(ckpt_dirs) <= self.keep_last:
                return
            to_delete = ckpt_dirs[:-self.keep_last]
            for d in to_delete:
                for f in [p for p in files if p.startswith(f"{d}/")]:
                    try:
                        self.api.delete_file(path=f, repo_id=self.repo_id, repo_type="model", token=self.token)
                    except Exception as e:
                        print(f"Failed deleting {f}: {e}")
        except Exception as e:
            print(f"Keep-last enforcement failed: {e}")

    def on_train_begin(self, args, state, control, **kwargs):
        # Prefer Trainer-computed max_steps
        if state.max_steps and state.max_steps > 0:
            self._half_step = max(1, state.max_steps // 2)
            self._final_step = state.max_steps
            print(f"Two-point checkpointing: half={self._half_step}, final={self._final_step}")
        else:
            # Best-effort fallback using dataloader length and grad accumulation if available
            td = kwargs.get("train_dataloader")
            if td is not None and args.gradient_accumulation_steps > 0:
                steps_per_epoch = math.ceil(len(td) / args.gradient_accumulation_steps)
                self._final_step = steps_per_epoch * int(args.num_train_epochs)
                self._half_step = max(1, self._final_step // 2)
                print(f"Two-point checkpointing (approx): half={self._half_step}, final={self._final_step}")

    def on_step_end(self, args, state, control, **kwargs):
        if not self._final_step:
            return control
        gs = state.global_step
        if gs == self._half_step and gs not in self._saved_steps:
            control.should_save = True
            self._pending_push_for_step = gs
        if gs == self._final_step and gs not in self._saved_steps:
            control.should_save = True
            self._pending_push_for_step = gs
        return control

    def on_save(self, args, state, control, **kwargs):
        # Push only when we triggered this save
        if self._pending_push_for_step is None:
            return control
        step = self._pending_push_for_step
        self._pending_push_for_step = None
        self._saved_steps.add(step)

        ckpt_dirname = f"checkpoint-{step}"
        try:
            upload_folder(
                repo_id=self.repo_id,
                folder_path=args.output_dir,
                repo_type="model",
                token=self.token,
                commit_message=f"upload {ckpt_dirname}",
                allow_patterns=[f"{ckpt_dirname}/**", "trainer_state.json"],
            )
            self._enforce_keep_last()
            print(f"Pushed {ckpt_dirname} to {self.repo_id}")
        except Exception as e:
            print(f"Hub upload failed for {ckpt_dirname}: {e}")
        return control


def train_stage(
    stage_name: str,
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    data_collator,
    out_dir: str,
    epochs: int,
    hub_repo: Optional[str] = None,
):
    """
    Train a single stage
    """
    print(f"\n{'='*60}")
    print(f"Training {stage_name}")
    print(f"{'='*60}")
    
    # Auto-select Hub repo by stage if not provided
    if hub_repo is None:
        s = (stage_name or "").lower()
        if s.startswith("stage1"):
            hub_repo = HUB_REPO_S1
        elif s.startswith("stage2"):
            hub_repo = HUB_REPO_S2
        elif s.startswith("stage3"):
            hub_repo = HUB_REPO_S3

    args = make_training_args(out_dir, epochs, two_point_hub=(hub_repo is not None))
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=args,
        data_collator=data_collator,
    )

    # Train-loss early stop (match test_overfit behavior)
    class TrainLossStopCallback(TrainerCallback):
        def __init__(self, threshold: float = 1.0):
            self.threshold = float(threshold)
            self.triggered = False

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return control
            loss = logs.get("loss")
            if loss is not None and loss < self.threshold and state.global_step > 0 and not self.triggered:
                self.triggered = True
                print(f"\nTrain-loss early stop: loss={loss:.4f} < {self.threshold}")
                control.should_training_stop = True
            return control

    trainer.add_callback(TrainLossStopCallback(threshold=1.0))
    
    # Add two-point Hub checkpoint uploader if configured
    if hub_repo:
        # Pass token if available to avoid auth prompts in Kaggle/Colab
        token = HF_TOKEN if isinstance(HF_TOKEN, str) and len(HF_TOKEN) > 0 else None
        trainer.add_callback(TwoPointHubCheckpointCallback(hub_repo, token=token))

    # Train (with auto-resume from Hub if available)
    resume_path = latest_hub_checkpoint(hub_repo) if hub_repo else None
    if resume_path:
        print(f"Resuming from Hub checkpoint: {resume_path}")
        trainer.train(resume_from_checkpoint=resume_path)
    else:
        print(f"Starting training for {stage_name}...")
        trainer.train()
    
    # Evaluate
    print(f"Evaluating {stage_name}...")
    metrics = trainer.evaluate()
    
    # Compute perplexity
    eval_loss = metrics.get("eval_loss", float("nan"))
    ppl = math.exp(eval_loss) if not math.isnan(eval_loss) else float("nan")
    
    print(f"\n{stage_name} Results:")
    print(f"  eval_loss: {eval_loss:.4f}")
    print(f"  perplexity: {ppl:.3f}")
    
    # Save model (optional - can be commented out to save space)
    # trainer.save_model(out_dir)
    # print(f"Model saved to {out_dir}")
    
    return metrics


def save_model_to_hub(model, tokenizer, repo_id: str, stage_name: str):
    """
    Save model and tokenizer to HuggingFace Hub
    """
    print(f"\nSaving {stage_name} to HuggingFace Hub: {repo_id}")
    model.push_to_hub(repo_id, commit_message=f"Upload {stage_name}")
    tokenizer.push_to_hub(repo_id, commit_message=f"Upload {stage_name}")
    print(f"Successfully saved {stage_name}")


def load_model_from_hub(repo_id: str):
    """
    Load model and tokenizer from HuggingFace Hub
    """
    from unsloth import FastLanguageModel
    from config import MAX_SEQ_LEN, DTYPE
    
    print(f"\nLoading model from HuggingFace Hub: {repo_id}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=repo_id,
        max_seq_length=MAX_SEQ_LEN,
        dtype=DTYPE,
        load_in_4bit=True,
    )
    print(f"Successfully loaded model from {repo_id}")
    return model, tokenizer
