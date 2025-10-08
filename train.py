"""
Training utilities and functions
"""
import math
import os
import re
from typing import Optional

import torch
from transformers import TrainingArguments, Trainer
from transformers.trainer_callback import TrainerCallback
from huggingface_hub import HfApi, upload_folder, snapshot_download
from config import (
    BATCH_TRAIN, BATCH_EVAL, GRAD_ACCUM, LR, WARMUP,
    LOG_STEPS, EVAL_STEPS, SAVE_STEPS, SEED, DTYPE,
    HUB_REPO_S1, HUB_REPO_S2, HUB_REPO_S3, HF_TOKEN
)


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
        optim="paged_adamw_8bit",
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