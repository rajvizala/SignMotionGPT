"""
Training script for Experiment 3: Semantic-Aware Regularised LLM Training.

This wraps the existing sentence-level training pipeline with the three
regularisation techniques from model.py.

Usage
-----
    python -m agent_experiments.exp3_semantic_regularization.train \
        --dataset-path /path/to/sentence_dataset.json \
        --vqvae-ckpt /path/to/vqvae_checkpoint.pt \
        --output-dir ./agent_experiments/outputs/exp3 \
        [--lambda-contrastive 0.1] \
        [--lambda-temporal 0.01] \
        [--label-smooth-alpha 0.1] \
        [--epochs 40] \
        [--batch-size 16]

WORKFLOW
-------
1. Load sentence-level data and VQ-VAE codebook
2. Set up LLM (Qwen3-0.6B) with motion vocabulary
3. Initialise motion token embeddings from VQ-VAE codebook (semantic bridge)
4. Train with augmented loss = CE + contrastive + temporal + label-smoothing
5. Evaluate on held-out validation split
"""

import argparse
import json
import os
import random
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from agent_experiments.shared.base_config import ExperimentConfig
from agent_experiments.shared.logging_utils import ExperimentLogger
from agent_experiments.shared.data_utils import load_json_data, split_by_type, extract_motion_vocab
from agent_experiments.exp3_semantic_regularization.model import SemanticRegularisedTrainer

warnings.filterwarnings("ignore")

M_START = "<M_START>"
M_END = "<M_END>"
PAD_TOKEN = "<PAD>"
LEN_SHORT = "<|LEN_SHORT|>"
LEN_MEDIUM = "<|LEN_MEDIUM|>"
LEN_LONG = "<|LEN_LONG|>"

TEMPLATES = [
    "Instruction: Generate sign language motion for the sentence: '{text}'",
    "Instruction: Translate this text to sign language: '{text}'",
    "Instruction: How would you sign '{text}'?",
    "Instruction: Create a sign language animation for: '{text}'",
]


class RegularisedSentenceDataset(Dataset):
    """Sentence-level dataset for regularised training."""

    def __init__(self, data: List[Dict], templates: List[str] = None):
        self.items = []
        templates = templates or TEMPLATES
        for item in data:
            text = (item.get("text") or item.get("sentence", "")).strip()
            tokens_str = item.get("motion_tokens", "")
            if not text or not tokens_str.strip():
                continue
            motion_len = len(tokens_str.split())
            if motion_len < 30:
                len_tok = LEN_SHORT
            elif motion_len < 80:
                len_tok = LEN_MEDIUM
            else:
                len_tok = LEN_LONG
            wrapped = " ".join(f"<M{t}>" for t in tokens_str.split())
            target = f"{M_START} {wrapped} {M_END}"
            for tmpl in templates:
                instruction = tmpl.format(text=text)
                prompt = f"{instruction} (Length: {len_tok})\nMotion: "
                self.items.append({
                    "prompt": prompt,
                    "full_text": prompt + target,
                    "text": text,
                    "motion_length": motion_len,
                })
        print(f"[Dataset] {len(self.items)} training items from {len(data)} sentences")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def create_collate_fn(tokenizer, max_length=256):
    pad_id = tokenizer.pad_token_id
    m_start_id = tokenizer.convert_tokens_to_ids(M_START)

    def collate_fn(batch):
        full_texts = [item["full_text"] for item in batch]
        tokenized = tokenizer(full_texts, truncation=True, max_length=max_length,
                              padding=True, return_tensors="pt")
        labels = tokenized["input_ids"].clone()
        for i in range(len(full_texts)):
            ids = tokenized["input_ids"][i]
            positions = (ids == m_start_id).nonzero(as_tuple=True)[0]
            prompt_len = positions[0].item() if len(positions) > 0 else len(ids)
            labels[i, :prompt_len] = -100
            if pad_id is not None:
                labels[i, ids == pad_id] = -100
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }
    return collate_fn


def load_codebook(ckpt_path: str) -> Optional[torch.Tensor]:
    """Extract codebook tensor from VQ-VAE checkpoint."""
    if not os.path.exists(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    for key in ["vqvae.quantizer.codebook", "quantizer.codebook",
                "vqvae.quantizer.embedding.weight", "quantizer.embedding.weight"]:
        if key in sd:
            return sd[key]
    for key in sd:
        if "codebook" in key.lower():
            return sd[key]
    return None


def main():
    parser = argparse.ArgumentParser(description="Exp3: Semantic Regularised LLM")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--vqvae-ckpt", required=True)
    parser.add_argument("--output-dir", default="./agent_experiments/outputs/exp3")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--lambda-contrastive", type=float, default=0.1)
    parser.add_argument("--lambda-temporal", type=float, default=0.01)
    parser.add_argument("--label-smooth-alpha", type=float, default=0.1)
    parser.add_argument("--label-smooth-tau", type=float, default=1.0)
    parser.add_argument("--no-contrastive", action="store_true")
    parser.add_argument("--no-temporal", action="store_true")
    parser.add_argument("--no-label-smooth", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = ExperimentLogger(args.output_dir, "exp3_semantic_reg")
    logger.log_hyperparameters(vars(args))

    # -- Data ---------------------------------------------------------------
    raw_data = load_json_data(args.dataset_path)
    _, sentence_data = split_by_type(raw_data)
    if not sentence_data:
        sentence_data = raw_data

    n_val = max(1, int(len(sentence_data) * 0.1))
    rng = random.Random(args.seed)
    indices = list(range(len(sentence_data)))
    rng.shuffle(indices)
    val_data = [sentence_data[i] for i in indices[:n_val]]
    train_data = [sentence_data[i] for i in indices[n_val:]]

    motion_tokens = extract_motion_vocab(sentence_data)

    # -- Model --------------------------------------------------------------
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.add_special_tokens({
        "pad_token": PAD_TOKEN,
        "additional_special_tokens": [M_START, M_END, LEN_SHORT, LEN_MEDIUM, LEN_LONG],
    })
    tokenizer.add_tokens(motion_tokens, special_tokens=True)

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,
                                                  attn_implementation="sdpa")
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        model = model.to(torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.to(device)

    codebook = load_codebook(args.vqvae_ckpt)
    hidden_dim = model.config.hidden_size

    trainer = SemanticRegularisedTrainer(
        model=model,
        tokenizer=tokenizer,
        codebook=codebook,
        hidden_dim=hidden_dim,
        lambda_contrastive=args.lambda_contrastive,
        lambda_temporal=args.lambda_temporal,
        label_smooth_alpha=args.label_smooth_alpha,
        label_smooth_tau=args.label_smooth_tau,
        use_contrastive=not args.no_contrastive,
        use_label_smoothing=not args.no_label_smooth,
        use_temporal_consistency=not args.no_temporal,
    )

    train_ds = RegularisedSentenceDataset(train_data)
    val_ds = RegularisedSentenceDataset(val_data)
    collate_fn = create_collate_fn(tokenizer)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=2, pin_memory=True)

    # -- Optimiser ----------------------------------------------------------
    all_params = list(model.parameters())
    if hasattr(trainer, 'contrastive_loss'):
        all_params += list(trainer.contrastive_loss.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # -- Training -----------------------------------------------------------
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        loss_accum = {}
        n_batches = len(train_loader)
        t0 = time.time()

        for i, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss, breakdown = trainer.compute_loss(input_ids, attention_mask, labels)

            loss.backward()
            if i % max(1, args.batch_size // 16) == 0:
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += breakdown["total_loss"]
            for k, v in breakdown.items():
                loss_accum[k] = loss_accum.get(k, 0) + v

            if i == 1 or i % max(1, n_batches // 10) == 0:
                print(f"\r  [Epoch {epoch}] {i}/{n_batches} loss={breakdown['total_loss']:.4f} "
                      f"ce={breakdown['ce_loss']:.4f}", end="", flush=True)
        print()
        scheduler.step()

        avg_loss = total_loss / n_batches
        avg_breakdown = {k: v / n_batches for k, v in loss_accum.items()}

        # Validation
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                n_val_batches += 1
        val_loss /= max(1, n_val_batches)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch}/{args.epochs}: train={avg_loss:.4f} val={val_loss:.4f} "
              f"breakdown={avg_breakdown} lr={scheduler.get_last_lr()[0]:.2e} ({elapsed:.1f}s)")

        logger.log_epoch(epoch, avg_loss, val_loss, metrics=avg_breakdown,
                         lr=scheduler.get_last_lr()[0])

        if val_loss < best_val:
            best_val = val_loss
            model.save_pretrained(os.path.join(args.output_dir, "best_model"))
            tokenizer.save_pretrained(os.path.join(args.output_dir, "best_model"))
            print(f"    -> New best val loss: {val_loss:.4f}")

        if epoch % 10 == 0:
            model.save_pretrained(os.path.join(args.output_dir, f"epoch_{epoch}"))
            tokenizer.save_pretrained(os.path.join(args.output_dir, f"epoch_{epoch}"))

    logger.log_final({"best_val_loss": best_val})
    print(f"\nExperiment 3 complete. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
