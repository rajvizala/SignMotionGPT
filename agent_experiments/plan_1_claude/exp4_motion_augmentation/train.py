"""
Training script for Experiment 4: Motion-Level Data Augmentation.

Usage
-----
    python -m agent_experiments.exp4_motion_augmentation.train \
        --dataset-path /path/to/sentence_dataset.json \
        --vqvae-ckpt /path/to/vqvae_checkpoint.pt \
        --output-dir ./agent_experiments/outputs/exp4 \
        [--dropout-prob 0.15] \
        [--substitute-prob 0.1] \
        [--aug-per-sample 2] \
        [--epochs 40]

This builds on the existing sentence-level pipeline, wrapping it with the
token augmentation from model.py.
"""

import argparse
import json
import os
import random
import sys
import time
import warnings
from typing import Dict, List, Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from agent_experiments.shared.logging_utils import ExperimentLogger
from agent_experiments.shared.data_utils import load_json_data, split_by_type, extract_motion_vocab
from agent_experiments.exp4_motion_augmentation.model import (
    TokenAugmentor, AugmentedMotionDataset,
)

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


def prepare_base_items(data: List[Dict[str, Any]]) -> List[Dict]:
    """Prepare base items with prompt and motion tokens."""
    items = []
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

        for tmpl in TEMPLATES:
            instruction = tmpl.format(text=text)
            prompt = f"{instruction} (Length: {len_tok})\nMotion: "
            items.append({
                "prompt": prompt,
                "full_text": prompt + target,
                "text": text,
                "motion_tokens": tokens_str,
                "motion_length": motion_len,
            })
    return items


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


def load_codebook_distances(ckpt_path: str):
    """Load codebook and compute pairwise distances."""
    if not os.path.exists(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    codebook = None
    for key in ["vqvae.quantizer.codebook", "quantizer.codebook",
                "vqvae.quantizer.embedding.weight"]:
        if key in sd:
            codebook = sd[key]
            break
    if codebook is None:
        for key in sd:
            if "codebook" in key.lower():
                codebook = sd[key]
                break
    if codebook is None:
        return None
    return torch.cdist(codebook.float(), codebook.float(), p=2)


def main():
    parser = argparse.ArgumentParser(description="Exp4: Augmented LLM Training")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--vqvae-ckpt", required=True)
    parser.add_argument("--output-dir", default="./agent_experiments/outputs/exp4")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--dropout-prob", type=float, default=0.15)
    parser.add_argument("--substitute-prob", type=float, default=0.1)
    parser.add_argument("--crop-prob", type=float, default=0.2)
    parser.add_argument("--aug-per-sample", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = ExperimentLogger(args.output_dir, "exp4_augmentation")
    logger.log_hyperparameters(vars(args))

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

    # Setup tokenizer and model
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

    # Setup augmentor
    cb_distances = load_codebook_distances(args.vqvae_ckpt)
    augmentor = TokenAugmentor(
        codebook_size=512,
        dropout_prob=args.dropout_prob,
        substitute_prob=args.substitute_prob,
        crop_prob=args.crop_prob,
        codebook_distances=cb_distances,
    )

    base_train_items = prepare_base_items(train_data)
    base_val_items = prepare_base_items(val_data)

    train_ds = AugmentedMotionDataset(
        base_train_items, augmentor,
        augmentations_per_sample=args.aug_per_sample,
    )
    collate_fn = create_collate_fn(tokenizer)

    print(f"[Data] Train: {len(base_train_items)} base + {len(train_ds) - len(base_train_items)} "
          f"augmented = {len(train_ds)} total")
    print(f"[Data] Val: {len(base_val_items)} (no augmentation)")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(base_val_items, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = len(train_loader)
        t0 = time.time()

        for i, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if i == 1 or i % max(1, n_batches // 10) == 0:
                print(f"\r  [Epoch {epoch}] {i}/{n_batches} loss={loss.item():.4f}",
                      end="", flush=True)
        print()
        scheduler.step()

        avg_loss = total_loss / n_batches

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                n_val += 1
        val_loss /= max(1, n_val)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch}/{args.epochs}: train={avg_loss:.4f} val={val_loss:.4f} "
              f"lr={scheduler.get_last_lr()[0]:.2e} ({elapsed:.1f}s)")
        logger.log_epoch(epoch, avg_loss, val_loss, lr=scheduler.get_last_lr()[0])

        if val_loss < best_val:
            best_val = val_loss
            model.save_pretrained(os.path.join(args.output_dir, "best_model"))
            tokenizer.save_pretrained(os.path.join(args.output_dir, "best_model"))
            print(f"    -> New best val loss: {val_loss:.4f}")

    logger.log_final({"best_val_loss": best_val})
    print(f"\nExperiment 4 complete. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
