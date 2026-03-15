"""
Exp 3: LLM Training with Masked Denoising + RAG + SWA.

Usage:
    python -m agent_experiments.plan_1_claude.exp_3.train \
        --sentence-data /path/to/sentence_dataset.json \
        --word-data /path/to/word_dataset.json \
        --vqvae-ckpt /path/to/best_vqvae.pt \
        [--epochs 40] [--dry-run]
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from agent_experiments.plan_1_claude.exp_3.config import Exp3Config
from agent_experiments.plan_1_claude.exp_3.model import (
    WordSignDictionary, retrieve_context, apply_motion_mask,
)

M_START = "<M_START>"
M_END = "<M_END>"
PAD_TOKEN = "<PAD>"
M_MASK = "<M_MASK>"
LEN_SHORT = "<|LEN_SHORT|>"
LEN_MEDIUM = "<|LEN_MEDIUM|>"
LEN_LONG = "<|LEN_LONG|>"

TEMPLATES = [
    "Instruction: Generate sign language motion for the sentence: '{text}'",
    "Instruction: Translate this text to sign language: '{text}'",
    "Instruction: How would you sign '{text}'?",
    "Instruction: Create a sign language animation for: '{text}'",
]


class SentenceDataset(Dataset):
    def __init__(self, data, dictionary, rag_prob=0.8, max_words=5, max_tokens=15):
        self.items = []
        for item in data:
            text = (item.get("text") or item.get("sentence", "")).strip()
            tokens_str = item.get("motion_tokens", "")
            if not text or not tokens_str.strip():
                continue
            ml = len(tokens_str.split())
            lt = LEN_SHORT if ml < 30 else (LEN_MEDIUM if ml < 80 else LEN_LONG)
            wrapped = " ".join(f"<M{t}>" for t in tokens_str.split())
            target = f"{M_START} {wrapped} {M_END}"
            ctx = retrieve_context(text, dictionary, max_words, max_tokens) if dictionary else ""
            for tmpl in TEMPLATES:
                inst = tmpl.format(text=text)
                self.items.append({"instruction": inst, "len_tok": lt, "context": ctx,
                                   "target": target, "text": text, "rag_prob": rag_prob})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        if item["context"] and random.random() < item["rag_prob"]:
            prompt = f"{item['context']}\n{item['instruction']} (Length: {item['len_tok']})\nMotion: "
        else:
            prompt = f"{item['instruction']} (Length: {item['len_tok']})\nMotion: "
        return {"prompt": prompt, "full_text": prompt + item["target"], "text": item["text"]}


def create_collate_fn(tokenizer, max_length=384):
    pad_id = tokenizer.pad_token_id
    m_start_id = tokenizer.convert_tokens_to_ids(M_START)

    def fn(batch):
        texts = [x["full_text"] for x in batch]
        tok = tokenizer(texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        labels = tok["input_ids"].clone()
        for i in range(len(texts)):
            ids = tok["input_ids"][i]
            pos = (ids == m_start_id).nonzero(as_tuple=True)[0]
            pl = pos[0].item() if len(pos) > 0 else len(ids)
            labels[i, :pl] = -100
            if pad_id is not None:
                labels[i, ids == pad_id] = -100
        return {"input_ids": tok["input_ids"], "attention_mask": tok["attention_mask"], "labels": labels}
    return fn


def main():
    parser = argparse.ArgumentParser(description="Exp 3: LLM Denoising+RAG+SWA")
    cfg = Exp3Config()
    parser.add_argument("--sentence-data", default=cfg.sentence_data_path, required=not bool(cfg.sentence_data_path))
    parser.add_argument("--word-data", default=cfg.word_data_path)
    parser.add_argument("--vqvae-ckpt", default=cfg.vqvae_ckpt)
    parser.add_argument("--output-dir", default=cfg.output_dir)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--patience", type=int, default=cfg.patience)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=cfg.seed)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  Exp 3: LLM with Masked Denoising + RAG + SWA")
    print("=" * 60)

    dictionary = None
    if args.word_data and os.path.exists(args.word_data):
        with open(args.word_data) as f:
            wd = json.load(f)
        if isinstance(wd, dict):
            wd = wd.get("samples", wd.get("data", []))
        dictionary = WordSignDictionary()
        dictionary.build_from_data(wd)

    with open(args.sentence_data) as f:
        sd = json.load(f)
    if isinstance(sd, dict):
        sd = sd.get("samples", sd.get("data", []))
    sentences = [x for x in sd if x.get("type", "").lower() == "sentence"]
    if not sentences:
        sentences = sd

    rng = random.Random(args.seed)
    idx = list(range(len(sentences)))
    rng.shuffle(idx)
    nv = max(1, int(len(sentences) * 0.1))
    val_data = [sentences[i] for i in idx[:nv]]
    train_data = [sentences[i] for i in idx[nv:]]

    all_motion_tokens = set()
    for item in sentences:
        for t in item.get("motion_tokens", "").split():
            if t.strip():
                all_motion_tokens.add(f"<M{t.strip()}>")
    motion_tokens = sorted(all_motion_tokens)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    extra_special = [M_START, M_END, LEN_SHORT, LEN_MEDIUM, LEN_LONG, M_MASK,
                     "[SIGN_CONTEXT]", "[/SIGN_CONTEXT]"]
    tokenizer.add_special_tokens({"pad_token": PAD_TOKEN, "additional_special_tokens": extra_special})
    tokenizer.add_tokens(motion_tokens, special_tokens=True)

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, trust_remote_code=True, attn_implementation="sdpa")
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        model = model.to(torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.to(device)

    train_ds = SentenceDataset(train_data, dictionary, cfg.rag_context_prob, cfg.rag_max_words, cfg.rag_max_tokens_per_word)
    val_ds = SentenceDataset(val_data, dictionary, 1.0, cfg.rag_max_words, cfg.rag_max_tokens_per_word)
    collate_fn = create_collate_fn(tokenizer, cfg.max_seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    swa_model = AveragedModel(model)
    swa_start = int(args.epochs * cfg.swa_start_pct)

    best_worst_group = float("inf")
    best_avg = float("inf")
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_denoise, nb = 0.0, 0.0, 0
        t0 = time.time()

        for i, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(input_ids=ids, attention_mask=attn, labels=labels)
            ce_loss = out.loss

            denoise_loss = torch.tensor(0.0, device=device)
            if cfg.denoise_mask_rate > 0:
                masked_ids, is_masked = apply_motion_mask(ids, tokenizer, cfg.denoise_mask_rate)
                if is_masked.any():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        den_out = model(input_ids=masked_ids, attention_mask=attn)
                    den_logits = den_out.logits[:, :-1].contiguous()
                    den_targets = ids[:, 1:].contiguous()
                    den_mask = is_masked[:, 1:]
                    if den_mask.any():
                        dl = F.cross_entropy(den_logits[den_mask], den_targets[den_mask])
                        denoise_loss = dl * cfg.denoise_weight

            loss = ce_loss + denoise_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += ce_loss.item()
            total_denoise += denoise_loss.item()
            nb += 1
            if args.dry_run and i >= 1:
                break

        if epoch >= swa_start:
            swa_model.update_parameters(model)
        scheduler.step()

        avg_loss = total_loss / max(nb, 1)
        avg_den = total_denoise / max(nb, 1)

        if epoch % cfg.eval_every == 0 or epoch == args.epochs or args.dry_run:
            model.eval()
            vl, vn = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    ids = batch["input_ids"].to(device)
                    attn = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    out = model(input_ids=ids, attention_mask=attn, labels=labels)
                    vl += out.loss.item()
                    vn += 1
                    if args.dry_run and vn >= 1:
                        break
            va = vl / max(vn, 1)
            wg = va
            elapsed = time.time() - t0
            print(f"  Epoch {epoch}/{args.epochs}: train_ce={avg_loss:.4f} denoise={avg_den:.4f} "
                  f"val={va:.4f} worst={wg:.4f} lr={scheduler.get_last_lr()[0]:.2e} ({elapsed:.1f}s)")

            if avg_loss < 0.5 and va > 3.0:
                print("  [WARNING] Severe overfitting detected. Consider stopping.")

            if wg < best_worst_group:
                best_worst_group = wg
                model.save_pretrained(os.path.join(args.output_dir, "best_worst_group"))
                tokenizer.save_pretrained(os.path.join(args.output_dir, "best_worst_group"))
                no_improve = 0
            else:
                no_improve += cfg.eval_every

            if va < best_avg:
                best_avg = va
                model.save_pretrained(os.path.join(args.output_dir, "best_avg"))
                tokenizer.save_pretrained(os.path.join(args.output_dir, "best_avg"))

            if no_improve >= args.patience:
                print(f"  [EARLY STOP] No improvement for {args.patience} epochs.")
                break

        if args.dry_run:
            print("[DRY RUN] Done.")
            break

    if epoch >= swa_start:
        swa_dir = os.path.join(args.output_dir, "swa_model")
        os.makedirs(swa_dir, exist_ok=True)
        swa_model.module.save_pretrained(swa_dir)
        tokenizer.save_pretrained(swa_dir)

    print(f"\n[Exp3] Complete. best_worst_group={best_worst_group:.4f} best_avg={best_avg:.4f}")


if __name__ == "__main__":
    main()
