from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from agent_experiments.experiments.config_schema import ExperimentConfig
from agent_experiments.experiments.domains import summarize_group_losses, worst_group_score
from agent_experiments.experiments.losses import (
    EWCState,
    estimate_fisher_diagonal,
    ewc_penalty,
    label_smoothed_nll_loss,
    symmetric_kl,
)
from agent_experiments.experiments.optim import SAM, SWAAccumulator

from training.sentence_level.pipeline import (
    M_END,
    M_START,
    initialize_embeddings_from_vqvae,
    setup_model_and_tokenizer,
)


SENTENCE_TEMPLATES = [
    "Instruction: Generate sign language motion for the sentence: '{text}'",
    "Instruction: Translate this text to sign language: '{text}'",
    "Instruction: How would you sign '{text}'?",
    "Instruction: Create a sign language animation for: '{text}'",
]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "samples" in data and isinstance(data["samples"], list):
            data = data["samples"]
        elif "data" in data and isinstance(data["data"], list):
            data = data["data"]
        else:
            data = []
    return list(data)


def _dedupe_sentence_data(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for item in rows:
        if str(item.get("type", "")).lower() != "sentence":
            continue
        text = (item.get("text") or item.get("sentence") or "").strip().lower()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(item)
    return out


def _clean_word_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for item in rows:
        word = str(item.get("word", "")).strip().lower()
        pid = str(item.get("participant_id", "")).strip()
        motion = str(item.get("motion_tokens", "")).strip()
        if not word or not pid or not motion:
            continue
        out.append(item)
    return out


def _prepare_motion_vocab(sentence_rows: List[Dict[str, Any]], word_rows: List[Dict[str, Any]]) -> List[str]:
    ids = set()
    for item in sentence_rows + word_rows:
        for tok in str(item.get("motion_tokens", "")).split():
            if tok.strip().isdigit():
                ids.add(int(tok.strip()))
    return [f"<M{i}>" for i in sorted(ids)]


def _sentence_prompt(text: str) -> str:
    tmpl = random.choice(SENTENCE_TEMPLATES)
    return f"{tmpl.format(text=text)}\nMotion: "


def _word_prompt(word: str) -> str:
    return f"Instruction: Generate sign language motion for word '{word}'.\nMotion: "


def _wrap_motion_tokens(raw: str) -> str:
    motion = " ".join(f"<M{t}>" for t in str(raw).split() if t.strip())
    return f"{M_START} {motion} {M_END}"


class PromptMotionDataset(Dataset):
    def __init__(self, items: List[Dict[str, Any]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


def _build_sentence_items(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for r in rows:
        text = (r.get("text") or r.get("sentence") or "").strip()
        motion = str(r.get("motion_tokens", "")).strip()
        if not text or not motion:
            continue
        prompt = _sentence_prompt(text)
        target = _wrap_motion_tokens(motion)
        items.append(
            {
                "prompt": prompt,
                "full_text": prompt + target,
                "text": text,
                "motion_tokens": motion,
                "participant_id": r.get("participant_id", "unknown"),
                "source": "sentence",
            }
        )
    return items


def _build_word_items(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for r in rows:
        word = str(r.get("word", "")).strip().lower()
        motion = str(r.get("motion_tokens", "")).strip()
        if not word or not motion:
            continue
        prompt = _word_prompt(word)
        target = _wrap_motion_tokens(motion)
        items.append(
            {
                "prompt": prompt,
                "full_text": prompt + target,
                "text": word,
                "motion_tokens": motion,
                "participant_id": r.get("participant_id", "unknown"),
                "source": "word_replay",
            }
        )
    return items


def _split_train_val(items: List[Dict[str, Any]], ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    indices = list(range(len(items)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val = max(1, int(len(indices) * ratio))
    val_set = set(indices[:n_val])
    train = [items[i] for i in indices if i not in val_set]
    val = [items[i] for i in indices if i in val_set]
    return train, val


def _build_epoch_train_items(
    sentence_train: List[Dict[str, Any]],
    replay_items: List[Dict[str, Any]],
    replay_ratio: float,
    seed: int,
) -> List[Dict[str, Any]]:
    if replay_ratio <= 0 or not replay_items:
        return list(sentence_train)
    rng = random.Random(seed)
    n_sentence = len(sentence_train)
    n_replay = int((replay_ratio / max(1e-8, 1.0 - replay_ratio)) * n_sentence)
    sampled = [replay_items[rng.randrange(0, len(replay_items))] for _ in range(max(1, n_replay))]
    merged = list(sentence_train) + sampled
    rng.shuffle(merged)
    return merged


def _collate_builder(tokenizer, max_len: int):
    pad_id = tokenizer.pad_token_id

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        full_texts = [b["full_text"] for b in batch]
        prompts = [b["prompt"] for b in batch]

        tok_full = tokenizer(
            full_texts,
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        tok_prompt = tokenizer(
            prompts,
            truncation=True,
            max_length=max_len,
            padding=False,
            add_special_tokens=False,
        )

        labels = tok_full["input_ids"].clone()
        for i in range(labels.size(0)):
            prompt_len = min(len(tok_prompt["input_ids"][i]), labels.size(1))
            labels[i, :prompt_len] = -100
            if pad_id is not None:
                labels[i, tok_full["input_ids"][i] == pad_id] = -100

        return {
            "input_ids": tok_full["input_ids"],
            "attention_mask": tok_full["attention_mask"],
            "labels": labels,
            "meta": batch,
        }

    return collate


def _compute_batch_loss(
    model,
    batch: Dict[str, Any],
    device: torch.device,
    smoothing: float,
    use_rdrop: bool,
    rdrop_alpha: float,
    ewc_state: Optional[EWCState],
    ewc_lambda: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    valid_mask = labels.ne(-100)

    if use_rdrop:
        out1 = model(input_ids=input_ids, attention_mask=attention_mask)
        out2 = model(input_ids=input_ids, attention_mask=attention_mask)
        ce1 = label_smoothed_nll_loss(out1.logits, labels, smoothing=smoothing)
        ce2 = label_smoothed_nll_loss(out2.logits, labels, smoothing=smoothing)
        ce = 0.5 * (ce1 + ce2)
        kl = symmetric_kl(out1.logits, out2.logits, mask=valid_mask)
        loss = ce + rdrop_alpha * kl
        logs = {"ce": float(ce.detach().cpu()), "kl": float(kl.detach().cpu())}
    else:
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        ce = label_smoothed_nll_loss(out.logits, labels, smoothing=smoothing)
        loss = ce
        logs = {"ce": float(ce.detach().cpu()), "kl": 0.0}

    if ewc_state is not None and ewc_lambda > 0:
        penalty = ewc_penalty(model, ewc_state)
        loss = loss + ewc_lambda * penalty
        logs["ewc"] = float(penalty.detach().cpu())
    else:
        logs["ewc"] = 0.0

    return loss, logs


@torch.no_grad()
def _evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    smoothing: float,
    group_keys: List[str],
) -> Dict[str, Any]:
    model.eval()
    all_losses: List[float] = []
    all_meta: List[Dict[str, Any]] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits

        vocab = logits.size(-1)
        token_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, vocab),
            labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(labels.size(0), labels.size(1))

        valid = labels.ne(-100).float()
        per_sample = (token_loss * valid).sum(-1) / valid.sum(-1).clamp_min(1.0)

        all_losses.extend([float(x) for x in per_sample.detach().cpu().tolist()])
        all_meta.extend(batch["meta"])

    avg = sum(all_losses) / max(1, len(all_losses))
    group_stats = summarize_group_losses(all_meta, all_losses, group_keys)
    worst_group = worst_group_score(group_stats)

    model.train()
    return {
        "val_loss": avg,
        "worst_group_loss": worst_group,
        "group_stats": group_stats,
    }


def _save_checkpoint(output_dir: str, model, tokenizer, step: int, tag: str) -> str:
    ckpt_dir = os.path.join(output_dir, "checkpoints", f"{tag}_step_{step}")
    _ensure_dir(ckpt_dir)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    return ckpt_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="ASL OOD generalization experiments on fixed VQ-VAE + LLM pipeline.")
    parser.add_argument("--config", required=True, type=str, help="Path to TOML config.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config key, format section.key=value. Can be passed multiple times.",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig.from_toml(args.config, overrides=args.set)
    _ensure_dir(cfg.data.output_dir)
    _set_seed(cfg.data.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device={device}")

    sentence_rows = _dedupe_sentence_data(_load_json(cfg.data.sentence_json))
    word_rows = _clean_word_rows(_load_json(cfg.data.word_json)) if cfg.data.word_json else []
    motion_tokens = _prepare_motion_vocab(sentence_rows, word_rows)

    print(f"[Data] sentence_rows={len(sentence_rows)}")
    print(f"[Data] word_rows={len(word_rows)}")
    print(f"[Data] motion_vocab={len(motion_tokens)}")

    sentence_items = _build_sentence_items(sentence_rows)
    word_items = _build_word_items(word_rows)
    sentence_train, sentence_val = _split_train_val(sentence_items, cfg.data.val_ratio, cfg.data.split_seed)

    model, tokenizer = setup_model_and_tokenizer(cfg.model.model_name, motion_tokens)
    if cfg.model.vqvae_ckpt:
        initialize_embeddings_from_vqvae(model, tokenizer, cfg.model.vqvae_ckpt, device)
    model.to(device)

    if cfg.model.grad_checkpointing:
        model.gradient_checkpointing_enable()

    collate_fn = _collate_builder(tokenizer, max_len=cfg.data.max_seq_len)
    val_loader = DataLoader(
        PromptMotionDataset(sentence_val),
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    total_train_steps = math.ceil(len(sentence_train) / cfg.training.batch_size) * cfg.training.epochs

    if cfg.regularization.use_sam:
        optimizer = SAM(
            model.parameters(),
            AdamW,
            rho=cfg.regularization.sam_rho,
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )
    else:
        optimizer = AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    warmup_steps = int(total_train_steps * cfg.training.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer.base_optimizer if isinstance(optimizer, SAM) else optimizer,
        lr_lambda=lambda step: min(1.0, (step + 1) / max(1, warmup_steps)),
    )

    ewc_state: Optional[EWCState] = None
    if cfg.regularization.use_ewc and word_items:
        ewc_dataset = PromptMotionDataset(word_items[: cfg.regularization.ewc_samples])
        ewc_loader = DataLoader(
            ewc_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.training.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        ewc_state = estimate_fisher_diagonal(
            model,
            ewc_loader,
            device=device,
            max_batches=max(1, cfg.regularization.ewc_samples // max(1, cfg.training.batch_size)),
        )
        print("[EWC] fisher estimated")

    swa_start = int(total_train_steps * cfg.regularization.swa_start_ratio) if cfg.regularization.use_swa else total_train_steps + 1
    swa_acc = SWAAccumulator(start_step=swa_start)

    best_metric = float("inf")
    best_step = 0
    no_improve = 0
    global_step = 0
    history: List[Dict[str, Any]] = []

    model.train()
    for epoch in range(cfg.training.epochs):
        epoch_items = _build_epoch_train_items(
            sentence_train=sentence_train,
            replay_items=word_items,
            replay_ratio=cfg.training.replay_ratio,
            seed=cfg.data.seed + epoch,
        )
        train_loader = DataLoader(
            PromptMotionDataset(epoch_items),
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.training.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        t0 = time.time()
        running = {"loss": 0.0, "ce": 0.0, "kl": 0.0, "ewc": 0.0, "n": 0}
        for batch in train_loader:
            global_step += 1

            if isinstance(optimizer, SAM):
                def closure():
                    optimizer.zero_grad(set_to_none=True)
                    loss, _ = _compute_batch_loss(
                        model=model,
                        batch=batch,
                        device=device,
                        smoothing=cfg.regularization.label_smoothing,
                        use_rdrop=cfg.regularization.use_rdrop,
                        rdrop_alpha=cfg.regularization.rdrop_alpha,
                        ewc_state=ewc_state,
                        ewc_lambda=cfg.regularization.ewc_lambda,
                    )
                    loss.backward()
                    return loss

                loss = optimizer.step(closure)
                with torch.no_grad():
                    _, logs = _compute_batch_loss(
                        model=model,
                        batch=batch,
                        device=device,
                        smoothing=cfg.regularization.label_smoothing,
                        use_rdrop=False,
                        rdrop_alpha=0.0,
                        ewc_state=None,
                        ewc_lambda=0.0,
                    )
            else:
                optimizer.zero_grad(set_to_none=True)
                loss, logs = _compute_batch_loss(
                    model=model,
                    batch=batch,
                    device=device,
                    smoothing=cfg.regularization.label_smoothing,
                    use_rdrop=cfg.regularization.use_rdrop,
                    rdrop_alpha=cfg.regularization.rdrop_alpha,
                    ewc_state=ewc_state,
                    ewc_lambda=cfg.regularization.ewc_lambda,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                optimizer.step()

            scheduler.step()
            swa_acc.update(model, global_step)

            running["loss"] += float(loss.detach().cpu())
            running["ce"] += float(logs["ce"])
            running["kl"] += float(logs["kl"])
            running["ewc"] += float(logs["ewc"])
            running["n"] += 1

            if global_step % cfg.training.eval_every_steps == 0:
                eval_out = _evaluate(
                    model=model,
                    loader=val_loader,
                    device=device,
                    smoothing=cfg.regularization.label_smoothing,
                    group_keys=cfg.validation.group_by,
                )
                metric = eval_out["worst_group_loss"]
                rec = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "train_loss": running["loss"] / max(1, running["n"]),
                    "train_ce": running["ce"] / max(1, running["n"]),
                    "train_kl": running["kl"] / max(1, running["n"]),
                    "train_ewc": running["ewc"] / max(1, running["n"]),
                    "val_loss": eval_out["val_loss"],
                    "worst_group_loss": metric,
                    "group_stats": eval_out["group_stats"],
                    "lr": float((optimizer.base_optimizer if isinstance(optimizer, SAM) else optimizer).param_groups[0]["lr"]),
                }
                history.append(rec)
                print(
                    f"[Eval] step={global_step} train={rec['train_loss']:.4f} "
                    f"val={rec['val_loss']:.4f} worst_group={metric:.4f}"
                )

                if metric < best_metric:
                    best_metric = metric
                    best_step = global_step
                    no_improve = 0
                    _save_checkpoint(cfg.data.output_dir, model, tokenizer, global_step, tag="best")
                else:
                    no_improve += 1

                if no_improve >= cfg.validation.early_stop_patience:
                    print("[Stop] early stopping triggered")
                    break

            if cfg.training.max_steps > 0 and global_step >= cfg.training.max_steps:
                print("[Stop] max_steps reached")
                break

            if global_step % cfg.training.save_every_steps == 0:
                _save_checkpoint(cfg.data.output_dir, model, tokenizer, global_step, tag="latest")

        epoch_sec = time.time() - t0
        print(
            f"[Epoch {epoch+1}/{cfg.training.epochs}] "
            f"loss={running['loss']/max(1, running['n']):.4f} "
            f"ce={running['ce']/max(1, running['n']):.4f} "
            f"time={epoch_sec:.1f}s"
        )

        if no_improve >= cfg.validation.early_stop_patience:
            break
        if cfg.training.max_steps > 0 and global_step >= cfg.training.max_steps:
            break

    if cfg.regularization.use_swa:
        print("[SWA] applying averaged weights")
        swa_acc.apply_to(model)
        _save_checkpoint(cfg.data.output_dir, model, tokenizer, global_step, tag="swa")

    final_dir = os.path.join(cfg.data.output_dir, "final_model")
    _ensure_dir(final_dir)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    with open(os.path.join(cfg.data.output_dir, "config_resolved.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    with open(os.path.join(cfg.data.output_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(cfg.data.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_step": best_step,
                "best_worst_group_loss": best_metric,
                "total_steps": global_step,
                "num_eval_points": len(history),
                "notes": cfg.notes,
            },
            f,
            indent=2,
        )

    print("[Done] training complete")
    print(f"[Done] output_dir={cfg.data.output_dir}")


if __name__ == "__main__":
    main()

