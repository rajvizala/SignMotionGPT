from __future__ import annotations

import json
import os
import random
from collections import defaultdict

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import build_parser, from_args
from .model import RepairLossWeights, SingleStreamRepairModel, compute_repair_loss, hand_jitter, temporal_warp

from agent_experiments.plan_gpt_5_4.shared.data_utils import (
    MotionFolderDataset,
    build_word_lexicon_from_dir,
    collate_motion_items,
    compute_mse_breakdown,
    decode_codes_batch,
    load_stats,
    load_vqvae_checkpoint,
    normalize_motion,
    pad_motion_batch,
    unnormalize_motion,
)
from agent_experiments.plan_gpt_5_4.shared.eval_harness import BUCKET_KEYS, empty_result
from agent_experiments.plan_gpt_5_4.shared.training_utils import (
    EarlyStopper,
    diagnostic_warnings,
    load_checkpoint_if_exists,
    model_payload,
    save_checkpoint,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(model, dataloader, mean, std, device, max_batches=None):
    model.eval()
    result = empty_result()
    bucket_values = defaultdict(list)
    overall_sum = 0.0
    hand_sum = 0.0
    body_sum = 0.0
    total_samples = 0
    code_ids = set()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            motion = batch["motion"].to(device)
            lengths = batch["lengths"]
            normalized = normalize_motion(motion, mean, std)
            codes, _ = model.encode(normalized)
            recon = unnormalize_motion(decode_codes_batch(model, codes), mean, std).cpu()
            for row_idx, length in enumerate(lengths.tolist()):
                metrics = compute_mse_breakdown(recon[row_idx], batch["motion"][row_idx], length)
                overall_sum += metrics["overall_mse"]
                hand_sum += metrics["hand_mse"]
                body_sum += metrics["body_mse"]
                total_samples += 1
                for bucket_name in batch["buckets"][row_idx]:
                    bucket_values[bucket_name].append(metrics["overall_mse"])
            code_ids.update(int(code) for code in codes.reshape(-1).tolist())

    if total_samples == 0:
        return result

    result["overall_mse"] = overall_sum / total_samples
    result["hand_mse"] = hand_sum / total_samples
    result["body_mse"] = body_sum / total_samples
    result["codebook_coverage_pct"] = 100.0 * len(code_ids) / 512.0
    non_empty = []
    for bucket_name in BUCKET_KEYS:
        values = bucket_values.get(bucket_name, [])
        mse = sum(values) / len(values) if values else 0.0
        result["buckets"][bucket_name] = {"mse": mse, "n_samples": len(values)}
        if values:
            non_empty.append(mse)
    result["worst_group_score"] = max(non_empty) if non_empty else result["overall_mse"]
    return result


def update_average_model(avg_model, source_model, num_updates: int) -> int:
    source_state = source_model.state_dict()
    avg_state = avg_model.state_dict()
    count = num_updates + 1
    for key, value in source_state.items():
        avg_state[key].mul_(num_updates / count).add_(value.detach(), alpha=1.0 / count)
    avg_model.load_state_dict(avg_state, strict=False)
    return count


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = from_args(args)
    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = load_stats(config.stats_path)
    mean = mean.to(device)
    std = std.to(device)

    if not config.train_dir or not config.val_dir or not config.word_data_dir:
        raise ValueError("--train-dir, --val-dir, and --word-data-dir are required for exp_3.")

    word_lexicon = build_word_lexicon_from_dir(config.word_data_dir, max_samples=config.max_samples)
    train_dataset = MotionFolderDataset(config.train_dir, word_lexicon=word_lexicon, max_samples=config.max_samples)
    val_dataset = MotionFolderDataset(config.val_dir, word_lexicon=word_lexicon, max_samples=config.max_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_motion_items,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collate_motion_items,
        num_workers=config.num_workers,
    )

    model = SingleStreamRepairModel().to(device)
    base_vq = load_vqvae_checkpoint(config.vqvae_ckpt, device=device)
    model.vqvae.load_state_dict(base_vq.state_dict(), strict=False)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_weights = RepairLossWeights(
        vq_loss_weight=config.vq_loss_weight,
        velocity_weight=config.velocity_weight,
        hand_weight=config.hand_weight,
        codebook_balance_weight=config.codebook_balance_weight,
    )

    resume_path = os.path.join(config.output_dir, "resume_state.pt")
    resume_state = load_checkpoint_if_exists(resume_path)
    start_epoch = 1
    best_worst = None
    best_avg = None
    swa_model = SingleStreamRepairModel().to(device)
    swa_model.load_state_dict(model.state_dict(), strict=False)
    swa_updates = 0
    start_swa_epoch = max(1, int(config.epochs * 0.8))

    if resume_state is not None:
        model.load_state_dict(resume_state["model_state_dict"], strict=False)
        if resume_state.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        if resume_state.get("swa_state_dict") is not None:
            swa_model.load_state_dict(resume_state["swa_state_dict"], strict=False)
        swa_updates = int(resume_state.get("swa_updates", 0))
        start_epoch = int(resume_state.get("epoch", 0)) + 1
        best_worst = resume_state.get("best_worst")
        best_avg = resume_state.get("best_avg")
        print(f"Resuming from epoch {start_epoch}.")

    early_stopper = EarlyStopper(config.patience, mode="min")
    if best_worst is not None:
        early_stopper.best_value = best_worst
    prev_eval = None

    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        train_metrics = []
        max_batches = 2 if config.dry_run else None
        for batch_idx, batch in enumerate(train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            motion = batch["motion"]
            augmented_rows = []
            for row_idx, length in enumerate(batch["lengths"].tolist()):
                row = motion[row_idx, :length].clone()
                row = temporal_warp(row, config.temporal_warp_prob)
                row = hand_jitter(row, config.hand_jitter_std)
                augmented_rows.append(row)
            padded_motion, padded_lengths = pad_motion_batch(augmented_rows)
            normalized = normalize_motion(padded_motion.to(device), mean, std)
            loss, metrics = compute_repair_loss(model, normalized, padded_lengths, loss_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            train_metrics.append(metrics)

        if epoch >= start_swa_epoch:
            swa_updates = update_average_model(swa_model, model, swa_updates)

        eval_model = swa_model if epoch >= start_swa_epoch else model
        val_metrics = evaluate_model(
            eval_model,
            val_loader,
            mean=mean,
            std=std,
            device=device,
            max_batches=2 if config.dry_run else None,
        )
        if prev_eval is not None:
            val_metrics["prev_hand_mse"] = prev_eval["hand_mse"]
            val_metrics["prev_body_mse"] = prev_eval["body_mse"]
        diagnostic_warnings(val_metrics | {"vq_loss": train_metrics[-1]["vq_loss"] if train_metrics else None}, prefix=f"epoch={epoch}")
        prev_eval = val_metrics

        avg_train = {
            key: sum(metric[key] for metric in train_metrics) / max(1, len(train_metrics))
            for key in train_metrics[0].keys()
        } if train_metrics else {}
        log_payload = {
            "epoch": epoch,
            "train_total_loss": avg_train.get("total_loss", 0.0),
            "train_vq_loss": avg_train.get("vq_loss", 0.0),
            "val_overall_mse": val_metrics["overall_mse"],
            "val_worst_group": val_metrics["worst_group_score"],
            "val_codebook_coverage_pct": val_metrics["codebook_coverage_pct"],
            "hand_mse": val_metrics["hand_mse"],
            "body_mse": val_metrics["body_mse"],
        }
        if epoch % 5 == 0 or config.dry_run:
            print(json.dumps({"epoch": epoch, "buckets": val_metrics["buckets"]}, indent=2))

        checkpoint_extra = {
            "best_worst": best_worst,
            "best_avg": best_avg,
            "swa_state_dict": swa_model.state_dict(),
            "swa_updates": swa_updates,
        }
        if best_worst is None or val_metrics["worst_group_score"] < best_worst:
            best_worst = val_metrics["worst_group_score"]
            save_checkpoint(
                os.path.join(config.output_dir, "best_worst_group.pt"),
                model_payload(eval_model, optimizer, epoch, config, log_payload, extra=checkpoint_extra),
            )
        if best_avg is None or val_metrics["overall_mse"] < best_avg:
            best_avg = val_metrics["overall_mse"]
            save_checkpoint(
                os.path.join(config.output_dir, "best_avg.pt"),
                model_payload(eval_model, optimizer, epoch, config, log_payload, extra=checkpoint_extra),
            )
        save_checkpoint(
            resume_path,
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "swa_state_dict": swa_model.state_dict(),
                "swa_updates": swa_updates,
                "best_worst": best_worst,
                "best_avg": best_avg,
                "config": vars(args),
            },
        )

        print(json.dumps(log_payload, indent=2))
        should_stop = early_stopper.step(val_metrics["worst_group_score"])
        if should_stop:
            print("Early stopping triggered on worst-group reconstruction.")
            break
        if config.dry_run:
            print("Dry run finished after two batches.")
            break


if __name__ == "__main__":
    main()
