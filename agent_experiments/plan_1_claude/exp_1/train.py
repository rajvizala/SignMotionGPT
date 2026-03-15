"""
Exp 1: Part-Aware VQ-VAE with Feature Augmentation.

Usage:
    python -m agent_experiments.plan_1_claude.exp_1.train \
        --data-dir /path/to/train_npz \
        --val-dir /path/to/val_npz \
        [--stats-path computed_stats.pt] \
        [--epochs 200] [--dry-run]
"""

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from agent_experiments.plan_1_claude.shared.data_utils import (
    NpzMotionDataset, collate_pad, train_val_split, HAND_SLICE, BODY_SLICE,
)
from agent_experiments.plan_1_claude.shared.eval_harness import (
    load_npz_sequences, evaluate_vqvae,
)
from agent_experiments.plan_1_claude.exp_1.model import (
    PartAwareVQVae, split_by_parts, PART_NAMES,
)
from agent_experiments.plan_1_claude.exp_1.config import Exp1Config

PART_LOSS_WEIGHTS = {"body": 10.0, "lhand": 12.0, "rhand": 12.0, "face_meta": 5.0}


def augment_features(x, cfg):
    if random.random() < cfg.temporal_warp_prob:
        B, T, C = x.shape
        factor = random.uniform(0.85, 1.15)
        new_T = max(8, int(T * factor))
        new_T = math.ceil(new_T / 8) * 8
        x = x.permute(0, 2, 1)
        x = F.interpolate(x, size=new_T, mode='linear', align_corners=False)
        x = x.permute(0, 2, 1)
    if random.random() < cfg.noise_prob:
        x = x + torch.randn_like(x) * cfg.noise_std
    if random.random() < cfg.hand_jitter_prob:
        noise = torch.zeros_like(x)
        noise[:, :, 73:163] = torch.randn(x.shape[0], x.shape[1], 90, device=x.device) * cfg.hand_jitter_std
        x = x + noise
    if random.random() < cfg.mirror_prob:
        xm = x.clone()
        lh = x[:, :, 73:118].clone()
        rh = x[:, :, 118:163].clone()
        xm[:, :, 73:118] = rh
        xm[:, :, 118:163] = lh
        x = xm
    return x


def compute_part_loss(x_recon, x_target, mask):
    pred = split_by_parts(x_recon)
    gt = split_by_parts(x_target)
    total = 0.0
    for name in PART_NAMES:
        diff = F.smooth_l1_loss(pred[name], gt[name], reduction='none')
        if mask is not None:
            diff = diff * mask.unsqueeze(-1)
            n = mask.sum() * diff.shape[-1]
        else:
            n = diff.numel()
        total = total + (diff.sum() / n.clamp(min=1)) * PART_LOSS_WEIGHTS[name]
    return total


def main():
    parser = argparse.ArgumentParser(description="Exp 1: Part-Aware VQ-VAE")
    cfg = Exp1Config()
    parser.add_argument("--data-dir", default=cfg.data_dir, required=not bool(cfg.data_dir))
    parser.add_argument("--val-dir", default=cfg.val_dir)
    parser.add_argument("--stats-path", default=cfg.stats_path)
    parser.add_argument("--output-dir", default=cfg.output_dir)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--patience", type=int, default=cfg.patience)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--vqvae-ckpt", default="")
    parser.add_argument("--seed", type=int, default=cfg.seed)
    args = parser.parse_args()
    cfg.dry_run = args.dry_run

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  Exp 1: Part-Aware VQ-VAE with Feature Augmentation")
    print("=" * 60)

    dataset = NpzMotionDataset(args.data_dir, args.stats_path)
    train_ds, val_ds = train_val_split(dataset, val_ratio=0.1, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_pad, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_pad, num_workers=1, pin_memory=True)

    model = PartAwareVQVae(
        code_num=cfg.code_num, code_dim=cfg.code_dim, hidden_dim=cfg.hidden_dim,
        down_t=cfg.down_t, stride_t=cfg.stride_t, depth=cfg.depth,
        dilation_growth_rate=cfg.dilation_growth_rate,
        fusion_heads=cfg.fusion_heads, fusion_layers=cfg.fusion_layers,
        commitment_weight=cfg.commitment_weight,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Model] Part-Aware VQ-VAE: {n_params:.2f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    swa_model = AveragedModel(model)
    swa_start = int(args.epochs * cfg.swa_start_pct)

    ckpt_path = os.path.join(args.output_dir, "latest_ckpt.pt")
    start_epoch = 1
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[Resume] From epoch {start_epoch}")

    best_worst_group = float("inf")
    best_avg = float("inf")
    no_improve_count = 0

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for i, (batch, lengths) in enumerate(train_loader):
            if batch is None:
                continue
            batch = batch.to(device)
            mask = torch.zeros(batch.shape[0], batch.shape[1], device=device)
            for bi, l in enumerate(lengths):
                mask[bi, :l] = 1.0

            batch = augment_features(batch, cfg)
            x_recon, commit_loss, info = model(batch)
            recon_loss = compute_part_loss(x_recon, batch, mask)
            loss = recon_loss + commit_loss

            if loss.item() > 3.0:
                print(f"  [WARNING] Loss {loss.item():.4f} > 3.0 at epoch {epoch} batch {i}. Consider stopping.")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

            if args.dry_run and i >= 1:
                break

        if epoch >= swa_start:
            swa_model.update_parameters(model)
        scheduler.step()

        avg_loss = total_loss / max(n_batches, 1)

        if epoch % cfg.eval_every == 0 or epoch == args.epochs or args.dry_run:
            model.eval()
            val_loss = 0.0
            val_hand = 0.0
            val_body = 0.0
            vn = 0
            with torch.no_grad():
                for batch, lengths in val_loader:
                    if batch is None:
                        continue
                    batch = batch.to(device)
                    mask = torch.zeros(batch.shape[0], batch.shape[1], device=device)
                    for bi, l in enumerate(lengths):
                        mask[bi, :l] = 1.0
                    x_recon, _, info = model(batch)
                    val_loss += F.mse_loss(x_recon, batch).item()
                    val_hand += F.mse_loss(x_recon[:, :, HAND_SLICE], batch[:, :, HAND_SLICE]).item()
                    val_body += F.mse_loss(x_recon[:, :, BODY_SLICE], batch[:, :, BODY_SLICE]).item()
                    vn += 1
                    if args.dry_run and vn >= 1:
                        break

            val_avg = val_loss / max(vn, 1)
            val_hand_avg = val_hand / max(vn, 1)
            val_body_avg = val_body / max(vn, 1)
            worst_group = max(val_hand_avg, val_body_avg, val_avg)

            perps = {k: f"{v.item():.1f}" for k, v in info["perplexities"].items()}
            elapsed = time.time() - t0
            print(f"  Epoch {epoch}/{args.epochs}: train={avg_loss:.4f} val={val_avg:.6f} "
                  f"hand={val_hand_avg:.6f} body={val_body_avg:.6f} worst={worst_group:.6f} "
                  f"perp={perps} lr={scheduler.get_last_lr()[0]:.2e} ({elapsed:.1f}s)")

            for pname, pval in info["perplexities"].items():
                if pval.item() < 10:
                    print(f"  [WARNING] Codebook collapse detected: {pname} perplexity = {pval.item():.1f}")

            if val_hand_avg > val_body_avg * 1.5 and epoch > 20:
                print(f"  [DIAGNOSTIC] Hand MSE >> Body MSE. Hand bottleneck persists.")

            if worst_group < best_worst_group:
                best_worst_group = worst_group
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                             "worst_group": worst_group}, os.path.join(args.output_dir, "best_worst_group.pt"))
                no_improve_count = 0
            else:
                no_improve_count += cfg.eval_every

            if val_avg < best_avg:
                best_avg = val_avg
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                             "avg_val": val_avg}, os.path.join(args.output_dir, "best_avg.pt"))

            if no_improve_count >= args.patience:
                print(f"  [EARLY STOP] No worst-group improvement for {args.patience} epochs.")
                break

        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                     "optimizer_state_dict": optimizer.state_dict()}, ckpt_path)

        if args.dry_run:
            print("[DRY RUN] Completed 1 epoch. Exiting.")
            break

    if epoch >= swa_start:
        torch.save({"model_state_dict": swa_model.module.state_dict()},
                    os.path.join(args.output_dir, "swa_model.pt"))
        print(f"[SWA] Saved averaged model.")

    print(f"\n[Exp1] Complete. Best worst-group={best_worst_group:.6f}, best avg={best_avg:.6f}")
    print(f"  Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
