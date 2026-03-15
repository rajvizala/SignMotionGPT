"""
Exp 2: Residual VQ-VAE (RQ-VAE).

Usage:
    python -m agent_experiments.plan_1_claude.exp_2.train \
        --data-dir /path/to/train_npz \
        --val-dir /path/to/val_npz \
        [--stats-path computed_stats.pt] \
        [--num-rq-levels 4] [--epochs 200] [--dry-run]
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
from torch.optim.swa_utils import AveragedModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from agent_experiments.plan_1_claude.shared.data_utils import (
    NpzMotionDataset, collate_pad, train_val_split, build_loss_weights,
    HAND_SLICE, BODY_SLICE,
)
from agent_experiments.plan_1_claude.exp_2.model import RQVae
from agent_experiments.plan_1_claude.exp_2.config import Exp2Config


def main():
    parser = argparse.ArgumentParser(description="Exp 2: RQ-VAE")
    cfg = Exp2Config()
    parser.add_argument("--data-dir", default=cfg.data_dir, required=not bool(cfg.data_dir))
    parser.add_argument("--val-dir", default=cfg.val_dir)
    parser.add_argument("--stats-path", default=cfg.stats_path)
    parser.add_argument("--output-dir", default=cfg.output_dir)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--num-rq-levels", type=int, default=cfg.num_rq_levels)
    parser.add_argument("--patience", type=int, default=cfg.patience)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--vqvae-ckpt", default="")
    parser.add_argument("--seed", type=int, default=cfg.seed)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_weights = build_loss_weights().to(device)

    print("=" * 60)
    print("  Exp 2: Residual VQ-VAE (RQ-VAE)")
    print("=" * 60)

    dataset = NpzMotionDataset(args.data_dir, args.stats_path)
    train_ds, val_ds = train_val_split(dataset, val_ratio=0.1, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_pad, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_pad, num_workers=1, pin_memory=True)

    model = RQVae(
        nfeats=cfg.nfeats, num_rq_levels=args.num_rq_levels,
        code_num=cfg.code_num, code_dim=cfg.code_dim,
        output_emb_width=cfg.output_emb_width, width=cfg.width,
        down_t=cfg.down_t, stride_t=cfg.stride_t, depth=cfg.depth,
        dilation_growth_rate=cfg.dilation_growth_rate,
        commitment_weight=cfg.commitment_weight, level_dropout=cfg.level_dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Model] RQ-VAE: {n_params:.2f}M params, {args.num_rq_levels} levels")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    swa_model = AveragedModel(model)
    swa_start = int(args.epochs * cfg.swa_start_pct)
    loss_fn = torch.nn.SmoothL1Loss(reduction='none')

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
    no_improve = 0

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        nb = 0
        t0 = time.time()

        for i, (batch, lengths) in enumerate(train_loader):
            if batch is None:
                continue
            batch = batch.to(device)
            mask = torch.zeros(batch.shape[0], batch.shape[1], device=device)
            for bi, l in enumerate(lengths):
                mask[bi, :l] = 1.0
            x_recon, commit, info = model(batch)
            rec = loss_fn(x_recon, batch) * loss_weights.unsqueeze(0).unsqueeze(0)
            rec = (rec * mask.unsqueeze(-1)).sum() / mask.sum().clamp(min=1) / batch.shape[-1]
            loss = rec + commit
            if loss.item() > 3.0:
                print(f"  [WARNING] Loss {loss.item():.4f} > 3.0 at epoch {epoch}")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            nb += 1
            if args.dry_run and i >= 1:
                break

        if epoch >= swa_start:
            swa_model.update_parameters(model)
        scheduler.step()
        avg_loss = total_loss / max(nb, 1)

        if epoch % cfg.eval_every == 0 or epoch == args.epochs or args.dry_run:
            model.eval()
            vl, vh, vb, vn = 0.0, 0.0, 0.0, 0
            with torch.no_grad():
                for batch, lengths in val_loader:
                    if batch is None:
                        continue
                    batch = batch.to(device)
                    x_r, _, info = model(batch)
                    vl += F.mse_loss(x_r, batch).item()
                    vh += F.mse_loss(x_r[:, :, HAND_SLICE], batch[:, :, HAND_SLICE]).item()
                    vb += F.mse_loss(x_r[:, :, BODY_SLICE], batch[:, :, BODY_SLICE]).item()
                    vn += 1
                    if args.dry_run and vn >= 1:
                        break
            va, vha, vba = vl / max(vn, 1), vh / max(vn, 1), vb / max(vn, 1)
            wg = max(vha, vba, va)
            perps = [f"{p.item():.1f}" for p in info["perplexities"]]
            print(f"  Epoch {epoch}/{args.epochs}: train={avg_loss:.4f} val={va:.6f} "
                  f"hand={vha:.6f} body={vba:.6f} worst={wg:.6f} perp={perps} "
                  f"lr={scheduler.get_last_lr()[0]:.2e} ({time.time()-t0:.1f}s)")
            for li, p in enumerate(info["perplexities"]):
                if p.item() < 10:
                    print(f"  [WARNING] Level {li} perplexity = {p.item():.1f} -- collapse risk")
            if vha > vba * 1.5 and epoch > 20:
                print(f"  [DIAGNOSTIC] Hand MSE >> Body MSE.")
            if wg < best_worst_group:
                best_worst_group = wg
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                             "worst_group": wg}, os.path.join(args.output_dir, "best_worst_group.pt"))
                no_improve = 0
            else:
                no_improve += cfg.eval_every
            if va < best_avg:
                best_avg = va
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                             "avg_val": va}, os.path.join(args.output_dir, "best_avg.pt"))
            if no_improve >= args.patience:
                print(f"  [EARLY STOP] No improvement for {args.patience} epochs.")
                break

        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                     "optimizer_state_dict": optimizer.state_dict()}, ckpt_path)
        if args.dry_run:
            print("[DRY RUN] Done.")
            break

    if epoch >= swa_start:
        torch.save({"model_state_dict": swa_model.module.state_dict()},
                    os.path.join(args.output_dir, "swa_model.pt"))
    print(f"\n[Exp2] Complete. best_worst_group={best_worst_group:.6f} best_avg={best_avg:.6f}")


if __name__ == "__main__":
    main()
