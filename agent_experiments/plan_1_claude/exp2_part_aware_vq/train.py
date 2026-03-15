"""
Training script for Experiment 2: Part-Aware VQ-VAE.

Usage
-----
    python -m agent_experiments.exp2_part_aware_vq.train \
        --data-dir /path/to/npz_word_data \
        --output-dir ./agent_experiments/outputs/exp2 \
        [--sentence-dir /path/to/how2sign_pkl] \
        [--code-num 256] \
        [--code-dim 128] \
        [--epochs 300] \
        [--batch-size 32]
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from agent_experiments.shared.base_config import ExperimentConfig
from agent_experiments.shared.logging_utils import ExperimentLogger
from agent_experiments.exp2_part_aware_vq.model import PartAwareVQVae, split_by_parts
from agent_experiments.exp1_residual_vq.train import (
    NPZMotionDataset, PKLMotionDataset, collate_pad,
)

warnings.filterwarnings("ignore")

PART_LOSS_WEIGHTS = {
    "body": 10.0,
    "lhand": 12.0,
    "rhand": 12.0,
    "face_and_meta": 5.0,
}


def compute_part_recon_loss(x_recon, x_target, mask):
    """Per-part weighted reconstruction loss."""
    parts_pred = split_by_parts(x_recon)
    parts_gt = split_by_parts(x_target)
    total = 0.0
    for name in parts_pred:
        diff = F.smooth_l1_loss(parts_pred[name], parts_gt[name], reduction='none')
        if mask is not None:
            diff = diff * mask.unsqueeze(-1)
            n_valid = mask.sum() * diff.shape[-1]
        else:
            n_valid = diff.numel()
        part_loss = diff.sum() / n_valid.clamp(min=1)
        total = total + part_loss * PART_LOSS_WEIGHTS.get(name, 1.0)
    return total


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_commit = 0.0
    n_batches = len(dataloader)

    for i, (batch, mask) in enumerate(dataloader, 1):
        batch = batch.to(device)
        mask = mask.to(device)

        x_recon, commit_loss, info = model(batch)
        recon_loss = compute_part_recon_loss(x_recon, batch, mask)
        loss = recon_loss + commit_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_commit += commit_loss.item()

        if i == 1 or i % max(1, n_batches // 10) == 0 or i == n_batches:
            perps = {k: f"{v.item():.1f}" for k, v in info["perplexities"].items()}
            print(
                f"\r  [Epoch {epoch}] {i}/{n_batches} "
                f"loss={loss.item():.4f} recon={recon_loss.item():.4f} "
                f"commit={commit_loss.item():.4f} perp={perps}",
                end="", flush=True,
            )
    print()
    return {
        "loss": total_loss / n_batches,
        "recon_loss": total_recon / n_batches,
        "commit_loss": total_commit / n_batches,
    }


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total = 0.0
    n = 0
    for batch, mask in dataloader:
        batch = batch.to(device)
        mask = mask.to(device)
        x_recon, _, _ = model(batch)
        total += compute_part_recon_loss(x_recon, batch, mask).item()
        n += 1
    return {"val_recon_loss": total / max(n, 1)}


def main():
    parser = argparse.ArgumentParser(description="Exp2: Train Part-Aware VQ-VAE")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--sentence-dir", default="")
    parser.add_argument("--output-dir", default="./agent_experiments/outputs/exp2")
    parser.add_argument("--code-num", type=int, default=256)
    parser.add_argument("--code-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--fusion-heads", type=int, default=4)
    parser.add_argument("--fusion-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = ExperimentConfig(seed=args.seed)
    device = torch.device(cfg.device)

    logger = ExperimentLogger(args.output_dir, "exp2_part_aware_vq")
    logger.log_hyperparameters(vars(args))

    dataset = NPZMotionDataset(args.data_dir)
    n_val = max(1, int(len(dataset) * 0.05))
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [len(dataset) - n_val, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_pad, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_pad, num_workers=2, pin_memory=True)

    model = PartAwareVQVae(
        code_num=args.code_num, code_dim=args.code_dim,
        hidden_dim=args.hidden_dim, fusion_heads=args.fusion_heads,
        fusion_layers=args.fusion_layers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Model] Part-Aware VQ-VAE: {n_params:.2f}M params, "
          f"4 codebooks x {args.code_num} codes x {args.code_dim} dim")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        val_loss = val_metrics["val_recon_loss"]
        elapsed = time.time() - t0
        print(f"  Epoch {epoch}/{args.epochs}: train={train_metrics['loss']:.4f} "
              f"val={val_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e} ({elapsed:.1f}s)")

        logger.log_epoch(epoch, train_metrics["loss"], val_loss,
                         metrics=train_metrics, lr=scheduler.get_last_lr()[0])

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "val_loss": val_loss, "config": vars(args),
            }, os.path.join(args.output_dir, "best_part_aware_vqvae.pt"))
            print(f"    -> New best val loss saved.")

        if epoch % 50 == 0:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                         "config": vars(args)},
                        os.path.join(args.output_dir, f"part_vqvae_epoch_{epoch}.pt"))

    if args.sentence_dir and os.path.isdir(args.sentence_dir):
        print("\n" + "=" * 60)
        print("  Fine-tuning Part-Aware VQ-VAE on sentence data")
        print("=" * 60)
        sent_ds = PKLMotionDataset(args.sentence_dir)
        sent_loader = DataLoader(sent_ds, batch_size=max(1, args.batch_size // 2),
                                 shuffle=True, collate_fn=collate_pad, num_workers=4)
        ft_opt = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.5)
        for ep in range(1, 51):
            m = train_one_epoch(model, sent_loader, ft_opt, device, ep)
            print(f"  FT Epoch {ep}/50: loss={m['loss']:.4f}")

        torch.save({"model_state_dict": model.state_dict(), "config": vars(args)},
                    os.path.join(args.output_dir, "part_vqvae_sentence_finetuned.pt"))

    logger.log_final({"best_val_loss": best_val})
    print(f"\nExperiment 2 complete. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
