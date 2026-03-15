"""
Training script for Experiment 1: Residual VQ-VAE (RQ-VAE).

Usage
-----
    python -m agent_experiments.exp1_residual_vq.train \
        --data-dir /path/to/npz_word_data \
        --output-dir ./agent_experiments/outputs/exp1 \
        [--sentence-dir /path/to/how2sign_pkl] \
        [--num-rq-levels 4] \
        [--epochs 300] \
        [--batch-size 32] \
        [--lr 2e-4]

The script trains an RQ-VAE on word-level ASL Citizen data first, then
optionally fine-tunes on sentence-level How2Sign data.  It logs per-epoch
metrics (reconstruction loss, perplexity per RQ level, codebook utilisation)
and saves checkpoints.
"""

import argparse
import glob
import math
import os
import pickle
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from agent_experiments.shared.base_config import ExperimentConfig
from agent_experiments.shared.logging_utils import ExperimentLogger
from agent_experiments.exp1_residual_vq.model import RQVae

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NPZMotionDataset(Dataset):
    """Load word-level motion data from NPZ files (same format as train_vqvae.py)."""

    def __init__(self, data_dir: str, min_frames: int = 32, max_frames: int = 512):
        self.samples = []
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        for npz_path in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
            data = np.load(npz_path, allow_pickle=True)
            features = data.get("features", data.get("motion", None))
            if features is None:
                continue
            if len(features.shape) == 1:
                continue
            T = features.shape[0]
            if T < min_frames or T > max_frames:
                continue
            self.samples.append(features.astype(np.float32))
        print(f"[NPZMotionDataset] Loaded {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.from_numpy(self.samples[idx])


class PKLMotionDataset(Dataset):
    """Load sentence-level SMPL-X data from PKL folders (How2Sign format)."""

    PARAM_KEYS = [
        "smplx_shape", "smplx_body_pose", "smplx_lhand_pose",
        "smplx_rhand_pose", "smplx_jaw_pose", "smplx_expr",
        "smplx_root_pose", "cam_trans",
    ]
    PARAM_DIMS = [10, 63, 45, 45, 3, 10, 3, 3]

    def __init__(
        self,
        data_dir: str,
        min_frames: int = 32,
        max_frames: int = 512,
        mean: np.ndarray = None,
        std: np.ndarray = None,
    ):
        self.samples = []
        self.mean = mean
        self.std = std
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Sentence data dir not found: {data_dir}")

        sentence_dirs = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])
        for sdir in sentence_dirs:
            full_dir = os.path.join(data_dir, sdir)
            pkl_files = sorted(glob.glob(os.path.join(full_dir, "*_3D.pkl")))
            if len(pkl_files) < min_frames:
                continue
            frames = []
            for pf in pkl_files[:max_frames]:
                with open(pf, "rb") as f:
                    frame = pickle.load(f)
                vec = []
                for key in self.PARAM_KEYS:
                    v = frame.get(key, np.zeros(1))
                    if hasattr(v, 'numpy'):
                        v = v.numpy()
                    vec.append(np.array(v).flatten())
                frames.append(np.concatenate(vec))
            features = np.stack(frames).astype(np.float32)
            self.samples.append(features)

        print(f"[PKLMotionDataset] Loaded {len(self.samples)} sentence clips from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat = self.samples[idx]
        if self.mean is not None and self.std is not None:
            feat = (feat - self.mean) / (self.std + 1e-8)
        return torch.from_numpy(feat)


def collate_pad(batch):
    """Pad variable-length sequences to the longest in the batch."""
    max_t = max(x.shape[0] for x in batch)
    # Round up to nearest multiple of 8 for VQ-VAE temporal downsampling
    factor = 8
    max_t = ((max_t + factor - 1) // factor) * factor
    C = batch[0].shape[1]
    padded = torch.zeros(len(batch), max_t, C)
    masks = torch.zeros(len(batch), max_t)
    for i, x in enumerate(batch):
        T = x.shape[0]
        padded[i, :T] = x
        masks[i, :T] = 1.0
    return padded, masks


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def compute_weighted_recon_loss(x_recon, x_target, mask, loss_weights):
    """Weighted SmoothL1 loss per parameter group, masked for padding."""
    diff = F.smooth_l1_loss(x_recon, x_target, reduction='none')  # (B, T, C)
    diff = diff * loss_weights.unsqueeze(0).unsqueeze(0).to(diff.device)
    if mask is not None:
        diff = diff * mask.unsqueeze(-1)
        n_valid = mask.sum() * diff.shape[-1]
    else:
        n_valid = diff.numel()
    return diff.sum() / n_valid.clamp(min=1)


def train_one_epoch(model, dataloader, optimizer, device, loss_weights, epoch):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_commit = 0.0
    total_perp = [0.0] * model.num_rq_levels
    n_batches = len(dataloader)

    for i, (batch, mask) in enumerate(dataloader, 1):
        batch = batch.to(device)
        mask = mask.to(device)

        x_recon, commit_loss, info = model(batch)
        recon_loss = compute_weighted_recon_loss(x_recon, batch, mask, loss_weights)
        loss = recon_loss + commit_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_commit += commit_loss.item()
        for lv, perp in enumerate(info["perplexities"]):
            total_perp[lv] += perp.item()

        if i == 1 or i % max(1, n_batches // 10) == 0 or i == n_batches:
            print(
                f"\r  [Epoch {epoch}] {i}/{n_batches} "
                f"loss={loss.item():.4f} recon={recon_loss.item():.4f} "
                f"commit={commit_loss.item():.4f} "
                f"perp_L0={info['perplexities'][0].item():.1f}",
                end="", flush=True,
            )
    print()

    return {
        "loss": total_loss / n_batches,
        "recon_loss": total_recon / n_batches,
        "commit_loss": total_commit / n_batches,
        "perplexities": [p / n_batches for p in total_perp],
    }


@torch.no_grad()
def evaluate(model, dataloader, device, loss_weights):
    model.eval()
    total_recon = 0.0
    n = 0
    for batch, mask in dataloader:
        batch = batch.to(device)
        mask = mask.to(device)
        x_recon, _, info = model(batch)
        recon = compute_weighted_recon_loss(x_recon, batch, mask, loss_weights)
        total_recon += recon.item()
        n += 1
    return {"val_recon_loss": total_recon / max(n, 1)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Exp1: Train RQ-VAE")
    parser.add_argument("--data-dir", required=True, help="Path to NPZ word-level data")
    parser.add_argument("--sentence-dir", default="", help="Path to How2Sign PKL folders (optional)")
    parser.add_argument("--output-dir", default="./agent_experiments/outputs/exp1")
    parser.add_argument("--num-rq-levels", type=int, default=4)
    parser.add_argument("--codebook-size", type=int, default=512)
    parser.add_argument("--code-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--level-dropout", type=float, default=0.1)
    parser.add_argument("--commitment-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = ExperimentConfig(seed=args.seed)
    device = torch.device(cfg.device)
    loss_weights = cfg.get_loss_weights()

    logger = ExperimentLogger(args.output_dir, "exp1_rqvae")
    logger.log_hyperparameters(vars(args))

    # -- Data ---------------------------------------------------------------
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

    # -- Model --------------------------------------------------------------
    model = RQVae(
        nfeats=cfg.smplx_dim,
        num_rq_levels=args.num_rq_levels,
        code_num=args.codebook_size,
        code_dim=args.code_dim,
        commitment_weight=args.commitment_weight,
        level_dropout=args.level_dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Model] RQ-VAE with {args.num_rq_levels} levels, {n_params:.2f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # -- Training -----------------------------------------------------------
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, loss_weights, epoch)
        val_metrics = evaluate(model, val_loader, device, loss_weights)
        scheduler.step()

        elapsed = time.time() - t0
        val_loss = val_metrics["val_recon_loss"]

        print(
            f"  Epoch {epoch}/{args.epochs}: "
            f"train={train_metrics['loss']:.4f} val={val_loss:.4f} "
            f"perp={[f'{p:.1f}' for p in train_metrics['perplexities']]} "
            f"lr={scheduler.get_last_lr()[0]:.2e} ({elapsed:.1f}s)"
        )

        logger.log_epoch(
            epoch=epoch,
            train_loss=train_metrics["loss"],
            val_loss=val_loss,
            metrics={
                "recon_loss": train_metrics["recon_loss"],
                "commit_loss": train_metrics["commit_loss"],
                **{f"perplexity_L{i}": p for i, p in enumerate(train_metrics["perplexities"])},
            },
            lr=scheduler.get_last_lr()[0],
        )

        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(args.output_dir, "best_rqvae.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": vars(args),
            }, ckpt_path)
            print(f"    -> New best val loss. Saved to {ckpt_path}")

        if epoch % 50 == 0:
            ckpt_path = os.path.join(args.output_dir, f"rqvae_epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": vars(args),
            }, ckpt_path)

    # -- Optional: fine-tune on sentence data ------------------------------
    if args.sentence_dir and os.path.isdir(args.sentence_dir):
        print("\n" + "=" * 60)
        print("  Fine-tuning RQ-VAE on sentence-level data")
        print("=" * 60)
        sent_ds = PKLMotionDataset(args.sentence_dir)
        sent_loader = DataLoader(sent_ds, batch_size=max(1, args.batch_size // 2),
                                 shuffle=True, collate_fn=collate_pad,
                                 num_workers=4, pin_memory=True)
        ft_optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.5, weight_decay=1e-4)
        ft_epochs = max(50, args.epochs // 5)
        for epoch in range(1, ft_epochs + 1):
            metrics = train_one_epoch(model, sent_loader, ft_optimizer, device, loss_weights, epoch)
            print(f"  FT Epoch {epoch}/{ft_epochs}: loss={metrics['loss']:.4f}")
            logger.log_epoch(epoch=args.epochs + epoch, train_loss=metrics["loss"],
                             metrics={"phase": "sentence_finetune"})

        ckpt_path = os.path.join(args.output_dir, "rqvae_sentence_finetuned.pt")
        torch.save({"model_state_dict": model.state_dict(), "config": vars(args)}, ckpt_path)

    logger.log_final({"best_val_loss": best_val})
    print(f"\nExperiment 1 complete. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
