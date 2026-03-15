"""
Shared data utilities for Plan 1 experiments.
Loads NPZ motion data, normalizes using computed_stats.pt, provides
collate functions compatible with the existing VQ-VAE architecture.
"""

import os
import sys
import glob
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

SMPLX_DIM = 182
PARAM_DIMS = [10, 63, 45, 45, 3, 10, 3, 3]
PARAM_NAMES = [
    "shape", "body_pose", "lhand_pose", "rhand_pose",
    "jaw_pose", "expression", "root_pose", "cam_trans",
]
PARAM_STARTS = list(np.cumsum([0] + PARAM_DIMS[:-1]))

HAND_SLICE = slice(73, 163)
BODY_SLICE = slice(10, 73)
FACE_META_SLICES = [slice(0, 10), slice(163, 176), slice(176, 182)]


def build_loss_weights(pose_w=10.0, shape_w=5.0, expr_w=8.0):
    w = torch.ones(SMPLX_DIM)
    w[PARAM_STARTS[1]:PARAM_STARTS[1] + PARAM_DIMS[1]] = pose_w
    w[PARAM_STARTS[2]:PARAM_STARTS[2] + PARAM_DIMS[2]] = pose_w
    w[PARAM_STARTS[3]:PARAM_STARTS[3] + PARAM_DIMS[3]] = pose_w
    w[PARAM_STARTS[0]:PARAM_STARTS[0] + PARAM_DIMS[0]] = shape_w
    w[PARAM_STARTS[5]:PARAM_STARTS[5] + PARAM_DIMS[5]] = expr_w
    return w


class NpzMotionDataset(Dataset):
    """Load motion data from NPZ files with optional normalization."""

    def __init__(self, root_dir, stats_path=None, min_seq_len=32, max_seq_len=512):
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.mean = 0
        self.std = 1

        pattern = os.path.join(root_dir, "**", "*.npz")
        self.files = sorted(glob.glob(pattern, recursive=True))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {root_dir}")

        if stats_path and os.path.exists(stats_path):
            stats = torch.load(stats_path, map_location="cpu", weights_only=False)
            self.mean = stats["mean"]
            self.std = stats["std"]
            print(f"[Data] Loaded stats from {stats_path}")
        else:
            print("[Data] WARNING: No stats file, proceeding without normalization")

        print(f"[Data] Found {len(self.files)} npz files in {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            data = np.load(self.files[idx])
            motion = data["motion"]
        except Exception:
            return None
        if motion.ndim != 2 or motion.shape[0] < self.min_seq_len:
            return None
        motion = motion[: self.max_seq_len]
        seq = torch.tensor(motion, dtype=torch.float32)
        seq = (seq - self.mean) / (self.std + 1e-8)
        return seq

    def get_filepath(self, idx):
        return self.files[idx]


def collate_pad(batch, factor=8, max_len=256):
    """Pad variable-length sequences to nearest multiple of factor."""
    batch = [x for x in batch if x is not None]
    if not batch:
        return None, None
    batch.sort(key=lambda x: x.shape[0], reverse=True)
    longest = min(batch[0].shape[0], max_len)
    padded_len = math.ceil(longest / factor) * factor
    C = batch[0].shape[1]
    padded = torch.zeros(len(batch), padded_len, C)
    lengths = []
    for i, x in enumerate(batch):
        T = min(x.shape[0], padded_len)
        padded[i, :T] = x[:T]
        lengths.append(T)
    return padded, torch.tensor(lengths)


def train_val_split(dataset, val_ratio=0.1, seed=42):
    """Deterministic train/val index split."""
    n = len(dataset)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val = max(1, int(n * val_ratio))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)
