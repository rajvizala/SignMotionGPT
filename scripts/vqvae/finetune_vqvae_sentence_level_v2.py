"""
finetune_vqvae_sentence_level_v2.py  --  Improved fine-tuning script

Changes from v1 / earlier drafts
---------------------------------
* VQ loss weight raised  0.25 -> 2.0   (prevents encoder-codebook drift)
* EMA momentum lowered   0.99 -> 0.95  (codebook tracks encoder faster)
* Linear LR warmup over first N epochs (prevents early destabilisation)
* Periodic codebook reset from encoder outputs (revives dead codes)
* Removed double-penalisation: recon uses flat weights, component losses
  (hand, body, vel, geo) provide the part-specific supervision
* Removed jerk loss (was not contributing to training)
* Added train_recon_simple (MSE) for direct comparison with val recon
* Augmentation noise disabled during warmup, reduced default sigma
* Cosine annealing with warmup as default scheduler

All changes are backward-compatible: the same CLI flags work, and
checkpoints from the old script can be resumed.
"""

import os
import sys
import glob
import pickle
import argparse
import json
import math
import random
import re
from collections import defaultdict
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from tqdm import tqdm

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from models.vqvae import VQVae

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PARAM_DIMS = [10, 63, 45, 45, 3, 10, 3, 3]  # Total: 182
PARAM_NAMES = ["shape", "body_pose", "lhand_pose", "rhand_pose",
               "jaw_pose", "expression", "root_pose", "cam_trans"]
SMPL_DIM = sum(PARAM_DIMS)  # 182

VQ_CONFIG = {
    "nfeats": SMPL_DIM,
    "code_num": 512,
    "code_dim": 512,
    "output_emb_width": 512,
    "down_t": 2,
    "stride_t": 2,
    "width": 512,
    "depth": 3,
    "dilation_growth_rate": 3,
    "activation": "relu",
    "norm": None,
    "quantizer": "ema_reset",
}

DOWNSAMPLE_FACTOR = 2 ** VQ_CONFIG["down_t"]
GDRIVE_CHECKPOINT_DIR = "/content/drive/MyDrive/finetune_combine_checkpoint"

# Slice indices for body parts inside the 182-dim vector
_HAND_SLICE = slice(73, 163)   # lhand + rhand  (90 dims)
_BODY_SLICE = slice(10, 73)    # body_pose       (63 dims)

# Indices to KEEP for sign language (body_pose + hands + root_pose)
# body_pose(10:73) + lhand(73:118) + rhand(118:163) + root_pose(176:179)
KEEP_INDICES_SIGN = list(range(10, 163)) + list(range(176, 179))  # 156 dims
SIGN_DIM = len(KEEP_INDICES_SIGN)  # 156


def strip_irrelevant(motion: torch.Tensor,
                     keep_idx=KEEP_INDICES_SIGN) -> torch.Tensor:
    """Extract only the relevant dims from the full 182-dim vector."""
    return motion[:, :, keep_idx]


def restore_full(stripped: torch.Tensor,
                 keep_idx=KEEP_INDICES_SIGN,
                 full_dim=SMPL_DIM) -> torch.Tensor:
    """Pad stripped tensor back to 182 dims (zeros for removed dims)."""
    B, T, _ = stripped.shape
    full = torch.zeros(B, T, full_dim, device=stripped.device,
                       dtype=stripped.dtype)
    idx = torch.tensor(keep_idx, device=stripped.device)
    full[:, :, idx] = stripped
    return full


# ===================================================================
# Checkpoint management
# ===================================================================

def find_checkpoint_files(directory: str) -> List[Tuple[str, int]]:
    if not os.path.exists(directory):
        return []
    checkpoints = []
    pattern = re.compile(r'vqvae_finetuned_epoch_(\d+)\.pt$')
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            epoch = int(match.group(1))
            checkpoints.append((os.path.join(directory, filename), epoch))
    checkpoints.sort(key=lambda x: x[1], reverse=True)
    return checkpoints


def find_latest_checkpoint(local_dir: str,
                           gdrive_dir: str = GDRIVE_CHECKPOINT_DIR
                           ) -> Optional[Tuple[str, int]]:
    print("\n[Checkpoint Search]")
    local_ckpts = find_checkpoint_files(local_dir)
    if local_ckpts:
        print(f"  Local ({local_dir}): Found {len(local_ckpts)} checkpoints, latest epoch {local_ckpts[0][1]}")
    else:
        print(f"  Local ({local_dir}): No checkpoints found")

    gdrive_ckpts = find_checkpoint_files(gdrive_dir)
    if gdrive_ckpts:
        print(f"  GDrive ({gdrive_dir}): Found {len(gdrive_ckpts)} checkpoints, latest epoch {gdrive_ckpts[0][1]}")
    else:
        print(f"  GDrive ({gdrive_dir}): No checkpoints found")

    all_ckpts = local_ckpts + gdrive_ckpts
    if not all_ckpts:
        print("  No existing checkpoints found. Starting from scratch.")
        return None
    all_ckpts.sort(key=lambda x: x[1], reverse=True)
    latest = all_ckpts[0]
    print(f"\n  -> Using checkpoint: {latest[0]} (epoch {latest[1]})")
    return latest


def delete_old_gdrive_checkpoints(gdrive_dir: str, keep_epoch: int):
    if not os.path.exists(gdrive_dir):
        return
    for filepath, epoch in find_checkpoint_files(gdrive_dir):
        if epoch != keep_epoch:
            try:
                os.remove(filepath)
                print(f"  Deleted old GDrive checkpoint: epoch {epoch}")
            except Exception:
                pass


# ===================================================================
# Model wrapper
# ===================================================================

class VQVAEWrapper(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.vqvae = VQVae(**config)

    def forward(self, x):
        return self.vqvae(x)

    def encode(self, x):
        return self.vqvae.encode(x)

    def decode(self, codes):
        return self.vqvae.decode(codes)


def load_vqvae_checkpoint(checkpoint_path: str, config: dict,
                          device=DEVICE) -> VQVAEWrapper:
    print(f"\nLoading VQ-VAE checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)

    has_prefix = any(k.startswith('vqvae.') for k in state_dict)
    model = VQVAEWrapper(config).to(device)

    if has_prefix:
        new_sd = {}
        for k, v in state_dict.items():
            new_sd[k[6:] if k.startswith('vqvae.') else k] = v
        result = model.vqvae.load_state_dict(new_sd, strict=False)
    else:
        result = model.vqvae.load_state_dict(state_dict, strict=False)

    print(f"  Loaded checkpoint - Missing: {len(result.missing_keys)}, "
          f"Unexpected: {len(result.unexpected_keys)}")
    return model


def set_quantizer_ema_mu(model: VQVAEWrapper, mu: float):
    """Override the EMA momentum on the quantizer for fine-tuning."""
    q = model.vqvae.quantizer
    if hasattr(q, 'mu'):
        old_mu = q.mu
        q.mu = mu
        print(f"  Quantizer EMA mu: {old_mu} -> {mu}")


# ===================================================================
# Codebook maintenance
# ===================================================================

@torch.no_grad()
def reset_dead_codes(model: VQVAEWrapper, dataloader: DataLoader,
                     device=DEVICE, usage_threshold: int = 2,
                     num_batches: int = 50):
    """Reset underutilized codebook entries from fresh encoder outputs.

    Samples num_batches from the dataloader, counts how often each code
    is used, and replaces codes used fewer than usage_threshold times
    with fresh encoder vectors.
    """
    model.eval()
    quantizer = model.vqvae.quantizer
    codebook_size = quantizer.nb_code

    all_codes = []
    encoder_vecs = []
    for i, batch in enumerate(dataloader):
        if i >= num_batches or batch is None:
            break
        motion = batch["motion"].to(device)
        x_in = model.vqvae.preprocess(motion)
        z_e = model.vqvae.encoder(x_in)
        z_flat = z_e.permute(0, 2, 1).reshape(-1, z_e.shape[1])
        encoder_vecs.append(z_flat)

        codes, _ = model.encode(motion)
        all_codes.append(codes.cpu().numpy().flatten())

    model.train()
    if not all_codes:
        print("  [Codebook] No data available for codebook analysis")
        return 0

    all_codes_np = np.concatenate(all_codes)
    counts = np.bincount(all_codes_np, minlength=codebook_size)
    total_tokens = len(all_codes_np)
    active_codes = int((counts > 0).sum())
    low_use_codes = int((counts < usage_threshold).sum())
    zero_use_codes = int((counts == 0).sum())

    print(f"  [Codebook] Scanned {total_tokens} tokens from {num_batches} batches")
    print(f"    Active: {active_codes}/{codebook_size} "
          f"({100*active_codes/codebook_size:.1f}%) | "
          f"Zero-use: {zero_use_codes} | "
          f"Below threshold (<{usage_threshold}): {low_use_codes}")

    dead_idx = np.where(counts < usage_threshold)[0]
    num_dead = len(dead_idx)
    if num_dead == 0:
        print(f"    All codes healthy -- no reset needed")
        return 0

    z_pool = torch.cat(encoder_vecs, dim=0)
    if z_pool.shape[0] < num_dead:
        print(f"    Not enough encoder vectors ({z_pool.shape[0]}) "
              f"to reset {num_dead} codes -- skipping")
        return 0

    perm = torch.randperm(z_pool.shape[0])[:num_dead]
    dead_idx_t = torch.from_numpy(dead_idx).to(device)
    quantizer.codebook.data[dead_idx_t] = z_pool[perm]

    if hasattr(quantizer, 'code_sum') and quantizer.code_sum is not None:
        quantizer.code_sum.data[dead_idx_t] = z_pool[perm]
        quantizer.code_count.data[dead_idx_t] = 1.0

    print(f"    Reset {num_dead} codes from encoder outputs")
    return num_dead


# ===================================================================
# Geodesic rotation loss
# ===================================================================
ROT_PARAMS = {
    "body_pose":  (10,  73,  21, 1.0),
    "lhand_pose": (73,  118, 15, 2.0),
    "rhand_pose": (118, 163, 15, 2.0),
    "jaw_pose":   (163, 166,  1, 0.3),
    "root_pose":  (176, 179,  1, 0.3),
}


def _axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    angle = torch.norm(aa, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
    axis = aa / angle
    cos_a = torch.cos(angle).unsqueeze(-1)
    sin_a = torch.sin(angle).unsqueeze(-1)
    x, y, z = axis.unbind(dim=-1)
    zero = torch.zeros_like(x)
    K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero],
                    dim=-1).reshape(*aa.shape[:-1], 3, 3)
    outer = axis.unsqueeze(-1) * axis.unsqueeze(-2)
    I = torch.eye(3, device=aa.device, dtype=aa.dtype).expand(
        *aa.shape[:-1], 3, 3)
    return cos_a * I + (1.0 - cos_a) * outer + sin_a * K


def geodesic_rotation_loss(pred, target, mask):
    p = pred.reshape(-1, 3)
    t = target.reshape(-1, 3)
    R_p = _axis_angle_to_matrix(p)
    R_t = _axis_angle_to_matrix(t)
    R_diff = R_p.transpose(-1, -2) @ R_t
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    cos_angle = ((trace - 1.0) / 2.0).clamp(-1 + 1e-6, 1 - 1e-6)
    B, T, C = pred.shape
    angles = torch.acos(cos_angle).reshape(B, T, C // 3)
    mask_exp = mask.unsqueeze(-1).expand_as(angles)
    return (angles * mask_exp).sum() / (mask_exp.sum() + 1e-8)


def compute_geodesic_loss(pred, target, mask):
    total = torch.tensor(0.0, device=pred.device)
    w_total = 0.0
    for _, (s, e, _, w) in ROT_PARAMS.items():
        total = total + w * geodesic_rotation_loss(
            pred[:, :, s:e], target[:, :, s:e], mask)
        w_total += w
    return total / w_total


# ===================================================================
# Jerk loss
# ===================================================================

def compute_jerk_loss(recon, target, mask,
                      hand_slice=_HAND_SLICE):
    def _jerk(x):
        acc = x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]
        return acc[:, 1:] - acc[:, :-1]
    jr = _jerk(recon[:, :, hand_slice])
    jt = _jerk(target[:, :, hand_slice])
    m = mask[:, 3:].unsqueeze(-1).expand_as(jr)
    return ((jr - jt) ** 2 * m).sum() / (m.sum() + 1e-8)


# ===================================================================
# Augmentation (gentler than v1)
# ===================================================================

def augment_motion(motion: torch.Tensor, epoch: int,
                   warmup_epochs: int = 10,
                   noise_sigma: float = 0.005) -> torch.Tensor:
    """Light augmentation; noise is disabled during warmup."""
    motion = motion.clone()
    if epoch > warmup_epochs and noise_sigma > 0:
        scale = min(1.0, (epoch - warmup_epochs) / 20.0)
        motion = motion + torch.randn_like(motion) * noise_sigma * scale
    return motion


# ===================================================================
# Dataset classes
# ===================================================================

class WordLevelDataset(Dataset):
    """NPZ motion data (word- or sentence-level)."""

    def __init__(self, root_dir: str, stats_path: Optional[str] = None,
                 min_seq_len: int = 16, max_seq_len: int = 256,
                 data_type: str = "word"):
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.data_type = data_type

        print(f"\n[NPZDataset ({data_type})] Loading from: {root_dir}")
        self.files = glob.glob(os.path.join(root_dir, "**", "*.npz"),
                               recursive=True)
        if not self.files:
            print(f"  Warning: No .npz files found in {root_dir}")
        else:
            print(f"  Found {len(self.files)} NPZ files")

        self.mean, self.std = self._load_stats(stats_path)

    def _load_stats(self, stats_path):
        if stats_path and os.path.exists(stats_path):
            stats = torch.load(stats_path, map_location='cpu')
            mean = stats.get('mean', torch.zeros(SMPL_DIM))
            std = stats.get('std', torch.ones(SMPL_DIM))
            print(f"  Loaded stats from {stats_path}")
        else:
            print(f"  Warning: Stats not found, using default")
            mean = torch.zeros(SMPL_DIM)
            std = torch.ones(SMPL_DIM)
        if not torch.is_tensor(mean):
            mean = torch.tensor(mean)
        if not torch.is_tensor(std):
            std = torch.tensor(std)
        return mean.float(), std.float()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            with np.load(self.files[idx]) as data:
                motion = data['motion']
        except Exception:
            return None
        if motion.shape[0] < self.min_seq_len:
            return None
        if motion.shape[0] > self.max_seq_len:
            motion = motion[:self.max_seq_len]
        t = torch.tensor(motion, dtype=torch.float32)
        t = (t - self.mean) / (self.std + 1e-8)
        return {"motion": t, "type": self.data_type, "source": "npz"}


class SentenceLevelDataset(Dataset):
    """How2Sign PKL-based sentence dataset."""

    def __init__(self, root_dir: str, stats_path: Optional[str] = None,
                 min_seq_len: int = 32, max_seq_len: int = 512):
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        print(f"\n[SentenceLevelDataset] Loading from: {root_dir}")
        self.sequence_folders = self._find_folders(root_dir)
        print(f"  Found {len(self.sequence_folders)} sequence folders")
        self.mean, self.std = WordLevelDataset._load_stats(self, stats_path)

    @staticmethod
    def _find_folders(root_dir):
        folders = []
        if not os.path.exists(root_dir):
            return folders
        if glob.glob(os.path.join(root_dir, "*_3D.pkl")):
            folders.append(root_dir)
        for item in sorted(os.listdir(root_dir)):
            p = os.path.join(root_dir, item)
            if os.path.isdir(p) and glob.glob(os.path.join(p, "*_3D.pkl")):
                folders.append(p)
        return folders

    def _load_sequence(self, folder):
        pkl_files = glob.glob(os.path.join(folder, "*_3D.pkl"))
        if not pkl_files:
            return None

        def _frame(f):
            m = re.search(r'_(\d+)_3D\.pkl$', os.path.basename(f))
            return int(m.group(1)) if m else 0
        pkl_files.sort(key=_frame)

        agg = defaultdict(list)
        for f in pkl_files:
            try:
                with open(f, 'rb') as fp:
                    d = pickle.load(fp)
                for k, v in d.items():
                    if torch.is_tensor(v):
                        v = v.cpu().numpy()
                    agg[k].append(v)
            except Exception:
                pass
        if not agg:
            return None
        try:
            def _get(key, dim, T):
                a = np.array(agg.get(key, []))
                if a.ndim == 3 and a.shape[1] == 1:
                    a = a.squeeze(1)
                if len(a) == 0 or a.shape[-1] == 0:
                    return np.zeros((T, dim), dtype=np.float32)
                if a.shape[-1] > dim:
                    a = a[..., :dim]
                elif a.shape[-1] < dim:
                    a = np.pad(a, ((0, 0), (0, dim - a.shape[-1])))
                return a.astype(np.float32)

            body = _get('smplx_body_pose', 63, 0)
            if len(body) == 0:
                return None
            T = len(body)
            parts = [
                _get('smplx_shape', 10, T), body,
                _get('smplx_lhand_pose', 45, T),
                _get('smplx_rhand_pose', 45, T),
                _get('smplx_jaw_pose', 3, T),
                _get('smplx_expr', 10, T),
                _get('smplx_root_pose', 3, T),
                _get('cam_trans', 3, T),
            ]
            vec = np.concatenate(parts, axis=1)
            return vec if vec.shape[1] == SMPL_DIM else None
        except Exception:
            return None

    def __len__(self):
        return len(self.sequence_folders)

    def __getitem__(self, idx):
        motion = self._load_sequence(self.sequence_folders[idx])
        if motion is None or motion.shape[0] < self.min_seq_len:
            return None
        if motion.shape[0] > self.max_seq_len:
            motion = motion[:self.max_seq_len]
        t = torch.tensor(motion, dtype=torch.float32)
        t = (t - self.mean) / (self.std + 1e-8)
        return {"motion": t, "type": "sentence", "source": "how2sign"}


class PreloadedDataset(Dataset):
    def __init__(self, data: List[Dict], data_type: str = "preloaded"):
        self.data = data
        self.data_type = data_type
        print(f"[PreloadedDataset] {len(self.data)} samples in memory")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def preload_word_level_data(root_dir, stats_path=None,
                            min_seq_len=16, max_seq_len=256,
                            show_progress=True):
    print(f"\n  PRELOADING WORD-LEVEL DATA\n  Source: {root_dir}")
    if stats_path and os.path.exists(stats_path):
        stats = torch.load(stats_path, map_location='cpu')
        mean = stats.get('mean', torch.zeros(SMPL_DIM)).float()
        std = stats.get('std', torch.ones(SMPL_DIM)).float()
        if not torch.is_tensor(mean):
            mean = torch.tensor(mean).float()
        if not torch.is_tensor(std):
            std = torch.tensor(std).float()
    else:
        mean, std = torch.zeros(SMPL_DIM), torch.ones(SMPL_DIM)

    files = glob.glob(os.path.join(root_dir, "**", "*.npz"), recursive=True)
    if not files:
        return PreloadedDataset([], "word")

    data = []
    it = tqdm(files, desc="Loading word data") if show_progress else files
    for fp in it:
        try:
            with np.load(fp) as d:
                m = d['motion']
            if m.shape[0] < min_seq_len:
                continue
            if m.shape[0] > max_seq_len:
                m = m[:max_seq_len]
            t = torch.tensor(m, dtype=torch.float32)
            t = (t - mean) / (std + 1e-8)
            data.append({"motion": t, "type": "word", "source": "npz"})
        except Exception:
            continue
    print(f"  Loaded: {len(data)} samples\n")
    return PreloadedDataset(data, "word")


def preload_sentence_level_data(root_dir, stats_path=None,
                                min_seq_len=32, max_seq_len=512,
                                show_progress=True):
    print(f"\n  PRELOADING SENTENCE-LEVEL DATA\n  Source: {root_dir}")
    if stats_path and os.path.exists(stats_path):
        stats = torch.load(stats_path, map_location='cpu')
        mean = stats.get('mean', torch.zeros(SMPL_DIM)).float()
        std = stats.get('std', torch.ones(SMPL_DIM)).float()
        if not torch.is_tensor(mean):
            mean = torch.tensor(mean).float()
        if not torch.is_tensor(std):
            std = torch.tensor(std).float()
    else:
        mean, std = torch.zeros(SMPL_DIM), torch.ones(SMPL_DIM)

    folders = SentenceLevelDataset._find_folders(root_dir)
    if not folders:
        return PreloadedDataset([], "sentence")

    ds = SentenceLevelDataset.__new__(SentenceLevelDataset)
    ds.mean, ds.std = mean, std
    ds.min_seq_len = min_seq_len
    ds.max_seq_len = max_seq_len
    ds.sequence_folders = folders

    data = []
    it = tqdm(folders, desc="Loading sentence data") if show_progress else folders
    for i, _ in enumerate(it):
        sample = ds[i]
        if sample is not None:
            data.append(sample)
    print(f"  Loaded: {len(data)} samples\n")
    return PreloadedDataset(data, "sentence")


# ===================================================================
# Collate / dataloader helpers
# ===================================================================

def collate_fn_mixed(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    batch.sort(key=lambda x: x["motion"].shape[0], reverse=True)
    max_len = batch[0]["motion"].shape[0]
    padded_len = math.ceil(max_len / DOWNSAMPLE_FACTOR) * DOWNSAMPLE_FACTOR
    B = len(batch)
    padded = torch.zeros(B, padded_len, SMPL_DIM)
    lengths, types = [], []
    for i, item in enumerate(batch):
        sl = item["motion"].shape[0]
        padded[i, :sl, :] = item["motion"]
        lengths.append(sl)
        types.append(item.get("type", "unknown"))
    return {"motion": padded, "lengths": torch.tensor(lengths), "types": types}


def create_mixed_dataloader(word_ds, sentence_ds, word_ratio=0.2,
                            batch_size=8, num_workers=0):
    combined = ConcatDataset([word_ds, sentence_ds])
    wl, sl = len(word_ds), len(sentence_ds)
    total = wl + sl
    ww = word_ratio / wl if wl > 0 else 0
    sw = (1 - word_ratio) / sl if sl > 0 else 0
    weights = [ww] * wl + [sw] * sl
    sampler = WeightedRandomSampler(weights, num_samples=total,
                                    replacement=True)
    return DataLoader(combined, batch_size=batch_size, sampler=sampler,
                      collate_fn=collate_fn_mixed, num_workers=num_workers,
                      pin_memory=True, drop_last=False)


def create_val_dataloader(val_ds, batch_size=8, num_workers=0):
    return DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                      collate_fn=collate_fn_mixed, num_workers=num_workers,
                      pin_memory=True, drop_last=False)


# ===================================================================
# Warmup + cosine LR scheduler
# ===================================================================

def cosine_warmup_lambda(epoch, warmup_epochs, total_epochs, min_lr_ratio=0.01):
    if epoch < warmup_epochs:
        return max(min_lr_ratio, epoch / max(1, warmup_epochs))
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))


# ===================================================================
# Trainer
# ===================================================================

class FinetuneTrainer:
    def __init__(self, model: VQVAEWrapper, dataloader: DataLoader,
                 output_dir: str, *,
                 learning_rate=1e-5, vq_loss_weight=2.5,
                 vel_loss_weight=0.5, hand_loss_weight=3.0,
                 body_loss_weight=1.0, geo_loss_weight=0.2,
                 grad_clip=1.0, gdrive_dir=GDRIVE_CHECKPOINT_DIR,
                 scheduler_type="cosine_warmup", num_epochs=100,
                 warmup_epochs=10, restart_period=100,
                 codebook_reset_every=5, codebook_reset_threshold=2,
                 noise_sigma=0.005, zero_irrelevant=False,
                 device=DEVICE):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.output_dir = output_dir
        self.gdrive_dir = gdrive_dir
        self.device = device

        self.vq_loss_weight = vq_loss_weight
        self.vel_loss_weight = vel_loss_weight
        self.hand_loss_weight = hand_loss_weight
        self.body_loss_weight = body_loss_weight
        self.geo_loss_weight = geo_loss_weight
        self.zero_irrelevant = zero_irrelevant
        self.grad_clip = grad_clip
        self.learning_rate = learning_rate
        self.scheduler_type = scheduler_type
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.restart_period = restart_period
        self.codebook_reset_every = codebook_reset_every
        self.codebook_reset_threshold = codebook_reset_threshold
        self.noise_sigma = noise_sigma
        self.start_epoch = 1

        os.makedirs(output_dir, exist_ok=True)
        if gdrive_dir:
            os.makedirs(gdrive_dir, exist_ok=True)

        self.optimizer = AdamW(model.parameters(), lr=learning_rate,
                               weight_decay=1e-4)
        self.scheduler = self._create_scheduler()
        self.recon_loss_fn = nn.SmoothL1Loss(reduction='none')

        self.history = {
            "epoch": [], "total_loss": [], "recon_loss": [], "vq_loss": [],
            "vel_loss": [], "hand_loss": [], "body_loss": [],
            "perplexity": [], "word_loss": [], "sentence_loss": [],
            "codebook_usage": [], "val_loss": [], "geo_loss": [],
            "lr": [], "train_recon_simple": [],
        }

    # ---------------------------------------------------------------
    def _create_scheduler(self):
        st = self.scheduler_type
        if st == "cosine_warmup":
            lam = lambda ep: cosine_warmup_lambda(
                ep, self.warmup_epochs, self.num_epochs)
            return LambdaLR(self.optimizer, lr_lambda=lam)
        if st == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)
        if st == "cosine_restart":
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            return CosineAnnealingWarmRestarts(
                self.optimizer, T_0=self.restart_period, T_mult=1,
                eta_min=1e-6)
        if st == "cosine_restart_gradual":
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            return CosineAnnealingWarmRestarts(
                self.optimizer, T_0=self.restart_period, T_mult=2,
                eta_min=1e-6)
        if st == "step":
            from torch.optim.lr_scheduler import StepLR
            return StepLR(self.optimizer, step_size=self.restart_period,
                          gamma=0.5)
        return LambdaLR(self.optimizer, lr_lambda=lambda _: 1.0)

    # ---------------------------------------------------------------
    def reset_learning_rate(self, new_lr=None, warmup_epochs=5):
        lr = new_lr if new_lr is not None else self.learning_rate
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
            pg['initial_lr'] = lr
        self.scheduler = self._create_scheduler()
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    # ---------------------------------------------------------------
    def load_checkpoint(self, checkpoint_path: str) -> int:
        print(f"\n[Loading Checkpoint]\n  Path: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            return 0
        ckpt = torch.load(checkpoint_path, map_location=self.device,
                          weights_only=False)
        if 'model_state_dict' in ckpt:
            self.model.load_state_dict(ckpt['model_state_dict'], strict=False)
        if 'optimizer_state_dict' in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except Exception:
                print("  Warning: Could not load optimizer state")
        if 'scheduler_state_dict' in ckpt:
            try:
                self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            except Exception:
                pass
        if 'history' in ckpt:
            loaded = ckpt['history']
            for key in self.history:
                if key in loaded:
                    self.history[key] = loaded[key]
                else:
                    self.history[key] = [0.0] * len(loaded.get('epoch', []))
        epoch = ckpt.get('epoch', 0)
        self.start_epoch = epoch + 1
        return epoch

    # ---------------------------------------------------------------
    def compute_loss(self, batch):
        motion = batch["motion"].to(self.device)
        lengths = batch["lengths"]
        types = batch["types"]

        if self.zero_irrelevant:
            motion = motion.clone()
            motion = strip_irrelevant(motion)
            motion = restore_full(motion)

        x_recon, vq_loss, perplexity = self.model(motion)

        mask = torch.zeros(motion.shape[0], motion.shape[1],
                           device=self.device)
        for i, l in enumerate(lengths):
            mask[i, :l] = 1.0

        if self.zero_irrelevant:
            keep = torch.tensor(KEEP_INDICES_SIGN, device=self.device)
            motion_k = motion[:, :, keep]
            recon_k = x_recon[:, :, keep]
        else:
            motion_k = motion
            recon_k = x_recon

        C = motion_k.shape[2]

        # -- Reconstruction (only on kept dims) --
        recon_raw = self.recon_loss_fn(recon_k, motion_k)
        mask_exp = mask.unsqueeze(-1).expand_as(recon_raw)
        recon_loss = (recon_raw * mask_exp).sum() / (mask_exp.sum() + 1e-8)

        # -- Velocity (only on kept dims) --
        vel_t = motion_k[:, 1:] - motion_k[:, :-1]
        vel_r = recon_k[:, 1:] - recon_k[:, :-1]
        vm = mask[:, 1:].unsqueeze(-1).expand_as(vel_t)
        vel_loss = ((vel_r - vel_t) ** 2 * vm).sum() / (vm.sum() + 1e-8)

        # -- Hand (always full-tensor indices, unaffected by stripping) --
        h_t = motion[:, :, _HAND_SLICE]
        h_r = x_recon[:, :, _HAND_SLICE]
        hm = mask.unsqueeze(-1).expand(-1, -1, 90)
        hand_loss = ((h_r - h_t) ** 2 * hm).sum() / (hm.sum() + 1e-8)

        # -- Body --
        b_t = motion[:, :, _BODY_SLICE]
        b_r = x_recon[:, :, _BODY_SLICE]
        bm = mask.unsqueeze(-1).expand(-1, -1, 63)
        body_loss = ((b_r - b_t) ** 2 * bm).sum() / (bm.sum() + 1e-8)

        # -- Geodesic (uses full-tensor indices internally) --
        geo_loss = compute_geodesic_loss(x_recon, motion, mask)

        total = (recon_loss
                 + self.vq_loss_weight * vq_loss
                 + self.vel_loss_weight * vel_loss
                 + self.hand_loss_weight * hand_loss
                 + self.body_loss_weight * body_loss
                 + self.geo_loss_weight * geo_loss)

        # Simple MSE recon (kept dims only, same metric as val)
        simple_recon = ((recon_k - motion_k) ** 2 * mask_exp).sum() / (
            mask_exp.sum() + 1e-8)

        # Per-type monitoring
        wm = torch.tensor([1.0 if t == "word" else 0.0 for t in types],
                          device=self.device)
        sm = 1.0 - wm
        per_sample = (recon_raw * mask_exp).sum(dim=(1, 2)) / (
            mask_exp.sum(dim=(1, 2)) + 1e-8)
        word_loss = (per_sample * wm).sum() / (wm.sum() + 1e-8)
        sent_loss = (per_sample * sm).sum() / (sm.sum() + 1e-8)

        metrics = {
            "total_loss": total.item(), "recon_loss": recon_loss.item(),
            "vq_loss": vq_loss.item(), "vel_loss": vel_loss.item(),
            "hand_loss": hand_loss.item(), "body_loss": body_loss.item(),
            "perplexity": perplexity.item(),
            "word_loss": word_loss.item(), "sentence_loss": sent_loss.item(),
            "geo_loss": geo_loss.item(),
            "train_recon_simple": simple_recon.item(),
        }
        return total, metrics

    # ---------------------------------------------------------------
    def compute_codebook_usage(self, num_batches=50):
        self.model.eval()
        all_codes = []
        with torch.no_grad():
            for i, batch in enumerate(self.dataloader):
                if i >= num_batches or batch is None:
                    break
                codes, _ = self.model.encode(
                    batch["motion"].to(self.device))
                all_codes.append(codes.cpu().numpy().flatten())
        self.model.train()
        if not all_codes:
            return 0.0
        return (len(np.unique(np.concatenate(all_codes)))
                / VQ_CONFIG["code_num"]) * 100

    # ---------------------------------------------------------------
    def train_epoch(self, epoch):
        self.model.train()
        epoch_metrics = defaultdict(list)
        type_counts = {"word": 0, "sentence": 0}

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            if batch is None:
                continue
            for t in batch["types"]:
                if t in type_counts:
                    type_counts[t] += 1

            batch = dict(batch)
            batch["motion"] = augment_motion(
                batch["motion"], epoch,
                warmup_epochs=self.warmup_epochs,
                noise_sigma=self.noise_sigma)

            self.optimizer.zero_grad()
            loss, metrics = self.compute_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.grad_clip)
            self.optimizer.step()

            for k, v in metrics.items():
                epoch_metrics[k].append(v)
            pbar.set_postfix({
                "loss": f"{metrics['total_loss']:.4f}",
                "recon": f"{metrics['recon_loss']:.4f}",
                "vq": f"{metrics['vq_loss']:.4f}",
                "hand": f"{metrics['hand_loss']:.4f}",
                "body": f"{metrics['body_loss']:.4f}",
                "ppl": f"{metrics['perplexity']:.0f}",
            })

        self.scheduler.step()

        avg = {k: float(np.mean(v)) for k, v in epoch_metrics.items()}
        avg["word_count"] = type_counts["word"]
        avg["sentence_count"] = type_counts["sentence"]
        avg["codebook_usage"] = self.compute_codebook_usage()
        avg["lr"] = self.optimizer.param_groups[0]['lr']
        return avg

    # ---------------------------------------------------------------
    def validate_epoch(self, val_dl):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in val_dl:
                if batch is None:
                    continue
                _, m = self.compute_loss(batch)
                losses.append(m["train_recon_simple"])
        self.model.train()
        return float(np.mean(losses)) if losses else float("inf")

    # ---------------------------------------------------------------
    def save_checkpoint(self, epoch, metrics, save_to_gdrive=True):
        data = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics, "history": self.history,
            "config": VQ_CONFIG,
            "timestamp": datetime.now().isoformat(),
        }
        path = os.path.join(self.output_dir,
                            f"vqvae_finetuned_epoch_{epoch:03d}.pt")
        torch.save(data, path)
        torch.save(data, os.path.join(self.output_dir,
                                      "vqvae_finetuned_latest.pt"))
        if save_to_gdrive and self.gdrive_dir:
            try:
                os.makedirs(self.gdrive_dir, exist_ok=True)
                gp = os.path.join(self.gdrive_dir,
                                  f"vqvae_finetuned_epoch_{epoch:03d}.pt")
                torch.save(data, gp)
                delete_old_gdrive_checkpoints(self.gdrive_dir, epoch)
            except Exception:
                pass

    # ---------------------------------------------------------------
    def plot_training_history(self):
        if not HAS_MATPLOTLIB or not self.history["epoch"]:
            return
        ep = self.history["epoch"]
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        axes[0, 0].plot(ep, self.history["total_loss"])
        axes[0, 0].set_title("Total Loss")
        axes[0, 1].plot(ep, self.history["train_recon_simple"], label="Train")
        if any(v > 0 for v in self.history["val_loss"]):
            axes[0, 1].plot(ep, self.history["val_loss"], label="Val")
        axes[0, 1].legend()
        axes[0, 1].set_title("Train vs Val Recon (MSE)")
        axes[0, 2].plot(ep, self.history["vq_loss"])
        axes[0, 2].set_title("VQ Loss")
        axes[0, 3].plot(ep, self.history["lr"])
        axes[0, 3].set_title("Learning Rate")

        axes[1, 0].plot(ep, self.history["hand_loss"], label="Hand")
        axes[1, 0].plot(ep, self.history["body_loss"], label="Body")
        axes[1, 0].legend()
        axes[1, 0].set_title("Component Losses")
        axes[1, 1].plot(ep, self.history["word_loss"], label="Word")
        axes[1, 1].plot(ep, self.history["sentence_loss"], label="Sentence")
        axes[1, 1].legend()
        axes[1, 1].set_title("Loss by Type")
        axes[1, 2].plot(ep, self.history["codebook_usage"])
        axes[1, 2].set_title("Codebook Usage %")
        axes[1, 3].plot(ep, self.history["perplexity"])
        axes[1, 3].set_title("Perplexity")

        for ax in axes.flat:
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "training_history.png"),
                    dpi=150)
        plt.close()

    # ---------------------------------------------------------------
    def train(self, num_epochs, save_every=10, val_dataloader=None):
        target = self.start_epoch + num_epochs - 1
        print(f"\n  Resuming from epoch {self.start_epoch}. "
              f"Target epoch: {target}\n")
        if self.start_epoch > target:
            return

        best_val = float("inf")

        for epoch in range(self.start_epoch, target + 1):
            metrics = self.train_epoch(epoch)

            # --- Codebook reset ---
            if (self.codebook_reset_every > 0
                    and epoch % self.codebook_reset_every == 0):
                reset_dead_codes(
                    self.model, self.dataloader, self.device,
                    usage_threshold=self.codebook_reset_threshold)

            # --- Validation ---
            val_loss = 0.0
            if val_dataloader is not None:
                val_loss = self.validate_epoch(val_dataloader)
                if val_loss < best_val:
                    best_val = val_loss
                    bp = os.path.join(self.output_dir, "vqvae_best.pt")
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "val_loss": val_loss, "config": VQ_CONFIG,
                    }, bp)
                    print(f"  [*] New best val loss: {val_loss:.6f}")

            # --- History ---
            self.history["epoch"].append(epoch)
            self.history["val_loss"].append(val_loss)
            for k in ["total_loss", "recon_loss", "vq_loss", "vel_loss",
                       "hand_loss", "body_loss", "perplexity", "word_loss",
                       "sentence_loss", "codebook_usage", "geo_loss",
                       "lr", "train_recon_simple"]:
                self.history[k].append(metrics.get(k, 0))

            # --- Logging ---
            lr_now = self.optimizer.param_groups[0]['lr']
            cb_usage = metrics.get('codebook_usage', 0)
            cb_active = int(cb_usage / 100 * VQ_CONFIG["code_num"])
            cb_total = VQ_CONFIG["code_num"]
            ppl = metrics.get('perplexity', 0)
            train_simple = metrics.get('train_recon_simple', 0)

            print(f"\n[Epoch {epoch}/{target}]  LR: {lr_now:.2e}  "
                  f"Perplexity: {ppl:.1f}  "
                  f"Codebook: {cb_active}/{cb_total} "
                  f"({cb_usage:.1f}%)")
            print(f"  Train -> Total: {metrics['total_loss']:.4f} | "
                  f"Recon: {metrics['recon_loss']:.4f} | "
                  f"VQ: {metrics['vq_loss']:.4f}")
            print(f"           Hand: {metrics['hand_loss']:.4f} | "
                  f"Body: {metrics['body_loss']:.4f} | "
                  f"Geo: {metrics['geo_loss']:.4f}")
            if val_loss > 0:
                print(f"  Recon  -> Train: {train_simple:.6f} | "
                      f"Val: {val_loss:.6f}")
            else:
                print(f"  Recon  -> Train: {train_simple:.6f}")

            # --- Checkpoint ---
            if epoch % save_every == 0 or epoch == target:
                self.save_checkpoint(epoch, metrics, save_to_gdrive=True)
                self.plot_training_history()

            with open(os.path.join(self.output_dir,
                                   "training_history.json"), 'w') as f:
                json.dump(self.history, f, indent=2)

        # Final save
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": VQ_CONFIG, "history": self.history,
            "timestamp": datetime.now().isoformat(),
        }, os.path.join(self.output_dir, "vqvae_finetuned_final.pt"))


# ===================================================================
# Validation-only mode
# ===================================================================

def validate_checkpoint(model, word_ds, sentence_ds, num_samples=20,
                        device=DEVICE):
    model.eval()

    def _sample_loss(motion_tensor):
        with torch.no_grad():
            T = motion_tensor.shape[0]
            pl = math.ceil(T / DOWNSAMPLE_FACTOR) * DOWNSAMPLE_FACTOR
            if pl > T:
                motion_tensor = torch.cat(
                    [motion_tensor,
                     torch.zeros(pl - T, SMPL_DIM)], dim=0)
            x = motion_tensor.unsqueeze(0).to(device)
            xr, _, _ = model(x)
            return F.mse_loss(xr[:, :T], x[:, :T]).item()

    for label, ds in [("Word", word_ds), ("Sentence", sentence_ds)]:
        n = min(num_samples, len(ds))
        losses = [_sample_loss(ds[i]["motion"])
                  for i in range(n) if ds[i] is not None]
        if losses:
            print(f"  {label}-level mean recon loss: {np.mean(losses):.4f}")


# ===================================================================
# Stats computation
# ===================================================================

def compute_and_save_stats(args):
    print(f"\n[Computing Stats] Calculating from data directories...")
    datasets = []
    if os.path.exists(args.word_data_dir):
        datasets.append(WordLevelDataset(
            args.word_data_dir, None, min_seq_len=1, max_seq_len=9999))
    if os.path.exists(args.sentence_data_dir):
        if glob.glob(os.path.join(args.sentence_data_dir, "**", "*.npz"),
                      recursive=True):
            datasets.append(WordLevelDataset(
                args.sentence_data_dir, None, data_type="sentence",
                min_seq_len=1, max_seq_len=9999))
        else:
            datasets.append(SentenceLevelDataset(
                args.sentence_data_dir, None, min_seq_len=1,
                max_seq_len=9999))
    if args.val_data_dir and os.path.exists(args.val_data_dir):
        if glob.glob(os.path.join(args.val_data_dir, "**", "*.npz"),
                      recursive=True):
            datasets.append(WordLevelDataset(
                args.val_data_dir, None, data_type="sentence",
                min_seq_len=1, max_seq_len=9999))
        else:
            datasets.append(SentenceLevelDataset(
                args.val_data_dir, None, min_seq_len=1, max_seq_len=9999))

    s = torch.zeros(SMPL_DIM, dtype=torch.float64)
    sq = torch.zeros(SMPL_DIM, dtype=torch.float64)
    count = 0
    for ds in datasets:
        if len(ds) == 0:
            continue
        idx = np.random.choice(len(ds), min(2000, len(ds)), replace=False)
        for i in tqdm(idx, desc="Sampling stats"):
            sample = ds[int(i)]
            if sample is None:
                continue
            m = sample["motion"]
            s += m.sum(dim=0).double()
            sq += (m ** 2).sum(dim=0).double()
            count += m.shape[0]
    if count == 0:
        return
    mean = (s / count).float()
    std = torch.sqrt(torch.clamp(sq / count - (s / count) ** 2,
                                 min=1e-8)).float()
    torch.save({"mean": mean, "std": std}, args.stats_path)
    print(f"  Saved stats to {args.stats_path}")


# ===================================================================
# Main
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune VQ-VAE (v2) on mixed word + sentence data")
    p.add_argument("--vqvae-ckpt", required=True)
    p.add_argument("--word-data-dir", required=True)
    p.add_argument("--sentence-data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--stats-path", default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--word-ratio", type=float, default=0.2)
    p.add_argument("--vq-loss-weight", type=float, default=2.5,
                   help="(default: 2.5)")
    p.add_argument("--vel-loss-weight", type=float, default=0.5)
    p.add_argument("--hand-loss-weight", type=float, default=3.0,
                   help="(default: 3.0, 3x body weight)")
    p.add_argument("--body-loss-weight", type=float, default=1.0)
    p.add_argument("--geo-loss-weight", type=float, default=0.2)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--preload", action="store_true")
    p.add_argument("--no-preload", dest="preload", action="store_false")
    p.add_argument("--word-only", action="store_true")
    p.add_argument("--sentence-only", action="store_true")
    p.add_argument("--validate-checkpoint", action="store_true")
    p.add_argument("--gdrive-dir", default=GDRIVE_CHECKPOINT_DIR)
    p.add_argument("--val-data-dir", default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--reset-lr", action="store_true")
    p.add_argument("--lr-scheduler",
                   choices=["cosine_warmup", "cosine", "cosine_restart",
                            "cosine_restart_gradual", "constant", "step"],
                   default="cosine_warmup",
                   help="(v2 default: cosine_warmup)")
    p.add_argument("--lr-warmup-epochs", type=int, default=10)
    p.add_argument("--lr-restart-period", type=int, default=100)
    p.add_argument("--ema-mu", type=float, default=0.95,
                   help="EMA momentum for codebook (v2 default: 0.95, "
                        "original: 0.99). Lower = faster tracking.")
    p.add_argument("--codebook-reset-every", type=int, default=5,
                   help="Reset dead codebook entries every N epochs "
                        "(0 to disable)")
    p.add_argument("--codebook-reset-threshold", type=int, default=2,
                   help="Codes used fewer than this many times get reset")
    p.add_argument("--noise-sigma", type=float, default=0.005,
                   help="Augmentation noise std (v2 default: 0.005, "
                        "was 0.02)")
    p.add_argument("--zero-irrelevant", action="store_true",
                   help="Zero out shape, jaw, expression, cam_trans "
                        "before forward pass (recommended for sign language)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.stats_path and not os.path.exists(args.stats_path):
        compute_and_save_stats(args)

    mode = "validate" if args.validate_checkpoint else (
        "word-only" if args.word_only else (
            "sentence-only" if args.sentence_only else "mixed"))

    os.makedirs(args.output_dir, exist_ok=True)
    if args.gdrive_dir:
        os.makedirs(args.gdrive_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "finetune_config.json"),
              'w') as f:
        json.dump(vars(args), f, indent=2)

    resume_ckpt = (find_latest_checkpoint(args.output_dir, args.gdrive_dir)
                   if not args.no_resume and mode != "validate" else None)
    model = load_vqvae_checkpoint(args.vqvae_ckpt, VQ_CONFIG, DEVICE)

    set_quantizer_ema_mu(model, args.ema_mu)

    if args.zero_irrelevant:
        print(f"  --zero-irrelevant: stripping shape/jaw/expression/cam_trans "
              f"({SMPL_DIM - SIGN_DIM} dims removed, {SIGN_DIM} active)")
        print(f"    Input: 182 dims -> strip to {SIGN_DIM} -> "
              f"pad back to 182 (zeros) -> model -> loss on {SIGN_DIM} dims")

    # --- Datasets ---
    if args.preload:
        word_ds = (preload_word_level_data(
            args.word_data_dir, args.stats_path,
            max_seq_len=args.max_seq_len)
            if mode in ("mixed", "word-only", "validate")
            else PreloadedDataset([], "word"))
        sentence_ds = (preload_sentence_level_data(
            args.sentence_data_dir, args.stats_path,
            max_seq_len=args.max_seq_len)
            if mode in ("mixed", "sentence-only", "validate")
            else PreloadedDataset([], "sentence"))
    else:
        word_ds = (WordLevelDataset(
            args.word_data_dir, args.stats_path,
            max_seq_len=args.max_seq_len)
            if mode in ("mixed", "word-only", "validate")
            else PreloadedDataset([], "word"))
        if mode in ("mixed", "sentence-only", "validate"):
            if glob.glob(os.path.join(args.sentence_data_dir, "**", "*.npz"),
                         recursive=True):
                sentence_ds = WordLevelDataset(
                    args.sentence_data_dir, args.stats_path,
                    max_seq_len=args.max_seq_len, data_type="sentence")
            else:
                sentence_ds = SentenceLevelDataset(
                    args.sentence_data_dir, args.stats_path,
                    max_seq_len=args.max_seq_len)
        else:
            sentence_ds = PreloadedDataset([], "sentence")

    if mode == "validate":
        validate_checkpoint(model, word_ds, sentence_ds, 20, DEVICE)
        return

    if mode == "word-only":
        args.word_ratio = 1.0
    elif mode == "sentence-only":
        args.word_ratio = 0.0
    if len(word_ds) == 0 and mode != "sentence-only":
        args.word_ratio = 0.0
    if len(sentence_ds) == 0 and mode != "word-only":
        args.word_ratio = 1.0

    # --- Val split ---
    sentence_train, sentence_val = sentence_ds, None
    if args.val_data_dir:
        if glob.glob(os.path.join(args.val_data_dir, "**", "*.npz"),
                      recursive=True):
            sentence_val = WordLevelDataset(
                args.val_data_dir, args.stats_path,
                max_seq_len=args.max_seq_len, data_type="sentence")
        elif args.preload:
            sentence_val = preload_sentence_level_data(
                args.val_data_dir, args.stats_path,
                max_seq_len=args.max_seq_len)
        else:
            sentence_val = SentenceLevelDataset(
                args.val_data_dir, args.stats_path,
                max_seq_len=args.max_seq_len)
    elif len(sentence_ds) > 20:
        n_val = max(1, int(len(sentence_ds) * 0.10))
        sentence_train, sentence_val = random_split(
            sentence_ds, [len(sentence_ds) - n_val, n_val],
            generator=torch.Generator().manual_seed(42))

    word_train, word_val = word_ds, None
    if len(word_ds) > 20:
        n_val = max(1, int(len(word_ds) * 0.10))
        word_train, word_val = random_split(
            word_ds, [len(word_ds) - n_val, n_val],
            generator=torch.Generator().manual_seed(42))

    val_parts = [d for d in [word_val, sentence_val] if d is not None]
    val_dl = (create_val_dataloader(
        ConcatDataset(val_parts), batch_size=args.batch_size,
        num_workers=0 if args.preload else args.num_workers)
        if val_parts else None)

    nw = 0 if args.preload else args.num_workers
    train_dl = create_mixed_dataloader(
        word_train, sentence_train, args.word_ratio,
        args.batch_size, nw)

    trainer = FinetuneTrainer(
        model, train_dl, args.output_dir,
        learning_rate=args.learning_rate,
        vq_loss_weight=args.vq_loss_weight,
        vel_loss_weight=args.vel_loss_weight,
        hand_loss_weight=args.hand_loss_weight,
        body_loss_weight=args.body_loss_weight,
        geo_loss_weight=args.geo_loss_weight,
        gdrive_dir=args.gdrive_dir,
        scheduler_type=args.lr_scheduler,
        num_epochs=args.epochs,
        warmup_epochs=args.lr_warmup_epochs,
        restart_period=args.lr_restart_period,
        codebook_reset_every=args.codebook_reset_every,
        codebook_reset_threshold=args.codebook_reset_threshold,
        noise_sigma=args.noise_sigma,
        zero_irrelevant=args.zero_irrelevant,
        device=DEVICE)

    if resume_ckpt:
        trainer.load_checkpoint(resume_ckpt[0])
        if args.reset_lr:
            trainer.reset_learning_rate(args.learning_rate,
                                        args.lr_warmup_epochs)

    trainer.train(args.epochs, args.save_every, val_dl)


if __name__ == "__main__":
    main()
