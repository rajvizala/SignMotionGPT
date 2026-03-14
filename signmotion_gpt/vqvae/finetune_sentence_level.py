"""
finetune_vqvae_sentence_level.py

Fine-tune the word-level VQ-VAE on a mixed dataset of:
- Word-level data (20%): Preserve existing knowledge
- Sentence-level data (80%): Adapt to How2Sign sequences

This script implements Phase 1 (Option A) of the sentence-level training plan.

Usage (Standard - loads data on-demand):
    python finetune_vqvae_sentence_level.py \
        --vqvae-ckpt path/to/word_level_vqvae.pt \
        --word-data-dir path/to/npz_data \
        --sentence-data-dir path/to/how2sign \
        --stats-path path/to/stats.pt \
        --output-dir path/to/output \
        --epochs 50 \
        --word-ratio 0.2

Usage (Preload - loads ALL data into RAM first, faster for slow storage like GDrive):
    python finetune_vqvae_sentence_level.py \
        --vqvae-ckpt path/to/word_level_vqvae.pt \
        --word-data-dir path/to/npz_data \
        --sentence-data-dir path/to/how2sign \
        --stats-path path/to/stats.pt \
        --output-dir path/to/output \
        --epochs 50 \
        --word-ratio 0.2 \
        --preload
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
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Visualizations will be skipped.")

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mGPT.archs.mgpt_vq import VQVae

# =============================================================================
# Configuration
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SMPL-X parameter configuration (MUST match training order)
PARAM_DIMS = [10, 63, 45, 45, 3, 10, 3, 3]  # Total: 182
PARAM_NAMES = ["shape", "body_pose", "lhand_pose", "rhand_pose",
               "jaw_pose", "expression", "root_pose", "cam_trans"]
SMPL_DIM = sum(PARAM_DIMS)  # 182

# VQ-VAE configuration (must match your checkpoint)
VQ_CONFIG = {
    "nfeats": SMPL_DIM,
    "code_num": 512,
    "code_dim": 512,
    "output_emb_width": 512,
    "down_t": 2,       # Based on checkpoint analysis
    "stride_t": 2,
    "width": 512,
    "depth": 3,
    "dilation_growth_rate": 3,
    "activation": "relu",
    "norm": None,
    "quantizer": "ema_reset"
}

# Temporal downsampling factor (2^down_t)
DOWNSAMPLE_FACTOR = 2 ** VQ_CONFIG["down_t"]

# Google Drive checkpoint directory (for Colab)
GDRIVE_CHECKPOINT_DIR = "/content/drive/MyDrive/finetune_combine_checkpoint"

# =============================================================================
# Checkpoint Management Functions
# =============================================================================

def find_checkpoint_files(directory: str) -> List[Tuple[str, int]]:
    """
    Find all VQ-VAE checkpoint files in a directory and extract their epoch numbers.
    
    Returns:
        List of (filepath, epoch_number) tuples, sorted by epoch (highest first)
    """
    if not os.path.exists(directory):
        return []
    
    checkpoints = []
    
    # Look for files matching pattern: vqvae_finetuned_epoch_XXX.pt
    pattern = re.compile(r'vqvae_finetuned_epoch_(\d+)\.pt$')
    
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            epoch = int(match.group(1))
            filepath = os.path.join(directory, filename)
            checkpoints.append((filepath, epoch))
    
    # Sort by epoch number (highest first)
    checkpoints.sort(key=lambda x: x[1], reverse=True)
    return checkpoints


def find_latest_checkpoint(local_dir: str, gdrive_dir: str = GDRIVE_CHECKPOINT_DIR) -> Optional[Tuple[str, int]]:
    """
    Find the latest checkpoint from both local and GDrive directories.
    
    Args:
        local_dir: Local output directory
        gdrive_dir: Google Drive checkpoint directory
    
    Returns:
        Tuple of (checkpoint_path, epoch_number) or None if no checkpoint found
    """
    print("\n[Checkpoint Search]")
    
    # Search local directory
    local_checkpoints = find_checkpoint_files(local_dir)
    if local_checkpoints:
        print(f"  Local ({local_dir}): Found {len(local_checkpoints)} checkpoints")
        print(f"    Latest: epoch {local_checkpoints[0][1]}")
    else:
        print(f"  Local ({local_dir}): No checkpoints found")
    
    # Search GDrive directory
    gdrive_checkpoints = find_checkpoint_files(gdrive_dir)
    if gdrive_checkpoints:
        print(f"  GDrive ({gdrive_dir}): Found {len(gdrive_checkpoints)} checkpoints")
        print(f"    Latest: epoch {gdrive_checkpoints[0][1]}")
    else:
        print(f"  GDrive ({gdrive_dir}): No checkpoints found")
    
    # Combine and find the latest
    all_checkpoints = local_checkpoints + gdrive_checkpoints
    
    if not all_checkpoints:
        print("  No existing checkpoints found. Starting from scratch.")
        return None
    
    # Sort by epoch and return the latest
    all_checkpoints.sort(key=lambda x: x[1], reverse=True)
    latest = all_checkpoints[0]
    print(f"\n  -> Using checkpoint: {latest[0]} (epoch {latest[1]})")
    
    return latest


def delete_old_gdrive_checkpoints(gdrive_dir: str, keep_epoch: int):
    """
    Delete old checkpoints from GDrive, keeping only the specified epoch.
    
    Args:
        gdrive_dir: Google Drive checkpoint directory
        keep_epoch: The epoch number to keep (delete all others)
    """
    if not os.path.exists(gdrive_dir):
        return
    
    checkpoints = find_checkpoint_files(gdrive_dir)
    
    for filepath, epoch in checkpoints:
        if epoch != keep_epoch:
            try:
                os.remove(filepath)
                print(f"  Deleted old GDrive checkpoint: epoch {epoch}")
            except Exception as e:
                print(f"  Warning: Could not delete {filepath}: {e}")


# =============================================================================
# Model Wrapper
# =============================================================================

class VQVAEWrapper(nn.Module):
    """Wrapper around VQVae with loss weight computation."""
    
    def __init__(self, config: dict):
        super().__init__()
        self.vqvae = VQVae(**config)
        
        # Loss weights for different parameter groups
        param_starts = np.cumsum([0] + PARAM_DIMS[:-1]).tolist()
        loss_weights = torch.ones(SMPL_DIM)
        
        # Higher weight for pose parameters (body, hands)
        loss_weights[param_starts[1]:param_starts[5]] = 10.0  # body + hands
        loss_weights[param_starts[0]:param_starts[1]] = 5.0   # shape
        loss_weights[param_starts[5]:param_starts[6]] = 8.0   # expression
        
        self.register_buffer('loss_weights', loss_weights)
    
    def forward(self, x):
        """Forward pass returning reconstruction, VQ loss, and perplexity."""
        return self.vqvae(x)
    
    def encode(self, x):
        """Encode input to discrete codes."""
        return self.vqvae.encode(x)
    
    def decode(self, codes):
        """Decode discrete codes to motion."""
        return self.vqvae.decode(codes)


def load_vqvae_checkpoint(checkpoint_path: str, config: dict, device=DEVICE) -> VQVAEWrapper:
    """Load VQ-VAE from checkpoint."""
    print(f"\nLoading VQ-VAE checkpoint from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
    
    # Check if checkpoint has 'vqvae.' prefix
    has_vqvae_prefix = any(k.startswith('vqvae.') for k in state_dict.keys())
    
    # Create model
    model = VQVAEWrapper(config).to(device)
    
    if has_vqvae_prefix:
        # Keys have 'vqvae.' prefix - load with wrapper structure
        # Remove 'vqvae.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('vqvae.'):
                new_state_dict[k[6:]] = v  # Remove 'vqvae.' prefix
            else:
                new_state_dict[k] = v
        result = model.vqvae.load_state_dict(new_state_dict, strict=False)
    else:
        # Raw VQVae checkpoint - load directly into vqvae submodule
        result = model.vqvae.load_state_dict(state_dict, strict=False)
    
    print(f"  Loaded checkpoint - Missing: {len(result.missing_keys)}, Unexpected: {len(result.unexpected_keys)}")
    if result.missing_keys:
        print(f"    Missing keys (first 5): {result.missing_keys[:5]}")
    if result.unexpected_keys:
        print(f"    Unexpected keys (first 5): {result.unexpected_keys[:5]}")
    
    return model


# =============================================================================
# Dataset Classes
# =============================================================================

class WordLevelDataset(Dataset):
    """
    Dataset for word-level motion data (NPZ format).
    
    Expected NPZ structure:
    - 'motion': (T, 182) array of SMPL-X parameters
    """
    
    def __init__(self, root_dir: str, stats_path: Optional[str] = None,
                 min_seq_len: int = 16, max_seq_len: int = 256):
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        
        print(f"\n[WordLevelDataset] Loading from: {root_dir}")
        
        # Find all NPZ files
        glob_pattern = os.path.join(root_dir, "**", "*.npz")
        self.files = glob.glob(glob_pattern, recursive=True)
        
        if not self.files:
            print(f"  Warning: No .npz files found in {root_dir}")
        else:
            print(f"  Found {len(self.files)} NPZ files")
        
        # Load normalization stats
        self.mean, self.std = self._load_stats(stats_path)
    
    def _load_stats(self, stats_path: Optional[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load normalization statistics."""
        if stats_path and os.path.exists(stats_path):
            stats = torch.load(stats_path, map_location='cpu')
            mean = stats.get('mean', torch.zeros(SMPL_DIM))
            std = stats.get('std', torch.ones(SMPL_DIM))
            print(f"  Loaded stats from {stats_path}")
        else:
            print(f"  Warning: Stats not found, using default (mean=0, std=1)")
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
                motion = data['motion']  # (T, 182)
        except Exception as e:
            print(f"Error loading {self.files[idx]}: {e}")
            return None
        
        # Check sequence length
        if motion.shape[0] < self.min_seq_len:
            return None
        
        # Truncate if too long
        if motion.shape[0] > self.max_seq_len:
            motion = motion[:self.max_seq_len]
        
        # Convert to tensor and normalize
        motion_tensor = torch.tensor(motion, dtype=torch.float32)
        motion_normalized = (motion_tensor - self.mean) / (self.std + 1e-8)
        
        return {
            "motion": motion_normalized,
            "type": "word",
            "source": "npz"
        }


class SentenceLevelDataset(Dataset):
    """
    Dataset for sentence-level motion data (How2Sign PKL format).
    
    Expected structure:
    - root_dir/sequence_folder/*_3D.pkl files
    - Each PKL contains: smplx_shape, smplx_body_pose, etc.
    """
    
    def __init__(self, root_dir: str, stats_path: Optional[str] = None,
                 min_seq_len: int = 32, max_seq_len: int = 512):
        self.root_dir = root_dir
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        
        print(f"\n[SentenceLevelDataset] Loading from: {root_dir}")
        
        # Find all sequence folders
        self.sequence_folders = self._find_sequence_folders(root_dir)
        print(f"  Found {len(self.sequence_folders)} sequence folders")
        
        # Load normalization stats
        self.mean, self.std = self._load_stats(stats_path)
    
    def _find_sequence_folders(self, root_dir: str) -> List[str]:
        """Find all folders containing *_3D.pkl files."""
        sequence_folders = []
        
        if not os.path.exists(root_dir):
            print(f"  Warning: Directory not found: {root_dir}")
            return sequence_folders
        
        # Check if root_dir itself contains PKL files
        pkl_files = glob.glob(os.path.join(root_dir, "*_3D.pkl"))
        if pkl_files:
            sequence_folders.append(root_dir)
        
        # Check subdirectories
        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)
            if os.path.isdir(item_path):
                pkl_files = glob.glob(os.path.join(item_path, "*_3D.pkl"))
                if pkl_files:
                    sequence_folders.append(item_path)
        
        return sorted(sequence_folders)
    
    def _load_stats(self, stats_path: Optional[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load normalization statistics."""
        if stats_path and os.path.exists(stats_path):
            stats = torch.load(stats_path, map_location='cpu')
            mean = stats.get('mean', torch.zeros(SMPL_DIM))
            std = stats.get('std', torch.ones(SMPL_DIM))
            print(f"  Loaded stats from {stats_path}")
        else:
            print(f"  Warning: Stats not found, using default (mean=0, std=1)")
            mean = torch.zeros(SMPL_DIM)
            std = torch.ones(SMPL_DIM)
        
        if not torch.is_tensor(mean):
            mean = torch.tensor(mean)
        if not torch.is_tensor(std):
            std = torch.tensor(std)
        
        return mean.float(), std.float()
    
    def _load_sequence_from_folder(self, folder_path: str) -> Optional[np.ndarray]:
        """Load and aggregate all frames from a sequence folder."""
        import re
        
        # Find all *_3D.pkl files
        pkl_files = glob.glob(os.path.join(folder_path, "*_3D.pkl"))
        if not pkl_files:
            return None
        
        # Sort by frame number
        def extract_frame_num(filepath):
            fname = os.path.basename(filepath)
            match = re.search(r'_(\d+)_3D\.pkl$', fname)
            return int(match.group(1)) if match else 0
        
        pkl_files.sort(key=extract_frame_num)
        
        # Aggregate frames
        aggregated = defaultdict(list)
        for f in pkl_files:
            try:
                with open(f, 'rb') as fp:
                    data = pickle.load(fp)
                for k, v in data.items():
                    if torch.is_tensor(v):
                        v = v.cpu().numpy()
                    aggregated[k].append(v)
            except Exception as e:
                continue
        
        if not aggregated:
            return None
        
        # Convert to 182-dim vector (MUST match training order!)
        try:
            # How2Sign uses 'smplx_' prefix
            parts = []
            
            # 1. Shape (10 dims)
            shape = np.array(aggregated.get('smplx_shape', []))
            if shape.ndim == 3 and shape.shape[1] == 1:
                shape = shape.squeeze(1)
            if shape.shape[-1] != 10:
                shape = shape[..., :10] if shape.shape[-1] > 10 else np.pad(shape, ((0,0), (0, 10-shape.shape[-1])))
            parts.append(shape)
            
            # 2. Body pose (63 dims)
            body_pose = np.array(aggregated.get('smplx_body_pose', []))
            if body_pose.ndim == 3 and body_pose.shape[1] == 1:
                body_pose = body_pose.squeeze(1)
            parts.append(body_pose)
            
            # 3. Left hand (45 dims)
            lhand = np.array(aggregated.get('smplx_lhand_pose', []))
            if lhand.ndim == 3 and lhand.shape[1] == 1:
                lhand = lhand.squeeze(1)
            parts.append(lhand)
            
            # 4. Right hand (45 dims)
            rhand = np.array(aggregated.get('smplx_rhand_pose', []))
            if rhand.ndim == 3 and rhand.shape[1] == 1:
                rhand = rhand.squeeze(1)
            parts.append(rhand)
            
            # 5. Jaw pose (3 dims)
            jaw = np.array(aggregated.get('smplx_jaw_pose', []))
            if jaw.ndim == 3 and jaw.shape[1] == 1:
                jaw = jaw.squeeze(1)
            if len(jaw) == 0 or jaw.shape[-1] == 0:
                jaw = np.zeros((len(body_pose), 3), dtype=np.float32)
            parts.append(jaw)
            
            # 6. Expression (10 dims)
            expr = np.array(aggregated.get('smplx_expr', []))
            if expr.ndim == 3 and expr.shape[1] == 1:
                expr = expr.squeeze(1)
            if len(expr) == 0 or expr.shape[-1] == 0:
                expr = np.zeros((len(body_pose), 10), dtype=np.float32)
            elif expr.shape[-1] > 10:
                expr = expr[..., :10]
            parts.append(expr)
            
            # 7. Root pose (3 dims)
            root = np.array(aggregated.get('smplx_root_pose', []))
            if root.ndim == 3 and root.shape[1] == 1:
                root = root.squeeze(1)
            if len(root) == 0 or root.shape[-1] == 0:
                root = np.zeros((len(body_pose), 3), dtype=np.float32)
            parts.append(root)
            
            # 8. Camera translation (3 dims)
            trans = np.array(aggregated.get('cam_trans', []))
            if trans.ndim == 3 and trans.shape[1] == 1:
                trans = trans.squeeze(1)
            if len(trans) == 0 or trans.shape[-1] == 0:
                trans = np.zeros((len(body_pose), 3), dtype=np.float32)
            parts.append(trans)
            
            # Concatenate all parts
            full_vec = np.concatenate([p.astype(np.float32) for p in parts], axis=1)
            
            if full_vec.shape[1] != SMPL_DIM:
                print(f"  Warning: Expected {SMPL_DIM} dims, got {full_vec.shape[1]}")
                return None
            
            return full_vec
            
        except Exception as e:
            return None
    
    def __len__(self):
        return len(self.sequence_folders)
    
    def __getitem__(self, idx):
        folder = self.sequence_folders[idx]
        motion = self._load_sequence_from_folder(folder)
        
        if motion is None:
            return None
        
        # Check sequence length
        if motion.shape[0] < self.min_seq_len:
            return None
        
        # Truncate if too long
        if motion.shape[0] > self.max_seq_len:
            motion = motion[:self.max_seq_len]
        
        # Convert to tensor and normalize
        motion_tensor = torch.tensor(motion, dtype=torch.float32)
        motion_normalized = (motion_tensor - self.mean) / (self.std + 1e-8)
        
        return {
            "motion": motion_normalized,
            "type": "sentence",
            "source": "how2sign"
        }


# =============================================================================
# Preloaded Dataset Classes (for faster training with slow storage like GDrive)
# =============================================================================

class PreloadedDataset(Dataset):
    """
    A dataset that holds all data in memory.
    Use this when loading from slow storage (GDrive, network drives).
    
    All data is loaded upfront, making training iterations much faster.
    """
    
    def __init__(self, data: List[Dict], data_type: str = "preloaded"):
        """
        Args:
            data: List of dictionaries with 'motion' tensors
            data_type: Label for this dataset type
        """
        self.data = data
        self.data_type = data_type
        print(f"[PreloadedDataset] Loaded {len(self.data)} samples into memory")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def preload_word_level_data(
    root_dir: str,
    stats_path: Optional[str] = None,
    min_seq_len: int = 16,
    max_seq_len: int = 256,
    show_progress: bool = True
) -> PreloadedDataset:
    """
    Load ALL word-level NPZ files into memory.
    
    Args:
        root_dir: Directory containing NPZ files
        stats_path: Path to normalization stats
        min_seq_len: Minimum sequence length
        max_seq_len: Maximum sequence length
        show_progress: Show progress bar
    
    Returns:
        PreloadedDataset with all valid samples in memory
    """
    print(f"\n{'='*60}")
    print(f"  PRELOADING WORD-LEVEL DATA INTO MEMORY")
    print(f"{'='*60}")
    print(f"  Source: {root_dir}")
    
    # Load stats
    if stats_path and os.path.exists(stats_path):
        stats = torch.load(stats_path, map_location='cpu')
        mean = stats.get('mean', torch.zeros(SMPL_DIM))
        std = stats.get('std', torch.ones(SMPL_DIM))
        if not torch.is_tensor(mean):
            mean = torch.tensor(mean)
        if not torch.is_tensor(std):
            std = torch.tensor(std)
        mean = mean.float()
        std = std.float()
        print(f"  Loaded stats from {stats_path}")
    else:
        print(f"  Warning: Stats not found, using default (mean=0, std=1)")
        mean = torch.zeros(SMPL_DIM)
        std = torch.ones(SMPL_DIM)
    
    # Find all NPZ files
    glob_pattern = os.path.join(root_dir, "**", "*.npz")
    files = glob.glob(glob_pattern, recursive=True)
    print(f"  Found {len(files)} NPZ files")
    
    if not files:
        print(f"  Warning: No files found!")
        return PreloadedDataset([], "word")
    
    # Load all files
    loaded_data = []
    skipped_short = 0
    skipped_error = 0
    
    iterator = tqdm(files, desc="Loading word-level data") if show_progress else files
    
    for file_path in iterator:
        try:
            with np.load(file_path) as data:
                motion = data['motion']  # (T, 182)
            
            # Check sequence length
            if motion.shape[0] < min_seq_len:
                skipped_short += 1
                continue
            
            # Truncate if too long
            if motion.shape[0] > max_seq_len:
                motion = motion[:max_seq_len]
            
            # Convert to tensor and normalize
            motion_tensor = torch.tensor(motion, dtype=torch.float32)
            motion_normalized = (motion_tensor - mean) / (std + 1e-8)
            
            loaded_data.append({
                "motion": motion_normalized,
                "type": "word",
                "source": "npz"
            })
            
        except Exception as e:
            skipped_error += 1
            continue
    
    print(f"\n  Loaded: {len(loaded_data)} samples")
    print(f"  Skipped (too short): {skipped_short}")
    print(f"  Skipped (errors): {skipped_error}")
    print(f"  Memory usage: ~{sum(d['motion'].numel() * 4 for d in loaded_data) / 1024 / 1024:.1f} MB")
    print(f"{'='*60}\n")
    
    return PreloadedDataset(loaded_data, "word")


def preload_sentence_level_data(
    root_dir: str,
    stats_path: Optional[str] = None,
    min_seq_len: int = 32,
    max_seq_len: int = 512,
    show_progress: bool = True
) -> PreloadedDataset:
    """
    Load ALL sentence-level How2Sign data into memory.
    
    Args:
        root_dir: Directory containing sequence folders
        stats_path: Path to normalization stats
        min_seq_len: Minimum sequence length
        max_seq_len: Maximum sequence length
        show_progress: Show progress bar
    
    Returns:
        PreloadedDataset with all valid samples in memory
    """
    import re
    
    print(f"\n{'='*60}")
    print(f"  PRELOADING SENTENCE-LEVEL DATA INTO MEMORY")
    print(f"{'='*60}")
    print(f"  Source: {root_dir}")
    
    # Load stats
    if stats_path and os.path.exists(stats_path):
        stats = torch.load(stats_path, map_location='cpu')
        mean = stats.get('mean', torch.zeros(SMPL_DIM))
        std = stats.get('std', torch.ones(SMPL_DIM))
        if not torch.is_tensor(mean):
            mean = torch.tensor(mean)
        if not torch.is_tensor(std):
            std = torch.tensor(std)
        mean = mean.float()
        std = std.float()
        print(f"  Loaded stats from {stats_path}")
    else:
        print(f"  Warning: Stats not found, using default (mean=0, std=1)")
        mean = torch.zeros(SMPL_DIM)
        std = torch.ones(SMPL_DIM)
    
    # Find all sequence folders
    sequence_folders = []
    
    if not os.path.exists(root_dir):
        print(f"  Warning: Directory not found!")
        return PreloadedDataset([], "sentence")
    
    # Check if root_dir itself contains PKL files
    pkl_files = glob.glob(os.path.join(root_dir, "*_3D.pkl"))
    if pkl_files:
        sequence_folders.append(root_dir)
    
    # Check subdirectories
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            pkl_files = glob.glob(os.path.join(item_path, "*_3D.pkl"))
            if pkl_files:
                sequence_folders.append(item_path)
    
    sequence_folders = sorted(sequence_folders)
    print(f"  Found {len(sequence_folders)} sequence folders")
    
    if not sequence_folders:
        print(f"  Warning: No sequence folders found!")
        return PreloadedDataset([], "sentence")
    
    # Helper function to load a sequence
    def load_sequence_from_folder(folder_path: str) -> Optional[np.ndarray]:
        pkl_files = glob.glob(os.path.join(folder_path, "*_3D.pkl"))
        if not pkl_files:
            return None
        
        # Sort by frame number
        def extract_frame_num(filepath):
            fname = os.path.basename(filepath)
            match = re.search(r'_(\d+)_3D\.pkl$', fname)
            return int(match.group(1)) if match else 0
        
        pkl_files.sort(key=extract_frame_num)
        
        # Aggregate frames
        aggregated = defaultdict(list)
        for f in pkl_files:
            try:
                with open(f, 'rb') as fp:
                    data = pickle.load(fp)
                for k, v in data.items():
                    if torch.is_tensor(v):
                        v = v.cpu().numpy()
                    aggregated[k].append(v)
            except Exception:
                continue
        
        if not aggregated:
            return None
        
        try:
            parts = []
            
            # 1. Shape (10 dims)
            shape = np.array(aggregated.get('smplx_shape', []))
            if shape.ndim == 3 and shape.shape[1] == 1:
                shape = shape.squeeze(1)
            if len(shape) == 0:
                return None
            if shape.shape[-1] != 10:
                if shape.shape[-1] > 10:
                    shape = shape[..., :10]
                else:
                    shape = np.pad(shape, ((0,0), (0, 10-shape.shape[-1])))
            parts.append(shape)
            
            # 2. Body pose (63 dims)
            body_pose = np.array(aggregated.get('smplx_body_pose', []))
            if body_pose.ndim == 3 and body_pose.shape[1] == 1:
                body_pose = body_pose.squeeze(1)
            if len(body_pose) == 0:
                return None
            parts.append(body_pose)
            
            T = len(body_pose)
            
            # 3. Left hand (45 dims)
            lhand = np.array(aggregated.get('smplx_lhand_pose', []))
            if lhand.ndim == 3 and lhand.shape[1] == 1:
                lhand = lhand.squeeze(1)
            if len(lhand) == 0:
                lhand = np.zeros((T, 45), dtype=np.float32)
            parts.append(lhand)
            
            # 4. Right hand (45 dims)
            rhand = np.array(aggregated.get('smplx_rhand_pose', []))
            if rhand.ndim == 3 and rhand.shape[1] == 1:
                rhand = rhand.squeeze(1)
            if len(rhand) == 0:
                rhand = np.zeros((T, 45), dtype=np.float32)
            parts.append(rhand)
            
            # 5. Jaw pose (3 dims)
            jaw = np.array(aggregated.get('smplx_jaw_pose', []))
            if jaw.ndim == 3 and jaw.shape[1] == 1:
                jaw = jaw.squeeze(1)
            if len(jaw) == 0 or jaw.shape[-1] == 0:
                jaw = np.zeros((T, 3), dtype=np.float32)
            parts.append(jaw)
            
            # 6. Expression (10 dims)
            expr = np.array(aggregated.get('smplx_expr', []))
            if expr.ndim == 3 and expr.shape[1] == 1:
                expr = expr.squeeze(1)
            if len(expr) == 0 or expr.shape[-1] == 0:
                expr = np.zeros((T, 10), dtype=np.float32)
            elif expr.shape[-1] > 10:
                expr = expr[..., :10]
            parts.append(expr)
            
            # 7. Root pose (3 dims)
            root = np.array(aggregated.get('smplx_root_pose', []))
            if root.ndim == 3 and root.shape[1] == 1:
                root = root.squeeze(1)
            if len(root) == 0 or root.shape[-1] == 0:
                root = np.zeros((T, 3), dtype=np.float32)
            parts.append(root)
            
            # 8. Camera translation (3 dims)
            trans = np.array(aggregated.get('cam_trans', []))
            if trans.ndim == 3 and trans.shape[1] == 1:
                trans = trans.squeeze(1)
            if len(trans) == 0 or trans.shape[-1] == 0:
                trans = np.zeros((T, 3), dtype=np.float32)
            parts.append(trans)
            
            # Concatenate all parts
            full_vec = np.concatenate([p.astype(np.float32) for p in parts], axis=1)
            
            if full_vec.shape[1] != SMPL_DIM:
                return None
            
            return full_vec
            
        except Exception:
            return None
    
    # Load all sequences
    loaded_data = []
    skipped_short = 0
    skipped_error = 0
    
    iterator = tqdm(sequence_folders, desc="Loading sentence-level data") if show_progress else sequence_folders
    
    for folder in iterator:
        try:
            motion = load_sequence_from_folder(folder)
            
            if motion is None:
                skipped_error += 1
                continue
            
            # Check sequence length
            if motion.shape[0] < min_seq_len:
                skipped_short += 1
                continue
            
            # Truncate if too long
            if motion.shape[0] > max_seq_len:
                motion = motion[:max_seq_len]
            
            # Convert to tensor and normalize
            motion_tensor = torch.tensor(motion, dtype=torch.float32)
            motion_normalized = (motion_tensor - mean) / (std + 1e-8)
            
            loaded_data.append({
                "motion": motion_normalized,
                "type": "sentence",
                "source": "how2sign"
            })
            
        except Exception:
            skipped_error += 1
            continue
    
    print(f"\n  Loaded: {len(loaded_data)} samples")
    print(f"  Skipped (too short): {skipped_short}")
    print(f"  Skipped (errors): {skipped_error}")
    print(f"  Memory usage: ~{sum(d['motion'].numel() * 4 for d in loaded_data) / 1024 / 1024:.1f} MB")
    print(f"{'='*60}\n")
    
    return PreloadedDataset(loaded_data, "sentence")


# =============================================================================
# Data Loading Utilities
# =============================================================================

def collate_fn_mixed(batch):
    """
    Collate function for mixed batch of word and sentence level data.
    
    Handles:
    - Filtering None samples
    - Padding to same length (multiple of DOWNSAMPLE_FACTOR)
    - Tracking data types for logging
    """
    # Filter out None samples
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    # Sort by sequence length (longest first)
    batch.sort(key=lambda x: x["motion"].shape[0], reverse=True)
    
    # Determine padded length (multiple of DOWNSAMPLE_FACTOR)
    max_len = batch[0]["motion"].shape[0]
    padded_len = math.ceil(max_len / DOWNSAMPLE_FACTOR) * DOWNSAMPLE_FACTOR
    
    # Create padded tensor and length tracker
    batch_size = len(batch)
    padded_motion = torch.zeros(batch_size, padded_len, SMPL_DIM)
    lengths = []
    types = []
    
    for i, item in enumerate(batch):
        seq_len = item["motion"].shape[0]
        padded_motion[i, :seq_len, :] = item["motion"]
        lengths.append(seq_len)
        types.append(item.get("type", "unknown"))
    
    return {
        "motion": padded_motion,
        "lengths": torch.tensor(lengths),
        "types": types
    }


def create_mixed_dataloader(
    word_dataset: Dataset,
    sentence_dataset: Dataset,
    word_ratio: float = 0.2,
    batch_size: int = 8,
    num_workers: int = 0
) -> DataLoader:
    """
    Create a dataloader that samples from both datasets with specified ratio.
    
    Args:
        word_dataset: Word-level dataset
        sentence_dataset: Sentence-level dataset
        word_ratio: Fraction of word-level samples (0.2 = 20%)
        batch_size: Batch size
        num_workers: Number of data loading workers
    
    Returns:
        DataLoader with mixed sampling
    """
    # Combine datasets
    combined_dataset = ConcatDataset([word_dataset, sentence_dataset])
    
    # Create weights for sampling
    word_len = len(word_dataset)
    sentence_len = len(sentence_dataset)
    total_len = word_len + sentence_len
    
    # Calculate weights to achieve desired ratio
    # word_ratio samples from word dataset, (1-word_ratio) from sentence
    word_weight = word_ratio / word_len if word_len > 0 else 0
    sentence_weight = (1 - word_ratio) / sentence_len if sentence_len > 0 else 0
    
    weights = [word_weight] * word_len + [sentence_weight] * sentence_len
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=total_len,
        replacement=True
    )
    
    print(f"\n[Mixed DataLoader]")
    print(f"  Word-level samples: {word_len}")
    print(f"  Sentence-level samples: {sentence_len}")
    print(f"  Word ratio: {word_ratio:.1%}")
    print(f"  Expected word samples per epoch: ~{int(total_len * word_ratio)}")
    print(f"  Expected sentence samples per epoch: ~{int(total_len * (1 - word_ratio))}")
    
    return DataLoader(
        combined_dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn_mixed,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )


# =============================================================================
# Training Functions
# =============================================================================

class FinetuneTrainer:
    """Trainer class for fine-tuning VQ-VAE."""
    
    def __init__(
        self,
        model: VQVAEWrapper,
        dataloader: DataLoader,
        output_dir: str,
        learning_rate: float = 1e-5,
        vq_loss_weight: float = 0.25,
        vel_loss_weight: float = 0.5,
        hand_loss_weight: float = 0.5,
        grad_clip: float = 1.0,
        gdrive_dir: str = GDRIVE_CHECKPOINT_DIR,
        scheduler_type: str = "constant",
        num_epochs: int = 100,
        restart_period: int = 100,
        device=DEVICE
    ):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.output_dir = output_dir
        self.gdrive_dir = gdrive_dir
        self.device = device
        self.vq_loss_weight = vq_loss_weight
        self.vel_loss_weight = vel_loss_weight
        self.hand_loss_weight = hand_loss_weight
        self.grad_clip = grad_clip
        self.learning_rate = learning_rate
        self.scheduler_type = scheduler_type
        self.num_epochs = num_epochs
        self.restart_period = restart_period
        self.start_epoch = 1  # Will be updated if resuming from checkpoint
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        if gdrive_dir:
            os.makedirs(gdrive_dir, exist_ok=True)
        
        # Optimizer
        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Scheduler (create based on type)
        self.scheduler = self._create_scheduler(scheduler_type, num_epochs, restart_period)
        
        # Loss function
        self.recon_loss_fn = nn.SmoothL1Loss(reduction='none')
        
        # Metrics tracking
        self.history = {
            "epoch": [],
            "total_loss": [],
            "recon_loss": [],
            "vq_loss": [],
            "vel_loss": [],
            "hand_loss": [],
            "perplexity": [],
            "word_loss": [],
            "sentence_loss": [],
            "codebook_usage": []
        }
    
    def _create_scheduler(self, scheduler_type: str, num_epochs: int, restart_period: int = 50):
        """Create learning rate scheduler based on type."""
        if scheduler_type == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        elif scheduler_type == "cosine_restart":
            # Cosine annealing with warm restarts
            # WARNING: Can cause loss spikes when LR restarts!
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            return CosineAnnealingWarmRestarts(
                self.optimizer, T_0=restart_period, T_mult=1, eta_min=1e-6
            )
        elif scheduler_type == "cosine_restart_gradual":
            # Cosine with restarts, but each cycle is 2x longer (less frequent spikes)
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            return CosineAnnealingWarmRestarts(
                self.optimizer, T_0=restart_period, T_mult=2, eta_min=1e-6
            )
        elif scheduler_type == "constant":
            # Constant learning rate (no decay) - BEST FOR LATE-STAGE TRAINING
            from torch.optim.lr_scheduler import LambdaLR
            return LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1.0)
        elif scheduler_type == "step":
            # Step decay: reduce LR by 0.5 every restart_period epochs
            from torch.optim.lr_scheduler import StepLR
            return StepLR(self.optimizer, step_size=restart_period, gamma=0.5)
        else:
            # Default to constant for stability
            from torch.optim.lr_scheduler import LambdaLR
            return LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1.0)
    
    def reset_learning_rate(self, new_lr: Optional[float] = None, warmup_epochs: int = 5):
        """
        Reset the learning rate and scheduler. Use this when resuming training
        if the LR has decayed to zero.
        
        Args:
            new_lr: New learning rate (default: use original LR)
            warmup_epochs: Number of epochs to warm up to full LR
        """
        lr = new_lr if new_lr is not None else self.learning_rate
        
        print(f"\n[Resetting Learning Rate]")
        print(f"Old LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        print(f"New LR: {lr:.2e}")
        print(f"Scheduler: {self.scheduler_type}")
        print(f"Restart Period: {self.restart_period} epochs")
        
        # IMPORTANT: Set LR BEFORE creating scheduler so scheduler stores correct base_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            # Also update initial_lr which some schedulers use as base
            param_group['initial_lr'] = lr
        
        # Create new scheduler (will use current LR as base)
        self.scheduler = self._create_scheduler(self.scheduler_type, self.num_epochs, self.restart_period)
        
        # IMPORTANT: Set LR AGAIN after scheduler creation to ensure it sticks
        # Some schedulers may modify LR during initialization
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        print(f"  Current LR after reset: {self.optimizer.param_groups[0]['lr']:.2e}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load model, optimizer, and scheduler states from a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        
        Returns:
            The epoch number from the checkpoint (training will resume from epoch+1)
        """
        print(f"\n[Loading Checkpoint]")
        print(f"  Path: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            print(f"  Warning: Checkpoint not found!")
            return 0
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Loaded model state")
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"  Loaded optimizer state")
        
        # Load scheduler state (with error handling for scheduler type mismatch)
        if 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"  Loaded scheduler state")
            except (KeyError, TypeError) as e:
                print(f"  Warning: Could not load scheduler state (scheduler type mismatch)")
                print(f"    This is normal when changing scheduler types (e.g., cosine_restart -> constant)")
                print(f"    A new scheduler will be used instead.")
        
        # Load history if available
        if 'history' in checkpoint:
            loaded_history = checkpoint['history']
            # Merge with current history structure (handles new keys like vel_loss, hand_loss)
            for key in self.history.keys():
                if key in loaded_history:
                    self.history[key] = loaded_history[key]
                else:
                    # New key not in checkpoint - fill with zeros for past epochs
                    num_past_epochs = len(loaded_history.get('epoch', []))
                    self.history[key] = [0.0] * num_past_epochs
                    print(f"  Note: Added new metric '{key}' (filled with 0s for past epochs)")
            print(f"  Loaded training history ({len(self.history['epoch'])} epochs)")
        
        # Get epoch number
        epoch = checkpoint.get('epoch', 0)
        self.start_epoch = epoch + 1
        
        # Print metrics from checkpoint
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            print(f"  Checkpoint metrics:")
            print(f"    Total Loss: {metrics.get('total_loss', 'N/A')}")
            print(f"    Recon Loss: {metrics.get('recon_loss', 'N/A')}")
        
        print(f"  Will resume from epoch {self.start_epoch}")
        return epoch
    
    def compute_loss(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Compute reconstruction, VQ, velocity, and hand-specific losses.
        
        Loss components:
        - recon_loss: MSE reconstruction loss (weighted by dimension)
        - vq_loss: Vector quantization commitment loss
        - vel_loss: Velocity (temporal derivative) loss for smooth motion
        - hand_loss: Extra weight on hand reconstruction (critical for sign language)
        """
        motion = batch["motion"].to(self.device)
        lengths = batch["lengths"]
        types = batch["types"]
        
        # Forward pass
        x_recon, vq_loss, perplexity = self.model(motion)
        
        # Create mask for valid frames
        mask = torch.zeros_like(motion[:, :, 0], device=self.device)
        for i, length in enumerate(lengths):
            mask[i, :length] = 1.0
        
        # ============================================================
        # 1. Position Reconstruction Loss (original)
        # ============================================================
        recon_loss_raw = self.recon_loss_fn(x_recon, motion) * self.model.loss_weights
        mask_expanded = mask.unsqueeze(-1).expand_as(recon_loss_raw)
        recon_loss = (recon_loss_raw * mask_expanded).sum() / mask_expanded.sum()
        
        # ============================================================
        # 2. Velocity Loss (temporal smoothness)
        # ============================================================
        # Compute velocities (temporal derivatives)
        vel_target = motion[:, 1:, :] - motion[:, :-1, :]  # (B, T-1, C)
        vel_recon = x_recon[:, 1:, :] - x_recon[:, :-1, :]
        
        # Velocity mask (one frame shorter)
        vel_mask = mask[:, 1:].unsqueeze(-1).expand_as(vel_target)
        
        vel_loss_raw = (vel_recon - vel_target) ** 2
        vel_loss = (vel_loss_raw * vel_mask).sum() / (vel_mask.sum() + 1e-8)
        
        # ============================================================
        # 3. Hand-Specific Loss (critical for sign language)
        # ============================================================
        # SMPL-X parameter indices:
        # lhand_pose: 73:118 (45 dims)
        # rhand_pose: 118:163 (45 dims)
        hand_target = motion[:, :, 73:163]  # Both hands: 90 dims
        hand_recon = x_recon[:, :, 73:163]
        
        hand_mask = mask.unsqueeze(-1).expand(-1, -1, 90)
        hand_loss_raw = (hand_recon - hand_target) ** 2
        hand_loss = (hand_loss_raw * hand_mask).sum() / (hand_mask.sum() + 1e-8)
        
        # ============================================================
        # 4. Combined Total Loss
        # ============================================================
        # Loss weights (tunable)
        vel_weight = getattr(self, 'vel_loss_weight', 0.5)
        hand_weight = getattr(self, 'hand_loss_weight', 0.5)
        
        total_loss = (
            recon_loss + 
            self.vq_loss_weight * vq_loss +
            vel_weight * vel_loss +
            hand_weight * hand_loss
        )
        
        # ============================================================
        # Per-type losses for monitoring
        # ============================================================
        word_mask_type = torch.tensor([1.0 if t == "word" else 0.0 for t in types], device=self.device)
        sentence_mask_type = 1.0 - word_mask_type
        
        # Compute per-sample losses (position reconstruction only for comparison)
        per_sample_loss = (recon_loss_raw * mask_expanded).sum(dim=(1, 2)) / (mask_expanded.sum(dim=(1, 2)) + 1e-8)
        
        word_loss = (per_sample_loss * word_mask_type).sum() / (word_mask_type.sum() + 1e-8)
        sentence_loss = (per_sample_loss * sentence_mask_type).sum() / (sentence_mask_type.sum() + 1e-8)
        
        metrics = {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "vq_loss": vq_loss.item(),
            "vel_loss": vel_loss.item(),
            "hand_loss": hand_loss.item(),
            "perplexity": perplexity.item(),
            "word_loss": word_loss.item(),
            "sentence_loss": sentence_loss.item()
        }
        
        return total_loss, metrics
    
    def compute_codebook_usage(self, num_batches: int = 50) -> float:
        """Compute codebook utilization percentage."""
        self.model.eval()
        all_codes = []
        
        with torch.no_grad():
            for i, batch in enumerate(self.dataloader):
                if i >= num_batches or batch is None:
                    break
                
                motion = batch["motion"].to(self.device)
                codes, _ = self.model.encode(motion)
                all_codes.append(codes.cpu().numpy().flatten())
        
        self.model.train()
        
        if not all_codes:
            return 0.0
        
        all_codes = np.concatenate(all_codes)
        unique_codes = len(np.unique(all_codes))
        total_codes = VQ_CONFIG["code_num"]
        
        return (unique_codes / total_codes) * 100
    
    def train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch."""
        self.model.train()
        
        epoch_metrics = defaultdict(list)
        type_counts = {"word": 0, "sentence": 0}
        
        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue
            
            # Update type counts
            for t in batch["types"]:
                if t in type_counts:
                    type_counts[t] += 1
            
            # Forward and backward
            self.optimizer.zero_grad()
            loss, metrics = self.compute_loss(batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            # Track metrics
            for k, v in metrics.items():
                epoch_metrics[k].append(v)
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{metrics['total_loss']:.4f}",
                "recon": f"{metrics['recon_loss']:.4f}",
                "vel": f"{metrics['vel_loss']:.4f}",
                "hand": f"{metrics['hand_loss']:.4f}",
                "ppl": f"{metrics['perplexity']:.1f}"
            })
        
        # Step scheduler
        self.scheduler.step()
        
        # Compute epoch averages
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        avg_metrics["word_count"] = type_counts["word"]
        avg_metrics["sentence_count"] = type_counts["sentence"]
        
        # Compute codebook usage
        avg_metrics["codebook_usage"] = self.compute_codebook_usage()
        
        return avg_metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict, save_to_gdrive: bool = True):
        """
        Save model checkpoint to local directory and optionally to GDrive.
        
        Args:
            epoch: Current epoch number
            metrics: Training metrics dictionary
            save_to_gdrive: Whether to also save to GDrive (and delete old checkpoints)
        """
        # Prepare checkpoint data
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "history": self.history,
            "config": VQ_CONFIG,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to local directory
        checkpoint_path = os.path.join(self.output_dir, f"vqvae_finetuned_epoch_{epoch:03d}.pt")
        torch.save(checkpoint_data, checkpoint_path)
        print(f"  Saved local checkpoint: {checkpoint_path}")
        
        # Also save as 'latest' locally
        latest_path = os.path.join(self.output_dir, "vqvae_finetuned_latest.pt")
        torch.save(checkpoint_data, latest_path)
        
        # Save to GDrive if enabled
        if save_to_gdrive and self.gdrive_dir:
            try:
                # Create GDrive directory if it doesn't exist
                os.makedirs(self.gdrive_dir, exist_ok=True)
                
                # Save checkpoint to GDrive
                gdrive_checkpoint_path = os.path.join(
                    self.gdrive_dir, f"vqvae_finetuned_epoch_{epoch:03d}.pt"
                )
                torch.save(checkpoint_data, gdrive_checkpoint_path)
                print(f"  Saved GDrive checkpoint: {gdrive_checkpoint_path}")
                
                # Delete old GDrive checkpoints (keep only the current one)
                delete_old_gdrive_checkpoints(self.gdrive_dir, keep_epoch=epoch)
                
            except Exception as e:
                print(f"  Warning: Could not save to GDrive: {e}")
    
    def plot_training_history(self):
        """Plot and save training metrics."""
        if not HAS_MATPLOTLIB or len(self.history["epoch"]) == 0:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Total loss
        axes[0, 0].plot(self.history["epoch"], self.history["total_loss"])
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Total Loss")
        axes[0, 0].set_title("Total Loss")
        axes[0, 0].grid(True)
        
        # Reconstruction loss
        axes[0, 1].plot(self.history["epoch"], self.history["recon_loss"])
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Recon Loss")
        axes[0, 1].set_title("Reconstruction Loss")
        axes[0, 1].grid(True)
        
        # VQ loss
        axes[0, 2].plot(self.history["epoch"], self.history["vq_loss"])
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].set_ylabel("VQ Loss")
        axes[0, 2].set_title("VQ Commitment Loss")
        axes[0, 2].grid(True)
        
        # Perplexity
        axes[1, 0].plot(self.history["epoch"], self.history["perplexity"])
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Perplexity")
        axes[1, 0].set_title("Codebook Perplexity")
        axes[1, 0].grid(True)
        
        # Word vs Sentence loss
        axes[1, 1].plot(self.history["epoch"], self.history["word_loss"], label="Word")
        axes[1, 1].plot(self.history["epoch"], self.history["sentence_loss"], label="Sentence")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Loss")
        axes[1, 1].set_title("Loss by Data Type")
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Codebook usage
        axes[1, 2].plot(self.history["epoch"], self.history["codebook_usage"])
        axes[1, 2].set_xlabel("Epoch")
        axes[1, 2].set_ylabel("Usage %")
        axes[1, 2].set_title("Codebook Utilization")
        axes[1, 2].axhline(y=30, color='g', linestyle='--', label='Target (30%)')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "training_history.png"), dpi=150)
        plt.close()
    
    def train(self, num_epochs: int, save_every: int = 10):
        """
        Full training loop with checkpoint resumption support.
        
        Args:
            num_epochs: Total number of epochs to train (will continue from start_epoch)
            save_every: Save checkpoint every N epochs (default: 10)
        """
        print(f"\n{'='*60}")
        if self.start_epoch > 1:
            print(f"  RESUMING Fine-tuning from epoch {self.start_epoch}")
            print(f"  Target: {num_epochs} total epochs")
        else:
            print(f"  Starting Fine-tuning for {num_epochs} epochs")
        print(f"{'='*60}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  GDrive directory: {self.gdrive_dir}")
        print(f"  Learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")
        print(f"  VQ loss weight: {self.vq_loss_weight}")
        print(f"  Velocity loss weight: {self.vel_loss_weight}")
        print(f"  Hand loss weight: {self.hand_loss_weight}")
        print(f"  Save every: {save_every} epochs")
        print(f"  Device: {self.device}")
        print(f"{'='*60}\n")
        
        # Check if we've already completed training
        if self.start_epoch > num_epochs:
            print(f"Training already completed! (start_epoch={self.start_epoch} > num_epochs={num_epochs})")
            return
        
        for epoch in range(self.start_epoch, num_epochs + 1):
            metrics = self.train_epoch(epoch)
            
            # Update history
            self.history["epoch"].append(epoch)
            for k in ["total_loss", "recon_loss", "vq_loss", "vel_loss", "hand_loss",
                      "perplexity", "word_loss", "sentence_loss", "codebook_usage"]:
                self.history[k].append(metrics.get(k, 0))
            
            # Print epoch summary
            print(f"\n[Epoch {epoch}/{num_epochs}]")
            print(f"  Total Loss: {metrics['total_loss']:.4f}")
            print(f"  Recon Loss: {metrics['recon_loss']:.4f}")
            print(f"  Vel Loss: {metrics['vel_loss']:.4f}")
            print(f"  Hand Loss: {metrics['hand_loss']:.4f}")
            print(f"  VQ Loss: {metrics['vq_loss']:.6f}")
            print(f"  Perplexity: {metrics['perplexity']:.2f}")
            print(f"  Word Loss: {metrics['word_loss']:.4f} ({metrics['word_count']} samples)")
            print(f"  Sentence Loss: {metrics['sentence_loss']:.4f} ({metrics['sentence_count']} samples)")
            print(f"  Codebook Usage: {metrics['codebook_usage']:.1f}%")
            
            # Save checkpoint (to both local and GDrive)
            if epoch % save_every == 0 or epoch == num_epochs:
                self.save_checkpoint(epoch, metrics, save_to_gdrive=True)
                self.plot_training_history()
            
            # Save history
            history_path = os.path.join(self.output_dir, "training_history.json")
            with open(history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"  Fine-tuning Complete!")
        print(f"{'='*60}")
        
        # Final save
        final_path = os.path.join(self.output_dir, "vqvae_finetuned_final.pt")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": VQ_CONFIG,
            "history": self.history,
            "timestamp": datetime.now().isoformat()
        }, final_path)
        print(f"Final model saved to: {final_path}")


# =============================================================================
# Checkpoint Validation (No Training)
# =============================================================================

def validate_checkpoint(
    model: nn.Module,
    word_dataset: Dataset,
    sentence_dataset: Dataset,
    num_samples: int = 20,
    device=DEVICE
):
    """
    Validate checkpoint by running inference on a few samples.
    This helps diagnose if the model loads correctly and what the baseline losses are.
    """
    print("\n" + "="*70)
    print("  CHECKPOINT VALIDATION (No Training)")
    print("="*70)
    
    model.eval()
    
    # Helper to compute loss on a single sample
    def compute_sample_loss(motion_tensor):
        with torch.no_grad():
            # Pad to multiple of DOWNSAMPLE_FACTOR
            T = motion_tensor.shape[0]
            padded_len = math.ceil(T / DOWNSAMPLE_FACTOR) * DOWNSAMPLE_FACTOR
            if padded_len > T:
                padding = torch.zeros(padded_len - T, SMPL_DIM)
                motion_padded = torch.cat([motion_tensor, padding], dim=0)
            else:
                motion_padded = motion_tensor
            
            # Add batch dimension and move to device
            x = motion_padded.unsqueeze(0).to(device)  # (1, T, 182)
            
            # Forward pass
            x_recon, vq_loss, perplexity = model(x)
            
            # Compute reconstruction loss (only on valid frames)
            x_valid = x[:, :T, :]
            x_recon_valid = x_recon[:, :T, :]
            recon_loss = F.mse_loss(x_recon_valid, x_valid)
            
            return {
                "recon_loss": recon_loss.item(),
                "vq_loss": vq_loss.item() if torch.is_tensor(vq_loss) else vq_loss,
                "perplexity": perplexity.item() if torch.is_tensor(perplexity) else perplexity,
                "seq_len": T
            }
    
    # Test word-level samples
    print("\n[Word-Level Samples]")
    word_losses = []
    word_count = min(num_samples, len(word_dataset))
    
    for i in range(word_count):
        sample = word_dataset[i]
        if sample is None:
            continue
        metrics = compute_sample_loss(sample["motion"])
        word_losses.append(metrics["recon_loss"])
        if i < 5:  # Show first 5
            print(f"  Sample {i+1}: recon={metrics['recon_loss']:.4f}, "
                  f"vq={metrics['vq_loss']:.4f}, ppl={metrics['perplexity']:.1f}, "
                  f"len={metrics['seq_len']}")
    
    if word_losses:
        print(f"\n  Word-level mean recon loss: {np.mean(word_losses):.4f}")
        print(f"  Word-level std recon loss:  {np.std(word_losses):.4f}")
        print(f"  Word-level min/max: {np.min(word_losses):.4f} / {np.max(word_losses):.4f}")
    
    # Test sentence-level samples
    print("\n[Sentence-Level Samples]")
    sentence_losses = []
    sentence_count = min(num_samples, len(sentence_dataset))
    
    for i in range(sentence_count):
        sample = sentence_dataset[i]
        if sample is None:
            continue
        metrics = compute_sample_loss(sample["motion"])
        sentence_losses.append(metrics["recon_loss"])
        if i < 5:  # Show first 5
            print(f"  Sample {i+1}: recon={metrics['recon_loss']:.4f}, "
                  f"vq={metrics['vq_loss']:.4f}, ppl={metrics['perplexity']:.1f}, "
                  f"len={metrics['seq_len']}")
    
    if sentence_losses:
        print(f"\n  Sentence-level mean recon loss: {np.mean(sentence_losses):.4f}")
        print(f"  Sentence-level std recon loss:  {np.std(sentence_losses):.4f}")
        print(f"  Sentence-level min/max: {np.min(sentence_losses):.4f} / {np.max(sentence_losses):.4f}")
    
    # Summary
    print("\n" + "-"*70)
    print("  VALIDATION SUMMARY")
    print("-"*70)
    if word_losses:
        print(f"  Word-level recon loss:     {np.mean(word_losses):.4f}")
        print(f"    -> Expected: ~0.01-0.1 if checkpoint is correct")
    if sentence_losses:
        print(f"  Sentence-level recon loss: {np.mean(sentence_losses):.4f}")
        print(f"    -> If MUCH higher than word-level, this confirms domain shift")
    
    if word_losses and sentence_losses:
        ratio = np.mean(sentence_losses) / np.mean(word_losses)
        print(f"\n  Sentence/Word loss ratio: {ratio:.1f}x")
        if ratio > 10:
            print("    -> SEVERE domain shift! Consider:")
            print("       1. Recompute normalization stats on combined data")
            print("       2. Use higher learning rate (1e-4 instead of 1e-5)")
            print("       3. Train for more epochs (200-500)")
        elif ratio > 3:
            print("    -> Moderate domain shift. Fine-tuning should help.")
        else:
            print("    -> Mild domain shift. Fine-tuning should work well.")
    
    print("="*70 + "\n")
    
    return {
        "word_mean_loss": np.mean(word_losses) if word_losses else None,
        "sentence_mean_loss": np.mean(sentence_losses) if sentence_losses else None
    }


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune VQ-VAE on mixed word-level and sentence-level data"
    )
    
    # Required arguments
    parser.add_argument("--vqvae-ckpt", required=True, 
                       help="Path to pre-trained VQ-VAE checkpoint")
    parser.add_argument("--word-data-dir", required=True,
                       help="Directory containing word-level NPZ files")
    parser.add_argument("--sentence-data-dir", required=True,
                       help="Directory containing sentence-level How2Sign data")
    parser.add_argument("--output-dir", required=True,
                       help="Output directory for checkpoints and logs")
    
    # Optional arguments
    parser.add_argument("--stats-path", default=None,
                       help="Path to normalization stats file")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size (default: 8)")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                       help="Learning rate (default: 1e-5)")
    parser.add_argument("--word-ratio", type=float, default=0.2,
                       help="Ratio of word-level data in each batch (default: 0.2)")
    parser.add_argument("--vq-loss-weight", type=float, default=0.25,
                       help="Weight for VQ commitment loss (default: 0.25)")
    parser.add_argument("--vel-loss-weight", type=float, default=0.5,
                       help="Weight for velocity (temporal smoothness) loss (default: 0.5)")
    parser.add_argument("--hand-loss-weight", type=float, default=0.5,
                       help="Weight for hand reconstruction loss (default: 0.5)")
    parser.add_argument("--save-every", type=int, default=5,
                       help="Save checkpoint every N epochs (default: 5)")
    parser.add_argument("--max-seq-len", type=int, default=512,
                       help="Maximum sequence length (default: 512)")
    parser.add_argument("--num-workers", type=int, default=0,
                       help="Number of data loading workers (default: 0)")
    
    # Preload option for faster training with slow storage
    parser.add_argument("--preload", action="store_true",
                       help="Preload ALL data into RAM before training (~4.5GB RAM needed). "
                            "RECOMMENDED for slow storage (GDrive). "
                            "WITHOUT this flag: data loads on-the-fly (slower but less RAM).")
    parser.add_argument("--no-preload", dest="preload", action="store_false",
                       help="Explicitly disable preloading (load data on-the-fly). "
                            "Use this if running low on RAM.")
    
    # Debug/validation modes
    parser.add_argument("--word-only", action="store_true",
                       help="Train on word-level data ONLY. Use this to validate "
                            "that the checkpoint loads correctly and loss is low (~0.01).")
    parser.add_argument("--sentence-only", action="store_true",
                       help="Train on sentence-level data ONLY. Use this to see "
                            "the baseline loss on sentence data without word mixing.")
    parser.add_argument("--validate-checkpoint", action="store_true",
                       help="Just validate the checkpoint by running inference on a few samples. "
                            "No training, just shows reconstruction loss.")
    
    # GDrive checkpoint options
    parser.add_argument("--gdrive-dir", default=GDRIVE_CHECKPOINT_DIR,
                       help=f"GDrive directory for saving checkpoints (default: {GDRIVE_CHECKPOINT_DIR})")
    parser.add_argument("--no-resume", action="store_true",
                       help="Do not resume from existing checkpoint. Start fresh from the base VQ-VAE.")
    
    # Learning rate options for continued training
    parser.add_argument("--reset-lr", action="store_true",
                       help="Reset learning rate and scheduler when resuming. Use this if LR has decayed to zero.")
    parser.add_argument("--lr-scheduler", 
                       choices=["cosine", "cosine_restart", "cosine_restart_gradual", "constant", "step"], 
                       default="constant",
                       help="Learning rate scheduler type (default: constant). "
                            "Options: 'constant' (stable, recommended for late-stage), "
                            "'cosine_restart' (restarts can cause spikes!), "
                            "'cosine_restart_gradual' (each cycle 2x longer), "
                            "'step' (halve LR every N epochs)")
    parser.add_argument("--lr-warmup-epochs", type=int, default=5,
                       help="Number of warmup epochs when resetting LR (default: 5)")
    parser.add_argument("--lr-restart-period", type=int, default=100,
                       help="Period for LR restart/step schedulers (default: 100 epochs)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine mode
    mode = "mixed"
    if args.validate_checkpoint:
        mode = "validate"
    elif args.word_only:
        mode = "word-only"
    elif args.sentence_only:
        mode = "sentence-only"
    
    print("\n" + "="*70)
    print("  VQ-VAE Fine-tuning for Sentence-Level Sign Language Motion")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  VQ-VAE Checkpoint: {args.vqvae_ckpt}")
    print(f"  Word Data Dir: {args.word_data_dir}")
    print(f"  Sentence Data Dir: {args.sentence_data_dir}")
    print(f"  Stats Path: {args.stats_path}")
    print(f"  Output Dir: {args.output_dir}")
    print(f"  GDrive Dir: {args.gdrive_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.learning_rate:.2e}")
    print(f"  Word Ratio: {args.word_ratio:.1%}")
    print(f"  VQ Loss Weight: {args.vq_loss_weight}")
    print(f"  Vel Loss Weight: {args.vel_loss_weight}")
    print(f"  Hand Loss Weight: {args.hand_loss_weight}")
    print(f"  Preload Mode: {'ENABLED (all data in RAM)' if args.preload else 'DISABLED (load on-the-fly)'}")
    print(f"  Auto-Resume: {'DISABLED (--no-resume)' if args.no_resume else 'ENABLED'}")
    print(f"  Training Mode: {mode.upper()}")
    print(f"  Device: {DEVICE}")
    print("="*70)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    if args.gdrive_dir:
        os.makedirs(args.gdrive_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(args.output_dir, "finetune_config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # =========================================================================
    # Check for existing checkpoints to resume from
    # =========================================================================
    resume_checkpoint = None
    if not args.no_resume and mode not in ["validate"]:
        resume_checkpoint = find_latest_checkpoint(args.output_dir, args.gdrive_dir)
    
    # Load base model (will be overwritten if resuming)
    model = load_vqvae_checkpoint(args.vqvae_ckpt, VQ_CONFIG, DEVICE)
    
    # Create datasets (with or without preloading)
    if args.preload:
        # =====================================================================
        # PRELOAD MODE: Load ALL data into RAM first
        # =====================================================================
        print("\n" + "="*70)
        print("  PRELOAD MODE: Loading all data into RAM...")
        print("  This may take a while but training will be MUCH faster!")
        print("="*70)
        
        import time
        preload_start = time.time()
        
        # Only load what we need based on mode
        if mode in ["mixed", "word-only", "validate"]:
            word_dataset = preload_word_level_data(
                root_dir=args.word_data_dir,
                stats_path=args.stats_path,
                max_seq_len=args.max_seq_len,
                show_progress=True
            )
        else:
            word_dataset = PreloadedDataset([], "word")
        
        if mode in ["mixed", "sentence-only", "validate"]:
            sentence_dataset = preload_sentence_level_data(
                root_dir=args.sentence_data_dir,
                stats_path=args.stats_path,
                max_seq_len=args.max_seq_len,
                show_progress=True
            )
        else:
            sentence_dataset = PreloadedDataset([], "sentence")
        
        preload_time = time.time() - preload_start
        total_samples = len(word_dataset) + len(sentence_dataset)
        print(f"\n{'='*70}")
        print(f"  PRELOAD COMPLETE!")
        print(f"  Total samples in RAM: {total_samples}")
        print(f"  Word-level: {len(word_dataset)} samples")
        print(f"  Sentence-level: {len(sentence_dataset)} samples")
        print(f"  Preload time: {preload_time:.1f}s")
        print(f"{'='*70}\n")
        
    else:
        # =====================================================================
        # STANDARD MODE: Load data on-demand during training
        # =====================================================================
        if mode in ["mixed", "word-only", "validate"]:
            word_dataset = WordLevelDataset(
                root_dir=args.word_data_dir,
                stats_path=args.stats_path,
                max_seq_len=args.max_seq_len
            )
        else:
            word_dataset = PreloadedDataset([], "word")
        
        if mode in ["mixed", "sentence-only", "validate"]:
            sentence_dataset = SentenceLevelDataset(
                root_dir=args.sentence_data_dir,
                stats_path=args.stats_path,
                max_seq_len=args.max_seq_len
            )
        else:
            sentence_dataset = PreloadedDataset([], "sentence")
    
    # =========================================================================
    # VALIDATE-CHECKPOINT MODE: Just run inference to check losses
    # =========================================================================
    if mode == "validate":
        validate_checkpoint(model, word_dataset, sentence_dataset, num_samples=20, device=DEVICE)
        print("\nValidation complete! No training performed.")
        return
    
    # =========================================================================
    # Adjust word_ratio based on mode
    # =========================================================================
    if mode == "word-only":
        print("\n*** WORD-ONLY MODE: Training on word-level data only ***")
        print("    This validates that the checkpoint loads correctly.\n")
        args.word_ratio = 1.0  # 100% word data
        
    elif mode == "sentence-only":
        print("\n*** SENTENCE-ONLY MODE: Training on sentence-level data only ***")
        print("    This shows baseline loss on sentence data.\n")
        args.word_ratio = 0.0  # 0% word data
    
    # Check if we have data
    if len(word_dataset) == 0 and len(sentence_dataset) == 0:
        print("\nError: No data found in either dataset!")
        return
    
    if len(word_dataset) == 0 and mode != "sentence-only":
        print("\nWarning: No word-level data found. Using sentence-level only.")
        args.word_ratio = 0.0
    
    if len(sentence_dataset) == 0 and mode != "word-only":
        print("\nWarning: No sentence-level data found. Using word-level only.")
        args.word_ratio = 1.0
    
    # Create mixed dataloader
    # Note: When preloading, num_workers=0 is fine since data is already in RAM
    effective_num_workers = 0 if args.preload else args.num_workers
    
    dataloader = create_mixed_dataloader(
        word_dataset=word_dataset,
        sentence_dataset=sentence_dataset,
        word_ratio=args.word_ratio,
        batch_size=args.batch_size,
        num_workers=effective_num_workers
    )
    
    # Create trainer
    trainer = FinetuneTrainer(
        model=model,
        dataloader=dataloader,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        vq_loss_weight=args.vq_loss_weight,
        vel_loss_weight=args.vel_loss_weight,
        hand_loss_weight=args.hand_loss_weight,
        gdrive_dir=args.gdrive_dir,
        scheduler_type=args.lr_scheduler,
        num_epochs=args.epochs,
        restart_period=args.lr_restart_period,
        device=DEVICE
    )
    
    # Load checkpoint if resuming
    if resume_checkpoint is not None:
        checkpoint_path, checkpoint_epoch = resume_checkpoint
        trainer.load_checkpoint(checkpoint_path)
        
        # Reset learning rate if requested (important when LR has decayed to zero!)
        if args.reset_lr:
            trainer.reset_learning_rate(args.learning_rate, args.lr_warmup_epochs)
            print(f"\n*** Resuming training from epoch {checkpoint_epoch + 1} with RESET learning rate ***\n")
        else:
            current_lr = trainer.optimizer.param_groups[0]['lr']
            if current_lr < 1e-7:
                print(f"\n*** WARNING: Learning rate is very low ({current_lr:.2e})! ***")
                print(f"*** Consider using --reset-lr to reset the learning rate ***\n")
            print(f"\n*** Resuming training from epoch {checkpoint_epoch + 1} ***\n")
    
    # Train
    trainer.train(num_epochs=args.epochs, save_every=args.save_every)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

