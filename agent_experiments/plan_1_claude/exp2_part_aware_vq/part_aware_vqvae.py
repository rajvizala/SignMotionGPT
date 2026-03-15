"""
Experiment 2: Part-Aware VQ-VAE for Sign Language Motion

MOTIVATION:
In sign language, different body parts carry different semantic information:
- Hands: primary semantic content (handshapes, movements)
- Upper body: grammatical markers, spatial references
- Face: non-manual markers (mouthing, expressions, eyebrow raises)

The current VQ-VAE quantizes all 182 dims together, forcing a single codebook
to represent all body parts simultaneously. This creates coupling: a change
in hand shape forces a completely different codebook entry even if the body
pose is identical. This coupling hurts generalization because the model
cannot recombine body part patterns independently.

APPROACH:
Split the SMPL-X features into semantically meaningful body part groups,
each with its own encoder-quantizer-decoder branch:
  1. Upper body: shape(10) + body_pose(63) + root_pose(3) + cam_trans(3) = 79 dims
  2. Left hand: lhand_pose(45) = 45 dims
  3. Right hand: rhand_pose(45) = 45 dims
  4. Face: jaw_pose(3) + expression(10) = 13 dims

Each branch has:
- Independent encoder (smaller, adapted to part dimensionality)
- Independent codebook (smaller K, allowing better utilization)
- Independent decoder

A lightweight cross-part attention module fuses information between parts
after encoding, ensuring that hand-body coordination is maintained.

REFERENCES:
- MotionGPT-2 (2024, arXiv:2410.21747):
  Introduced Part-Aware VQVAE with separate codebooks for body and hands,
  enabling fine-grained holistic motion generation.
- SOKE / Signs as Tokens (ICCV 2025, Zuo et al.):
  Used decoupled tokenizer with separate upper body, left hand, right hand
  branches and multi-head decoding for sign language generation.
- T2S-GPT (ACL 2024): Dynamic vector quantization for sign language with
  adaptive information density per body part.

HYPOTHESIS:
Decoupling body parts will enable compositional generalization: the model
can recombine known hand patterns with new body poses to generate motions
for unseen sentences. Each part's codebook will be smaller but more
efficiently utilized, reducing codebook collapse.

For the LLM, we emit tokens from all parts in a structured sequence:
[UPPER_t, LHAND_t, RHAND_t, FACE_t] for each timestep t, with separate
token ranges per part. This teaches the LLM the compositional structure
of sign language explicitly.

EXPECTED OUTCOMES:
- Higher per-part codebook utilization (>95% for hands)
- Better hand reconstruction (lower L2 for lhand/rhand)
- Compositional generalization: combine known parts in new ways
- Reduced total codebook entries needed (4x128=512 vs 1x512)
"""

import os
import sys
import math
import json
import time
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from models.resnet import Resnet1D
from models.quantize_cnn import QuantizeEMAReset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.utils import (
    SMPLX_TOTAL_DIM, SMPLX_PARAM_DIMS, BODY_PART_GROUPS,
    set_seeds, compute_reconstruction_metrics, compute_codebook_utilization,
    print_experiment_banner, save_experiment_results, EarlyStopping,
)


# Part definitions with their feature dimension ranges
PART_CONFIGS = {
    'upper_body': {
        'params': ['shape', 'body_pose', 'root_pose', 'cam_trans'],
        'codebook_size': 256,
        'code_dim': 256,
        'width': 256,
        'depth': 2,
    },
    'left_hand': {
        'params': ['lhand_pose'],
        'codebook_size': 128,
        'code_dim': 128,
        'width': 128,
        'depth': 2,
    },
    'right_hand': {
        'params': ['rhand_pose'],
        'codebook_size': 128,
        'code_dim': 128,
        'width': 128,
        'depth': 2,
    },
    'face': {
        'params': ['jaw_pose', 'expression'],
        'codebook_size': 64,
        'code_dim': 64,
        'width': 64,
        'depth': 2,
    },
}


def get_part_feature_indices(part_config: Dict) -> List[int]:
    """Get the feature indices for a body part group."""
    indices = []
    for param_name in part_config['params']:
        start, end = SMPLX_PARAM_DIMS[param_name]
        indices.extend(range(start, end))
    return sorted(indices)


def get_part_dim(part_config: Dict) -> int:
    """Get total feature dimension for a body part."""
    return len(get_part_feature_indices(part_config))


class PartEncoder(nn.Module):
    """Encoder for a single body part."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        width: int = 128,
        depth: int = 2,
        down_t: int = 3,
        stride_t: int = 2,
        dilation_growth_rate: int = 3,
        activation: str = 'relu',
        norm: str = None,
    ):
        super().__init__()
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_dim, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for _ in range(down_t):
            block = nn.Sequential(
                nn.Conv1d(width, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate,
                         activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_dim, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class PartDecoder(nn.Module):
    """Decoder for a single body part."""

    def __init__(
        self,
        output_dim: int,
        input_dim: int,
        width: int = 128,
        depth: int = 2,
        down_t: int = 3,
        stride_t: int = 2,
        dilation_growth_rate: int = 3,
        activation: str = 'relu',
        norm: str = None,
    ):
        super().__init__()
        blocks = []
        blocks.append(nn.Conv1d(input_dim, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for _ in range(down_t):
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate,
                         reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, width, 3, 1, 1),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, output_dim, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class CrossPartAttention(nn.Module):
    """
    Lightweight cross-attention module to maintain coordination between
    body parts after independent encoding.

    Each part's latent representation attends to all other parts, allowing
    the model to capture dependencies like:
    - Hand movement correlating with body lean
    - Facial expressions coordinating with hand signs

    Uses multi-head attention with residual connections.

    Reference: Similar to the cross-attention in MotionGPT-2's part fusion.
    """

    def __init__(self, part_dims: Dict[str, int], num_heads: int = 4):
        super().__init__()

        total_dim = sum(part_dims.values())

        # Project each part to a common dimension for attention
        self.common_dim = max(part_dims.values())

        self.part_projections = nn.ModuleDict({
            name: nn.Linear(dim, self.common_dim)
            for name, dim in part_dims.items()
        })

        self.attention = nn.MultiheadAttention(
            embed_dim=self.common_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.part_back_projections = nn.ModuleDict({
            name: nn.Linear(self.common_dim, dim)
            for name, dim in part_dims.items()
        })

        self.norm = nn.LayerNorm(self.common_dim)

    def forward(self, part_latents: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            part_latents: Dict mapping part_name -> (B, C_part, T) tensors
        Returns:
            Updated part latents with cross-part information
        """
        # Project all parts to common dim: (B, T, common_dim)
        projected = {}
        part_order = sorted(part_latents.keys())

        for name in part_order:
            x = part_latents[name]  # (B, C, T)
            x = x.permute(0, 2, 1)  # (B, T, C)
            projected[name] = self.part_projections[name](x)  # (B, T, common_dim)

        # Concatenate along sequence dimension for cross-attention
        B, T, _ = projected[part_order[0]].shape
        concat = torch.cat([projected[name] for name in part_order], dim=1)  # (B, N_parts*T, common_dim)
        concat = self.norm(concat)

        # Self-attention across all parts
        attended, _ = self.attention(concat, concat, concat)

        # Split back and project to original dimensions
        result = {}
        offset = 0
        for name in part_order:
            part_attended = attended[:, offset:offset + T, :]  # (B, T, common_dim)
            part_orig = projected[name]
            # Residual connection
            fused = part_orig + 0.1 * part_attended
            # Project back to part dimension
            result[name] = self.part_back_projections[name](fused).permute(0, 2, 1)  # (B, C, T)
            offset += T

        return result


class PartAwareVQVAE(nn.Module):
    """
    Part-Aware VQ-VAE with independent codebooks per body part.

    Architecture:
    1. Split input into body parts
    2. Encode each part independently
    3. Cross-part attention for coordination
    4. Independent quantization per part
    5. Decode each part independently
    6. Reassemble into full body
    """

    def __init__(
        self,
        part_configs: Dict[str, Dict] = None,
        down_t: int = 3,
        stride_t: int = 2,
        dilation_growth_rate: int = 3,
        activation: str = 'relu',
        norm: str = None,
        mu: float = 0.99,
        use_cross_attention: bool = True,
    ):
        super().__init__()

        if part_configs is None:
            part_configs = PART_CONFIGS

        self.part_configs = part_configs
        self.use_cross_attention = use_cross_attention

        # Build per-part indices
        self.part_indices = {}
        self.part_dims = {}

        for part_name, config in part_configs.items():
            indices = get_part_feature_indices(config)
            self.part_indices[part_name] = indices
            self.part_dims[part_name] = len(indices)

        # Encoders
        self.encoders = nn.ModuleDict()
        for part_name, config in part_configs.items():
            feat_dim = self.part_dims[part_name]
            self.encoders[part_name] = PartEncoder(
                input_dim=feat_dim,
                output_dim=config['code_dim'],
                width=config['width'],
                depth=config['depth'],
                down_t=down_t,
                stride_t=stride_t,
                dilation_growth_rate=dilation_growth_rate,
                activation=activation,
                norm=norm,
            )

        # Quantizers
        self.quantizers = nn.ModuleDict()
        for part_name, config in part_configs.items():
            self.quantizers[part_name] = QuantizeEMAReset(
                config['codebook_size'], config['code_dim'], mu
            )

        # Cross-part attention
        if use_cross_attention:
            code_dims = {name: config['code_dim'] for name, config in part_configs.items()}
            self.cross_attention = CrossPartAttention(code_dims)

        # Decoders
        self.decoders = nn.ModuleDict()
        for part_name, config in part_configs.items():
            feat_dim = self.part_dims[part_name]
            self.decoders[part_name] = PartDecoder(
                output_dim=feat_dim,
                input_dim=config['code_dim'],
                width=config['width'],
                depth=config['depth'],
                down_t=down_t,
                stride_t=stride_t,
                dilation_growth_rate=dilation_growth_rate,
                activation=activation,
                norm=norm,
            )

    def split_features(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Split (B, T, D) features into per-part tensors."""
        parts = {}
        for part_name, indices in self.part_indices.items():
            parts[part_name] = features[:, :, indices]  # (B, T, D_part)
        return parts

    def reassemble_features(self, parts: Dict[str, torch.Tensor], B: int, T: int) -> torch.Tensor:
        """Reassemble per-part tensors into (B, T, D) features."""
        output = torch.zeros(B, T, SMPLX_TOTAL_DIM, device=next(iter(parts.values())).device)
        for part_name, indices in self.part_indices.items():
            for i, idx in enumerate(indices):
                output[:, :, idx] = parts[part_name][:, :, i]
        return output

    def forward(self, features: torch.Tensor):
        """
        Args:
            features: (B, T, 182) full body SMPL-X features
        Returns:
            x_out: (B, T, 182) reconstructed features
            total_loss: sum of per-part commitment losses
            info: per-part diagnostics
        """
        B, T, D = features.shape

        # 1. Split into parts
        parts = self.split_features(features)

        # 2. Encode each part
        encoded = {}
        for part_name in self.part_configs:
            x = parts[part_name].permute(0, 2, 1)  # (B, D_part, T)
            encoded[part_name] = self.encoders[part_name](x)  # (B, code_dim, T')

        # 3. Cross-part attention
        if self.use_cross_attention:
            encoded = self.cross_attention(encoded)

        # 4. Quantize each part
        quantized = {}
        total_loss = torch.tensor(0.0, device=features.device)
        info = {}

        for part_name in self.part_configs:
            x_enc = encoded[part_name]
            x_q, commit_loss, perplexity = self.quantizers[part_name](x_enc)
            quantized[part_name] = x_q
            total_loss = total_loss + commit_loss
            info[part_name] = {
                'commit_loss': commit_loss.item(),
                'perplexity': perplexity.item() if torch.is_tensor(perplexity) else perplexity,
            }

        # 5. Decode each part
        decoded_parts = {}
        for part_name in self.part_configs:
            x_dec = self.decoders[part_name](quantized[part_name])  # (B, D_part, T)
            decoded_parts[part_name] = x_dec.permute(0, 2, 1)  # (B, T, D_part)

        # 6. Reassemble
        x_out = self.reassemble_features(decoded_parts, B, T)

        return x_out, total_loss, info

    def encode(self, features: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], None]:
        """Encode to per-part code indices."""
        B, T, D = features.shape
        parts = self.split_features(features)

        all_codes = {}
        for part_name in self.part_configs:
            x = parts[part_name].permute(0, 2, 1)
            x_enc = self.encoders[part_name](x)

            if self.use_cross_attention:
                pass  # Skip cross-attention for encoding (optional)

            # Get code indices
            N, C, T_enc = x_enc.shape
            x_flat = x_enc.permute(0, 2, 1).contiguous().view(-1, C)
            code_idx = self.quantizers[part_name].quantize(x_flat)
            all_codes[part_name] = code_idx.view(N, T_enc)

        return all_codes, None

    def decode_from_codes(self, codes: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode from per-part code indices."""
        decoded_parts = {}
        B = None
        T_out = None

        for part_name, code_idx in codes.items():
            N, T_enc = code_idx.shape
            B = N
            code_dim = self.part_configs[part_name]['code_dim']

            x_d = self.quantizers[part_name].dequantize(code_idx.view(-1))
            x_d = x_d.view(N, T_enc, code_dim).permute(0, 2, 1).contiguous()

            x_dec = self.decoders[part_name](x_d)
            decoded_parts[part_name] = x_dec.permute(0, 2, 1)
            T_out = x_dec.shape[2]

        return self.reassemble_features(decoded_parts, B, T_out)


def flatten_part_codes_for_llm(
    codes: Dict[str, torch.Tensor],
    part_configs: Dict[str, Dict] = None,
) -> torch.Tensor:
    """
    Flatten per-part codes into a single token sequence for LLM.

    Strategy: For each timestep t, emit tokens in order:
    [UPPER_t, LHAND_t, RHAND_t, FACE_t]

    Each part's codes are offset to use distinct token ranges:
    - upper_body: [0, K_upper)
    - left_hand: [K_upper, K_upper + K_lhand)
    - right_hand: [K_upper + K_lhand, K_upper + K_lhand + K_rhand)
    - face: [..., ... + K_face)

    Total vocabulary for motion tokens = sum of all codebook sizes.
    """
    if part_configs is None:
        part_configs = PART_CONFIGS

    part_order = ['upper_body', 'left_hand', 'right_hand', 'face']

    # Compute offsets
    offset = 0
    offsets = {}
    for name in part_order:
        offsets[name] = offset
        offset += part_configs[name]['codebook_size']

    # Interleave: for each timestep, emit all parts
    N, T = codes[part_order[0]].shape
    parts_per_step = len(part_order)

    flat = torch.zeros(N, T * parts_per_step, dtype=torch.long, device=codes[part_order[0]].device)

    for t in range(T):
        for p_idx, name in enumerate(part_order):
            flat[:, t * parts_per_step + p_idx] = codes[name][:, t] + offsets[name]

    return flat


def unflatten_part_codes_from_llm(
    flat_codes: torch.Tensor,
    part_configs: Dict[str, Dict] = None,
) -> Dict[str, torch.Tensor]:
    """Reverse of flatten_part_codes_for_llm."""
    if part_configs is None:
        part_configs = PART_CONFIGS

    part_order = ['upper_body', 'left_hand', 'right_hand', 'face']
    parts_per_step = len(part_order)

    N, total_len = flat_codes.shape
    T = total_len // parts_per_step

    offset = 0
    offsets = {}
    for name in part_order:
        offsets[name] = offset
        offset += part_configs[name]['codebook_size']

    codes = {}
    for p_idx, name in enumerate(part_order):
        part_flat = flat_codes[:, p_idx::parts_per_step][:, :T]
        codes[name] = (part_flat - offsets[name]).clamp(0, part_configs[name]['codebook_size'] - 1)

    return codes


# ============================================================================
# Dataset
# ============================================================================

class NpzMotionDataset(Dataset):
    """Dataset that loads motion from .npz files (motion key, shape T x 182)."""

    def __init__(self, root_dir: str, stats_path: Optional[str] = None,
                 min_seq_len: int = 16, max_seq_len: int = 256):
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len

        glob_pattern = os.path.join(root_dir, '**', '*.npz')
        self.files = glob.glob(glob_pattern, recursive=True)
        if not self.files:
            raise FileNotFoundError(f"No .npz files found at '{glob_pattern}'")
        print(f"[Dataset] Found {len(self.files)} .npz files in {root_dir}")

        if stats_path and os.path.exists(stats_path):
            stats = torch.load(stats_path, map_location='cpu')
            self.mean = stats.get('mean', torch.zeros(SMPLX_TOTAL_DIM))
            self.std = stats.get('std', torch.ones(SMPLX_TOTAL_DIM))
            if not torch.is_tensor(self.mean):
                self.mean = torch.tensor(self.mean, dtype=torch.float32)
            if not torch.is_tensor(self.std):
                self.std = torch.tensor(self.std, dtype=torch.float32)
        else:
            self.mean = torch.zeros(SMPLX_TOTAL_DIM)
            self.std = torch.ones(SMPLX_TOTAL_DIM)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            with np.load(self.files[idx]) as data:
                motion = data['motion']
        except Exception as e:
            print(f"Error loading {self.files[idx]}: {e}")
            return None

        if motion.shape[0] < self.min_seq_len:
            return None
        if motion.shape[0] > self.max_seq_len:
            motion = motion[:self.max_seq_len]
        if motion.shape[1] != SMPLX_TOTAL_DIM:
            return None

        motion_tensor = torch.tensor(motion, dtype=torch.float32)
        normalized = (motion_tensor - self.mean) / (self.std + 1e-8)
        return normalized


def collate_fn(batch):
    """Collate and pad sequences to same length (multiple of 8 for downsampling)."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    batch.sort(key=lambda x: x.shape[0], reverse=True)
    max_len = min(batch[0].shape[0], 256)
    padded_len = math.ceil(max_len / 8) * 8
    padded = torch.zeros(len(batch), padded_len, SMPLX_TOTAL_DIM)
    for i, x in enumerate(batch):
        length = min(x.shape[0], padded_len)
        padded[i, :length] = x[:length]
    return padded


# ============================================================================
# Training
# ============================================================================

def train_part_aware_vqvae(
    model: PartAwareVQVAE,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int = 300,
    lr: float = 2e-4,
    beta: float = 0.25,
    output_dir: str = "./exp2_output",
):
    """Training loop for Part-Aware VQ-VAE."""
    os.makedirs(output_dir, exist_ok=True)

    # Per-parameter loss weights
    loss_weights = torch.ones(SMPLX_TOTAL_DIM, device=device)
    for param_name, (start, end) in SMPLX_PARAM_DIMS.items():
        if param_name in ['body_pose', 'lhand_pose', 'rhand_pose']:
            loss_weights[start:end] = 10.0
        elif param_name == 'shape':
            loss_weights[start:end] = 5.0
        elif param_name == 'expression':
            loss_weights[start:end] = 8.0

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    early_stop = EarlyStopping(patience=50, min_delta=1e-5)

    model.to(device)
    best_val_loss = float('inf')
    history = []

    print_experiment_banner("Part-Aware VQ-VAE Training", f"Parts={len(model.part_configs)}, Epochs={epochs}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_start = time.time()
        last_info = {}

        for batch_data in train_loader:
            if batch_data is None:
                continue
            if isinstance(batch_data, (list, tuple)):
                features = batch_data[0].to(device)
            else:
                features = batch_data.to(device)

            x_recon, commit_loss, info = model(features)
            last_info = info

            diff = features - x_recon
            weighted_diff = diff * loss_weights.unsqueeze(0).unsqueeze(0)
            recon_loss = F.smooth_l1_loss(weighted_diff, torch.zeros_like(weighted_diff))

            total_loss = recon_loss + beta * commit_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += total_loss.item()

        scheduler.step()
        avg_loss = epoch_loss / max(len(train_loader), 1)

        # Validation
        val_loss = None
        if val_loader is not None and (epoch + 1) % 10 == 0:
            model.eval()
            val_total = 0
            val_count = 0
            with torch.no_grad():
                for batch_data in val_loader:
                    if batch_data is None:
                        continue
                    if isinstance(batch_data, (list, tuple)):
                        features = batch_data[0].to(device)
                    else:
                        features = batch_data.to(device)
                    x_recon, commit_loss, info = model(features)
                    diff = features - x_recon
                    weighted_diff = diff * loss_weights.unsqueeze(0).unsqueeze(0)
                    recon_loss = F.smooth_l1_loss(weighted_diff, torch.zeros_like(weighted_diff))
                    val_total += (recon_loss + beta * commit_loss).item()
                    val_count += 1
            if val_count > 0:
                val_loss = val_total / val_count

            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_loss,
                }, os.path.join(output_dir, 'best_model.pt'))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            msg = f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}"
            if val_loss is not None:
                msg += f" | Val: {val_loss:.6f}"
            if last_info:
                perps = [f"{last_info[name]['perplexity']:.0f}" for name in sorted(last_info.keys())]
                msg += f" | Perp: [{', '.join(perps)}]"
            print(msg)

        history.append({'epoch': epoch + 1, 'train_loss': avg_loss, 'val_loss': val_loss})

        if val_loss is not None and early_stop(val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break

    torch.save({'model_state_dict': model.state_dict(), 'history': history},
               os.path.join(output_dir, 'final_model.pt'))

    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 2: Part-Aware VQ-VAE")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with motion .npz files")
    parser.add_argument("--output-dir", type=str, default="./exp2_output")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--no-cross-attention", action="store_true")
    parser.add_argument("--stats-path", type=str, default=None, help="Path to normalization stats (optional)")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PartAwareVQVAE(
        part_configs=PART_CONFIGS,
        use_cross_attention=not args.no_cross_attention,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
    print(f"Total parameters: {total_params:,}")

    # Print per-part info
    total_codebook = 0
    for name, config in PART_CONFIGS.items():
        dim = get_part_dim(config)
        total_codebook += config['codebook_size']
        print(f"  {name}: {dim} dims, codebook={config['codebook_size']}")
    print(f"  Total codebook entries: {total_codebook}")
    print(f"  Effective capacity: {math.prod(c['codebook_size'] for c in PART_CONFIGS.values()):,}")

    # Load dataset and create loaders
    dataset = NpzMotionDataset(
        args.data_dir,
        stats_path=args.stats_path,
        min_seq_len=16,
        max_seq_len=256,
    )
    n = len(dataset)
    val_size = max(1, int(n * args.val_split))
    train_size = n - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    print(f"Train samples: {train_size}, Val samples: {val_size}")

    model, history = train_part_aware_vqvae(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        output_dir=args.output_dir,
    )

    print(f"\nTraining complete. Best model and history saved to {args.output_dir}")
