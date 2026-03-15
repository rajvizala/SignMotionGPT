"""
Experiment 2 -- Part-Aware VQ-VAE with Multi-Head Decoding

REFERENCES
----------
[1] Jiang et al., "MotionGPT-2: A General-Purpose Motion-Language Model for
    Motion Generation and Understanding", arXiv:2410.21747, 2024.
[2] Zuo et al., "Signs as Tokens: A Retrieval-Enhanced Multilingual Sign
    Language Generator" (SOKE), ICCV 2025.
[3] T2S-GPT: "Dynamic Vector Quantization for Autoregressive Sign Language
    Production from Text", ACL 2024.

CORE IDEA
---------
The baseline VQ-VAE quantizes the full 182-dim SMPL-X vector as a monolithic
unit.  This means that a single codebook must simultaneously encode:
  - body pose (63-dim), which defines the overall posture
  - hand poses (45-dim each), which carry most of the linguistic information
  - face expression (10-dim), which conveys non-manual markers
  - shape (10-dim), root (3-dim), cam (3-dim) -- signer identity / viewpoint

For ASL, hands are the primary carrier of meaning.  Entangling hand and body
in a single code forces the model to memorise exact combinations of hand+body
seen during training.  An unseen sentence with a *new* body-hand combination
gets mapped to the nearest training combination, producing artifacts.

The Part-Aware VQ-VAE splits the feature vector into **four anatomical groups**,
each quantized by its own codebook.  A cross-part attention fusion layer then
lets the groups interact before decoding.

WHY THIS SHOULD HELP GENERALIZATION
------------------------------------
1. **Combinatorial compositionality**: with K codes for body and K' for hands,
   the model can represent K*K' combinations without ever having seen them all.
2. **Independent hand precision**: sign language meaning is overwhelmingly in
   the hands.  A dedicated hand codebook preserves fine finger articulation
   that gets averaged out in a shared codebook.
3. **Signer-invariant body codes**: shape and root pose carry signer identity.
   Separating them lets the model factor out signer style from sign content,
   following the decoupled-motion approach of [3] for signer independence.
4. **Multi-head LLM generation**: the LLM can predict part tokens in parallel
   (as in SOKE [2]), reducing sequence length and error accumulation.
"""

import math
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from models.resnet import Resnet1D


# ---------------------------------------------------------------------------
# Part Definition
# ---------------------------------------------------------------------------

PART_GROUPS = {
    "body": {"start": 10, "end": 73, "dim": 63},
    "lhand": {"start": 73, "end": 118, "dim": 45},
    "rhand": {"start": 118, "end": 163, "dim": 45},
    "face_and_meta": {"start": 0, "end": 10, "dim": 10,
                      "extras": [(163, 176, 13), (176, 182, 6)]},
}


def split_by_parts(x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Split (B, T, 182) into per-part tensors."""
    parts = {}
    parts["body"] = x[:, :, 10:73]
    parts["lhand"] = x[:, :, 73:118]
    parts["rhand"] = x[:, :, 118:163]
    # face_and_meta: shape(10) + jaw(3) + expression(10) + root(3) + cam(3) = 29
    parts["face_and_meta"] = torch.cat([
        x[:, :, 0:10],     # shape
        x[:, :, 163:176],  # jaw + expression
        x[:, :, 176:182],  # root + cam
    ], dim=-1)
    return parts


def merge_parts(parts: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Reconstruct (B, T, 182) from per-part tensors."""
    fm = parts["face_and_meta"]
    shape = fm[:, :, :10]
    jaw_expr = fm[:, :, 10:23]
    root_cam = fm[:, :, 23:29]
    body = parts["body"]
    lhand = parts["lhand"]
    rhand = parts["rhand"]
    return torch.cat([shape, body, lhand, rhand, jaw_expr, root_cam], dim=-1)


# ---------------------------------------------------------------------------
# Per-Part Encoder / Quantizer / Decoder
# ---------------------------------------------------------------------------

class PartEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 code_dim: int = 128, down_t: int = 3, stride_t: int = 2,
                 depth: int = 3, dilation_growth_rate: int = 3):
        super().__init__()
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_dim, hidden_dim, 3, 1, 1))
        blocks.append(nn.GELU())
        for _ in range(down_t):
            blocks.append(nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, filter_t, stride_t, pad_t),
                Resnet1D(hidden_dim, depth, dilation_growth_rate,
                         activation='gelu', norm=None),
            ))
        blocks.append(nn.Conv1d(hidden_dim, code_dim, 3, 1, 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


class PartDecoder(nn.Module):
    def __init__(self, output_dim: int, hidden_dim: int = 256,
                 code_dim: int = 128, down_t: int = 3, stride_t: int = 2,
                 depth: int = 3, dilation_growth_rate: int = 3):
        super().__init__()
        blocks = []
        blocks.append(nn.Conv1d(code_dim, hidden_dim, 3, 1, 1))
        blocks.append(nn.GELU())
        for _ in range(down_t):
            blocks.append(nn.Sequential(
                Resnet1D(hidden_dim, depth, dilation_growth_rate,
                         reverse_dilation=True, activation='gelu', norm=None),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(hidden_dim, hidden_dim, 3, 1, 1),
            ))
        blocks.append(nn.Conv1d(hidden_dim, hidden_dim, 3, 1, 1))
        blocks.append(nn.GELU())
        blocks.append(nn.Conv1d(hidden_dim, output_dim, 3, 1, 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


class EMACodebook(nn.Module):
    """EMA-updated codebook with dead-code reset (matching baseline)."""

    def __init__(self, num_codes: int, code_dim: int, mu: float = 0.99):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.mu = mu
        self.register_buffer("codebook", torch.randn(num_codes, code_dim) * 0.02)
        self.register_buffer("code_count", torch.ones(num_codes))
        self.register_buffer("code_sum", torch.randn(num_codes, code_dim) * 0.02)
        self.register_buffer("initialized", torch.tensor(False))

    def _init(self, x):
        n = x.shape[0]
        if n < self.num_codes:
            reps = (self.num_codes + n - 1) // n
            x = x.repeat(reps, 1)[:self.num_codes]
            x = x + torch.randn_like(x) * 0.01
        self.codebook.copy_(x[:self.num_codes])
        self.code_sum.copy_(self.codebook)
        self.code_count.fill_(1.0)
        self.initialized.fill_(True)

    def quantize(self, x):
        dist = (
            x.pow(2).sum(-1, keepdim=True)
            - 2 * x @ self.codebook.t()
            + self.codebook.pow(2).sum(-1).unsqueeze(0)
        )
        return dist.argmin(dim=-1)

    def forward(self, x_nct):
        N, C, T = x_nct.shape
        x = x_nct.permute(0, 2, 1).reshape(-1, C)  # (NT, C)

        if self.training and not self.initialized:
            self._init(x)

        indices = self.quantize(x)
        quantized = F.embedding(indices, self.codebook)

        if self.training:
            onehot = F.one_hot(indices, self.num_codes).float()
            self.code_count.mul_(self.mu).add_(onehot.sum(0), alpha=1 - self.mu)
            self.code_sum.mul_(self.mu).add_(onehot.t() @ x, alpha=1 - self.mu)
            usage = (self.code_count >= 1.0).float().unsqueeze(-1)
            update = self.code_sum / self.code_count.unsqueeze(-1).clamp(min=1e-6)
            n = x.shape[0]
            reps = (self.num_codes + n - 1) // n
            rand = (x.repeat(reps, 1)[:self.num_codes]
                    + torch.randn(self.num_codes, C, device=x.device) * 0.01)
            self.codebook.copy_(usage * update + (1 - usage) * rand)

        commit_loss = F.mse_loss(x, quantized.detach())
        quantized_st = x + (quantized - x).detach()

        with torch.no_grad():
            avg = F.one_hot(indices, self.num_codes).float().mean(0)
            perplexity = torch.exp(-torch.sum(avg * torch.log(avg + 1e-7)))

        out = quantized_st.view(N, T, C).permute(0, 2, 1)
        return out, commit_loss, perplexity, indices.view(N, T)


# ---------------------------------------------------------------------------
# Cross-Part Fusion
# ---------------------------------------------------------------------------

class CrossPartFusion(nn.Module):
    """
    Lightweight transformer that lets part latents interact before decoding.
    At each timestep, the 4 part vectors attend to each other via multi-head
    self-attention.  This preserves part separation in the codebook while
    allowing the decoder to use cross-part context.
    """

    def __init__(self, code_dim: int, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=code_dim,
                nhead=num_heads,
                dim_feedforward=code_dim * 2,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

    def forward(self, part_latents: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        part_latents: list of 4 tensors, each (B, code_dim, T_compressed)
        Returns: list of 4 tensors, same shapes, fused.
        """
        B = part_latents[0].shape[0]
        C = part_latents[0].shape[1]
        T = part_latents[0].shape[2]
        n_parts = len(part_latents)

        stacked = torch.stack(part_latents, dim=2)  # (B, C, n_parts, T)
        stacked = stacked.permute(0, 3, 2, 1)  # (B, T, n_parts, C)
        stacked = stacked.reshape(B * T, n_parts, C)  # (B*T, n_parts, C)

        x = stacked
        for layer in self.layers:
            x = layer(x)

        x = x.view(B, T, n_parts, C)
        x = x.permute(0, 3, 2, 1)  # (B, C, n_parts, T)

        return [x[:, :, i, :] for i in range(n_parts)]


# ---------------------------------------------------------------------------
# Full Part-Aware VQ-VAE
# ---------------------------------------------------------------------------

PART_INPUT_DIMS = {
    "body": 63,
    "lhand": 45,
    "rhand": 45,
    "face_and_meta": 29,
}


class PartAwareVQVae(nn.Module):
    """
    Part-Aware VQ-VAE: independent encoder-quantizer per body part,
    cross-part fusion, then independent decoders.
    """

    PART_NAMES = ["body", "lhand", "rhand", "face_and_meta"]

    def __init__(
        self,
        code_num: int = 256,
        code_dim: int = 128,
        hidden_dim: int = 256,
        down_t: int = 3,
        stride_t: int = 2,
        depth: int = 3,
        dilation_growth_rate: int = 3,
        fusion_heads: int = 4,
        fusion_layers: int = 2,
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        self.commitment_weight = commitment_weight
        self.code_dim = code_dim
        self.code_num = code_num

        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        self.quantizers = nn.ModuleDict()

        for name in self.PART_NAMES:
            input_dim = PART_INPUT_DIMS[name]
            self.encoders[name] = PartEncoder(
                input_dim, hidden_dim, code_dim, down_t, stride_t,
                depth, dilation_growth_rate,
            )
            self.decoders[name] = PartDecoder(
                input_dim, hidden_dim, code_dim, down_t, stride_t,
                depth, dilation_growth_rate,
            )
            self.quantizers[name] = EMACodebook(code_num, code_dim)

        self.fusion = CrossPartFusion(code_dim, fusion_heads, fusion_layers)

    def forward(self, features: torch.Tensor):
        """
        features: (B, T, 182) raw SMPL-X
        Returns: (x_recon, total_commit_loss, info_dict)
        """
        parts = split_by_parts(features)
        encoded = {}
        quantized = {}
        commit_losses = {}
        perplexities = {}
        all_indices = {}

        for name in self.PART_NAMES:
            x_part = parts[name].permute(0, 2, 1)  # (B, C_part, T)
            enc = self.encoders[name](x_part)
            q, closs, perp, indices = self.quantizers[name](enc)
            encoded[name] = enc
            quantized[name] = q
            commit_losses[name] = closs
            perplexities[name] = perp
            all_indices[name] = indices

        # Cross-part fusion
        fused = self.fusion([quantized[n] for n in self.PART_NAMES])
        fused_dict = {name: fused[i] for i, name in enumerate(self.PART_NAMES)}

        # Decode each part
        decoded_parts = {}
        for name in self.PART_NAMES:
            dec = self.decoders[name](fused_dict[name])
            decoded_parts[name] = dec.permute(0, 2, 1)

        x_recon = merge_parts(decoded_parts)

        total_commit = sum(commit_losses.values()) * self.commitment_weight

        info = {
            "perplexities": perplexities,
            "commit_losses": commit_losses,
            "indices": all_indices,
        }
        return x_recon, total_commit, info

    def encode(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode to per-part indices."""
        parts = split_by_parts(features)
        result = {}
        for name in self.PART_NAMES:
            x_part = parts[name].permute(0, 2, 1)
            enc = self.encoders[name](x_part)
            _, _, _, indices = self.quantizers[name](enc)
            result[name] = indices
        return result

    def decode_from_indices(self, indices_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode from per-part indices."""
        quantized_list = []
        for name in self.PART_NAMES:
            idx = indices_dict[name]
            cb = self.quantizers[name].codebook
            q = F.embedding(idx, cb).permute(0, 2, 1)
            quantized_list.append(q)

        fused = self.fusion(quantized_list)
        decoded_parts = {}
        for i, name in enumerate(self.PART_NAMES):
            dec = self.decoders[name](fused[i])
            decoded_parts[name] = dec.permute(0, 2, 1)

        return merge_parts(decoded_parts)
