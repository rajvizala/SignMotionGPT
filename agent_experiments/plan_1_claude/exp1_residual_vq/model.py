"""
Experiment 1 -- Residual Vector Quantization VQ-VAE (RQ-VAE)

REFERENCES
----------
[1] Zeghidour et al., "SoundStream: An End-to-End Neural Audio Codec",
    IEEE/ACM Trans. Audio Speech Lang. Process., 2022.
[2] Lee et al., "Autoregressive Image Generation using Residual Quantization",
    CVPR 2022.
[3] MOGO: "Residual Quantized Hierarchical Causal Transformer for Real-Time
    and Infinite-Length 3D Human Motion Generation", arXiv:2506.05952, 2025.
[4] MoSa: "Motion Generation with Scalable Autoregressive Modeling",
    arXiv:2511.01200, 2024.

CORE IDEA
---------
The baseline VQ-VAE uses a single-level codebook of 512 codes.  When the
motion vocabulary is complex (word-level *and* sentence-level data), a flat
codebook either:
  (a) collapses -- many codes go unused, or
  (b) over-specialises -- codes become so specific they fail on unseen motions.

Residual Quantization fixes this by decomposing the latent vector into
*multiple* quantization levels.  Level 0 captures the coarse structure;
each subsequent level quantizes the *residual* error from the previous level.
The final representation is the sum of all quantized vectors.

WHY THIS SHOULD HELP GENERALIZATION
------------------------------------
1. **Information factorisation**: coarse codes capture motion *categories*
   (e.g. "hand raise"), fine codes capture *style* (e.g. speed, amplitude).
   An unseen sentence can re-use the coarse codes with novel fine codes.
2. **Higher effective codebook capacity**: with D levels and K codes each,
   the effective vocabulary is K^D without actually storing K^D vectors.
3. **Reduced codebook collapse**: each level only needs to represent
   residual information, so code utilisation is naturally higher.
4. **Multi-scale LLM generation**: the LLM can be trained to first predict
   coarse tokens, then refine -- matching the coarse-to-fine curriculum
   humans use when learning sign language.
"""

import math
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from models.resnet import Resnet1D


# ---------------------------------------------------------------------------
# Residual Quantizer
# ---------------------------------------------------------------------------

class ResidualQuantizer(nn.Module):
    """
    Multi-level residual vector quantization.

    At each level q in [0, num_levels), we quantize the current residual
    against codebook_q, subtract the quantized vector, and pass the new
    residual to the next level.  The final quantized vector is the sum
    over all levels.

    Parameters
    ----------
    num_levels : int
        Number of quantization levels (D).  Each adds K codes.
    num_codes : int
        Codes per level (K).
    code_dim : int
        Dimensionality of each code vector.
    mu : float
        EMA decay for codebook updates.
    commitment_weight : float
        Beta for the commitment loss (encoder -> codebook).
    level_dropout : float
        During training, randomly drop higher RQ levels with this
        probability to encourage the base level to be self-sufficient.
        Inspired by dropout regularisation in SoundStream [1].
    """

    def __init__(
        self,
        num_levels: int = 4,
        num_codes: int = 512,
        code_dim: int = 512,
        mu: float = 0.99,
        commitment_weight: float = 0.25,
        level_dropout: float = 0.0,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.mu = mu
        self.commitment_weight = commitment_weight
        self.level_dropout = level_dropout

        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(num_codes, code_dim) * 0.01)
            for _ in range(num_levels)
        ])

        # EMA statistics per level
        self.register_buffer(
            "code_count",
            torch.ones(num_levels, num_codes),
        )
        self.register_buffer(
            "code_sum",
            torch.randn(num_levels, num_codes, code_dim) * 0.01,
        )
        self.register_buffer("initialized", torch.zeros(num_levels, dtype=torch.bool))

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _tile_to(x: torch.Tensor, target_n: int) -> torch.Tensor:
        n, d = x.shape
        if n >= target_n:
            return x[:target_n]
        reps = (target_n + n - 1) // n
        out = x.repeat(reps, 1)[:target_n]
        out = out + torch.randn_like(out) * (0.01 / math.sqrt(d))
        return out

    def _init_codebook(self, level: int, x: torch.Tensor):
        tiled = self._tile_to(x, self.num_codes)
        self.codebooks[level].data.copy_(tiled)
        self.code_sum[level].copy_(tiled)
        self.code_count[level].fill_(1.0)
        self.initialized[level] = True

    def _quantize_level(self, x_flat: torch.Tensor, level: int):
        """Quantize flat (N, D) against codebook[level]."""
        cb = self.codebooks[level]  # (K, D)
        dist = (
            x_flat.pow(2).sum(-1, keepdim=True)
            - 2 * x_flat @ cb.t()
            + cb.pow(2).sum(-1, keepdim=True).t()
        )
        indices = dist.argmin(dim=-1)  # (N,)
        quantized = F.embedding(indices, cb)  # (N, D)
        return quantized, indices

    @torch.no_grad()
    def _update_codebook_ema(self, level: int, x_flat: torch.Tensor, indices: torch.Tensor):
        onehot = F.one_hot(indices, self.num_codes).float()  # (N, K)
        count = onehot.sum(0)  # (K,)
        code_sum = onehot.t() @ x_flat  # (K, D)

        self.code_count[level] = self.mu * self.code_count[level] + (1 - self.mu) * count
        self.code_sum[level] = self.mu * self.code_sum[level] + (1 - self.mu) * code_sum

        usage = (self.code_count[level] >= 1.0).float().unsqueeze(-1)
        update = self.code_sum[level] / self.code_count[level].unsqueeze(-1).clamp(min=1e-6)
        replacement = self._tile_to(x_flat, self.num_codes)
        self.codebooks[level].data.copy_(usage * update + (1 - usage) * replacement)

    # -- forward -----------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (N, C, T)   -- encoder output in channels-first layout.

        Returns
        -------
        x_quantized : (N, C, T) -- sum of all quantized levels.
        commit_loss : scalar
        info : dict with per-level perplexity, indices, etc.
        """
        N, C, T = x.shape
        x_flat = x.permute(0, 2, 1).reshape(-1, C)  # (N*T, C)

        # Decide how many levels to use this step (level dropout)
        if self.training and self.level_dropout > 0:
            max_levels = max(1, int(self.num_levels * (1 - self.level_dropout * torch.rand(1).item())))
        else:
            max_levels = self.num_levels

        residual = x_flat.clone()
        total_quantized = torch.zeros_like(x_flat)
        total_commit_loss = 0.0
        all_indices = []
        all_perplexities = []

        for lv in range(max_levels):
            if self.training and not self.initialized[lv]:
                self._init_codebook(lv, residual)

            quantized, indices = self._quantize_level(residual, lv)

            if self.training:
                self._update_codebook_ema(lv, residual, indices)

            commit_loss = F.mse_loss(residual, quantized.detach())
            total_commit_loss = total_commit_loss + commit_loss * self.commitment_weight

            # Straight-through estimator
            quantized_st = residual + (quantized - residual).detach()
            total_quantized = total_quantized + quantized_st
            residual = residual - quantized.detach()

            all_indices.append(indices.view(N, T))

            # Perplexity
            with torch.no_grad():
                onehot = F.one_hot(indices, self.num_codes).float()
                avg = onehot.mean(0)
                perp = torch.exp(-torch.sum(avg * torch.log(avg + 1e-7)))
                all_perplexities.append(perp)

        x_quantized = total_quantized.view(N, T, C).permute(0, 2, 1)  # (N, C, T)

        info = {
            "indices": all_indices,
            "perplexities": all_perplexities,
            "num_active_levels": max_levels,
        }
        return x_quantized, total_commit_loss, info

    # -- encode / decode helpers for inference ----------------------------

    def encode(self, x_flat: torch.Tensor) -> List[torch.Tensor]:
        """Encode (N*T, C) -> list of (N*T,) index tensors per level."""
        residual = x_flat
        all_idx = []
        for lv in range(self.num_levels):
            _, idx = self._quantize_level(residual, lv)
            quantized = F.embedding(idx, self.codebooks[lv])
            residual = residual - quantized
            all_idx.append(idx)
        return all_idx

    def decode(self, index_lists: List[torch.Tensor]) -> torch.Tensor:
        """Decode list of index tensors -> (N*T, C) reconstructed."""
        total = torch.zeros(index_lists[0].shape[0], self.code_dim,
                            device=index_lists[0].device)
        for lv, idx in enumerate(index_lists):
            total = total + F.embedding(idx, self.codebooks[lv])
        return total


# ---------------------------------------------------------------------------
# RQ-VAE Model
# ---------------------------------------------------------------------------

class RQVae(nn.Module):
    """
    VQ-VAE with Residual Quantization replacing the single-level quantizer.

    The encoder / decoder architecture is identical to the baseline VQVae
    from models/vqvae.py so that we can do controlled comparisons.
    """

    def __init__(
        self,
        nfeats: int = 182,
        num_rq_levels: int = 4,
        code_num: int = 512,
        code_dim: int = 512,
        output_emb_width: int = 512,
        down_t: int = 3,
        stride_t: int = 2,
        width: int = 512,
        depth: int = 3,
        dilation_growth_rate: int = 3,
        norm: Optional[str] = None,
        activation: str = "relu",
        commitment_weight: float = 0.25,
        level_dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.code_dim = code_dim
        self.num_rq_levels = num_rq_levels

        self.encoder = _Encoder(
            nfeats, output_emb_width, down_t, stride_t,
            width, depth, dilation_growth_rate, activation, norm,
        )
        self.decoder = _Decoder(
            nfeats, output_emb_width, down_t, stride_t,
            width, depth, dilation_growth_rate, activation, norm,
        )
        self.quantizer = ResidualQuantizer(
            num_levels=num_rq_levels,
            num_codes=code_num,
            code_dim=code_dim,
            commitment_weight=commitment_weight,
            level_dropout=level_dropout,
        )

    def preprocess(self, x):
        return x.permute(0, 2, 1)  # (B, T, C) -> (B, C, T)

    def postprocess(self, x):
        return x.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)

    def forward(self, features: torch.Tensor):
        x_in = self.preprocess(features)
        x_enc = self.encoder(x_in)
        x_q, commit_loss, info = self.quantizer(x_enc)
        x_dec = self.decoder(x_q)
        x_out = self.postprocess(x_dec)
        return x_out, commit_loss, info

    def encode(self, features: torch.Tensor) -> Tuple[List[torch.Tensor], int]:
        """Encode features -> list of code-index tensors per RQ level."""
        N, T, _ = features.shape
        x_in = self.preprocess(features)
        x_enc = self.encoder(x_in)
        x_enc_flat = x_enc.permute(0, 2, 1).reshape(-1, self.code_dim)
        indices = self.quantizer.encode(x_enc_flat)
        indices = [idx.view(N, -1) for idx in indices]
        return indices, x_enc.shape[2]  # indices, compressed_len

    def decode_from_indices(self, index_lists: List[torch.Tensor]) -> torch.Tensor:
        """Decode from per-level code indices -> reconstructed features."""
        N, T_compressed = index_lists[0].shape
        flat_indices = [idx.reshape(-1) for idx in index_lists]
        x_flat = self.quantizer.decode(flat_indices)  # (N*T_c, C)
        x_q = x_flat.view(N, T_compressed, self.code_dim).permute(0, 2, 1)
        x_dec = self.decoder(x_q)
        return self.postprocess(x_dec)


# ---------------------------------------------------------------------------
# Encoder / Decoder (mirrors models/vqvae.py exactly)
# ---------------------------------------------------------------------------

class _Encoder(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, down_t, stride_t,
                 width, depth, dilation_growth_rate, activation, norm):
        super().__init__()
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for _ in range(down_t):
            block = nn.Sequential(
                nn.Conv1d(width, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class _Decoder(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, down_t, stride_t,
                 width, depth, dilation_growth_rate, activation, norm):
        super().__init__()
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for _ in range(down_t):
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True,
                         activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, width, 3, 1, 1),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
