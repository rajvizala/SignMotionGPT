"""
Part-Aware VQ-VAE: independent encoder-quantizer per body part,
cross-part attention fusion, then independent decoders.

Ref: MotionGPT-2 (arXiv:2410.21747), SOKE (ICCV 2025)
"""

import math
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from models.resnet import Resnet1D

PART_INPUT_DIMS = {"body": 63, "lhand": 45, "rhand": 45, "face_meta": 29}
PART_NAMES = ["body", "lhand", "rhand", "face_meta"]


def split_by_parts(x):
    parts = {}
    parts["body"] = x[:, :, 10:73]
    parts["lhand"] = x[:, :, 73:118]
    parts["rhand"] = x[:, :, 118:163]
    parts["face_meta"] = torch.cat([x[:, :, 0:10], x[:, :, 163:176], x[:, :, 176:182]], dim=-1)
    return parts


def merge_parts(parts):
    fm = parts["face_meta"]
    return torch.cat([fm[:, :, :10], parts["body"], parts["lhand"],
                      parts["rhand"], fm[:, :, 10:23], fm[:, :, 23:29]], dim=-1)


class PartEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, code_dim=128,
                 down_t=3, stride_t=2, depth=3, dilation_growth_rate=3):
        super().__init__()
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_dim, hidden_dim, 3, 1, 1))
        blocks.append(nn.GELU())
        for _ in range(down_t):
            blocks.append(nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, filter_t, stride_t, pad_t),
                Resnet1D(hidden_dim, depth, dilation_growth_rate, activation='gelu'),
            ))
        blocks.append(nn.Conv1d(hidden_dim, code_dim, 3, 1, 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


class PartDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dim=256, code_dim=128,
                 down_t=3, stride_t=2, depth=3, dilation_growth_rate=3):
        super().__init__()
        blocks = []
        blocks.append(nn.Conv1d(code_dim, hidden_dim, 3, 1, 1))
        blocks.append(nn.GELU())
        for _ in range(down_t):
            blocks.append(nn.Sequential(
                Resnet1D(hidden_dim, depth, dilation_growth_rate,
                         reverse_dilation=True, activation='gelu'),
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
    def __init__(self, num_codes, code_dim, mu=0.99):
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

    def forward(self, x_nct):
        N, C, T = x_nct.shape
        x = x_nct.permute(0, 2, 1).reshape(-1, C)
        if self.training and not self.initialized:
            self._init(x)
        dist = x.pow(2).sum(-1, keepdim=True) - 2 * x @ self.codebook.t() + self.codebook.pow(2).sum(-1).unsqueeze(0)
        indices = dist.argmin(dim=-1)
        quantized = F.embedding(indices, self.codebook)
        if self.training:
            onehot = F.one_hot(indices, self.num_codes).float()
            self.code_count.mul_(self.mu).add_(onehot.sum(0), alpha=1 - self.mu)
            self.code_sum.mul_(self.mu).add_(onehot.t() @ x, alpha=1 - self.mu)
            usage = (self.code_count >= 1.0).float().unsqueeze(-1)
            update = self.code_sum / self.code_count.unsqueeze(-1).clamp(min=1e-6)
            n2 = x.shape[0]
            reps = (self.num_codes + n2 - 1) // n2
            rand = (x.repeat(reps, 1)[:self.num_codes] + torch.randn(self.num_codes, C, device=x.device) * 0.01)
            self.codebook.copy_(usage * update + (1 - usage) * rand)
        commit_loss = F.mse_loss(x, quantized.detach())
        quantized_st = x + (quantized - x).detach()
        with torch.no_grad():
            avg = F.one_hot(indices, self.num_codes).float().mean(0)
            perplexity = torch.exp(-torch.sum(avg * torch.log(avg + 1e-7)))
        out = quantized_st.view(N, T, C).permute(0, 2, 1)
        return out, commit_loss, perplexity, indices.view(N, T)


class CrossPartFusion(nn.Module):
    def __init__(self, code_dim, num_heads=4, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=code_dim, nhead=num_heads,
                dim_feedforward=code_dim * 2, dropout=0.1,
                activation='gelu', batch_first=True,
            ) for _ in range(num_layers)
        ])

    def forward(self, part_latents):
        B, C, T = part_latents[0].shape
        n_parts = len(part_latents)
        stacked = torch.stack(part_latents, dim=2).permute(0, 3, 2, 1).reshape(B * T, n_parts, C)
        x = stacked
        for layer in self.layers:
            x = layer(x)
        x = x.view(B, T, n_parts, C).permute(0, 3, 2, 1)
        return [x[:, :, i, :] for i in range(n_parts)]


class PartAwareVQVae(nn.Module):
    def __init__(self, code_num=256, code_dim=128, hidden_dim=256,
                 down_t=3, stride_t=2, depth=3, dilation_growth_rate=3,
                 fusion_heads=4, fusion_layers=2, commitment_weight=0.25):
        super().__init__()
        self.commitment_weight = commitment_weight
        self.code_dim = code_dim
        self.code_num = code_num
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        self.quantizers = nn.ModuleDict()
        for name in PART_NAMES:
            dim = PART_INPUT_DIMS[name]
            self.encoders[name] = PartEncoder(dim, hidden_dim, code_dim, down_t, stride_t, depth, dilation_growth_rate)
            self.decoders[name] = PartDecoder(dim, hidden_dim, code_dim, down_t, stride_t, depth, dilation_growth_rate)
            self.quantizers[name] = EMACodebook(code_num, code_dim)
        self.fusion = CrossPartFusion(code_dim, fusion_heads, fusion_layers)

    def forward(self, features):
        parts = split_by_parts(features)
        quantized_list, commit_losses, perplexities, all_indices = [], {}, {}, {}
        for name in PART_NAMES:
            enc = self.encoders[name](parts[name].permute(0, 2, 1))
            q, cl, perp, idx = self.quantizers[name](enc)
            quantized_list.append(q)
            commit_losses[name] = cl
            perplexities[name] = perp
            all_indices[name] = idx
        fused = self.fusion(quantized_list)
        decoded = {}
        for i, name in enumerate(PART_NAMES):
            decoded[name] = self.decoders[name](fused[i]).permute(0, 2, 1)
        x_recon = merge_parts(decoded)
        total_commit = sum(commit_losses.values()) * self.commitment_weight
        return x_recon, total_commit, {"perplexities": perplexities, "commit_losses": commit_losses, "indices": all_indices}
