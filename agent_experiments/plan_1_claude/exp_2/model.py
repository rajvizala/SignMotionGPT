"""
Residual VQ-VAE (RQ-VAE): multi-level residual quantisation.
Ref: MOGO (arXiv:2506.05952), SoundStream (IEEE TASLP 2022), MoSa (arXiv:2511.01200)
"""

import math
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from models.resnet import Resnet1D
from models.vqvae import Encoder, Decoder


class ResidualQuantizer(nn.Module):
    def __init__(self, num_levels=4, num_codes=512, code_dim=512,
                 mu=0.99, commitment_weight=0.25, level_dropout=0.0):
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
        self.register_buffer("code_count", torch.ones(num_levels, num_codes))
        self.register_buffer("code_sum", torch.randn(num_levels, num_codes, code_dim) * 0.01)
        self.register_buffer("initialized", torch.zeros(num_levels, dtype=torch.bool))

    @staticmethod
    def _tile(x, n):
        cur_n, d = x.shape
        if cur_n >= n:
            return x[:n]
        reps = (n + cur_n - 1) // cur_n
        return (x.repeat(reps, 1)[:n] + torch.randn(n, d, device=x.device) * 0.01 / math.sqrt(d))

    def _init_level(self, lv, x):
        tiled = self._tile(x, self.num_codes)
        self.codebooks[lv].data.copy_(tiled)
        self.code_sum[lv].copy_(tiled)
        self.code_count[lv].fill_(1.0)
        self.initialized[lv] = True

    def _quantize(self, x, lv):
        cb = self.codebooks[lv]
        dist = x.pow(2).sum(-1, keepdim=True) - 2 * x @ cb.t() + cb.pow(2).sum(-1).unsqueeze(0)
        idx = dist.argmin(dim=-1)
        return F.embedding(idx, cb), idx

    @torch.no_grad()
    def _update_ema(self, lv, x, idx):
        oh = F.one_hot(idx, self.num_codes).float()
        self.code_count[lv] = self.mu * self.code_count[lv] + (1 - self.mu) * oh.sum(0)
        self.code_sum[lv] = self.mu * self.code_sum[lv] + (1 - self.mu) * (oh.t() @ x)
        usage = (self.code_count[lv] >= 1.0).float().unsqueeze(-1)
        update = self.code_sum[lv] / self.code_count[lv].unsqueeze(-1).clamp(min=1e-6)
        replacement = self._tile(x, self.num_codes)
        self.codebooks[lv].data.copy_(usage * update + (1 - usage) * replacement)

    def forward(self, x):
        N, C, T = x.shape
        x_flat = x.permute(0, 2, 1).reshape(-1, C)
        if self.training and self.level_dropout > 0:
            max_lv = max(1, int(self.num_levels * (1 - self.level_dropout * torch.rand(1).item())))
        else:
            max_lv = self.num_levels
        residual = x_flat.clone()
        total_q = torch.zeros_like(x_flat)
        total_commit = 0.0
        all_idx, all_perp = [], []
        for lv in range(max_lv):
            if self.training and not self.initialized[lv]:
                self._init_level(lv, residual)
            q, idx = self._quantize(residual, lv)
            if self.training:
                self._update_ema(lv, residual, idx)
            total_commit = total_commit + F.mse_loss(residual, q.detach()) * self.commitment_weight
            q_st = residual + (q - residual).detach()
            total_q = total_q + q_st
            residual = residual - q.detach()
            all_idx.append(idx.view(N, T))
            with torch.no_grad():
                avg = F.one_hot(idx, self.num_codes).float().mean(0)
                all_perp.append(torch.exp(-(avg * torch.log(avg + 1e-7)).sum()))
        return total_q.view(N, T, C).permute(0, 2, 1), total_commit, {"indices": all_idx, "perplexities": all_perp, "active_levels": max_lv}

    def encode(self, x_flat):
        residual, idx_list = x_flat.clone(), []
        for lv in range(self.num_levels):
            q, idx = self._quantize(residual, lv)
            residual = residual - q
            idx_list.append(idx)
        return idx_list

    def decode(self, idx_list):
        total = torch.zeros(idx_list[0].shape[0], self.code_dim, device=idx_list[0].device)
        for lv, idx in enumerate(idx_list):
            total = total + F.embedding(idx, self.codebooks[lv])
        return total


class RQVae(nn.Module):
    def __init__(self, nfeats=182, num_rq_levels=4, code_num=512, code_dim=512,
                 output_emb_width=512, down_t=3, stride_t=2, width=512,
                 depth=3, dilation_growth_rate=3, commitment_weight=0.25,
                 level_dropout=0.1, **kwargs):
        super().__init__()
        self.code_dim = code_dim
        self.num_rq_levels = num_rq_levels
        self.encoder = Encoder(nfeats, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate)
        self.decoder = Decoder(nfeats, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate)
        self.quantizer = ResidualQuantizer(num_rq_levels, code_num, code_dim, commitment_weight=commitment_weight, level_dropout=level_dropout)

    def forward(self, features):
        x_in = features.permute(0, 2, 1)
        x_enc = self.encoder(x_in)
        x_q, commit, info = self.quantizer(x_enc)
        x_dec = self.decoder(x_q)
        x_out = x_dec.permute(0, 2, 1)
        return x_out, commit, info

    def encode(self, features):
        N, T, _ = features.shape
        x_in = features.permute(0, 2, 1)
        x_enc = self.encoder(x_in)
        x_flat = x_enc.permute(0, 2, 1).reshape(-1, self.code_dim)
        return self.quantizer.encode(x_flat)
