from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mGPT.archs.mgpt_vq import VQVae

from agent_experiments.plan_gpt_5_4.shared.data_utils import HAND_END, HAND_START, SMPL_DIM, VQ_CONFIG


@dataclass
class RepairLossWeights:
    vq_loss_weight: float = 0.25
    velocity_weight: float = 0.5
    hand_weight: float = 1.0
    codebook_balance_weight: float = 0.05


class SingleStreamRepairModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vqvae = VQVae(**VQ_CONFIG)
        loss_weights = torch.ones(SMPL_DIM)
        loss_weights[HAND_START:HAND_END] = 12.0
        self.register_buffer("loss_weights", loss_weights)

    def forward(self, x: torch.Tensor):
        return self.vqvae(x)

    def encode(self, x: torch.Tensor):
        return self.vqvae.encode(x)

    def decode(self, z: torch.Tensor):
        return self.vqvae.decode(z)


def temporal_warp(motion: torch.Tensor, probability: float) -> torch.Tensor:
    if random.random() > probability or motion.shape[0] < 4:
        return motion
    scale = random.uniform(0.9, 1.1)
    target_len = max(4, int(round(motion.shape[0] * scale)))
    motion_t = motion.transpose(0, 1).unsqueeze(0)
    warped = F.interpolate(motion_t, size=target_len, mode="linear", align_corners=False)
    return warped.squeeze(0).transpose(0, 1)


def hand_jitter(motion: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0:
        return motion
    jitter = torch.zeros_like(motion)
    jitter[:, HAND_START:HAND_END] = torch.randn_like(motion[:, HAND_START:HAND_END]) * std
    return motion + jitter


def compute_repair_loss(
    model: SingleStreamRepairModel,
    motion: torch.Tensor,
    lengths: torch.Tensor,
    weights: RepairLossWeights,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    reconstruction, vq_loss, perplexity = model(motion)
    mask = torch.zeros_like(motion[:, :, 0])
    for index, length in enumerate(lengths.tolist()):
        mask[index, :length] = 1.0

    recon_raw = F.smooth_l1_loss(reconstruction, motion, reduction="none") * model.loss_weights
    recon_mask = mask.unsqueeze(-1).expand_as(recon_raw)
    recon_loss = (recon_raw * recon_mask).sum() / recon_mask.sum().clamp_min(1.0)

    vel_target = motion[:, 1:, :] - motion[:, :-1, :]
    vel_recon = reconstruction[:, 1:, :] - reconstruction[:, :-1, :]
    vel_mask = mask[:, 1:].unsqueeze(-1).expand_as(vel_target)
    velocity_loss = ((vel_target - vel_recon) ** 2 * vel_mask).sum() / vel_mask.sum().clamp_min(1.0)

    hand_mask = mask.unsqueeze(-1).expand(-1, -1, HAND_END - HAND_START)
    hand_loss = (
        ((reconstruction[:, :, HAND_START:HAND_END] - motion[:, :, HAND_START:HAND_END]) ** 2 * hand_mask).sum()
        / hand_mask.sum().clamp_min(1.0)
    )

    codebook_penalty = torch.relu(torch.tensor(200.0, device=motion.device) - perplexity) / 200.0
    total_loss = (
        recon_loss
        + weights.vq_loss_weight * vq_loss
        + weights.velocity_weight * velocity_loss
        + weights.hand_weight * hand_loss
        + weights.codebook_balance_weight * codebook_penalty
    )
    metrics = {
        "total_loss": float(total_loss.item()),
        "recon_loss": float(recon_loss.item()),
        "vq_loss": float(vq_loss.item()),
        "velocity_loss": float(velocity_loss.item()),
        "hand_loss": float(hand_loss.item()),
        "perplexity": float(perplexity.item()),
    }
    return total_loss, metrics
