"""Configuration for Exp 1: Part-Aware VQ-VAE."""

import os
from dataclasses import dataclass


@dataclass
class Exp1Config:
    vqvae_ckpt: str = os.environ.get("VQVAE_CHECKPOINT", "")
    stats_path: str = os.environ.get("VQVAE_STATS_PATH", "computed_stats.pt")
    data_dir: str = ""
    val_dir: str = ""
    output_dir: str = "./agent_experiments/plan_1_claude/outputs/exp_1"

    nfeats: int = 182
    code_num: int = 256
    code_dim: int = 128
    hidden_dim: int = 256
    down_t: int = 3
    stride_t: int = 2
    depth: int = 3
    dilation_growth_rate: int = 3
    fusion_heads: int = 4
    fusion_layers: int = 2
    commitment_weight: float = 0.25

    epochs: int = 200
    batch_size: int = 32
    lr: float = 2e-4
    weight_decay: float = 1e-4
    seed: int = 42
    patience: int = 30

    temporal_warp_prob: float = 0.3
    noise_prob: float = 0.5
    noise_std: float = 0.02
    hand_jitter_prob: float = 0.4
    hand_jitter_std: float = 0.03
    mirror_prob: float = 0.3

    swa_start_pct: float = 0.8
    eval_every: int = 5
    dry_run: bool = False
