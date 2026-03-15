"""Configuration for Exp 2: Residual VQ-VAE."""

import os
from dataclasses import dataclass


@dataclass
class Exp2Config:
    vqvae_ckpt: str = os.environ.get("VQVAE_CHECKPOINT", "")
    stats_path: str = os.environ.get("VQVAE_STATS_PATH", "computed_stats.pt")
    data_dir: str = ""
    val_dir: str = ""
    output_dir: str = "./agent_experiments/plan_1_claude/outputs/exp_2"

    nfeats: int = 182
    num_rq_levels: int = 4
    code_num: int = 512
    code_dim: int = 512
    output_emb_width: int = 512
    width: int = 512
    down_t: int = 3
    stride_t: int = 2
    depth: int = 3
    dilation_growth_rate: int = 3
    commitment_weight: float = 0.25
    level_dropout: float = 0.1

    epochs: int = 200
    batch_size: int = 32
    lr: float = 2e-4
    weight_decay: float = 1e-4
    seed: int = 42
    patience: int = 30

    swa_start_pct: float = 0.8
    eval_every: int = 5
    dry_run: bool = False
