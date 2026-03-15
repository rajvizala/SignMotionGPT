"""Configuration for Exp 0: Baseline Evaluation."""

import os
from dataclasses import dataclass, field


@dataclass
class Exp0Config:
    vqvae_ckpt: str = os.environ.get("VQVAE_CHECKPOINT", "")
    stats_path: str = os.environ.get("VQVAE_STATS_PATH", "computed_stats.pt")
    val_dir: str = ""
    word_data_dir: str = ""
    output_dir: str = "./agent_experiments/plan_1_claude/outputs/exp_0"
    codebook_size: int = 512
    code_dim: int = 512
    nfeats: int = 182
