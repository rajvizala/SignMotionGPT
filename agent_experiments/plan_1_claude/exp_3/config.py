"""Configuration for Exp 3: LLM with Denoising + RAG + SWA."""

import os
from dataclasses import dataclass


@dataclass
class Exp3Config:
    vqvae_ckpt: str = os.environ.get("VQVAE_CHECKPOINT", "")
    stats_path: str = os.environ.get("VQVAE_STATS_PATH", "computed_stats.pt")
    sentence_data_path: str = ""
    word_data_path: str = ""
    output_dir: str = "./agent_experiments/plan_1_claude/outputs/exp_3"

    model_name: str = "Qwen/Qwen3-0.6B"
    max_seq_len: int = 384

    epochs: int = 40
    batch_size: int = 16
    lr: float = 3e-5
    weight_decay: float = 0.01
    seed: int = 42
    patience: int = 10

    denoise_mask_rate: float = 0.15
    denoise_weight: float = 0.3
    rag_context_prob: float = 0.8
    rag_max_words: int = 5
    rag_max_tokens_per_word: int = 15
    replay_ratio: float = 0.1
    swa_start_pct: float = 0.8

    eval_every: int = 2
    dry_run: bool = False
