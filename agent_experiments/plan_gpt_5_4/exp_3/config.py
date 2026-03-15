from __future__ import annotations

import argparse
from dataclasses import dataclass

from agent_experiments.plan_gpt_5_4.shared.data_utils import default_stats_path, default_vqvae_ckpt


@dataclass
class Exp3Config:
    train_dir: str = ""
    val_dir: str = ""
    word_data_dir: str = ""
    output_dir: str = "./agent_experiments/plan_gpt_5_4/runs/exp_3"
    vqvae_ckpt: str = default_vqvae_ckpt()
    stats_path: str = default_stats_path()
    epochs: int = 20
    batch_size: int = 8
    eval_batch_size: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    vq_loss_weight: float = 0.25
    velocity_weight: float = 0.5
    hand_weight: float = 1.0
    codebook_balance_weight: float = 0.05
    temporal_warp_prob: float = 0.5
    hand_jitter_std: float = 0.02
    max_samples: int = 2048
    patience: int = 4
    num_workers: int = 0
    seed: int = 42
    dry_run: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 3: narrow single-stream tokenizer rescue.")
    parser.add_argument("--train-dir", default=Exp3Config.train_dir)
    parser.add_argument("--val-dir", default=Exp3Config.val_dir)
    parser.add_argument("--word-data-dir", default=Exp3Config.word_data_dir)
    parser.add_argument("--output-dir", default=Exp3Config.output_dir)
    parser.add_argument("--vqvae-ckpt", default=Exp3Config.vqvae_ckpt)
    parser.add_argument("--stats-path", default=Exp3Config.stats_path)
    parser.add_argument("--epochs", type=int, default=Exp3Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Exp3Config.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=Exp3Config.eval_batch_size)
    parser.add_argument("--learning-rate", type=float, default=Exp3Config.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=Exp3Config.weight_decay)
    parser.add_argument("--max-grad-norm", type=float, default=Exp3Config.max_grad_norm)
    parser.add_argument("--vq-loss-weight", type=float, default=Exp3Config.vq_loss_weight)
    parser.add_argument("--velocity-weight", type=float, default=Exp3Config.velocity_weight)
    parser.add_argument("--hand-weight", type=float, default=Exp3Config.hand_weight)
    parser.add_argument("--codebook-balance-weight", type=float, default=Exp3Config.codebook_balance_weight)
    parser.add_argument("--temporal-warp-prob", type=float, default=Exp3Config.temporal_warp_prob)
    parser.add_argument("--hand-jitter-std", type=float, default=Exp3Config.hand_jitter_std)
    parser.add_argument("--max-samples", type=int, default=Exp3Config.max_samples)
    parser.add_argument("--patience", type=int, default=Exp3Config.patience)
    parser.add_argument("--num-workers", type=int, default=Exp3Config.num_workers)
    parser.add_argument("--seed", type=int, default=Exp3Config.seed)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def from_args(args: argparse.Namespace) -> Exp3Config:
    return Exp3Config(**vars(args))
