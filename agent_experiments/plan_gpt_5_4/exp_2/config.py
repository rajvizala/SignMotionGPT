from __future__ import annotations

import argparse
from dataclasses import dataclass

from agent_experiments.plan_gpt_5_4.shared.data_utils import default_json_path, default_stats_path, default_vqvae_ckpt


@dataclass
class Exp2Config:
    train_json: str = default_json_path()
    val_json: str = default_json_path()
    word_json: str = default_json_path()
    output_dir: str = "./agent_experiments/plan_gpt_5_4/runs/exp_2"
    model_name: str = "Qwen/Qwen3-0.6B"
    vqvae_ckpt: str = default_vqvae_ckpt()
    stats_path: str = default_stats_path()
    val_motion_dir: str = ""
    word_data_dir: str = ""
    eval_max_motion_samples: int = 128
    epochs: int = 12
    batch_size: int = 2
    eval_batch_size: int = 2
    max_seq_len: int = 384
    max_motion_tokens: int = 256
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    replay_ratio: float = 0.1
    denoise_weight: float = 0.3
    mask_rate: float = 0.15
    max_context_words: int = 5
    anchor_tokens: int = 6
    lexicon_context_prob: float = 0.8
    alignment_dim: int = 256
    alignment_weight: float = 0.05
    alignment_temperature: float = 0.07
    max_position_offset: int = 2048
    patience: int = 3
    num_workers: int = 0
    seed: int = 42
    dry_run: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 2: lexicon memory plus alignment and randomized positions.")
    parser.add_argument("--train-json", default=Exp2Config.train_json)
    parser.add_argument("--val-json", default=Exp2Config.val_json)
    parser.add_argument("--word-json", default=Exp2Config.word_json)
    parser.add_argument("--output-dir", default=Exp2Config.output_dir)
    parser.add_argument("--model-name", default=Exp2Config.model_name)
    parser.add_argument("--vqvae-ckpt", default=Exp2Config.vqvae_ckpt)
    parser.add_argument("--stats-path", default=Exp2Config.stats_path)
    parser.add_argument("--val-motion-dir", default=Exp2Config.val_motion_dir)
    parser.add_argument("--word-data-dir", default=Exp2Config.word_data_dir)
    parser.add_argument("--eval-max-motion-samples", type=int, default=Exp2Config.eval_max_motion_samples)
    parser.add_argument("--epochs", type=int, default=Exp2Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Exp2Config.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=Exp2Config.eval_batch_size)
    parser.add_argument("--max-seq-len", type=int, default=Exp2Config.max_seq_len)
    parser.add_argument("--max-motion-tokens", type=int, default=Exp2Config.max_motion_tokens)
    parser.add_argument("--learning-rate", type=float, default=Exp2Config.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=Exp2Config.weight_decay)
    parser.add_argument("--max-grad-norm", type=float, default=Exp2Config.max_grad_norm)
    parser.add_argument("--replay-ratio", type=float, default=Exp2Config.replay_ratio)
    parser.add_argument("--denoise-weight", type=float, default=Exp2Config.denoise_weight)
    parser.add_argument("--mask-rate", type=float, default=Exp2Config.mask_rate)
    parser.add_argument("--max-context-words", type=int, default=Exp2Config.max_context_words)
    parser.add_argument("--anchor-tokens", type=int, default=Exp2Config.anchor_tokens)
    parser.add_argument("--lexicon-context-prob", type=float, default=Exp2Config.lexicon_context_prob)
    parser.add_argument("--alignment-dim", type=int, default=Exp2Config.alignment_dim)
    parser.add_argument("--alignment-weight", type=float, default=Exp2Config.alignment_weight)
    parser.add_argument("--alignment-temperature", type=float, default=Exp2Config.alignment_temperature)
    parser.add_argument("--max-position-offset", type=int, default=Exp2Config.max_position_offset)
    parser.add_argument("--patience", type=int, default=Exp2Config.patience)
    parser.add_argument("--num-workers", type=int, default=Exp2Config.num_workers)
    parser.add_argument("--seed", type=int, default=Exp2Config.seed)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def from_args(args: argparse.Namespace) -> Exp2Config:
    return Exp2Config(**vars(args))
