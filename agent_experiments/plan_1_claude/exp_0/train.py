"""
Exp 0: Baseline Evaluation -- establishes bucketed metrics for current VQ-VAE.

Usage:
    python -m agent_experiments.plan_1_claude.exp_0.train \
        --vqvae-ckpt path/to/checkpoint.pt \
        --val-dir path/to/val_npz \
        [--stats-path computed_stats.pt] \
        [--dry-run]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from agent_experiments.plan_1_claude.shared.eval_harness import (
    load_vqvae, load_npz_sequences, evaluate_vqvae,
)
from agent_experiments.plan_1_claude.exp_0.config import Exp0Config


def main():
    parser = argparse.ArgumentParser(description="Exp 0: Baseline VQ-VAE Evaluation")
    cfg = Exp0Config()
    parser.add_argument("--vqvae-ckpt", default=cfg.vqvae_ckpt, required=not bool(cfg.vqvae_ckpt))
    parser.add_argument("--val-dir", default=cfg.val_dir, required=not bool(cfg.val_dir))
    parser.add_argument("--stats-path", default=cfg.stats_path)
    parser.add_argument("--output-dir", default=cfg.output_dir)
    parser.add_argument("--word-data-dir", default=cfg.word_data_dir)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    import torch
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  Exp 0: Baseline VQ-VAE Evaluation")
    print("=" * 60)

    model = load_vqvae(args.vqvae_ckpt, device, code_num=cfg.codebook_size)

    sequences = load_npz_sequences(args.val_dir, args.stats_path)
    print(f"[Exp0] Loaded {len(sequences)} validation sequences")

    if args.dry_run:
        sequences = sequences[:4]
        print("[DRY RUN] Using only 4 sequences")

    if not sequences:
        print("[ERROR] No sequences loaded. Exiting.")
        return

    results = evaluate_vqvae(model, sequences, device, cfg.codebook_size)

    out_path = os.path.join(args.output_dir, "baseline_eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Exp0] Baseline results saved to {out_path}")
    print(f"  Overall MSE:       {results['overall_mse']}")
    print(f"  Worst-group:       {results['worst_group_score']}")
    print(f"  Hand MSE:          {results['hand_mse']}")
    print(f"  Body MSE:          {results['body_mse']}")
    print(f"  Codebook coverage: {results['codebook_coverage_pct']}%")

    if results["codebook_coverage_pct"] < 60.0:
        print("\n  [WARNING] Codebook coverage below 60%% -- collapse risk")
    if results["hand_mse"] > 0 and results["body_mse"] > 0:
        ratio = results["hand_mse"] / results["body_mse"]
        if ratio > 1.5:
            print(f"  [DIAGNOSTIC] Hand/body MSE ratio = {ratio:.2f} -- hand bottleneck confirmed")


if __name__ == "__main__":
    main()
