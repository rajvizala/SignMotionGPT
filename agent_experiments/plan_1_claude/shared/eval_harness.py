"""
Domain-aware evaluation harness for VQ-VAE checkpoints.
Computes per-bucket reconstruction metrics and codebook coverage.

Standalone usage:
    python -m agent_experiments.plan_1_claude.shared.eval_harness \
        --vqvae-ckpt path/to/ckpt.pt \
        --val-dir path/to/val_npz \
        --output-dir path/to/results \
        [--stats-path path/to/computed_stats.pt] \
        [--word-data-dir path/to/word_npz]
"""

import argparse
import json
import math
import os
import sys
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from models.vqvae import VQVae

SMPLX_DIM = 182
HAND_SLICE = slice(73, 163)
BODY_SLICE = slice(10, 73)
LENGTH_SHORT_MAX = 80
LENGTH_MEDIUM_MAX = 160


def load_vqvae(ckpt_path, device, nfeats=182, code_num=512, code_dim=512):
    model = VQVae(nfeats=nfeats, code_num=code_num, code_dim=code_dim,
                  output_emb_width=code_dim, quantizer="ema_reset")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    clean_sd = {}
    for k, v in sd.items():
        key = k.replace("vqvae.", "") if k.startswith("vqvae.") else k
        clean_sd[key] = v
    model.load_state_dict(clean_sd, strict=False)
    model.to(device).eval()
    return model


def load_npz_sequences(data_dir, stats_path=None, max_len=256):
    mean, std = 0, 1
    if stats_path and os.path.exists(stats_path):
        stats = torch.load(stats_path, map_location="cpu", weights_only=False)
        mean, std = stats["mean"], stats["std"]
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.npz"), recursive=True))
    sequences = []
    for f in files:
        try:
            motion = np.load(f)["motion"]
        except Exception:
            continue
        if motion.ndim != 2 or motion.shape[0] < 32:
            continue
        seq = torch.tensor(motion[:max_len], dtype=torch.float32)
        seq = (seq - mean) / (std + 1e-8)
        sequences.append(seq)
    return sequences


def get_length_bucket(seq_len):
    if seq_len < LENGTH_SHORT_MAX:
        return "length_short"
    if seq_len < LENGTH_MEDIUM_MAX:
        return "length_medium"
    return "length_long"


def evaluate_vqvae(model, sequences, device, codebook_size=512):
    results = {
        "per_sample": [],
        "all_codes": set(),
        "hand_mse_sum": 0.0,
        "body_mse_sum": 0.0,
        "total_mse_sum": 0.0,
        "n_samples": 0,
    }
    buckets = {}

    model.eval()
    with torch.no_grad():
        for seq in sequences:
            T = seq.shape[0]
            pad_len = math.ceil(T / 8) * 8
            x = torch.zeros(1, pad_len, SMPLX_DIM)
            x[0, :T] = seq
            x = x.to(device)

            x_recon, _, perplexity = model(x)

            x_orig = x[0, :T]
            x_rec = x_recon[0, :T]

            total_mse = F.mse_loss(x_rec, x_orig).item()
            hand_mse = F.mse_loss(x_rec[:, HAND_SLICE], x_orig[:, HAND_SLICE]).item()
            body_mse = F.mse_loss(x_rec[:, BODY_SLICE], x_orig[:, BODY_SLICE]).item()

            code_idx, _ = model.encode(x)
            codes = code_idx[0].cpu().tolist()
            results["all_codes"].update(codes)

            bucket_name = get_length_bucket(T)
            if bucket_name not in buckets:
                buckets[bucket_name] = {"mse_sum": 0.0, "n": 0}
            buckets[bucket_name]["mse_sum"] += total_mse
            buckets[bucket_name]["n"] += 1

            results["total_mse_sum"] += total_mse
            results["hand_mse_sum"] += hand_mse
            results["body_mse_sum"] += body_mse
            results["n_samples"] += 1

    n = max(results["n_samples"], 1)
    coverage = len(results["all_codes"]) / codebook_size * 100.0

    output = {
        "overall_mse": results["total_mse_sum"] / n,
        "worst_group_score": 0.0,
        "buckets": {},
        "codebook_coverage_pct": round(coverage, 2),
        "hand_mse": results["hand_mse_sum"] / n,
        "body_mse": results["body_mse_sum"] / n,
    }

    for bname in ["length_short", "length_medium", "length_long"]:
        b = buckets.get(bname, {"mse_sum": 0.0, "n": 0})
        mse = b["mse_sum"] / max(b["n"], 1) if b["n"] > 0 else 0.0
        output["buckets"][bname] = {"mse": round(mse, 6), "n_samples": b["n"]}

    for placeholder in ["coverage_high", "coverage_medium", "coverage_low",
                         "novelty_seen_words", "novelty_novel_vocab"]:
        output["buckets"][placeholder] = {"mse": 0.0, "n_samples": 0}

    all_bucket_mses = [b["mse"] for b in output["buckets"].values() if b["n_samples"] > 0]
    output["worst_group_score"] = max(all_bucket_mses) if all_bucket_mses else output["overall_mse"]

    output["overall_mse"] = round(output["overall_mse"], 6)
    output["hand_mse"] = round(output["hand_mse"], 6)
    output["body_mse"] = round(output["body_mse"], 6)
    output["worst_group_score"] = round(output["worst_group_score"], 6)

    return output


def main():
    parser = argparse.ArgumentParser(description="Eval harness for VQ-VAE checkpoints")
    parser.add_argument("--vqvae-ckpt", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--output-dir", default="./eval_results")
    parser.add_argument("--stats-path", default="computed_stats.pt")
    parser.add_argument("--word-data-dir", default="")
    parser.add_argument("--codebook-size", type=int, default=512)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Eval] Loading VQ-VAE from {args.vqvae_ckpt}")
    model = load_vqvae(args.vqvae_ckpt, device, code_num=args.codebook_size)

    print(f"[Eval] Loading validation data from {args.val_dir}")
    sequences = load_npz_sequences(args.val_dir, args.stats_path)
    print(f"[Eval] Loaded {len(sequences)} sequences")

    if len(sequences) == 0:
        print("[Eval] ERROR: No sequences loaded. Check --val-dir and --stats-path")
        return

    results = evaluate_vqvae(model, sequences, device, args.codebook_size)

    out_path = os.path.join(args.output_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Eval] Results saved to {out_path}")
    print(f"  Overall MSE:        {results['overall_mse']}")
    print(f"  Worst-group score:  {results['worst_group_score']}")
    print(f"  Hand MSE:           {results['hand_mse']}")
    print(f"  Body MSE:           {results['body_mse']}")
    print(f"  Codebook coverage:  {results['codebook_coverage_pct']}%")
    for bname, bdata in results["buckets"].items():
        if bdata["n_samples"] > 0:
            print(f"  {bname}: mse={bdata['mse']:.6f} n={bdata['n_samples']}")

    if results["hand_mse"] > results["body_mse"] * 1.5:
        print("\n  [DIAGNOSTIC] Hand MSE is significantly worse than body MSE.")
        print("  This confirms the hand representation bottleneck.")

    if results["codebook_coverage_pct"] < 60.0:
        print("\n  [WARNING] Codebook coverage below 60% -- possible collapse.")


if __name__ == "__main__":
    main()
