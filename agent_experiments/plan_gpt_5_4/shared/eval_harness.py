from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from .data_utils import (
    MotionFolderDataset,
    decode_codes_batch,
    build_word_lexicon_from_dir,
    collate_motion_items,
    compute_mse_breakdown,
    default_stats_path,
    default_vqvae_ckpt,
    load_stats,
    load_vqvae_checkpoint,
    unnormalize_motion,
    normalize_motion,
)


BUCKET_KEYS = [
    "length_short",
    "length_medium",
    "length_long",
    "coverage_high",
    "coverage_medium",
    "coverage_low",
    "novelty_seen_words",
    "novelty_novel_vocab",
]


def empty_result() -> Dict[str, object]:
    return {
        "overall_mse": 0.0,
        "worst_group_score": 0.0,
        "buckets": {key: {"mse": 0.0, "n_samples": 0} for key in BUCKET_KEYS},
        "codebook_coverage_pct": 0.0,
        "hand_mse": 0.0,
        "body_mse": 0.0,
    }


def evaluate_vqvae_checkpoint(
    vqvae_ckpt: str,
    val_dir: str,
    word_data_dir: str,
    stats_path: str,
    batch_size: int = 8,
    max_samples: int | None = None,
) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    word_lexicon = build_word_lexicon_from_dir(word_data_dir, max_samples=max_samples)
    dataset = MotionFolderDataset(val_dir, word_lexicon=word_lexicon, max_samples=max_samples)
    if len(dataset) == 0:
        return empty_result()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_motion_items)
    model = load_vqvae_checkpoint(vqvae_ckpt, device=device)
    mean, std = load_stats(stats_path)
    mean = mean.to(device)
    std = std.to(device)

    overall_mse_sum = 0.0
    hand_mse_sum = 0.0
    body_mse_sum = 0.0
    total_samples = 0
    code_ids = set()
    bucket_values: Dict[str, List[float]] = defaultdict(list)

    for batch in dataloader:
        motions = batch["motion"].to(device)
        lengths = batch["lengths"]
        normalized = normalize_motion(motions, mean, std)
        with torch.no_grad():
            codes, _ = model.encode(normalized)
            decoded = decode_codes_batch(model, codes)
        reconstructed = unnormalize_motion(decoded, mean, std).detach().cpu()

        for row_idx, length in enumerate(lengths.tolist()):
            metrics = compute_mse_breakdown(reconstructed[row_idx], batch["motion"][row_idx], length)
            overall_mse_sum += metrics["overall_mse"]
            hand_mse_sum += metrics["hand_mse"]
            body_mse_sum += metrics["body_mse"]
            total_samples += 1
            for bucket_name in batch["buckets"][row_idx]:
                bucket_values[bucket_name].append(metrics["overall_mse"])

        code_ids.update(int(code) for code in codes.reshape(-1).tolist())

    result = empty_result()
    if total_samples == 0:
        return result

    result["overall_mse"] = overall_mse_sum / total_samples
    result["hand_mse"] = hand_mse_sum / total_samples
    result["body_mse"] = body_mse_sum / total_samples
    result["codebook_coverage_pct"] = 100.0 * len(code_ids) / 512.0

    non_empty_bucket_scores = []
    for bucket_name in BUCKET_KEYS:
        values = bucket_values.get(bucket_name, [])
        bucket_mse = sum(values) / len(values) if values else 0.0
        result["buckets"][bucket_name] = {"mse": bucket_mse, "n_samples": len(values)}
        if values:
            non_empty_bucket_scores.append(bucket_mse)
    result["worst_group_score"] = max(non_empty_bucket_scores) if non_empty_bucket_scores else result["overall_mse"]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Worst-group VQ-VAE evaluation harness.")
    parser.add_argument("--vqvae-ckpt", default=default_vqvae_ckpt(), help="Checkpoint to evaluate.")
    parser.add_argument("--val-dir", required=True, help="Validation directory containing NPZ files.")
    parser.add_argument("--word-data-dir", required=True, help="Word-level NPZ directory for lexical buckets.")
    parser.add_argument("--stats-path", default=default_stats_path(), help="Normalization stats path.")
    parser.add_argument("--output-dir", required=True, help="Output directory for JSON metrics.")
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample cap.")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate a tiny subset and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    max_samples = 8 if args.dry_run and args.max_samples is None else args.max_samples
    results = evaluate_vqvae_checkpoint(
        vqvae_ckpt=args.vqvae_ckpt,
        val_dir=args.val_dir,
        word_data_dir=args.word_data_dir,
        stats_path=args.stats_path,
        batch_size=args.batch_size,
        max_samples=max_samples,
    )
    output_path = os.path.join(args.output_dir, "eval_results.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
