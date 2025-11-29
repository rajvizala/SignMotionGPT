"""
Evaluate the SignMotionGPT model on a held-out SMPL-X test dataset.

The script can download Google Drive archives or consume an already extracted
directory of `video_data.pkl` files. Each sequence is converted into encoder
features via the project VQ-VAE utilities and compared against motions generated
by the LLM to compute FID/Diversity/Multimodality metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    TEST_EVAL_DOWNLOAD_DIR,
    TEST_EVAL_EXTRACT_DIR,
    TEST_EVAL_HF_REPO,
    TEST_EVAL_HF_SUBFOLDER,
    TEST_EVAL_MAX_ZIPS,
    TEST_EVAL_OUTPUT_DIR,
    TEST_EVAL_SAMPLE_LIMIT,
)

M_START = "<M_START>"
M_END = "<M_END>"
PAD_TOKEN = "<PAD>"

INFERENCE_REPETITION_PENALTY = 1.2
INFERENCE_TEMPERATURE = 0.7
INFERENCE_TOP_K = 50


# -----------------------------------------------------------------------------
# Download / extraction helpers
# -----------------------------------------------------------------------------
def try_import_gdown() -> bool:
    try:
        import gdown  # noqa: F401

        return True
    except Exception:
        return False


def download_drive_folder(folder_url_or_id: str, dest_dir: str) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    if not try_import_gdown():
        raise RuntimeError("gdown is required for Drive downloads. Install with `pip install gdown`.")
    import gdown

    if "drive.google.com" in folder_url_or_id:
        url = folder_url_or_id
    else:
        url = f"https://drive.google.com/drive/folders/{folder_url_or_id}"
    print(f"Downloading Drive folder to {dest_dir} ...")
    gdown.download_folder(url=url, output=dest_dir, quiet=False, use_cookies=False)
    print("Download complete.")


def list_zip_files(download_dir: str) -> List[str]:
    matches: List[str] = []
    for root, _dirs, files in os.walk(download_dir):
        for name in files:
            if name.lower().endswith(".zip"):
                matches.append(os.path.join(root, name))
    return sorted(matches)


def extract_zip_files(zip_paths: List[str], extract_dir: str, limit: Optional[int]) -> List[str]:
    os.makedirs(extract_dir, exist_ok=True)
    extracted_roots: List[str] = []
    for idx, zp in enumerate(zip_paths):
        if limit is not None and idx >= limit:
            break
        try:
            with zipfile.ZipFile(zp, "r") as archive:
                subdir = os.path.splitext(os.path.basename(zp))[0]
                target = os.path.join(extract_dir, subdir)
                os.makedirs(target, exist_ok=True)
                archive.extractall(target)
                extracted_roots.append(target)
        except Exception as exc:
            print(f"⚠️  Failed to extract {zp}: {exc}")
    print(f"Extracted {len(extracted_roots)} archives.")
    return extracted_roots


def find_video_pkl_paths(extracted_root: str) -> List[str]:
    matches: List[str] = []
    for root, _dirs, files in os.walk(extracted_root):
        for name in files:
            if name == "video_data.pkl":
                matches.append(os.path.join(root, name))
    return matches


def parse_word_from_path(path: str) -> str:
    base = os.path.basename(os.path.dirname(path))
    if "-" in base:
        word = base.split("-", 1)[1]
    else:
        word = base
    return word.strip().lower()


# -----------------------------------------------------------------------------
# SMPL-X helpers
# -----------------------------------------------------------------------------
def try_to_array(value) -> Optional[np.ndarray]:
    if isinstance(value, np.ndarray):
        return value
    try:
        return np.asarray(value)
    except Exception:
        return None


def load_smplx_params_from_pkl(pkl_path: str) -> Optional[np.ndarray]:
    try:
        with open(pkl_path, "rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        print(f"⚠️  Could not read {pkl_path}: {exc}")
        return None

    if not isinstance(payload, (list, tuple)) or len(payload) == 0:
        return None

    def get_vec(frame: dict, key: str, expected: int, allow_trim: bool = True) -> np.ndarray:
        val = frame.get(key)
        arr = try_to_array(val)
        if arr is None:
            return np.zeros((expected,), dtype=np.float32)
        arr = np.array(arr, dtype=np.float32).reshape(-1)
        if arr.size == expected:
            return arr
        if allow_trim and arr.size > expected:
            if key == "body_pose" and arr.size == 66 and expected == 63:
                return arr[3:3 + 63]
            return arr[:expected]
        if arr.size < expected:
            out = np.zeros((expected,), dtype=np.float32)
            out[: arr.size] = arr
            return out
        return arr[:expected]

    sequences: List[np.ndarray] = []
    for frame in payload:
        if not isinstance(frame, dict):
            continue
        vec = np.concatenate(
            [
                get_vec(frame, "shape", 10),
                get_vec(frame, "body_pose", 63),
                get_vec(frame, "lhand_pose", 45),
                get_vec(frame, "rhand_pose", 45),
                get_vec(frame, "cam_trans", 3),
                get_vec(frame, "expression", 10),
                get_vec(frame, "jaw_pose", 3),
                np.zeros((3,), dtype=np.float32),  # eye pose placeholder
            ],
            axis=0,
        )
        sequences.append(vec)
    if not sequences:
        return None
    return np.stack(sequences, axis=0).astype(np.float32)


def import_visualize_helpers():
    try:
        from visualize import (
            load_vqvae,
            load_stats,
            decode_tokens_to_params,
            VQVAE_CHECKPOINT as DEFAULT_VQ,
            STATS_PATH as DEFAULT_STATS,
        )

        return load_vqvae, load_stats, decode_tokens_to_params, DEFAULT_VQ, DEFAULT_STATS
    except Exception as exc:
        raise RuntimeError(f"Failed to import visualize helpers: {exc}") from exc


def _encode_params_to_feature(
    params: np.ndarray,
    vq_model,
    mean,
    std,
    device: torch.device,
) -> Optional[np.ndarray]:
    if params is None or params.size == 0:
        return None
    clip = torch.from_numpy(params.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        x_pre = None
        if hasattr(vq_model.vqvae, "preprocess"):
            try:
                x_pre = vq_model.vqvae.preprocess(clip)
            except Exception:
                x_pre = None
        if x_pre is None:
            if mean is not None and std is not None:
                mean_t = torch.from_numpy(np.array(mean, dtype=np.float32)).to(device).view(1, 1, -1)
                std_t = torch.from_numpy(np.array(std, dtype=np.float32)).to(device).view(1, 1, -1)
                clip = (clip - mean_t) / (std_t + 1e-8)
            x_pre = clip.transpose(1, 2).contiguous()
        latent = vq_model.vqvae.encoder(x_pre)
        if latent.dim() == 3:
            embed_dim = getattr(vq_model.vqvae, "output_emb_width", None)
            if embed_dim is not None:
                if latent.shape[1] == embed_dim:
                    axis = 2
                elif latent.shape[2] == embed_dim:
                    axis = 1
                else:
                    axis = 2 if latent.shape[2] < latent.shape[1] else 1
            else:
                axis = 2 if latent.shape[2] < latent.shape[1] else 1
            feat = latent.mean(dim=axis).squeeze(0)
        elif latent.dim() == 2:
            feat = latent.squeeze(0)
        else:
            feat = latent.view(1, -1).mean(dim=0)
        vec = feat.detach().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


# -----------------------------------------------------------------------------
# Metrics helpers
# -----------------------------------------------------------------------------
def calculate_activation_statistics_np(activations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_frechet_distance_np(mu1, sigma1, mu2, sigma2, eps=1e-6) -> float:
    from scipy.linalg import sqrtm

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape, "Mean vectors must match"
    assert sigma1.shape == sigma2.shape, "Covariance matrices must match"
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError("Covmean contains large imaginary components")
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def calculate_diversity_np(activation: np.ndarray, diversity_times: int = 200) -> float:
    assert activation.ndim == 2
    n = activation.shape[0]
    if n < 2:
        return float("nan")
    times = min(diversity_times, max(1, n - 1))
    idx1 = np.random.choice(n, times, replace=False)
    idx2 = np.random.choice(n, times, replace=False)
    diffs = activation[idx1] - activation[idx2]
    return float(np.linalg.norm(diffs, axis=1).mean())


def _to_label_tensor3(acts: np.ndarray, labels: List[str]) -> np.ndarray:
    label_to_indices: Dict[str, List[int]] = {}
    for idx, lbl in enumerate(labels):
        label_to_indices.setdefault(lbl, []).append(idx)
    counts = [len(v) for v in label_to_indices.values()]
    if not counts:
        raise ValueError("No labels available for multimodality computation.")
    min_count = max(2, min(counts))
    stacked = []
    for lbl in sorted(label_to_indices.keys()):
        stacked.append(acts[label_to_indices[lbl][:min_count]])
    return np.stack(stacked, axis=0)


def calculate_multimodality_np(activation: np.ndarray, multimodality_times: int = 20) -> float:
    assert activation.ndim == 3
    _, per_label, _ = activation.shape
    if per_label < 2:
        return float("nan")
    times = min(multimodality_times, max(1, per_label - 1))
    first = np.random.choice(per_label, times, replace=False)
    second = np.random.choice(per_label, times, replace=False)
    diffs = activation[:, first] - activation[:, second]
    return float(np.linalg.norm(diffs, axis=2).mean())


# -----------------------------------------------------------------------------
# Generation helpers
# -----------------------------------------------------------------------------
def extract_ids_from_sequence(seq: str) -> List[int]:
    content = seq
    if M_START in seq and M_END in seq:
        content = seq.split(M_START, 1)[-1].split(M_END, 1)[0]
    ids: List[int] = []
    for tok in content.split():
        if tok.startswith("<M") and tok.endswith(">"):
            payload = tok[2:-1]
            if payload.isdigit():
                ids.append(int(payload))
    return ids


def generate_motion_text(model, tokenizer, word: str, device: torch.device) -> str:
    model.eval()
    prompt = f"Instruction: Generate motion for word '{word}' with variant 'unknown'.\nMotion: "
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=INFERENCE_TEMPERATURE,
            top_k=INFERENCE_TOP_K,
            repetition_penalty=INFERENCE_REPETITION_PENALTY,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids(M_END),
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=False)
    if "Motion: " in decoded:
        return decoded.split("Motion: ", 1)[-1].strip()
    return decoded.strip()


# -----------------------------------------------------------------------------
# Core evaluation
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Evaluate the trained Stage 2 model on an unseen SMPL-X test dataset."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--drive-url", type=str, help="Google Drive folder URL to download archives from.")
    group.add_argument("--drive-id", type=str, help="Google Drive folder ID to download archives from.")
    group.add_argument(
        "--local-extracted-dir",
        type=str,
        help="Use an existing directory that already contains extracted `video_data.pkl` files.",
    )

    parser.add_argument("--max-zips", type=int, default=TEST_EVAL_MAX_ZIPS, help="Maximum number of zip files to extract.")
    parser.add_argument("--download-dir", type=str, default=TEST_EVAL_DOWNLOAD_DIR, help="Directory to store downloaded zips.")
    parser.add_argument("--extract-dir", type=str, default=TEST_EVAL_EXTRACT_DIR, help="Directory to extract archives into.")

    parser.add_argument("--hf-repo-id", type=str, default=TEST_EVAL_HF_REPO, help="Hugging Face repo containing the Stage 2 checkpoint.")
    parser.add_argument(
        "--hf-subfolder",
        type=str,
        default=TEST_EVAL_HF_SUBFOLDER,
        help="Subfolder inside the repo that hosts the Stage 2 model (e.g., `stage2_v2/epoch-020`).",
    )

    parser.add_argument("--vqvae-ckpt", type=str, default=None, help="Optional override for VQ-VAE checkpoint path.")
    parser.add_argument("--stats-path", type=str, default=None, help="Optional override for VQ-VAE stats file.")

    parser.add_argument("--output-dir", type=str, default=TEST_EVAL_OUTPUT_DIR, help="Directory to write metrics JSON.")
    parser.add_argument("--sample-limit", type=int, default=TEST_EVAL_SAMPLE_LIMIT, help="Maximum number of samples to evaluate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def run_evaluation(args: argparse.Namespace) -> Dict[str, object]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "metrics_test.json")

    print(f"Loading Stage 2 model from HF: {args.hf_repo_id} (subfolder='{args.hf_subfolder}')")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_repo_id, subfolder=args.hf_subfolder, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.hf_repo_id, subfolder=args.hf_subfolder, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    load_vqvae, load_stats, decode_tokens_to_params, DEFAULT_VQ, DEFAULT_STATS = import_visualize_helpers()
    vq_ckpt = args.vqvae_ckpt if args.vqvae_ckpt else os.getenv("VQVAE_CHECKPOINT", DEFAULT_VQ)
    stats_path = args.stats_path if args.stats_path else os.getenv("VQVAE_STATS_PATH", DEFAULT_STATS)
    print(f"Loading VQ-VAE from: {vq_ckpt}")
    vq_model = load_vqvae(vq_ckpt, device=device)
    print(f"Loading stats from: {stats_path}")
    mean, std = load_stats(stats_path)

    extracted_dirs: List[str] = []
    if args.local_extracted_dir:
        if not os.path.isdir(args.local_extracted_dir):
            raise FileNotFoundError(f"Local extracted dir not found: {args.local_extracted_dir}")
        extracted_dirs = [args.local_extracted_dir]
    else:
        folder_ref = args.drive_url if args.drive_url else args.drive_id
        download_drive_folder(folder_ref, args.download_dir)
        zips = list_zip_files(args.download_dir)
        if not zips:
            raise RuntimeError("No zip files found after download.")
        extracted_dirs = extract_zip_files(zips, args.extract_dir, limit=args.max_zips)

    samples: List[Tuple[str, str]] = []
    for root in extracted_dirs:
        for pkl_path in find_video_pkl_paths(root):
            samples.append((parse_word_from_path(pkl_path), pkl_path))
    if not samples:
        raise RuntimeError("No `video_data.pkl` files discovered in the extracted directories.")

    random.shuffle(samples)
    samples = samples[: args.sample_limit]
    print(f"Found {len(samples)} samples to evaluate.")

    gt_features: List[np.ndarray] = []
    gen_features: List[np.ndarray] = []
    labels: List[str] = []

    for idx, (word, pkl_path) in enumerate(samples, 1):
        params_gt = load_smplx_params_from_pkl(pkl_path)
        if params_gt is None or params_gt.ndim != 2:
            print(f"Skipping {pkl_path}: invalid SMPL-X payload.")
            continue
        try:
            feat_gt = _encode_params_to_feature(params_gt, vq_model, mean, std, device)
        except Exception as exc:
            print(f"Skipping {pkl_path}: encoder failed ({exc}).")
            continue
        if feat_gt is None:
            print(f"Skipping {pkl_path}: empty GT feature.")
            continue

        gen_text = generate_motion_text(model, tokenizer, word, device)
        token_ids = extract_ids_from_sequence(gen_text)
        if not token_ids:
            print(f"Skipping GEN for '{word}': no motion tokens produced.")
            continue
        try:
            params_gen = decode_tokens_to_params(token_ids, vq_model, mean, std, device=device)
        except Exception as exc:
            print(f"Skipping GEN for '{word}': decode failed ({exc}).")
            continue
        feat_gen = _encode_params_to_feature(params_gen, vq_model, mean, std, device)
        if feat_gen is None:
            print(f"Skipping GEN for '{word}': empty GEN feature.")
            continue

        gt_features.append(feat_gt)
        gen_features.append(feat_gen)
        labels.append(word)
        if idx % 25 == 0:
            print(f"Processed {idx} samples...")

    if len(gt_features) < 5 or len(gen_features) < 5:
        print("⚠️  Not enough samples to compute stable metrics; results may be noisy.")

    gt_feats = np.stack(gt_features, axis=0)
    gen_feats = np.stack(gen_features, axis=0)

    diversity_gt = calculate_diversity_np(gt_feats, diversity_times=min(200, max(4, gt_feats.shape[0] - 1)))
    diversity_gen = calculate_diversity_np(gen_feats, diversity_times=min(200, max(4, gen_feats.shape[0] - 1)))

    try:
        gt_lbl_tensor = _to_label_tensor3(gt_feats, labels)
        gen_lbl_tensor = _to_label_tensor3(gen_feats, labels)
        mim_gt = calculate_multimodality_np(
            gt_lbl_tensor, multimodality_times=min(20, max(3, gt_lbl_tensor.shape[1] - 1))
        )
        mim_gen = calculate_multimodality_np(
            gen_lbl_tensor, multimodality_times=min(20, max(3, gen_lbl_tensor.shape[1] - 1))
        )
    except Exception as exc:
        print(f"⚠️  Multimodality could not be computed reliably: {exc}")
        mim_gt = float("nan")
        mim_gen = float("nan")

    mu_gen, cov_gen = calculate_activation_statistics_np(gen_feats)
    mu_gt, cov_gt = calculate_activation_statistics_np(gt_feats)
    fid = calculate_frechet_distance_np(mu_gt, cov_gt, mu_gen, cov_gen)

    metrics_payload = {
        "source": "test_raw_smplx_encoder_features",
        "counts": {
            "samples_total": len(samples),
            "samples_used": int(gt_feats.shape[0]),
        },
        "fid": fid,
        "diversity": {
            "ground_truth": diversity_gt,
            "model": diversity_gen,
        },
        "multimodality": {
            "ground_truth": mim_gt,
            "model": mim_gen,
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved test metrics to {metrics_path}")
    return metrics_payload


def main() -> None:
    args = parse_args()
    try:
        run_evaluation(args)
    except Exception as exc:
        print(f"Evaluation failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

