"""
Evaluation metrics for motion generation
"""
import random
import os
import re
import json
import numpy as np
import scipy.linalg
import torch
from typing import List, Tuple, Dict, Optional, Any
from rapidfuzz.distance import Levenshtein
from collections import defaultdict
from data import motion_specials_to_ids
from config import (
    SEED, PIPELINE_OUTPUT_DIR, M_START, M_END,
    INFERENCE_TEMPERATURE, INFERENCE_TOP_K, INFERENCE_REPETITION_PENALTY
)

random.seed(SEED)

# ======================================================================================
# Logic from test_overfit.py (Metrics & Visualization)
# ======================================================================================

def calculate_activation_statistics_np(activations: np.ndarray):
    """
    Params:
    -- activations: num_samples x dim_feat (numpy)
    Returns:
    -- mu: dim_feat
    -- sigma: dim_feat x dim_feat
    """
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov

def calculate_frechet_distance_np(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def calculate_diversity_np(activation: np.ndarray, diversity_times: int = 200) -> float:
    """Mean pairwise L2 distance across random pairs."""
    assert len(activation.shape) == 2
    if activation.shape[0] < 2:
        return 0.0
    num_samples = activation.shape[0]
    effective_times = min(diversity_times, max(1, num_samples - 1))
    first_indices = np.random.choice(num_samples, effective_times, replace=False)
    second_indices = np.random.choice(num_samples, effective_times, replace=False)
    diffs = activation[first_indices] - activation[second_indices]
    dist = np.linalg.norm(diffs, axis=1)
    return float(dist.mean())

def calculate_multimodality_np(activation: np.ndarray, multimodality_times: int = 20) -> float:
    """
    activation: [num_labels, num_per_label, D]
    Returns mean pairwise within-label diversity (higher = more multimodal).
    """
    assert len(activation.shape) == 3
    num_labels, num_per_label, _ = activation.shape
    if num_per_label < 2:
        return float("nan")
    effective_times = min(multimodality_times, max(1, num_per_label - 1))
    first_dices = np.random.choice(num_per_label, effective_times, replace=False)
    second_dices = np.random.choice(num_per_label, effective_times, replace=False)
    diffs = activation[:, first_dices] - activation[:, second_dices]
    dist = np.linalg.norm(diffs, axis=2)
    return float(dist.mean())

# --------------------------------------------------------------------------------------
# Token sequence → activation (bag-of-motion-tokens) helpers
# --------------------------------------------------------------------------------------
def _extract_motion_tokens_from_sequence(seq: str) -> list[str]:
    # Expect tokens like <M123>, within M_START/M_END fences; keep only <M...>
    return [tok for tok in seq.split() if tok.startswith("<M") and tok.endswith(">")]

def _extract_ids_from_sequence(seq: str) -> list[int]:
    return [int(t[2:-1]) for t in _extract_motion_tokens_from_sequence(seq) if t[2:-1].isdigit()]

def _build_token_index(tokens_vocab: list[str]) -> Dict[str, int]:
    return {tok: idx for idx, tok in enumerate(tokens_vocab)}

def _sequence_to_activation(seq: str, token_to_index: Dict[str, int]) -> np.ndarray:
    vec = np.zeros((len(token_to_index),), dtype=np.float32)
    for tok in _extract_motion_tokens_from_sequence(seq):
        idx = token_to_index.get(tok)
        if idx is not None:
            vec[idx] += 1.0
    # Normalize to unit length to reduce length bias
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

def generate_motion(model, tokenizer, prompt, device):
    """Generates a motion sequence from a prompt using sampling."""
    model.eval()
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
            early_stopping=True
        )
    
    decoded = tokenizer.decode(output[0], skip_special_tokens=False)
    if "Motion: " in decoded:
        motion_part = decoded.split("Motion: ")[-1]
    else:
        motion_part = decoded
    return motion_part.strip()

def build_instruction_prompt(word: str, participant_id: Optional[str] = None, include_participant: bool = True) -> str:
    """
    Build the plain-text prompt used by the raw training/eval pipeline (Instruction/Motion format).

    Stage 2-style: "word + participant_id"
    Stage 3-style: "word only" (include_participant=False)
    """
    w = str(word)
    if include_participant:
        pid = "" if participant_id is None else str(participant_id)
        return f"Instruction: Generate motion for word '{w}' with variant '{pid}'.\nMotion: "
    return f"Instruction: Generate motion for word '{w}'.\nMotion: "

def _collect_eval_pairs(model, tokenizer, data, device, include_participant: bool = True) -> list[Tuple[str, str, str]]:
    """
    Returns list of (word, participant_id, gt_sequence, generated_sequence) for each sample in data.
    """
    results = []
    for sample in data:
        gt_tokens_str = sample.get("motion_tokens", "")
        gt_wrapped = " ".join([f"<M{t}>" for t in gt_tokens_str.split()])
        gt_sequence = f"{M_START} {gt_wrapped} {M_END}"
        prompt = build_instruction_prompt(
            word=sample.get("word", ""),
            participant_id=sample.get("participant_id", ""),
            include_participant=include_participant,
        )
        generated_sequence = generate_motion(model, tokenizer, prompt, device)
        pid = str(sample.get("participant_id", ""))
        results.append((sample["word"], pid, gt_sequence, generated_sequence))
    return results

def _activations_from_pairs(pairs: list[Tuple[str, str, str]], vocab_tokens: list[str]):
    """
    Build numpy activations and labels arrays from sequences.
    Returns:
      gt_acts: (N, D)
      gen_acts: (N, D)
      labels: list[str] length N (word labels)
    """
    token_to_index = _build_token_index(vocab_tokens)
    gt_vecs = []
    gen_vecs = []
    labels = []
    for pair in pairs:
        # Support both legacy 3-tuple (word, gt, gen) and new 4-tuple (word, pid, gt, gen)
        if len(pair) == 4:
            word, _pid, gt_seq, gen_seq = pair
        else:
            word, gt_seq, gen_seq = pair
        gt_vecs.append(_sequence_to_activation(gt_seq, token_to_index))
        gen_vecs.append(_sequence_to_activation(gen_seq, token_to_index))
        labels.append(word)
    return np.stack(gt_vecs, axis=0), np.stack(gen_vecs, axis=0), labels

def _to_label_tensor3(acts: np.ndarray, labels: list[str]) -> np.ndarray:
    """
    Convert N x D activations with string labels to [L, K, D] by truncating each label
    to the minimum count across labels.
    """
    label_to_indices: Dict[str, list[int]] = {}
    for i, lbl in enumerate(labels):
        label_to_indices.setdefault(lbl, []).append(i)
    per_label_counts = [len(idxs) for idxs in label_to_indices.values()]
    if len(per_label_counts) == 0:
        raise ValueError("No labels found for multimodality computation.")
    min_count = max(2, min(per_label_counts))
    label_names = sorted(label_to_indices.keys())
    stacked = []
    for lbl in label_names:
        idxs = label_to_indices[lbl][:min_count]
        stacked.append(acts[idxs])
    return np.stack(stacked, axis=0)  # [L, K, D]

def evaluate_metrics_motiongpt_style(model, tokenizer, eval_data, all_motion_tokens, device, include_participant: bool = True):
    """
    Computes:
      - Diversity: GT vs GEN (pair)
      - Multimodality (MIM): GT vs GEN (pair)
      - FID: between GT and GEN
    """
    print("\n" + "="*80)
    print("      METRICS EVALUATION (FID, Diversity, Multimodality)")
    print("="*80)
    pairs = _collect_eval_pairs(model, tokenizer, eval_data, device, include_participant=include_participant)
    gt_acts, gen_acts, labels = _activations_from_pairs(pairs, all_motion_tokens)
    # Diversity
    diversity_times = min(200, max(4, gt_acts.shape[0] - 1))
    diversity_gt = calculate_diversity_np(gt_acts, diversity_times=diversity_times)
    diversity_gen = calculate_diversity_np(gen_acts, diversity_times=diversity_times)
    # Multimodality (MIM)
    try:
        gt_lbl_tensor = _to_label_tensor3(gt_acts, labels)
        gen_lbl_tensor = _to_label_tensor3(gen_acts, labels)
        multimodality_times = min(20, max(3, gt_lbl_tensor.shape[1] - 1))
        mim_gt = calculate_multimodality_np(gt_lbl_tensor, multimodality_times=multimodality_times)
        mim_gen = calculate_multimodality_np(gen_lbl_tensor, multimodality_times=multimodality_times)
    except Exception as exc:
        print(f"⚠️  Multimodality could not be computed reliably: {exc}")
        mim_gt = float("nan")
        mim_gen = float("nan")
    # FID
    mu_gen, cov_gen = calculate_activation_statistics_np(gen_acts)
    mu_gt, cov_gt = calculate_activation_statistics_np(gt_acts)
    fid = calculate_frechet_distance_np(mu_gt, cov_gt, mu_gen, cov_gen)
    print(f"Diversity:    GT = {diversity_gt:.4f} | GEN = {diversity_gen:.4f}")
    print(f"Multimodality (MIM): GT = {mim_gt:.4f} | GEN = {mim_gen:.4f}")
    print(f"FID (GT vs GEN): {fid:.4f}")
    return {
        "diversity_gt": diversity_gt,
        "diversity_gen": diversity_gen,
        "mim_gt": mim_gt,
        "mim_gen": mim_gen,
        "fid": fid,
        "pairs": pairs,  # for visualization usage
    }

def _encode_params_to_feature(params: np.ndarray, vq_model, mean, std, device) -> np.ndarray:
    """
    Convert SMPL-X parameter sequence (T, D) into a single clip feature using
    the VQ-VAE encoder output BEFORE quantization. Average-pool over time to get (D_embed,).
    """
    if params.size == 0:
        return np.zeros((getattr(vq_model.vqvae, "output_emb_width", 512),), dtype=np.float32)
    x = torch.from_numpy(params.astype(np.float32)).to(device)  # [T, D]
    x = x.unsqueeze(0)  # [1, T, D]
    with torch.no_grad():
        # Normalize / preprocess
        x_pre = None
        if hasattr(vq_model.vqvae, "preprocess"):
            try:
                x_pre = vq_model.vqvae.preprocess(x)  # expected to return tensor ready for encoder
            except Exception:
                x_pre = None
        if x_pre is None:
            # Manual normalization with provided mean/std
            if mean is not None and std is not None:
                mean_t = torch.from_numpy(np.array(mean, dtype=np.float32)).to(device).view(1, 1, -1)
                std_t = torch.from_numpy(np.array(std, dtype=np.float32)).to(device).view(1, 1, -1)
                x_norm = (x - mean_t) / (std_t + 1e-8)
            else:
                x_norm = x
            # Some encoders expect [N, D, T]
            x_pre = x_norm.transpose(1, 2).contiguous()  # [1, D, T]
        # Encode to get pre-quant latent
        z_e = vq_model.vqvae.encoder(x_pre)
        # z_e could be [N, D_embed, T_q] or [N, T_q, D_embed]
        if z_e.dim() == 3:
            embed_dim_known = getattr(vq_model.vqvae, "output_emb_width", None)
            if embed_dim_known is not None:
                if z_e.shape[1] == embed_dim_known:
                    time_axis = 2  # [N, D_embed, T_q]
                elif z_e.shape[2] == embed_dim_known:
                    time_axis = 1  # [N, T_q, D_embed]
                else:
                    time_axis = 2 if z_e.shape[2] < z_e.shape[1] else 1
            else:
                time_axis = 2 if z_e.shape[2] < z_e.shape[1] else 1
            feat = z_e.mean(dim=time_axis).squeeze(0)
        elif z_e.dim() == 2:
            feat = z_e.squeeze(0)
        else:
            feat = z_e.view(1, -1).mean(dim=0)
        feat_np = feat.detach().cpu().numpy().astype(np.float32)
        # L2 normalize
        norm = np.linalg.norm(feat_np)
        if norm > 0:
            feat_np = feat_np / norm
        return feat_np

def evaluate_metrics_encoder_style(
    model,
    tokenizer,
    eval_data,
    device,
    vqvae_ckpt: Optional[str] = None,
    stats_path: Optional[str] = None,
    sample_limit: int = 100,
    include_participant: bool = True,
):
    """
    Computes FID, Diversity, and MIM using VQ-VAE encoder pre-quantization features.
    """
    print("\n" + "="*80)
    print("      METRICS EVALUATION (VQ-VAE Encoder Features)")
    print("="*80)
    # Lazy import to reuse your visualization utilities and stats
    try:
        from visualize import load_vqvae, load_stats, VQVAE_CHECKPOINT as DEFAULT_VQ, STATS_PATH as DEFAULT_STATS
        vq_ckpt = vqvae_ckpt or os.getenv("VQVAE_CHECKPOINT", DEFAULT_VQ)
        stats_p = stats_path or os.getenv("VQVAE_STATS_PATH", DEFAULT_STATS)
        vq_model = load_vqvae(vq_ckpt, device=device)
        mean, std = load_stats(stats_p)
        from visualize import decode_tokens_to_params
    except Exception as exc:
        print(f"⚠️  Could not set up VQ-VAE encoder metrics: {exc}")
        return {}
    # Collect GT/GEN token sequences for pairs (limit to speed-up)
    pairs = _collect_eval_pairs(model, tokenizer, eval_data[:sample_limit], device, include_participant=include_participant)
    # Build features
    gt_feats = []
    gen_feats = []
    labels = []
    for pair in pairs:
        if len(pair) == 4:
            word, _pid, gt_seq, gen_seq = pair
        else:
            word, gt_seq, gen_seq = pair
        # Decode to SMPL-X
        tokens_gt = _extract_ids_from_sequence(gt_seq)
        tokens_gen = _extract_ids_from_sequence(gen_seq)
        try:
            params_gt = decode_tokens_to_params(tokens_gt, vq_model, mean, std, device=device)  # (T, D) denorm
        except Exception:
            params_gt = np.zeros((0, 182), dtype=np.float32)
        try:
            params_gen = decode_tokens_to_params(tokens_gen, vq_model, mean, std, device=device)  # (T, D) denorm
        except Exception:
            params_gen = np.zeros((0, 182), dtype=np.float32)
        # Encode (pre-quant) -> pooled feature
        feat_gt = _encode_params_to_feature(params_gt, vq_model, mean, std, device)
        feat_gen = _encode_params_to_feature(params_gen, vq_model, mean, std, device)
        gt_feats.append(feat_gt)
        gen_feats.append(feat_gen)
        labels.append(word)
    gt_feats = np.stack(gt_feats, axis=0)
    gen_feats = np.stack(gen_feats, axis=0)
    # Diversity
    diversity_times = min(200, max(4, gt_feats.shape[0] - 1))
    diversity_gt = calculate_diversity_np(gt_feats, diversity_times=diversity_times)
    diversity_gen = calculate_diversity_np(gen_feats, diversity_times=diversity_times)
    # Multimodality (MIM)
    try:
        gt_lbl_tensor = _to_label_tensor3(gt_feats, labels)
        gen_lbl_tensor = _to_label_tensor3(gen_feats, labels)
        multimodality_times = min(20, max(3, gt_lbl_tensor.shape[1] - 1))
        mim_gt = calculate_multimodality_np(gt_lbl_tensor, multimodality_times=multimodality_times)
        mim_gen = calculate_multimodality_np(gen_lbl_tensor, multimodality_times=multimodality_times)
    except Exception as exc:
        print(f"⚠️  Multimodality could not be computed reliably: {exc}")
        mim_gt = float("nan")
        mim_gen = float("nan")
    # FID (on encoder features)
    mu_gen, cov_gen = calculate_activation_statistics_np(gen_feats)
    mu_gt, cov_gt = calculate_activation_statistics_np(gt_feats)
    fid = calculate_frechet_distance_np(mu_gt, cov_gt, mu_gen, cov_gen)
    print(f"Diversity (encoder feats):    GT = {diversity_gt:.4f} | GEN = {diversity_gen:.4f}")
    print(f"Multimodality (MIM, encoder): GT = {mim_gt:.4f} | GEN = {mim_gen:.4f}")
    print(f"FID (encoder feats, GT vs GEN): {fid:.4f}")
    return {
        "diversity_gt": diversity_gt,
        "diversity_gen": diversity_gen,
        "mim_gt": mim_gt,
        "mim_gen": mim_gen,
        "fid": fid,
        "pairs": pairs,
    }

def save_side_by_side_visualizations(pairs: list[Tuple[str, str, str]], output_dir: str, limit: int = 4):
    """
    Generate side-by-side 3D animations for GT vs GEN.
    """
    try:
        from visualize import (
            load_vqvae, load_stats, load_smplx_model,
            decode_tokens_to_params, params_to_vertices,
            VQVAE_CHECKPOINT as DEFAULT_VQ, STATS_PATH as DEFAULT_STATS, SMPLX_MODEL_DIR as DEFAULT_SMPLX
        )
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as exc:
        print(f"⚠️  Visualization skipped (missing dependencies): {exc}")
        return

    os.makedirs(output_dir, exist_ok=True)
    vqvae_ckpt = os.getenv("VQVAE_CHECKPOINT", DEFAULT_VQ)
    stats_path = os.getenv("VQVAE_STATS_PATH", DEFAULT_STATS)
    smplx_dir = os.getenv("SMPLX_MODEL_DIR", DEFAULT_SMPLX)

    print("Loading VQ-VAE, stats, SMPL-X ...")
    vq_model = load_vqvae(vqvae_ckpt)
    mean, std = load_stats(stats_path)
    smplx_model = load_smplx_model(smplx_dir)

    def animate_side_by_side(verts_left, faces, verts_right, fps=20, titles=("Ground Truth", "LLM Generated"), output_html=None):
        T = min(verts_left.shape[0], verts_right.shape[0])
        verts_left, verts_right = verts_left[:T], verts_right[:T]
        i, j, k = faces.T.tolist()
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scene'}, {'type': 'scene'}]],
            horizontal_spacing=0.05,
            subplot_titles=list(titles)
        )
        left_mesh = go.Mesh3d(x=verts_left[0,:,0], y=verts_left[0,:,1], z=verts_left[0,:,2], i=i,j=j,k=k,opacity=0.7,showscale=False)
        right_mesh = go.Mesh3d(x=verts_right[0,:,0], y=verts_right[0,:,1], z=verts_right[0,:,2], i=i,j=j,k=k,opacity=0.7,showscale=False)
        fig.add_trace(left_mesh, row=1, col=1)
        fig.add_trace(right_mesh, row=1, col=2)
        frames = []
        for t in range(T):
            frames.append(go.Frame(
                name=str(t),
                data=[
                    go.Mesh3d(x=verts_left[t,:,0], y=verts_left[t,:,1], z=verts_left[t,:,2], i=i,j=j,k=k,opacity=0.7,showscale=False,scene="scene"),
                    go.Mesh3d(x=verts_right[t,:,0], y=verts_right[t,:,1], z=verts_right[t,:,2], i=i,j=j,k=k,opacity=0.7,showscale=False,scene="scene2")
                ]
            ))
        fig.frames = frames
        fig.update_layout(
            showlegend=False,
            margin=dict(l=10, r=10, t=50, b=10),
            scene=dict(aspectmode='data',xaxis=dict(visible=False),yaxis=dict(visible=False),zaxis=dict(visible=False),
                       camera=dict(eye=dict(x=0,y=-2,z=0.7))),
            scene2=dict(aspectmode='data',xaxis=dict(visible=False),yaxis=dict(visible=False),zaxis=dict(visible=False),
                        camera=dict(eye=dict(x=0,y=-2,z=0.7))),
            updatemenus=[dict(
                type="buttons", x=0.5, xanchor="center", y=1.15, yanchor="top",
                buttons=[
                    dict(label="Play", method="animate", args=[None, {"frame": {"duration": max(1,1000//fps), "redraw": True}, "fromcurrent": True}]),
                    dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}}])
                ]
            )]
        )
        if output_html:
            fig.write_html(output_html)
            print(f"✅ Saved: {output_html}")
        return fig

    # Determine which words to include (up to `limit` distinct words)
    allowed_words = None
    if isinstance(limit, int) and limit > 0:
        ordered_unique_words = []
        for pair in pairs:
            word = pair[0]
            if word not in ordered_unique_words:
                ordered_unique_words.append(word)
            if len(ordered_unique_words) >= limit:
                break
        allowed_words = set(ordered_unique_words)

    for pair in pairs:
        try:
            if len(pair) == 4:
                word, pid, gt_seq, gen_seq = pair
            else:
                word, gt_seq, gen_seq = pair
                pid = "unknown"
            if allowed_words is not None and word not in allowed_words:
                continue
            tokens_gt = _extract_ids_from_sequence(gt_seq)
            tokens_gen = _extract_ids_from_sequence(gen_seq)
            params_gt = decode_tokens_to_params(tokens_gt, vq_model, mean, std)
            params_gen = decode_tokens_to_params(tokens_gen, vq_model, mean, std)
            verts_gt, faces = params_to_vertices(params_gt, smplx_model)
            verts_gen, _ = params_to_vertices(params_gen, smplx_model)
            out_dir = os.path.join(output_dir)
            os.makedirs(out_dir, exist_ok=True)
            # Sanitize for filesystem safety
            safe_word = re.sub(r'[^A-Za-z0-9_-]+', '_', str(word))
            safe_pid = re.sub(r'[^A-Za-z0-9_-]+', '_', str(pid))
            output_html = os.path.join(out_dir, f"word_{safe_word}_{safe_pid}_side_by_side.html")
            animate_side_by_side(
                verts_left=verts_gt,
                faces=faces,
                verts_right=verts_gen,
                fps=20,
                titles=("Ground Truth", "LLM Generated"),
                output_html=output_html
            )
        except Exception as exc:
            print(f"⚠️  Error creating visualization for word '{pair[0]}': {exc}")

def run_inference_on_all_samples(model, tokenizer, data, device, include_participant: bool = True):
    """
    Runs inference on ALL available samples for the trained words and compares 
    each one to its specific ground truth.
    """
    print("\n" + "="*80)
    print("      INFERENCE AND EVALUATION (ALL SAMPLES)")
    print("      Goal: Test the model's performance on every variant.")
    print("="*80)
    
    def compare_sequences(gt: str, gen: str):
        """Provides a simple visual diff of two sequences without external libraries."""
        gt_tokens = gt.split()
        gen_tokens = gen.split()

        print("\nDetailed Comparison (✅ = Match, ❌ = Mismatch/Missing/Added):")
        
        gt_str =   "  GT:  "
        gen_str =  "  GEN: "
        diff_str = "       "
        
        max_len = max(len(gt_tokens), len(gen_tokens))
        
        for i in range(max_len):
            gt_tok = gt_tokens[i] if i < len(gt_tokens) else "___"
            gen_tok = gen_tokens[i] if i < len(gen_tokens) else "___"
            
            max_tok_len = max(len(gt_tok), len(gen_tok))
            gt_tok_padded = gt_tok.ljust(max_tok_len)
            gen_tok_padded = gen_tok.ljust(max_tok_len)
            
            gt_str += gt_tok_padded + " "
            gen_str += gen_tok_padded + " "
            
            if gt_tok == gen_tok:
                diff_str += "✅".ljust(max_tok_len) + " "
            else:
                diff_str += "❌".ljust(max_tok_len) + " "
                
        print(gt_str)
        print(gen_str)
        print(diff_str)

    data_by_word = {}
    for item in data:
        word = item['word']
        if word not in data_by_word:
            data_by_word[word] = []
        data_by_word[word].append(item)

    for word, samples in data_by_word.items():
        print(f"\n\n{'='*25} TESTING WORD: '{word}' {'='*25}")
        num_correct = 0
        
        for i, sample in enumerate(samples):
            pid = sample.get("participant_id", "")
            if include_participant:
                print(f"\n--- Testing Variant {i+1}/{len(samples)}: '{pid}' ---")
            else:
                print(f"\n--- Testing Sample {i+1}/{len(samples)} (prompt is WORD-ONLY; PID ignored) ---")
            
            gt_tokens_str = sample.get("motion_tokens", "")
            gt_wrapped = " ".join([f"<M{t}>" for t in gt_tokens_str.split()])
            gt_sequence = f"{M_START} {gt_wrapped} {M_END}"
            print(f"Ground Truth:\n{gt_sequence}")

            prompt = build_instruction_prompt(
                word=sample.get("word", ""),
                participant_id=pid,
                include_participant=include_participant,
            )
            generated_sequence = generate_motion(model, tokenizer, prompt, device)
            print(f"\nLLM Generated:\n{generated_sequence}")
            
            compare_sequences(gt_sequence, generated_sequence)

            if gt_sequence.strip() == generated_sequence.strip():
                num_correct += 1
            
            print("-" * 80)
        
        accuracy = (num_correct / len(samples)) * 100
        print(f"\nSUMMARY FOR '{word}': {num_correct}/{len(samples)} correct ({accuracy:.1f}%)")


# ======================================================================================
# Existing Utilities (Compatibility)
# ======================================================================================
def seq_edit_distance(a_ids: List[int], b_ids: List[int]) -> int:
    """Token-level Levenshtein distance"""
    return Levenshtein.distance(a_ids, b_ids)

def best_ref_distance(pred_ids: List[int], refs: List[List[int]]) -> int:
    """Find minimum edit distance to any reference"""
    if not refs:
        return len(pred_ids)
    return min(seq_edit_distance(pred_ids, r) for r in refs)

def build_text_to_refs(dataset):
    """
    Build mapping from text prompts to list of reference motion sequences
    """
    text_to_refs = defaultdict(list)
    for ex in dataset:
        text_to_refs[ex["text_query"]].append(
            [int(x) for x in ex["motion_tokens"].split()]
        )
    return text_to_refs

def _concat(ids_list: List[List[int]]) -> List[int]:
    out = []
    for s in ids_list:
        out.extend(s)
    return out

def _distinct_n(ids_list: List[List[int]], n: int) -> float:
    if n <= 0:
        return 0.0
    total = 0
    uniq = set()
    for seq in ids_list:
        if len(seq) < n:
            continue
        total += (len(seq) - n + 1)
        for i in range(len(seq) - n + 1):
            uniq.add(tuple(seq[i:i+n]))
    if total == 0:
        return 0.0
    return len(uniq) / float(total)

def token_fid_diag(gens: List[List[int]], refs: List[List[int]], codebook_size: int) -> float:
    """
    Diagonal-covariance Fréchet distance between histograms of token usage.
    This is a lightweight proxy for FID using token distributions.
    """
    if len(gens) == 0 or len(refs) == 0:
        return float("nan")

    def feats(batch: List[List[int]]) -> np.ndarray:
        mats = []
        for seq in batch:
            hist = np.bincount([x for x in seq if 0 <= x < codebook_size], minlength=codebook_size).astype(np.float64)
            s = hist.sum()
            if s > 0:
                hist /= s
            mats.append(hist)
        return np.stack(mats, axis=0)

    G = feats(gens)
    R = feats(refs)
    mu_g = G.mean(axis=0)
    mu_r = R.mean(axis=0)
    var_g = G.var(axis=0)
    var_r = R.var(axis=0)
    mean_term = np.sum((mu_g - mu_r) ** 2)
    # Diagonal covariance approximation
    cov_term = np.sum(var_g + var_r - 2.0 * np.sqrt(np.clip(var_g * var_r, 0.0, None)))
    return float(mean_term + cov_term)

def compute_token_metrics(
    gen_by_text: Dict[str, List[int]],
    text_to_refs: Dict[str, List[List[int]]],
    codebook_size: int,
) -> Dict[str, float]:
    """
    Compute token-level metrics:
      - FID_diag: Fréchet distance between token histograms (diag cov)
      - MIM: average min edit distance to references
      - Diversity: distinct-1 and distinct-2
    """
    gens = list(gen_by_text.values())
    refs_all = _concat([v for v in text_to_refs.values()])
    # refs_all is concatenated list of ids; split sequences are needed
    ref_seqs = [r for refs in text_to_refs.values() for r in refs]

    fid_diag = token_fid_diag(gens, ref_seqs, codebook_size)

    # MIM: average best edit distance per prompt (only over prompts we generated)
    mim_dists = []
    for text, gen_ids in gen_by_text.items():
        refs = text_to_refs.get(text, [])
        mim_dists.append(best_ref_distance(gen_ids, refs))
    mim = float(sum(mim_dists) / len(mim_dists)) if mim_dists else float("nan")

    div1 = _distinct_n(gens, 1)
    div2 = _distinct_n(gens, 2)

    return {
        "FID_diag": fid_diag,
        "MIM": mim,
        "distinct_1": div1,
        "distinct_2": div2,
    }

def eval_t2m_set(
    model,
    tokenizer,
    sample_pairs: List[Tuple[str, List[List[int]]]],
    mot_begin_id: int,
    mot_end_id: int,
    motion_token_ids: list,
    length_stats_by_text: dict,
    global_median_len: int,
    prompt_vocab: dict = None,
    has_pid: bool = False,
    per_prompt_vocab: bool = True,
    n_eval: int = 100
):
    """
    Evaluate text-to-motion generation on a set of samples
    Returns a compact dict with avg_edit_dist & median_len; kept for pipeline compatibility.
    """
    random.shuffle(sample_pairs)
    subset = sample_pairs[:min(n_eval, len(sample_pairs))]
    
    dists = []
    lens = []
    
    for text, ref_list in subset:
        gen = generate_t2m(
            model=model,
            tokenizer=tokenizer,
            prompt_text=text,
            mot_begin_id=mot_begin_id,
            mot_end_id=mot_end_id,
            motion_token_ids=motion_token_ids,
            length_stats_by_text=length_stats_by_text,
            global_median_len=global_median_len,
            prompt_vocab=prompt_vocab,
            pid=None,
            has_pid=has_pid,
            per_prompt_vocab=per_prompt_vocab
        )
        span = gen.split("<MOT_BEGIN>")[-1]
        span = span.split("<MOT_END>")[0]
        pred_ids = motion_specials_to_ids(span)
        d = best_ref_distance(pred_ids, ref_list)
        dists.append(d)
        lens.append(len(pred_ids))
    
    if dists:
        avg_dist = sum(dists) / len(dists)
        median_len = sorted(lens)[len(lens)//2] if lens else 0
        print(f"Eval T2M: avg_edit_dist={avg_dist:.2f}, median_len={median_len}, n={len(dists)}")
        return {"avg_edit_dist": avg_dist, "median_len": median_len, "n_samples": len(dists)}
    else:
        print("Eval T2M: no samples")
        return {}

def _load_vqvae_helpers_for_metrics(device, vqvae_ckpt: Optional[str] = None, stats_path: Optional[str] = None):
    """
    Shared loader for Stage 3 multi-ref encoder-based evaluation.
    Returns: (vq_model, mean, std, decode_tokens_to_params)
    """
    from visualize import load_vqvae, load_stats, decode_tokens_to_params, VQVAE_CHECKPOINT as DEFAULT_VQ, STATS_PATH as DEFAULT_STATS
    vq_ckpt = vqvae_ckpt or os.getenv("VQVAE_CHECKPOINT", DEFAULT_VQ)
    stats_p = stats_path or os.getenv("VQVAE_STATS_PATH", DEFAULT_STATS)
    vq_model = load_vqvae(vq_ckpt, device=device)
    mean, std = load_stats(stats_p)
    return vq_model, mean, std, decode_tokens_to_params


def _wrap_gt_sequence_from_sample(sample: Dict[str, Any]) -> str:
    gt_tokens_str = str(sample.get("motion_tokens", "")).strip()
    gt_wrapped = " ".join([f"<M{t}>" for t in gt_tokens_str.split()])
    return f"{M_START} {gt_wrapped} {M_END}"


def _sequence_to_encoder_feature(seq: str, vq_model, mean, std, device, decode_tokens_to_params) -> Optional[np.ndarray]:
    """
    seq: string that may contain <M123> tokens (and other text).
    Returns L2-normalized encoder feature vector, or None on failure.
    """
    ids = _extract_ids_from_sequence(seq)
    if len(ids) == 0:
        return None
    try:
        params = decode_tokens_to_params(ids, vq_model, mean, std, device=device)
        feat = _encode_params_to_feature(params, vq_model, mean, std, device)
        return feat
    except Exception:
        return None


def _min_l2_to_refs(x: np.ndarray, ref_mat: np.ndarray) -> Tuple[float, int]:
    """
    Returns (min_l2_distance, argmin_index) between x and rows of ref_mat.
    Assumes both are float32 vectors, typically already L2-normalized.
    """
    d = np.linalg.norm(ref_mat - x.reshape(1, -1), axis=1)
    j = int(np.argmin(d))
    return float(d[j]), j


def evaluate_stage3_multiref_encoder_style(
    model,
    tokenizer,
    eval_data: List[Dict[str, Any]],
    device,
    *,
    k_samples: int = 10,
    vqvae_ckpt: Optional[str] = None,
    stats_path: Optional[str] = None,
    sample_limit: Optional[int] = None,
    seed: int = SEED,
):
    """
    Stage 3 (word-only, 1-to-many) evaluation using ONLY VQ-VAE encoder features.

    Option C metrics:
      - Quality-to-closest-ref:
          * avg_min_feat_dist: mean over (word, k) of min L2 distance to any GT ref for that word
          * avg_best_of_k_feat_dist: mean over words of best-of-K (min over k of min-ref distance)
      - Distribution match:
          * fid_per_word_mean: mean over words of FID(GT_feats(word), GEN_feats(word))
          * fid_global: FID over concatenated GT feats vs concatenated GEN feats (all words)

    Visualization helper output:
      - pairs_closest: one pair per word using GEN(best-of-K) vs GT(closest-ref to that GEN), with GT participant_id preserved.
    """
    subset = eval_data[:sample_limit] if (isinstance(sample_limit, int) and sample_limit > 0) else eval_data

    # Group by word (lower)
    by_word: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ex in subset:
        w = str(ex.get("word", "")).lower().strip()
        if not w:
            continue
        by_word[w].append(ex)

    if not by_word:
        return {"error": "No valid words in eval_data.", "pairs_closest": [], "per_word": {}}

    # Load VQ-VAE + stats + decoder
    try:
        vq_model, mean, std, decode_tokens_to_params = _load_vqvae_helpers_for_metrics(device, vqvae_ckpt=vqvae_ckpt, stats_path=stats_path)
    except Exception as exc:
        return {"error": f"Could not set up VQ-VAE encoder evaluation: {exc}", "pairs_closest": [], "per_word": {}}

    per_word: Dict[str, Dict[str, Any]] = {}
    pairs_closest: List[Tuple[str, str, str, str]] = []

    all_gt_feats = []
    all_gen_feats = []
    all_exact_matches = []
    all_gen_unique_ratios = []
    all_ref_coverages = []
    all_gen_diversities = []
    all_ref_diversities = []

    # Deterministic base RNG (generation is still stochastic, but this makes it repeatable run-to-run)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    for word, samples in sorted(by_word.items(), key=lambda kv: kv[0]):
        # Build GT ref feats for this word (ALL variants provided in eval_data)
        gt_feats = []
        gt_seqs = []
        gt_pids = []
        gt_id_keys = []
        for s in samples:
            gt_seq = _wrap_gt_sequence_from_sample(s)
            feat = _sequence_to_encoder_feature(gt_seq, vq_model, mean, std, device, decode_tokens_to_params)
            if feat is None:
                continue
            gt_feats.append(feat)
            gt_seqs.append(gt_seq)
            gt_pids.append(str(s.get("participant_id", "")))
            gt_id_keys.append(tuple(_extract_ids_from_sequence(gt_seq)))

        if len(gt_feats) < 2:
            # Too few refs to compute per-word cov/FID robustly
            per_word[word] = {
                "n_refs": len(gt_feats),
                "n_gens": 0,
                "n_refs_unique": len(set(gt_id_keys)),
                "avg_min_feat_dist": float("nan"),
                "best_of_k_feat_dist": float("nan"),
                "fid_word": float("nan"),
                "exact_match_rate": float("nan"),
                "n_gens_unique": 0,
                "gen_unique_ratio": float("nan"),
                "ref_coverage_ratio": float("nan"),
                "ref_diversity_feat": float("nan"),
                "gen_diversity_feat": float("nan"),
                "note": "Too few GT references (need >=2 encoder-features).",
            }
            continue

        gt_mat = np.stack(gt_feats, axis=0).astype(np.float32)
        all_gt_feats.append(gt_mat)

        # Reference diagnostics
        gt_key_to_index: Dict[tuple, int] = {}
        for j, key in enumerate(gt_id_keys):
            if key not in gt_key_to_index:
                gt_key_to_index[key] = j
        n_refs_unique = len(gt_key_to_index)
        # Mean pairwise distance in feature space (full, not sampled)
        try:
            diffs = gt_mat[:, None, :] - gt_mat[None, :, :]
            dmat = np.linalg.norm(diffs, axis=2)
            iu = np.triu_indices(dmat.shape[0], k=1)
            ref_div_feat = float(np.mean(dmat[iu])) if len(iu[0]) > 0 else float("nan")
        except Exception:
            ref_div_feat = float("nan")

        # Generate K samples for this word (word-only prompt)
        prompt = build_instruction_prompt(word=word, participant_id=None, include_participant=False)

        gen_feats = []
        gen_seqs = []
        gen_id_keys = []
        min_dists = []
        best_gen_i = None
        best_gen_dist = float("inf")
        best_gen_closest_ref_j = None
        exact_matches = 0
        matched_ref_keys = set()

        for k in range(int(k_samples)):
            # Make sampling reproducible per (word, k)
            # (Python hash is salted per process, so avoid hash(word))
            k_seed = int(seed + (k * 1000) + (sum(ord(c) for c in word) % 997))
            random.seed(k_seed)
            np.random.seed(k_seed % (2**32 - 1))
            torch.manual_seed(k_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(k_seed)

            gen_seq = generate_motion(model, tokenizer, prompt, device)
            feat_g = _sequence_to_encoder_feature(gen_seq, vq_model, mean, std, device, decode_tokens_to_params)
            if feat_g is None:
                continue
            gen_key = tuple(_extract_ids_from_sequence(gen_seq))
            gen_id_keys.append(gen_key)
            if gen_key in gt_key_to_index:
                exact_matches += 1
                matched_ref_keys.add(gen_key)

            d_min, j = _min_l2_to_refs(feat_g, gt_mat)
            gen_feats.append(feat_g)
            gen_seqs.append(gen_seq)
            min_dists.append(d_min)

            if d_min < best_gen_dist:
                best_gen_dist = d_min
                best_gen_i = len(gen_feats) - 1
                best_gen_closest_ref_j = j

        if len(gen_feats) < 2:
            per_word[word] = {
                "n_refs": int(gt_mat.shape[0]),
                "n_gens": len(gen_feats),
                "n_refs_unique": n_refs_unique,
                "avg_min_feat_dist": float("nan"),
                "best_of_k_feat_dist": float("nan"),
                "fid_word": float("nan"),
                "exact_match_rate": float("nan"),
                "n_gens_unique": len(set(gen_id_keys)),
                "gen_unique_ratio": float("nan"),
                "ref_coverage_ratio": float("nan"),
                "ref_diversity_feat": ref_div_feat,
                "gen_diversity_feat": float("nan"),
                "note": "Too few valid generated samples (need >=2 encoder-features).",
            }
            continue

        gen_mat = np.stack(gen_feats, axis=0).astype(np.float32)
        all_gen_feats.append(gen_mat)

        # Generation diagnostics
        n_gens = int(gen_mat.shape[0])
        n_gens_unique = len(set(gen_id_keys))
        gen_unique_ratio = float(n_gens_unique / max(1, n_gens))
        exact_match_rate = float(exact_matches / max(1, n_gens))
        ref_coverage_ratio = float(len(matched_ref_keys) / max(1, n_refs_unique))
        try:
            diffs_g = gen_mat[:, None, :] - gen_mat[None, :, :]
            dmat_g = np.linalg.norm(diffs_g, axis=2)
            iu_g = np.triu_indices(dmat_g.shape[0], k=1)
            gen_div_feat = float(np.mean(dmat_g[iu_g])) if len(iu_g[0]) > 0 else float("nan")
        except Exception:
            gen_div_feat = float("nan")

        # Option C: quality-to-closest-ref
        avg_min_dist = float(np.mean(min_dists)) if min_dists else float("nan")
        best_of_k = float(np.min(min_dists)) if min_dists else float("nan")

        # Option C: distribution match (per-word FID on encoder features)
        try:
            mu_g, cov_g = calculate_activation_statistics_np(gen_mat)
            mu_r, cov_r = calculate_activation_statistics_np(gt_mat)
            fid_word = float(calculate_frechet_distance_np(mu_r, cov_r, mu_g, cov_g))
        except Exception:
            fid_word = float("nan")

        per_word[word] = {
            "n_refs": int(gt_mat.shape[0]),
            "n_gens": n_gens,
            "n_refs_unique": n_refs_unique,
            "avg_min_feat_dist": avg_min_dist,
            "best_of_k_feat_dist": best_of_k,
            "fid_word": fid_word,
            "exact_match_rate": exact_match_rate,
            "n_gens_unique": n_gens_unique,
            "gen_unique_ratio": gen_unique_ratio,
            "ref_coverage_ratio": ref_coverage_ratio,
            "ref_diversity_feat": ref_div_feat,
            "gen_diversity_feat": gen_div_feat,
        }

        all_exact_matches.append(exact_match_rate)
        all_gen_unique_ratios.append(gen_unique_ratio)
        all_ref_coverages.append(ref_coverage_ratio)
        all_gen_diversities.append(gen_div_feat)
        all_ref_diversities.append(ref_div_feat)

        # Visualization pair: GEN(best-of-K) vs GT(closest ref to that GEN)
        if best_gen_i is not None and best_gen_closest_ref_j is not None:
            gt_seq_best = gt_seqs[best_gen_closest_ref_j]
            gt_pid_best = gt_pids[best_gen_closest_ref_j]
            gen_seq_best = gen_seqs[best_gen_i]
            pairs_closest.append((word, gt_pid_best, gt_seq_best, gen_seq_best))

    # Aggregate
    # mean over words (only those with finite values)
    def _mean_finite(xs: List[float]) -> float:
        xs2 = [float(x) for x in xs if x is not None and np.isfinite(x)]
        return float(np.mean(xs2)) if xs2 else float("nan")

    avg_min_feat_dist = _mean_finite([v.get("avg_min_feat_dist") for v in per_word.values()])
    avg_best_of_k_feat_dist = _mean_finite([v.get("best_of_k_feat_dist") for v in per_word.values()])
    fid_per_word_mean = _mean_finite([v.get("fid_word") for v in per_word.values()])
    exact_match_rate_mean = _mean_finite(all_exact_matches)
    gen_unique_ratio_mean = _mean_finite(all_gen_unique_ratios)
    ref_coverage_ratio_mean = _mean_finite(all_ref_coverages)
    gen_diversity_feat_mean = _mean_finite(all_gen_diversities)
    ref_diversity_feat_mean = _mean_finite(all_ref_diversities)

    # Global FID (concatenate all feats)
    try:
        gt_all = np.concatenate(all_gt_feats, axis=0) if all_gt_feats else None
        gen_all = np.concatenate(all_gen_feats, axis=0) if all_gen_feats else None
        if gt_all is None or gen_all is None or gt_all.shape[0] < 2 or gen_all.shape[0] < 2:
            fid_global = float("nan")
        else:
            mu_g, cov_g = calculate_activation_statistics_np(gen_all)
            mu_r, cov_r = calculate_activation_statistics_np(gt_all)
            fid_global = float(calculate_frechet_distance_np(mu_r, cov_r, mu_g, cov_g))
    except Exception:
        fid_global = float("nan")

    return {
        "source": "vqvae_encoder_stage3_multiref",
        "k_samples": int(k_samples),
        "avg_min_feat_dist": avg_min_feat_dist,
        "avg_best_of_k_feat_dist": avg_best_of_k_feat_dist,
        "fid_per_word_mean": fid_per_word_mean,
        "fid_global": fid_global,
        "exact_match_rate_mean": exact_match_rate_mean,
        "gen_unique_ratio_mean": gen_unique_ratio_mean,
        "ref_coverage_ratio_mean": ref_coverage_ratio_mean,
        "ref_diversity_feat_mean": ref_diversity_feat_mean,
        "gen_diversity_feat_mean": gen_diversity_feat_mean,
        "per_word": per_word,
        "pairs_closest": pairs_closest,
    }
