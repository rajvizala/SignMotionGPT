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
from signmotion_gpt.word_pipeline.data import motion_specials_to_ids
from signmotion_gpt.evaluation.generation import generate_t2m
from signmotion_gpt.common.config import (
    SEED, PIPELINE_OUTPUT_DIR, M_START, M_END,
    INFERENCE_TEMPERATURE, INFERENCE_TOP_K, INFERENCE_REPETITION_PENALTY
)

random.seed(SEED)

# ======================================================================================
# Length-conditioning bin thresholds - MUST match signmotion_gpt.sentence_pipeline.pipeline exactly
# ======================================================================================
LEN_CURRICULUM_SHORT_MAX  = 30   # < 30 tokens  -> LEN_SHORT
LEN_CURRICULUM_MEDIUM_MAX = 80   # 30-79 tokens -> LEN_MEDIUM
                                 # >= 80 tokens  -> LEN_LONG

# Exact V2 training templates (must stay in sync with train_sentence_pipeline_v2.TEMPLATES)
V2_EVAL_TEMPLATES = [
    "Instruction: Generate sign language motion for the sentence: '{text}'",
    "Instruction: Translate this text to sign language: '{text}'",
    "Instruction: How would you sign '{text}'?",
    "Instruction: Create a sign language animation for: '{text}'",
]

# ======================================================================================
# Lazy-loaded length predictor (loaded once, reused across calls)
# ======================================================================================
_len_predictor_cache: Dict[str, Any] = {}   # keys: 'model', 'encoder', 'device'

def _get_length_predictor(device):
    """
    Lazy-load the trained length predictor and SentenceTransformer encoder.
    Looks for the checkpoint at LENGTH_PREDICTOR_CKPT env variable, or at the
    default location './length_predictor_model/best_model.pt'.

    Returns (predictor_model, sentence_encoder) or (None, None) on failure.
    """
    if 'model' in _len_predictor_cache:
        return _len_predictor_cache['model'], _len_predictor_cache['encoder']

    ckpt_path = "/content/model_epoch_190.pt"

    if not os.path.exists(ckpt_path):
        print(f"  [LengthPredictor] Checkpoint not found at {ckpt_path}.")
        print("    Set LENGTH_PREDICTOR_CKPT env var or train with length_predictor.py.")
        print("    Falling back to GT-length binning.")
        _len_predictor_cache['model'] = None
        _len_predictor_cache['encoder'] = None
        return None, None

    try:
        from length_predictor import load_length_predictor, predict_length_for_sentences
        try:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as enc_err:
            print(f"  [LengthPredictor] Could not load SentenceTransformer: {enc_err}")
            _len_predictor_cache['model'] = None
            _len_predictor_cache['encoder'] = None
            return None, None

        predictor = load_length_predictor(ckpt_path, device=device)
        print(f"  [LengthPredictor] Loaded from {ckpt_path}")

        _len_predictor_cache['model'] = predictor
        _len_predictor_cache['encoder'] = encoder
        return predictor, encoder

    except Exception as e:
        print(f"  [LengthPredictor] Failed to load: {e}")
        _len_predictor_cache['model'] = None
        _len_predictor_cache['encoder'] = None
        return None, None


def _predict_len_token(sentence: str, predictor, encoder, device, fallback_len: int = None) -> str:
    """
    Predict the length-conditioning token (<|LEN_SHORT|>, <|LEN_MEDIUM|>, or <|LEN_LONG|>)
    for a given sentence using the trained length predictor.

    Falls back to GT-length binning when predictor is unavailable.
    """
    if predictor is not None and encoder is not None:
        try:
            from length_predictor import predict_length_for_sentences
            lengths = predict_length_for_sentences([sentence], predictor, encoder, device=device)
            predicted_len = int(lengths[0])
        except Exception:
            predicted_len = fallback_len if fallback_len is not None else LEN_CURRICULUM_MEDIUM_MAX
    else:
        # Graceful fallback: use GT token count if known, else assume medium
        predicted_len = fallback_len if fallback_len is not None else LEN_CURRICULUM_MEDIUM_MAX

    if predicted_len < LEN_CURRICULUM_SHORT_MAX:
        return "<|LEN_SHORT|>"
    elif predicted_len < LEN_CURRICULUM_MEDIUM_MAX:
        return "<|LEN_MEDIUM|>"
    else:
        return "<|LEN_LONG|>"

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


# ======================================================================================
# DTW-JPE / DTW-PA-JPE helpers  (SOKE paper metrics)
# ======================================================================================

def _procrustes_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rigid (rotation + translation, unit scale) Procrustes alignment.

    Aligns `source` onto `target` using SVD.  Both inputs are (N, 3) point sets
    for a *single frame* (or (T, J, 3) sequences).

    When sequences have different lengths (T_src ≠ T_tgt), `source` is
    resampled to `target`'s length before computing the rotation, so that the
    cross-covariance matrix H is well-defined.  The resulting rotation is then
    applied to the *original* (unresampled) source so that subsequent DTW still
    handles the length mismatch.

    A *global* rigid transform (single rotation + translation, no per-frame
    jitter) is computed — consistent with the sign-language evaluation standard.

    Args:
        source : (T_src, J, 3) or (J, 3)  — predicted / generated joints.
        target : (T_tgt, J, 3) or (J, 3)  — ground-truth joints.

    Returns:
        aligned_source : same shape as *source* (not resampled), rigid-aligned
                         to target's coordinate frame.
    """
    src = np.array(source, dtype=np.float64)
    tgt = np.array(target, dtype=np.float64)

    squeeze = src.ndim == 2
    if squeeze:
        src = src[np.newaxis]   # (1, J, 3)
        tgt = tgt[np.newaxis]

    T_src, J, _ = src.shape
    T_tgt        = tgt.shape[0]

    # Resample src to match tgt length for cross-covariance computation only.
    # This is necessary when T_src ≠ T_tgt so H = src_r.T @ tgt is (3,3).
    src_r = _resample_sequence(src, T_tgt)  # (T_tgt, J, 3)

    # Centre of mass over the resampled pair (temporal + joint mean)
    src_mean = src_r.mean(axis=(0, 1), keepdims=True)  # (1,1,3)
    tgt_mean =  tgt.mean(axis=(0, 1), keepdims=True)

    src_c = src_r - src_mean  # both (T_tgt, J, 3)
    tgt_c = tgt   - tgt_mean

    # 3×3 cross-covariance — shapes match because we resampled
    H = (src_c.reshape(-1, 3)).T @ (tgt_c.reshape(-1, 3))  # (3, 3)
    U, _S, Vt = np.linalg.svd(H)

    # Ensure proper rotation (det = +1)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T  # (3, 3)

    # Apply rotation to the ORIGINAL (unresampled) source and re-centre onto target.
    # We use the original src mean for centering (approximated from src_r mean).
    orig_src_mean = src.mean(axis=(0, 1), keepdims=True)  # (1,1,3)
    src_orig_c = src - orig_src_mean  # (T_src, J, 3)
    aligned = (src_orig_c.reshape(T_src * J, 3) @ R.T).reshape(T_src, J, 3) + tgt_mean

    if squeeze:
        aligned = aligned[0]
    return aligned.astype(np.float32)

def _resample_sequence(seq: np.ndarray, target_T: int) -> np.ndarray:
    """Linearly resample a joint sequence to a different number of frames.

    Args:
        seq      : (T_src, J, 3) or (T_src, D) sequence.
        target_T : desired number of output frames.

    Returns:
        Resampled array with shape (target_T, J, 3) or (target_T, D).
    """
    T_src = seq.shape[0]
    if T_src == target_T:
        return seq
    src_t = np.linspace(0, 1, T_src)
    dst_t = np.linspace(0, 1, target_T)
    original_shape = seq.shape[1:]  # everything after the time dimension
    flat = seq.reshape(T_src, -1)   # (T_src, D_flat)
    out = np.stack(
        [np.interp(dst_t, src_t, flat[:, d]) for d in range(flat.shape[1])],
        axis=1
    )  # (target_T, D_flat)
    return out.reshape(target_T, *original_shape).astype(seq.dtype)


def _dtw_distance(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    n_joints: int = 1,
) -> float:
    """Normalized Dynamic Time Warping distance between two joint sequences.

    Cost function:  mean per-joint L2 error for each frame pair.
    Normalization:  total accumulated cost / optimal path length (number of
                    steps).  The result is the *average per-frame mean-joint
                    displacement* — the same unit reported by SOKE and most
                    sign-language papers.

    Args:
        seq_a   : (T_a, D) — e.g. T_a frames × (J*3) flattened joints.
        seq_b   : (T_b, D)
        n_joints: number of joints J so we can compute mean-per-joint error.
                  D must equal J*3.  Default 1 keeps old L2-norm behaviour.

    Returns:
        Scalar normalized DTW distance (mean per-frame, per-joint L2).
    """
    T_a, D = seq_a.shape
    T_b = seq_b.shape[0]

    # Build per-frame-pair cost matrix
    if n_joints > 1 and D == n_joints * 3:
        # Mean per-joint L2: makes units comparable across body/hand and papers
        a3 = seq_a.reshape(T_a, n_joints, 3)
        b3 = seq_b.reshape(T_b, n_joints, 3)
        diff = a3[:, np.newaxis, :, :] - b3[np.newaxis, :, :, :]   # (T_a, T_b, J, 3)
        cost = np.mean(np.linalg.norm(diff, axis=3), axis=2)        # (T_a, T_b)
    else:
        cost = np.linalg.norm(seq_a[:, np.newaxis, :] - seq_b[np.newaxis, :, :], axis=2)

    # DP — track accumulated cost AND path-step count simultaneously
    INF = np.inf
    dtw   = np.full((T_a + 1, T_b + 1), INF, dtype=np.float64)
    steps = np.zeros((T_a + 1, T_b + 1), dtype=np.int32)
    dtw[0, 0] = 0.0
    for i in range(1, T_a + 1):
        for j in range(1, T_b + 1):
            c = float(cost[i - 1, j - 1])
            v_ins  = dtw[i - 1, j]
            v_del  = dtw[i,     j - 1]
            v_mat  = dtw[i - 1, j - 1]
            if v_ins <= v_del and v_ins <= v_mat:
                dtw[i, j]   = c + v_ins
                steps[i, j] = 1 + steps[i - 1, j]
            elif v_del <= v_mat:
                dtw[i, j]   = c + v_del
                steps[i, j] = 1 + steps[i, j - 1]
            else:
                dtw[i, j]   = c + v_mat
                steps[i, j] = 1 + steps[i - 1, j - 1]

    path_len = steps[T_a, T_b]
    if path_len == 0:
        return 0.0
    return float(dtw[T_a, T_b]) / path_len


def _tokens_to_smplx_joints(
    token_ids: list,
    vq_model,
    mean,
    std,
    smplx_model,
    device,
    decode_tokens_fn,
) -> tuple:
    """Decode token IDs → SMPL-X joint positions (body + hands).

    Uses the visualize.py `decode_tokens_to_params` helper to get 182-dim
    SMPL-X parameters, then runs the SMPL-X forward pass to obtain 3-D joints.

    SMPL-X joint layout (127 joints total when using default config):
      0      = pelvis
      1-21   = body joints (21 joints)
      22-24  = jaw / left-eye / right-eye
      25-54  = left-hand joints  (15 knuckles × 3 DOF in axis-angle, 25-39 = 15)
      55-84  = right-hand joints (55-69 = 15)

    We follow SOKE's split of ~11 upper-body + 30 hand joints:
      body  : indices 0–10 (pelvis + 10 upper-body joints)
      hands : indices 25–54 (left hand 15) + 55–69 (right hand 15 → 30 total)

    Args:
        token_ids       : list[int] — VQ-VAE token IDs.
        vq_model        : loaded VQ-VAE wrapper.
        mean, std       : normalisation stats.
        smplx_model     : loaded smplx.SMPLX instance.
        device          : torch device.
        decode_tokens_fn: visualize.decode_tokens_to_params function.

    Returns:
        (body_joints, hand_joints) — numpy arrays (T, J_body, 3) and (T, J_hand, 3),
        or (None, None) on failure.
    """
    if not token_ids or smplx_model is None:
        return None, None
    try:
        import torch
        params = decode_tokens_fn(token_ids, vq_model, mean, std, device=device)  # (T, 182)

        # -----------------------------------------------------------------------
        # SENTENCE-LEVEL SMPL-X layout (How2Sign / hybrid VQ-VAE training):
        #   [shape(10), body_pose(63), lhand(45), rhand(45), jaw(3),
        #    expression(10), root_pose(3), cam_trans(3)]  => total 182
        #
        # Index map:
        #   0-9    : shape / betas
        #   10-72  : body_pose (63-dim)
        #   73-117 : left_hand_pose (45-dim)
        #   118-162: right_hand_pose (45-dim)
        #   163-165: jaw_pose (3-dim)   <-- NOT transl!
        #   166-175: expression (10-dim)
        #   176-178: root_pose / global_orient (3-dim)
        #   179-181: cam_trans / transl (3-dim)
        # -----------------------------------------------------------------------
        T = params.shape[0]
        betas         = torch.from_numpy(params[:, 0:10].astype(np.float32)).to(device)     # (T, 10)
        body_p        = torch.from_numpy(params[:, 10:73].astype(np.float32)).to(device)    # (T, 63)
        lhand_p       = torch.from_numpy(params[:, 73:118].astype(np.float32)).to(device)   # (T, 45)
        rhand_p       = torch.from_numpy(params[:, 118:163].astype(np.float32)).to(device)  # (T, 45)
        jaw_p         = torch.from_numpy(params[:, 163:166].astype(np.float32)).to(device)  # (T, 3)
        expr_p        = torch.from_numpy(params[:, 166:176].astype(np.float32)).to(device)  # (T, 10)
        global_orient = torch.from_numpy(params[:, 176:179].astype(np.float32)).to(device)  # (T, 3)
        transl_p      = torch.from_numpy(params[:, 179:182].astype(np.float32)).to(device)  # (T, 3)
        # CRITICAL: SMPL-X stores default leye/reye of shape (1, 3) internally.
        # If we omit them, SMPL-X tries to cat (T,*) tensors with its (1,3) defaults
        # → "Expected size T but got size 1" error. We must always pass explicit zeros.
        zeros3 = torch.zeros((T, 3), dtype=torch.float32, device=device)

        # Process in batches to avoid OOM for long sequences (VQ-VAE upsamples 4x,
        # so 128 tokens → 512 frames)
        BATCH = 64
        all_joints = []
        with torch.no_grad():
            for s in range(0, T, BATCH):
                e = min(s + BATCH, T)
                out = smplx_model(
                    betas=betas[s:e],
                    global_orient=global_orient[s:e],
                    body_pose=body_p[s:e],
                    left_hand_pose=lhand_p[s:e],
                    right_hand_pose=rhand_p[s:e],
                    expression=expr_p[s:e],
                    jaw_pose=jaw_p[s:e],
                    leye_pose=zeros3[s:e],   # Must match batch size T; default is (1,3) → mismatch
                    reye_pose=zeros3[s:e],   # Same here
                    transl=transl_p[s:e],
                    return_verts=False,
                )
                all_joints.append(out.joints.cpu().numpy())  # (batch, 127, 3)

        joints = np.concatenate(all_joints, axis=0)  # (T, 127, 3)

        # -----------------------------------------------------------------------
        # ROOT-RELATIVE normalisation (standard for 3D pose metrics)
        # -----------------------------------------------------------------------
        # SMPL-X joints are in camera-space metres. The root joint (pelvis,
        # index 0) encodes global translation: person could be anywhere from
        # 3–8 m from the camera. This global offset dominates any DTW distance
        # and is meaningless for sign quality — two identical sign sequences
        # filmed from different distances would look "far apart".
        #
        # Solution: subtract pelvis at every frame → root-relative positions.
        # This is the standard convention used by HumanEva, MPI-INF-3DHP, and
        # every MPJPE paper. Values become relative displacements (metres).
        #
        # Then ×1000 → millimetres, matching the mm scale used by most pose
        # papers (including the SOKE equivalents you compare against).
        # -----------------------------------------------------------------------
        pelvis = joints[:, 0:1, :]                   # (T, 1, 3) — root
        joints_rel = (joints - pelvis) * 100.0       # root-relative, in cm  ← matches SOKE/NSA scale

        # Split body (indices 0–10) and hands (indices 25–54)
        body_joints = joints_rel[:, 0:11, :]    # (T, 11, 3) cm, root-relative
        hand_joints = joints_rel[:, 25:55, :]   # (T, 30, 3) cm, root-relative
        return body_joints, hand_joints

    except Exception as exc:
        print(f"  [DTW-JPE] _tokens_to_smplx_joints failed: {exc}")
        return None, None


def compute_dtw_jpe(
    gt_body: np.ndarray, gt_hand: np.ndarray,
    gen_body: np.ndarray, gen_hand: np.ndarray,
) -> dict:
    """Compute DTW-JPE and PA-JPE for a single (GT, GEN) pair.

    Matches the SOKE / NSA paper metric definitions:

      DTW-JPE   = Normalised DTW on raw root-relative joint positions.
                  Handles variable-length sequences natively.
                  Units: cm.  Lower is better.

      PA-JPE    = Procrustes-Aligned MPJPE.
                  Both sequences are resampled to the same length (max of the
                  two), one global 3-D rotation is found via Procrustes SVD,
                  the generated sequence is rotated, and mean per-joint L2
                  error is computed over all frames.
                  This approach guarantees PA-JPE ≤ DTW-JPE always.
                  Units: cm.  Lower is better.

    Args:
        gt_body  : (T_gt, 11, 3) — GT body joints, root-relative, cm.
        gt_hand  : (T_gt, 30, 3) — GT hand joints, root-relative, cm.
        gen_body : (T_gen, 11, 3) — Generated body joints.
        gen_hand : (T_gen, 30, 3) — Generated hand joints.

    Returns:
        Dict with keys: dtw_jpe_body, dtw_jpe_hand, dtw_pa_jpe_body, dtw_pa_jpe_hand.
        Values are floats; NaN on any failure.
    """
    nan = float("nan")
    out = {
        "dtw_jpe_body": nan,
        "dtw_jpe_hand": nan,
        "dtw_pa_jpe_body": nan,
        "dtw_pa_jpe_hand": nan,
    }

    try:
        J_body = gt_body.shape[1]   # 11
        J_hand = gt_hand.shape[1]   # 30

        def flat(arr):
            return arr.reshape(arr.shape[0], -1).astype(np.float64)

        # ------------------------------------------------------------------
        # 1. DTW-JPE — normalized DTW on raw root-relative sequences.
        #    Handles T_gen ≠ T_gt natively.
        # ------------------------------------------------------------------
        out["dtw_jpe_body"] = _dtw_distance(flat(gt_body), flat(gen_body), n_joints=J_body)
        out["dtw_jpe_hand"] = _dtw_distance(flat(gt_hand), flat(gen_hand), n_joints=J_hand)

        # ------------------------------------------------------------------
        # 2. DTW-PA-JPE — DTW on Procrustes-aligned sequences.
        #    Exactly matches the SOKE / NSA "DTW on procrustes-aligned JPE".
        #
        #    Since both sequences are ALREADY ROOT-RELATIVE (pelvis = 0,0,0
        #    every frame), there is NO global translation to correct — only a
        #    possible global rotation (e.g. person oriented slightly differently
        #    between clips).  Using a full translation Procrustes would subtract
        #    the mean-of-all-joints-across-all-frames and then re-add the GT
        #    mean, effectively MOVING the skeleton to a wrong position and
        #    blowing up the error.
        #
        #    Correct approach: ROTATION-ONLY Procrustes.
        #      H = gen_flat.T @ gt_flat  (no centering)
        #      R = Vt.T @ D @ U.T from SVD(H)
        #    This finds the rotation that best aligns the generated sequence
        #    onto the GT in a least-squares sense.  Then DTW handles the
        #    remaining temporal alignment.
        # ------------------------------------------------------------------
        def _dtw_pa_jpe(gt_seq, gen_seq, n_j):
            """Rotation-only Procrustes alignment then DTW-JPE.

            Args:
                gt_seq  : (T_gt, J, 3)  — ground truth, root-relative cm
                gen_seq : (T_gen, J, 3) — generated,     root-relative cm
                n_j     : number of joints J
            Returns:
                Scalar normalised DTW distance after rotation alignment.
            """
            T_gt  = gt_seq.shape[0]
            T_gen = gen_seq.shape[0]
            T_c   = max(T_gt, T_gen)

            # Resample ONLY to compute the rotation; keep original for DTW.
            gt_r  = _resample_sequence(gt_seq,  T_c).astype(np.float64)  # (T_c, J, 3)
            gen_r = _resample_sequence(gen_seq, T_c).astype(np.float64)  # (T_c, J, 3)

            # Rotation-only Procrustes: H = gen.T @ gt (no centroid subtraction)
            # Root-relative means pelvis=0 for every frame — no translation needed.
            H = gen_r.reshape(-1, 3).T @ gt_r.reshape(-1, 3)   # (3, 3)
            U, _S, Vt = np.linalg.svd(H)
            d = np.linalg.det(Vt.T @ U.T)
            D = np.diag([1.0, 1.0, d])
            R = Vt.T @ D @ U.T  # (3, 3)  — rotation only, det = +1

            # Apply rotation to the ORIGINAL-length generated sequence
            gen_rot = (gen_seq.astype(np.float64).reshape(-1, 3) @ R.T).reshape(T_gen, n_j, 3)

            # DTW on rotated gen vs original gt
            return _dtw_distance(
                gt_seq.reshape(T_gt, -1).astype(np.float64),
                gen_rot.reshape(T_gen, -1).astype(np.float64),
                n_joints=n_j,
            )

        out["dtw_pa_jpe_body"] = _dtw_pa_jpe(gt_body, gen_body, J_body)
        out["dtw_pa_jpe_hand"] = _dtw_pa_jpe(gt_hand, gen_hand, J_hand)

    except Exception as exc:
        print(f"  [compute_dtw_jpe] error: {exc}")
        import traceback; traceback.print_exc()

    return out


# Lazy cache for SMPL-X model (same device-keyed pattern as length predictor)
_smplx_model_cache: Dict[str, Any] = {}


def _get_smplx_model_for_metrics(device: str = "cpu"):
    """Lazy-load the SMPL-X model for DTW-JPE computation.

    Looks up the model directory in:
      1. SMPLX_MODEL_DIR environment variable
      2. visualize.SMPLX_MODEL_DIR default

    Returns the loaded smplx.SMPLX instance, or None if unavailable.
    """
    key = str(device)
    if key in _smplx_model_cache:
        return _smplx_model_cache[key]

    try:
        from visualize import load_smplx_model, SMPLX_MODEL_DIR as DEFAULT_SMPLX_DIR
        smplx_dir = os.environ.get("SMPLX_MODEL_DIR", DEFAULT_SMPLX_DIR)
        smplx_m = load_smplx_model(smplx_dir, device=device)
        _smplx_model_cache[key] = smplx_m
        print(f"  [DTW-JPE] SMPL-X model loaded from {smplx_dir}")
        return smplx_m
    except Exception as exc:
        print(f"  [DTW-JPE] SMPL-X model unavailable — DTW-JPE will be null. ({exc})")
        _smplx_model_cache[key] = None
        return None


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

def generate_motion(model, tokenizer, prompt, device, 
                    max_new_tokens: int = 150,
                    min_new_tokens: int = None,
                    use_greedy: bool = False,
                    temperature: float = None,
                    force_length: bool = False):
    """
    Generates a motion sequence from a prompt.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt text
        device: torch device
        max_new_tokens: Maximum tokens to generate (default 150 for sentences)
        min_new_tokens: Minimum tokens before allowing EOS (helps with short generation)
        use_greedy: If True, use greedy decoding (deterministic, often better for accuracy)
        temperature: Override default temperature (lower = more deterministic)
        force_length: If True, ignore EOS until min_new_tokens is reached
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Determine if this is a sentence prompt (longer expected output)
    is_sentence = "sentence:" in prompt.lower()
    
    # Adjust parameters based on prompt type
    if min_new_tokens is None:
        min_new_tokens = 20 if is_sentence else 5
    
    if temperature is None:
        temperature = INFERENCE_TEMPERATURE
    
    # For sentences, use slightly lower temperature for more consistent output
    if is_sentence and temperature > 0.5:
        temperature = 0.5
    
    m_end_id = tokenizer.convert_tokens_to_ids(M_END)
    
    with torch.no_grad():
        if use_greedy:
            # Greedy decoding - deterministic, often more accurate
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=m_end_id,
            )
        else:
            # Sampling with adjusted parameters
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=INFERENCE_TOP_K,
                top_p=0.9,  # Added nucleus sampling
                repetition_penalty=INFERENCE_REPETITION_PENALTY,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=m_end_id,
            )
    
    decoded = tokenizer.decode(output[0], skip_special_tokens=False)
    if "Motion: " in decoded:
        motion_part = decoded.split("Motion: ")[-1]
    else:
        motion_part = decoded
    return motion_part.strip()


def generate_motion_for_sentence(model, tokenizer, prompt, device, 
                                  expected_length: int = None):
    """
    Specialized generation for sentence-level prompts.
    Uses more conservative settings to encourage longer, more accurate sequences.
    
    Args:
        expected_length: If known, the expected output length (helps set min_new_tokens)
    """
    # For sentences, we want:
    # 1. Longer max generation
    # 2. Minimum length to prevent early termination
    # 3. Lower temperature for consistency
    # 4. Possibly greedy decoding for accuracy
    
    min_tokens = 25  # Sentences should be at least this long
    if expected_length:
        min_tokens = max(25, int(expected_length * 0.5))  # At least 50% of expected
    
    return generate_motion(
        model, tokenizer, prompt, device,
        max_new_tokens=180,  # Allow long sequences
        min_new_tokens=min_tokens,
        use_greedy=True,  # Greedy often works better for accuracy
        temperature=0.3,   # Low temperature for consistency
    )


# Length bucket tokens for v3 pipeline
# Must match train_sentence_pipeline_v3.LENGTH_BUCKETS (include 120 for 110-129 range)
LENGTH_BUCKETS_V3 = [10, 20, 30, 40, 50, 60, 80, 100, 120, 128]


def get_length_bucket_token(length: int) -> str:
    """Map motion sequence length to a length bucket token (v3 format)."""
    for bucket in LENGTH_BUCKETS_V3:
        if length <= bucket:
            return f"<LEN_{bucket}>"
    return f"<LEN_{LENGTH_BUCKETS_V3[-1]}>"


def build_sentence_prompt_v3(sentence: str, expected_length: int = None, include_length: bool = True) -> str:
    """
    Build prompt for v3 sentence-level pipeline with optional length conditioning.
    
    Args:
        sentence: The input sentence to generate motion for
        expected_length: Expected number of motion tokens (used for length bucket)
        include_length: If True and expected_length is provided, include length token
    
    Returns:
        Formatted prompt string
    """
    if include_length and expected_length is not None:
        length_token = get_length_bucket_token(expected_length)
        return f"Instruction: Generate sign language motion for: '{sentence}'\nExpected length: {length_token}\nMotion: "
    return f"Instruction: Generate sign language motion for: '{sentence}'\nMotion: "


def generate_motion_for_sentence_v3(
    model, tokenizer, sentence: str, device,
    expected_length: int = None,
    include_length_conditioning: bool = True,
    use_greedy: bool = True,
    temperature: float = 0.5,
    repetition_penalty: float = 1.2,
    top_p: float = 0.9,
):
    """
    Improved sentence-level generation with length conditioning (v3 pipeline).
    
    Key improvements over v1:
    1. Length conditioning via bucket tokens
    2. Repetition penalty to avoid mode collapse
    3. Better min/max token bounds based on expected length
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        sentence: The input sentence
        device: torch device
        expected_length: Expected number of motion tokens (for length conditioning)
        include_length_conditioning: Whether to include length token in prompt
        use_greedy: Use greedy decoding (more deterministic)
        temperature: Sampling temperature (lower = more deterministic)
        repetition_penalty: Penalty for repeating tokens (higher = less repetition)
        top_p: Nucleus sampling parameter
    
    Returns:
        Generated motion sequence string
    """
    model.eval()
    
    # Build prompt with optional length conditioning
    prompt = build_sentence_prompt_v3(
        sentence, 
        expected_length=expected_length,
        include_length=include_length_conditioning and expected_length is not None
    )
    
    # Set generation bounds based on expected length
    if expected_length is not None:
        max_new_tokens = min(expected_length + 40, 200)
        min_new_tokens = max(10, int(expected_length * 0.4))
    else:
        max_new_tokens = 180
        min_new_tokens = 20
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    m_end_id = tokenizer.convert_tokens_to_ids(M_END)
    
    with torch.no_grad():
        if use_greedy:
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=m_end_id,
                repetition_penalty=repetition_penalty,
            )
        else:
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=50,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=m_end_id,
            )
    
    decoded = tokenizer.decode(output[0], skip_special_tokens=False)
    if "Motion: " in decoded:
        motion_part = decoded.split("Motion: ")[-1]
    else:
        motion_part = decoded
    
    return motion_part.strip()


# =============================================================================
# Length Predictor Integration for Inference
# =============================================================================

# =============================================================================
# Integration with User's Length Predictor (length_predictor.py)
# =============================================================================

def load_trained_length_predictor(
    model_path: str,
    device: str = "cuda",
):
    """
    Load the trained MLP length predictor from length_predictor.py.
    
    Args:
        model_path: Path to the trained model (e.g., ./length_predictor_model/best_model.pt)
        device: Device to load model on
    
    Returns:
        Tuple of (model, sentence_encoder, predict_function)
    """
    try:
        from length_predictor import (
            load_length_predictor,
            predict_length_for_sentences,
        )
        from sentence_transformers import SentenceTransformer
        
        # Load model
        model = load_length_predictor(model_path, device=torch.device(device))
        
        # Load sentence encoder
        sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        def predict_fn(sentence: str) -> int:
            """Predict motion length for a single sentence."""
            lengths = predict_length_for_sentences(
                [sentence], model, sentence_encoder, device=torch.device(device)
            )
            return lengths[0]
        
        print(f"[LengthPredictor] Loaded from {model_path}")
        return model, sentence_encoder, predict_fn
        
    except ImportError as e:
        print(f"[LengthPredictor] Could not load: {e}")
        print("  Falling back to heuristic predictor")
        return None, None, None


def generate_with_trained_length_predictor(
    motion_model,
    motion_tokenizer,
    length_predictor_path: str,
    sentence: str,
    device: str = "cuda",
    use_greedy: bool = True,
    temperature: float = 0.5,
    repetition_penalty: float = 1.2,
):
    """
    Complete inference pipeline using your trained length predictor.
    
    Args:
        motion_model: The trained motion generation LLM
        motion_tokenizer: Tokenizer for the motion model
        length_predictor_path: Path to trained length predictor model
        sentence: Input sentence to generate motion for
        device: Device to use
        use_greedy: Use greedy decoding
        temperature: Sampling temperature
        repetition_penalty: Repetition penalty
    
    Returns:
        Dictionary with:
        - predicted_length: Predicted motion token count
        - length_bucket: Length bucket token used
        - motion: Generated motion sequence
    """
    # Load length predictor
    _, _, predict_fn = load_trained_length_predictor(length_predictor_path, device)
    
    if predict_fn is None:
        # Fallback to heuristic
        word_count = len(sentence.split())
        predicted_length = max(8, min(128, int(10 + 3.5 * word_count)))
    else:
        predicted_length = predict_fn(sentence)
    
    # Get bucket token
    length_bucket = get_length_bucket_token(predicted_length)
    
    # Generate motion
    motion = generate_motion_for_sentence_v3(
        motion_model, motion_tokenizer, sentence, device,
        expected_length=predicted_length,
        include_length_conditioning=True,
        use_greedy=use_greedy,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
    )
    
    return {
        "predicted_length": predicted_length,
        "length_bucket": length_bucket,
        "motion": motion,
    }


def create_heuristic_length_predictor(data: List[Dict[str, Any]]) -> callable:
    """
    Create a simple heuristic length predictor from training data.
    
    This is a lightweight alternative to the full neural length predictor.
    Uses linear regression: length = base + slope * word_count
    
    Args:
        data: Training data with 'text'/'sentence' and 'motion_tokens'
    
    Returns:
        A function that predicts length from a sentence string
    """
    word_counts = []
    motion_lengths = []
    
    for item in data:
        text = item.get("text") or item.get("sentence", "")
        motion_str = item.get("motion_tokens", "")
        
        if text.strip() and motion_str.strip():
            words = len(text.split())
            tokens = len(motion_str.split())
            word_counts.append(words)
            motion_lengths.append(tokens)
    
    if len(word_counts) == 0:
        # Fallback defaults
        base = 10
        slope = 3.0
    else:
        import numpy as np
        word_counts = np.array(word_counts)
        motion_lengths = np.array(motion_lengths)
        
        mean_words = np.mean(word_counts)
        mean_tokens = np.mean(motion_lengths)
        
        numerator = np.sum((word_counts - mean_words) * (motion_lengths - mean_tokens))
        denominator = np.sum((word_counts - mean_words) ** 2)
        
        if denominator > 0:
            slope = numerator / denominator
            base = mean_tokens - slope * mean_words
        else:
            base = 10
            slope = 3.0
    
    print(f"[Heuristic Length Predictor] Fitted: {slope:.2f} tokens/word + {base:.1f} base")
    
    def predict(sentence: str) -> int:
        word_count = len(sentence.split())
        predicted = base + slope * word_count
        return max(8, min(128, int(round(predicted))))
    
    return predict


def generate_sentence_motion_auto_length(
    model, tokenizer, sentence: str, device,
    length_predictor: callable = None,
    use_greedy: bool = True,
    temperature: float = 0.5,
    repetition_penalty: float = 1.2,
):
    """
    Generate motion for a sentence with automatic length prediction.
    
    This is the recommended inference function for sentence-level generation.
    
    Args:
        model: The motion generation LLM
        tokenizer: The tokenizer
        sentence: Input sentence
        device: torch device
        length_predictor: Function that takes sentence and returns expected length.
                         If None, uses a simple word-count heuristic.
        use_greedy: Use greedy decoding
        temperature: Sampling temperature
        repetition_penalty: Repetition penalty
    
    Returns:
        Tuple of (predicted_length, length_bucket, generated_motion)
    """
    # Default length predictor: simple word count heuristic
    if length_predictor is None:
        word_count = len(sentence.split())
        predicted_length = max(8, min(128, int(10 + 3.0 * word_count)))
    else:
        predicted_length = length_predictor(sentence)
    
    # Get length bucket
    length_bucket = get_length_bucket_token(predicted_length)
    
    # Generate with length conditioning
    motion = generate_motion_for_sentence_v3(
        model, tokenizer, sentence, device,
        expected_length=predicted_length,
        include_length_conditioning=True,
        use_greedy=use_greedy,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
    )
    
    return predicted_length, length_bucket, motion


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
        print(f"[Warning]  Multimodality could not be computed reliably: {exc}")
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
        print(f"[Warning]  Could not set up VQ-VAE encoder metrics: {exc}")
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
        print(f"[Warning]  Multimodality could not be computed reliably: {exc}")
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

def save_side_by_side_visualizations(
    pairs: list[Tuple[str, str, str]], 
    output_dir: str, 
    limit: int = 4,
    data_level: str = "word",
    output_format: str = "video"
):
    """
    Generate side-by-side 3D visualizations for GT vs GEN using PyRender video.
    
    Args:
        pairs: List of (word, gt_seq, gen_seq) or (word, pid, gt_seq, gen_seq) tuples
        output_dir: Directory to save video visualizations
        limit: Maximum number of distinct words to visualize
        data_level: "word" for word-level data, "sentence" for sentence-level (How2Sign/hybrid)
                   This affects how SMPL-X parameters are mapped.
        output_format: "video" (default, PyRender MP4) or "html" (Plotly interactive)
    """
    try:
        from visualize import (
            load_vqvae, load_stats, load_smplx_model,
            decode_tokens_to_params, params_to_vertices,
            VQVAE_CHECKPOINT as DEFAULT_VQ, STATS_PATH as DEFAULT_STATS, SMPLX_MODEL_DIR as DEFAULT_SMPLX
        )
    except Exception as exc:
        print(f"Visualization skipped (missing dependencies): {exc}")
        return

    # Try to import optional video rendering (may not exist in visualize.py)
    render_side_by_side_video = None
    try:
        from visualize import render_side_by_side_video, ensure_pyrender
    except ImportError:
        pass

    os.makedirs(output_dir, exist_ok=True)
    vqvae_ckpt = os.getenv("VQVAE_CHECKPOINT", DEFAULT_VQ)
    stats_path = os.getenv("VQVAE_STATS_PATH", DEFAULT_STATS)
    smplx_dir = os.getenv("SMPLX_MODEL_DIR", DEFAULT_SMPLX)

    print(f"Loading VQ-VAE, stats, SMPL-X (data_level={data_level})...")
    vq_model = load_vqvae(vqvae_ckpt)
    mean, std = load_stats(stats_path)
    smplx_model = load_smplx_model(smplx_dir)

    # Fall back to HTML if video rendering is not available
    if output_format == "video":
        if render_side_by_side_video is None:
            print("render_side_by_side_video not available in visualize.py. Falling back to HTML output.")
            output_format = "html"
        else:
            try:
                if not ensure_pyrender():
                    print("PyRender not available. Falling back to HTML output.")
                    output_format = "html"
            except Exception:
                print("PyRender check failed. Falling back to HTML output.")
                output_format = "html"

    # HTML fallback function (kept for backward compatibility)
    def animate_side_by_side_html(verts_left, faces, verts_right, fps=20, titles=("Ground Truth", "LLM Generated"), output_html=None):
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("Plotly not available for HTML output.")
            return None
        
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
            print(f"Saved HTML: {output_html}")
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

    video_count = 0
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
            # CRITICAL FIX: lock_trans=True to prevent jerking/instability in visualization
            verts_gt, faces = params_to_vertices(params_gt, smplx_model, data_level=data_level, lock_trans=True)
            verts_gen, _ = params_to_vertices(params_gen, smplx_model, data_level=data_level, lock_trans=True)
            
            # Print token comparison to console
            gt_token_strs = " ".join([f"<M{t}>" for t in tokens_gt])
            gen_token_strs = " ".join([f"<M{t}>" for t in tokens_gen])
            print(f"  Sentence: {repr(str(word)[:80])}")
            print(f"  GT  ({len(tokens_gt)} tokens): <M_START> {gt_token_strs} <M_END>")
            print(f"  GEN ({len(tokens_gen)} tokens): <M_START> {gen_token_strs} <M_END>")
            
            # Token-level match indicator
            matches = sum(1 for g, r in zip(tokens_gen, tokens_gt) if g == r)
            max_len = max(len(tokens_gt), len(tokens_gen), 1)
            accuracy = matches / max_len * 100
            print(f"  Token match: {matches}/{max_len} ({accuracy:.1f}%)")
            
            # Sanitize for filesystem safety
            safe_word = re.sub(r'[^A-Za-z0-9_-]+', '_', str(word))[:50]  # Truncate for sentences
            safe_pid = re.sub(r'[^A-Za-z0-9_-]+', '_', str(pid))
            # Use appropriate prefix based on content length (sentences are longer)
            is_sentence = len(str(word)) > 20 or ' ' in str(word)
            prefix = "sentence" if is_sentence else "word"
            
            if output_format == "video":
                # PyRender video output (default)
                output_video = os.path.join(output_dir, f"{prefix}_{safe_word}_{safe_pid}_comparison.mp4")
                print(f"\nRendering video: {prefix}_{safe_word}...")
                render_side_by_side_video(
                    verts_gt=verts_gt,
                    verts_gen=verts_gen,
                    faces=faces,
                    output_path=output_video,
                    labels=("Ground Truth", "LLM Generated"),
                    fps=15,
                    slowdown=2,
                    apply_rotation_fix=True,
                    trim_end_frames=True,
                    show_progress=True,
                    stabilize_motion=True
                )
                video_count += 1
            else:
                # HTML fallback
                output_html = os.path.join(output_dir, f"{prefix}_{safe_word}_{safe_pid}_side_by_side.html")
                animate_side_by_side_html(
                    verts_left=verts_gt,
                    faces=faces,
                    verts_right=verts_gen,
                    fps=20,
                    titles=("Ground Truth", "LLM Generated"),
                    output_html=output_html
                )
                video_count += 1
                
        except Exception as exc:
            print(f"Error creating visualization for '{pair[0]}': {exc}")
    
    print(f"\nSaved {video_count} visualization{'s' if video_count != 1 else ''} to {output_dir}")

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

        print("\nDetailed Comparison ([OK] = Match, [Error] = Mismatch/Missing/Added):")
        
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
                diff_str += "[OK]".ljust(max_tok_len) + " "
            else:
                diff_str += "[Error]".ljust(max_tok_len) + " "
                
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
    # For Stage-3 "classic" metrics (Diversity/MIM) computed on VQ-VAE encoder features
    gt_feat_list: List[np.ndarray] = []
    gt_labels: List[str] = []
    gen_feat_list: List[np.ndarray] = []
    gen_labels: List[str] = []
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
        for f in gt_feats:
            gt_feat_list.append(np.asarray(f, dtype=np.float32))
            gt_labels.append(word)

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
        for f in gen_feats:
            gen_feat_list.append(np.asarray(f, dtype=np.float32))
            gen_labels.append(word)

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

    # Stage-3 "classic" trio (Diversity/MIM) on encoder features.
    # These aggregate over ALL valid generated samples across all words (i.e., mean over K).
    try:
        if len(gt_feat_list) >= 2:
            gt_all_for_div = np.stack(gt_feat_list, axis=0)
            diversity_times = min(200, max(4, gt_all_for_div.shape[0] - 1))
            diversity_gt = calculate_diversity_np(gt_all_for_div, diversity_times=diversity_times)
        else:
            diversity_gt = float("nan")
    except Exception:
        diversity_gt = float("nan")

    try:
        if len(gen_feat_list) >= 2:
            gen_all_for_div = np.stack(gen_feat_list, axis=0)
            diversity_times = min(200, max(4, gen_all_for_div.shape[0] - 1))
            diversity_gen = calculate_diversity_np(gen_all_for_div, diversity_times=diversity_times)
        else:
            diversity_gen = float("nan")
    except Exception:
        diversity_gen = float("nan")

    try:
        if len(gt_feat_list) >= 2:
            gt_all_for_mim = np.stack(gt_feat_list, axis=0)
            gt_lbl_tensor = _to_label_tensor3(gt_all_for_mim, gt_labels)
            mim_times = min(20, max(3, gt_lbl_tensor.shape[1] - 1))
            mim_gt = calculate_multimodality_np(gt_lbl_tensor, multimodality_times=mim_times)
        else:
            mim_gt = float("nan")
    except Exception:
        mim_gt = float("nan")

    try:
        if len(gen_feat_list) >= 2:
            gen_all_for_mim = np.stack(gen_feat_list, axis=0)
            gen_lbl_tensor = _to_label_tensor3(gen_all_for_mim, gen_labels)
            mim_times = min(20, max(3, gen_lbl_tensor.shape[1] - 1))
            mim_gen = calculate_multimodality_np(gen_lbl_tensor, multimodality_times=mim_times)
        else:
            mim_gen = float("nan")
    except Exception:
        mim_gen = float("nan")

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
        "diversity": {
            "ground_truth": diversity_gt,
            "model": diversity_gen,
        },
        "multimodality": {
            "ground_truth": mim_gt,
            "model": mim_gen,
        },
        "exact_match_rate_mean": exact_match_rate_mean,
        "gen_unique_ratio_mean": gen_unique_ratio_mean,
        "ref_coverage_ratio_mean": ref_coverage_ratio_mean,
        "ref_diversity_feat_mean": ref_diversity_feat_mean,
        "gen_diversity_feat_mean": gen_diversity_feat_mean,
        "per_word": per_word,
        "pairs_closest": pairs_closest,
    }


# ======================================================================================
# Sentence-Level Evaluation (One-to-One Mapping)
# ======================================================================================

def evaluate_sentence_level_encoder_style(
    model,
    tokenizer,
    eval_data: List[Dict[str, Any]],
    device,
    *,
    vqvae_ckpt: Optional[str] = None,
    stats_path: Optional[str] = None,
    sample_limit: Optional[int] = None,
    seed: int = SEED,
    dtw_only: bool = False,
):
    """
    Sentence-level (one-to-one) evaluation using VQ-VAE encoder features.
    
    Unlike multi-reference word evaluation, each sentence has exactly ONE ground truth.
    
    Metrics:
      - avg_feat_dist: Mean L2 distance between GT and generated features
      - token_edit_distance: Mean Levenshtein distance between GT and generated token sequences
      - token_accuracy: Mean token-level accuracy
      - fid_global: FID over all GT feats vs all generated feats
      - dtw_jpe / dtw_pa_jpe: SOKE paper joint-position metrics (requires SMPL-X)
    
    Args:
      dtw_only: If True, skip LLM generation, FID, edit-distance and feature-distance.
                Only compute DTW-JPE / DTW-PA-JPE for each sentence pair. Much faster.
    
    Returns:
      - pairs_closest: List of (sentence_id, "sentence", gt_seq, gen_seq) for visualization
    """
    subset = eval_data[:sample_limit] if (isinstance(sample_limit, int) and sample_limit > 0) else eval_data
    
    if not subset:
        return {"error": "No sentence data provided.", "pairs_closest": [], "per_sentence": {}}
    
    # Load VQ-VAE helpers
    try:
        vq_model, mean, std, decode_tokens_to_params = _load_vqvae_helpers_for_metrics(
            device, vqvae_ckpt=vqvae_ckpt, stats_path=stats_path
        )
    except Exception as exc:
        return {"error": f"Could not set up VQ-VAE encoder: {exc}", "pairs_closest": [], "per_sentence": {}}
    
    # Deterministic RNG
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Results storage
    pairs_closest: List[Tuple[str, str, str, str]] = []
    per_sentence: Dict[str, Dict[str, Any]] = {}
    all_gt_feats = []
    all_gen_feats = []
    feat_distances = []
    edit_distances = []
    token_accuracies = []
    # DTW-JPE storage (SOKE paper metrics)
    dtw_jpe_body_list: List[float] = []
    dtw_jpe_hand_list: List[float] = []
    dtw_pa_jpe_body_list: List[float] = []
    dtw_pa_jpe_hand_list: List[float] = []

    # Lazy-load SMPL-X model once for DTW-JPE (graceful: None if unavailable)
    smplx_model_for_dtw = _get_smplx_model_for_metrics(str(device))

    if dtw_only:
        if smplx_model_for_dtw is None:
            print("  [DTW-only] SMPL-X model unavailable — cannot compute DTW. Aborting.")
            return {
                "source": "vqvae_encoder_sentence_level",
                "dtw_only": True,
                "error": "SMPL-X model unavailable",
                "dtw_jpe": {"body": float("nan"), "hand": float("nan")},
                "dtw_pa_jpe": {"body": float("nan"), "hand": float("nan")},
                "per_sentence": {},
                "pairs_closest": [],
            }
        print(f"  [DTW-only] Computing DTW-JPE on {len(subset)} sentences (skipping FID/edit-dist)...")
    
    # Get motion token info for generation
    mot_begin_id = tokenizer.convert_tokens_to_ids(M_START)
    mot_end_id = tokenizer.convert_tokens_to_ids(M_END)
    motion_token_ids = set()
    for tok_id in range(len(tokenizer)):
        tok_str = tokenizer.convert_ids_to_tokens(tok_id)
        if tok_str and tok_str.startswith("<M") and tok_str.endswith(">") and tok_str not in [M_START, M_END]:
            motion_token_ids.add(tok_id)
    
    print(f"  Evaluating {len(subset)} sentences...")

    # Load length predictor once (lazy, cached)
    len_predictor, len_encoder = _get_length_predictor(device)
    if len_predictor is not None:
        print("  [LengthPredictor] Using trained predictor for length conditioning.")
    else:
        print("  [LengthPredictor] No predictor available — will fallback to GT-length binning.")

    for i, sample in enumerate(subset):
        # Get sentence text
        sentence_full = sample.get("sentence_full") or sample.get("text") or sample.get("sentence", "")
        sentence_id = sample.get("word", sentence_full[:50])  # Use truncated as ID
        
        if not sentence_full:
            continue
        
        # Get GT sequence
        gt_seq = _wrap_gt_sequence_from_sample(sample)
        gt_tokens = _extract_ids_from_sequence(gt_seq)

        if dtw_only:
            # ---------------------------------------------------------------
            # DTW-only path: no LLM generation. We still need a gen_seq.
            # Use GT as both GT and GEN just to get the pipeline working, OR
            # generate one sequence per sentence. We do GENERATE here so the
            # DTW actually measures the model's error.
            # ---------------------------------------------------------------
            if not gt_tokens:
                continue
            try:
                if len_predictor is not None or True:  # always build prompt
                    len_token = _predict_len_token(
                        sentence_full, len_predictor, len_encoder, device,
                        fallback_len=len(gt_tokens),
                    )
                    template = random.choice(V2_EVAL_TEMPLATES)
                    instruction = template.format(text=sentence_full)
                    prompt = f"{instruction} (Length: {len_token})\nMotion: "
                else:
                    prompt = sentence_full

                gen_seq = _generate_single_sample_for_sentence(
                    model, tokenizer, prompt, mot_begin_id, mot_end_id, motion_token_ids, device
                )
                gen_tokens = _extract_ids_from_sequence(gen_seq)
            except Exception as e:
                print(f"  [DTW-only] Generation failed for sentence {i+1}: {e}")
                continue

            if not gen_tokens:
                continue

            # Compute DTW only
            dtw_metrics_sentence = {
                "dtw_jpe_body": float("nan"),
                "dtw_jpe_hand": float("nan"),
                "dtw_pa_jpe_body": float("nan"),
                "dtw_pa_jpe_hand": float("nan"),
            }
            try:
                gt_body, gt_hand = _tokens_to_smplx_joints(
                    gt_tokens, vq_model, mean, std, smplx_model_for_dtw, device, decode_tokens_to_params
                )
                gen_body, gen_hand = _tokens_to_smplx_joints(
                    gen_tokens, vq_model, mean, std, smplx_model_for_dtw, device, decode_tokens_to_params
                )
                if gt_body is None or gen_body is None:
                    print(f"  [DTW-JPE] Sentence {i+1}: joint extraction failed")
                else:
                    dtw_metrics_sentence = compute_dtw_jpe(gt_body, gt_hand, gen_body, gen_hand)
                    for key, lst in [
                        ("dtw_jpe_body", dtw_jpe_body_list),
                        ("dtw_jpe_hand", dtw_jpe_hand_list),
                        ("dtw_pa_jpe_body", dtw_pa_jpe_body_list),
                        ("dtw_pa_jpe_hand", dtw_pa_jpe_hand_list),
                    ]:
                        v = dtw_metrics_sentence.get(key, float("nan"))
                        if np.isfinite(v):
                            lst.append(v)
            except Exception as exc:
                print(f"  [DTW-JPE] Sentence {i+1} exception: {exc}")

            per_sentence[sentence_id] = {
                "gt_tokens": len(gt_tokens),
                "gen_tokens": len(gen_tokens),
                **dtw_metrics_sentence,
            }
            pairs_closest.append((sentence_id, "sentence", gt_seq, gen_seq))

            if (i + 1) % 5 == 0:
                print(f"    [DTW-only] Processed {i + 1}/{len(subset)} sentences...")
            continue  # skip the rest of the main loop body

        # ------------------------------------------------------------------
        # FULL METRICS PATH (default: dtw_only=False)
        # ------------------------------------------------------------------
        gt_feat = _sequence_to_encoder_feature(gt_seq, vq_model, mean, std, device, decode_tokens_to_params)
        
        if gt_feat is None:
            per_sentence[sentence_id] = {"error": "Could not extract GT features"}
            continue
        
        all_gt_feats.append(gt_feat)

        # --- Predict length token via trained predictor (realistic inference condition) ---
        # During real inference we don't know GT length, so we predict it from the sentence.
        # Fallback: if predictor unavailable, use GT token count (for debugging only).
        len_token = _predict_len_token(
            sentence_full,
            len_predictor,
            len_encoder,
            device,
            fallback_len=len(gt_tokens),  # only used when predictor is None
        )

        # Use a random training template (matches training distribution exactly)
        template = random.choice(V2_EVAL_TEMPLATES)
        instruction = template.format(text=sentence_full)
        prompt = f"{instruction} (Length: {len_token})\nMotion: "
        
        try:
            gen_seq = _generate_single_sample_for_sentence(
                model, tokenizer, prompt, mot_begin_id, mot_end_id, motion_token_ids, device
            )
            gen_tokens = _extract_ids_from_sequence(gen_seq)
            gen_feat = _sequence_to_encoder_feature(gen_seq, vq_model, mean, std, device, decode_tokens_to_params)
        except Exception as e:
            per_sentence[sentence_id] = {"error": f"Generation failed: {str(e)[:100]}"}
            continue
        
        if gen_feat is None or len(gen_tokens) == 0:
            per_sentence[sentence_id] = {
                "gt_tokens": len(gt_tokens),
                "gen_tokens": len(gen_tokens) if gen_tokens else 0,
                "error": "Could not extract generated features"
            }
            continue
        
        all_gen_feats.append(gen_feat)
        
        # Compute metrics for this sentence
        # 1. Feature distance (L2)
        feat_dist = float(np.linalg.norm(gt_feat - gen_feat))
        feat_distances.append(feat_dist)
        
        # 2. Token edit distance (normalized)
        edit_dist = Levenshtein.distance(gt_tokens, gen_tokens)
        max_len = max(len(gt_tokens), len(gen_tokens), 1)
        norm_edit_dist = edit_dist / max_len
        edit_distances.append(norm_edit_dist)
        
        # 3. Token accuracy
        matches = sum(1 for g, r in zip(gen_tokens, gt_tokens) if g == r)
        accuracy = matches / max_len if max_len > 0 else 0.0
        token_accuracies.append(accuracy)

        # 4. DTW-JPE / DTW-PA-JPE (SOKE paper metrics) — opt-in, requires SMPL-X
        dtw_metrics_sentence = {
            "dtw_jpe_body": float("nan"),
            "dtw_jpe_hand": float("nan"),
            "dtw_pa_jpe_body": float("nan"),
            "dtw_pa_jpe_hand": float("nan"),
        }
        if smplx_model_for_dtw is not None and gt_tokens and gen_tokens:
            try:
                gt_body, gt_hand = _tokens_to_smplx_joints(
                    gt_tokens, vq_model, mean, std, smplx_model_for_dtw, device, decode_tokens_to_params
                )
                gen_body, gen_hand = _tokens_to_smplx_joints(
                    gen_tokens, vq_model, mean, std, smplx_model_for_dtw, device, decode_tokens_to_params
                )
                if gt_body is None or gen_body is None:
                    print(f"  [DTW-JPE] Sentence {i+1}: joint extraction failed (gt_body={'None' if gt_body is None else 'OK'}, gen_body={'None' if gen_body is None else 'OK'})")
                else:
                    dtw_metrics_sentence = compute_dtw_jpe(gt_body, gt_hand, gen_body, gen_hand)
                    # Collect finite values for aggregation
                    for key, lst in [
                        ("dtw_jpe_body", dtw_jpe_body_list),
                        ("dtw_jpe_hand", dtw_jpe_hand_list),
                        ("dtw_pa_jpe_body", dtw_pa_jpe_body_list),
                        ("dtw_pa_jpe_hand", dtw_pa_jpe_hand_list),
                    ]:
                        v = dtw_metrics_sentence.get(key, float("nan"))
                        if np.isfinite(v):
                            lst.append(v)
            except Exception as dtw_exc:
                print(f"  [DTW-JPE] Sentence {i+1}: exception: {dtw_exc}")
        
        # Store per-sentence metrics
        per_sentence[sentence_id] = {
            "gt_tokens": len(gt_tokens),
            "gen_tokens": len(gen_tokens),
            "feat_dist": feat_dist,
            "edit_dist": edit_dist,
            "norm_edit_dist": norm_edit_dist,
            "token_accuracy": accuracy,
            "len_token_used": len_token,          # Which length bin was predicted
            "prompt_used": prompt,                 # Full prompt for debugging
            # DTW-JPE metrics (NaN if SMPL-X unavailable)
            **dtw_metrics_sentence,
        }
        
        # Add to visualization pairs - include full token sequences for printout
        pairs_closest.append((sentence_id, "sentence", gt_seq, gen_seq))
        
        if (i + 1) % 5 == 0:
            print(f"    Processed {i + 1}/{len(subset)} sentences...")
    
    # Aggregate metrics
    avg_feat_dist = float(np.mean(feat_distances)) if feat_distances else float("nan")
    avg_edit_dist = float(np.mean(edit_distances)) if edit_distances else float("nan")
    avg_token_accuracy = float(np.mean(token_accuracies)) if token_accuracies else float("nan")

    # Aggregate DTW-JPE metrics
    def _nanmean_list(lst):
        return float(np.mean(lst)) if lst else float("nan")

    avg_dtw_jpe_body     = _nanmean_list(dtw_jpe_body_list)
    avg_dtw_jpe_hand     = _nanmean_list(dtw_jpe_hand_list)
    avg_dtw_pa_jpe_body  = _nanmean_list(dtw_pa_jpe_body_list)
    avg_dtw_pa_jpe_hand  = _nanmean_list(dtw_pa_jpe_hand_list)

    # Compute global FID (GT vs GEN) and GT self-FID (sanity baseline ≈ 0)
    fid_global = float("nan")
    fid_gt_self = float("nan")
    try:
        if len(all_gt_feats) >= 2 and len(all_gen_feats) >= 2:
            gt_all = np.stack(all_gt_feats, axis=0)
            gen_all = np.stack(all_gen_feats, axis=0)
            mu_g, cov_g = calculate_activation_statistics_np(gen_all)
            mu_r, cov_r = calculate_activation_statistics_np(gt_all)
            fid_global = float(calculate_frechet_distance_np(mu_r, cov_r, mu_g, cov_g))

            # GT self-FID: split GT in half and measure FID between halves.
            # Should approach 0 with enough samples. Good sanity check for the
            # feature space and scale of the fid_global number.
            n = gt_all.shape[0]
            if n >= 4:
                half = n // 2
                mu_r1, cov_r1 = calculate_activation_statistics_np(gt_all[:half])
                mu_r2, cov_r2 = calculate_activation_statistics_np(gt_all[half:])
                fid_gt_self = float(calculate_frechet_distance_np(mu_r1, cov_r1, mu_r2, cov_r2))
    except Exception:
        pass

    print(f"  Completed: {len(pairs_closest)} successful generations out of {len(subset)} sentences")
    if dtw_only:
        print(f"  [DTW-only] avg DTW-JPE body={avg_dtw_jpe_body:.4f}, hand={avg_dtw_jpe_hand:.4f}")
        print(f"  [DTW-only] avg DTW-PA-JPE body={avg_dtw_pa_jpe_body:.4f}, hand={avg_dtw_pa_jpe_hand:.4f}")
    else:
        if not np.isnan(fid_global):
            print(f"  FID (encoder feats): GT-vs-GEN = {fid_global:.4f} | GT-self = {fid_gt_self:.4f}")
        if not np.isnan(avg_dtw_jpe_body):
            print(f"  DTW-JPE:    body = {avg_dtw_jpe_body:.4f} | hand = {avg_dtw_jpe_hand:.4f}")
            print(f"  DTW-PA-JPE: body = {avg_dtw_pa_jpe_body:.4f} | hand = {avg_dtw_pa_jpe_hand:.4f}")

    return {
        "source": "vqvae_encoder_sentence_level",
        "dtw_only": dtw_only,
        "num_sentences": len(subset),
        "num_successful": len(pairs_closest),
        "avg_feat_dist": avg_feat_dist,
        "avg_edit_dist": avg_edit_dist,
        "avg_token_accuracy": avg_token_accuracy,
        # --- FID (VQ-VAE encoder feature space) ---
        "fid_global": fid_global,
        "fid_gt_self": fid_gt_self,
        "fid_note": (
            "FID computed on VQ-VAE pre-quant encoder features (512-dim, L2-normalised). "
            "fid_global = GT vs GEN (lower is better). "
            "fid_gt_self = GT vs GT-split (sanity baseline, should be near 0 with many samples)."
            + (" [skipped in dtw_only mode]" if dtw_only else "")
        ),
        # --- DTW-JPE / DTW-PA-JPE (SOKE paper metrics) ---
        "dtw_jpe": {
            "body":    avg_dtw_jpe_body,
            "hand":    avg_dtw_jpe_hand,
            "note": (
                "DTW-JPE: mean per-frame per-joint L2 on root-relative SMPL-X joints. "
                "Units: cm (joints ×100 from SMPL-X metres). "
                "Body = 11 upper-body joints, Hand = 30 hand joints. Lower is better. "
                "Directly comparable to SOKE/NSA paper Table 1."
            ),
        },
        "dtw_pa_jpe": {
            "body":    avg_dtw_pa_jpe_body,
            "hand":    avg_dtw_pa_jpe_hand,
            "note": (
                "DTW-PA-JPE: DTW on rotation-only Procrustes-aligned sequences. "
                "Root-relative (pelvis=0) so NO translation centering — only rotation. "
                "Rotation computed on resampled pair, applied to original-length gen, then DTW. "
                "Units: cm. Should be ≤ DTW-JPE. Lower is better."
            ),
        },
        "per_sentence": per_sentence,
        "pairs_closest": pairs_closest,
    }


def _generate_single_sample_for_sentence(
    model, tokenizer, prompt: str, mot_begin_id: int, mot_end_id: int, 
    motion_token_ids: set, device, max_new_tokens: int = 200,
    use_greedy: bool = True
) -> str:
    """
    Generate a single motion sequence for a sentence prompt.
    
    Uses greedy decoding by default (consistent with best evaluation results
    shown in training logs). Set use_greedy=False to use sampling.
    """
    model.eval()
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        if use_greedy:
            # Greedy decoding - most faithful to training evaluation (do_sample=False)
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,               # Greedy
                repetition_penalty=1.2,        # Prevent repetition loops
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=mot_end_id,
            )
        else:
            # Sampling mode (for diversity)
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=INFERENCE_TEMPERATURE,
                top_k=INFERENCE_TOP_K,
                repetition_penalty=INFERENCE_REPETITION_PENALTY,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=mot_end_id,
            )
    
    # Extract generated tokens (after the prompt)
    generated_ids = outputs[0][input_ids.shape[1]:].tolist()
    
    # Build sequence string
    seq_parts = []
    started = False
    for tok_id in generated_ids:
        if tok_id == mot_begin_id:
            started = True
            seq_parts.append(M_START)
        elif tok_id == mot_end_id:
            seq_parts.append(M_END)
            break
        elif started and tok_id in motion_token_ids:
            tok_str = tokenizer.convert_ids_to_tokens(tok_id)
            seq_parts.append(tok_str)
    
    # If no M_START found, try to extract any motion tokens
    if not seq_parts:
        for tok_id in generated_ids:
            if tok_id in motion_token_ids:
                tok_str = tokenizer.convert_ids_to_tokens(tok_id)
                seq_parts.append(tok_str)
    
    return " ".join(seq_parts) if seq_parts else ""
