"""
Evaluation metrics for motion generation (token-level proxies)
"""
import random
from typing import List, Tuple, Dict
from rapidfuzz.distance import Levenshtein
from collections import defaultdict
import numpy as np
from data import motion_specials_to_ids
from generate import generate_t2m
from config import SEED

random.seed(SEED)


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



