"""
Evaluation metrics for motion generation
"""
import random
from typing import List, Tuple
from rapidfuzz.distance import Levenshtein
from collections import defaultdict
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
    """
    random.shuffle(sample_pairs)
    subset = sample_pairs[:min(n_eval, len(sample_pairs))]
    
    dists = []
    lens = []
    
    for text, ref_list in subset:
        # Generate
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
        
        # Extract motion span
        span = gen.split("<MOT_BEGIN>")[-1]
        span = span.split("<MOT_END>")[0]
        pred_ids = motion_specials_to_ids(span)
        
        # Compute distance to closest reference
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