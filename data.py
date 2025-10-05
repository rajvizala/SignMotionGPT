"""
Dataset loading and vocabulary building utilities
"""
import json
import random
from typing import List, Dict, Tuple
from collections import defaultdict
from datasets import Dataset
from config import SEED, MAX_EVAL_SAMPLES


def load_dataset(data_path: str) -> Dataset:
    """Load dataset from JSON file"""
    with open(data_path, "r") as f:
        data = json.load(f)
    return Dataset.from_list(data)


def build_motion_vocab(dataset: Dataset) -> Tuple[int, int]:
    """
    Build motion vocabulary by finding max token ID
    Returns: (codebook_size, max_token_id)
    """
    def max_token_in_example(ex):
        return max(int(x) for x in ex["motion_tokens"].split())
    
    global_max_id = 0
    for ex in dataset:
        global_max_id = max(global_max_id, max_token_in_example(ex))
    
    codebook_size = global_max_id + 1
    print(f"Max motion token id found: {global_max_id}")
    print(f"Codebook size: {codebook_size}")
    
    return codebook_size, global_max_id


def ids_to_motion_specials(s: str) -> str:
    """Convert space-separated IDs to motion special tokens"""
    return " ".join(f"<motion_{x}>" for x in s.split())


def motion_specials_to_ids(s: str) -> List[int]:
    """Extract motion IDs from special tokens"""
    toks = s.strip().split()
    ids = []
    for t in toks:
        if t.startswith("<motion_"):
            try:
                ids.append(int(t[8:-1]))
            except:
                pass
    return ids


def compute_length_stats(dataset: Dataset) -> Tuple[Dict[str, Dict[str, int]], int]:
    """
    Compute motion length statistics per prompt
    Returns: (stats_by_text, global_median_length)
    """
    by_text = defaultdict(list)
    for ex in dataset:
        by_text[ex["text_query"]].append(len(ex["motion_tokens"].split()))
    
    stats = {}
    all_lens = []
    for k, arr in by_text.items():
        arr_sorted = sorted(arr)
        n = len(arr_sorted)
        median = arr_sorted[n//2] if n % 2 == 1 else (arr_sorted[n//2 - 1] + arr_sorted[n//2]) // 2
        stats[k] = {"median": median, "min": arr_sorted[0], "max": arr_sorted[-1]}
        all_lens.extend(arr_sorted)
    
    all_lens = sorted(all_lens) or [16]
    global_median = all_lens[len(all_lens)//2]
    
    return stats, global_median


def build_prompt_vocab(dataset: Dataset) -> Dict[str, List[int]]:
    """Build per-prompt vocabulary (whitelist of tokens that appear with each prompt)"""
    table = defaultdict(set)
    for ex in dataset:
        for x in ex["motion_tokens"].split():
            table[ex["text_query"]].add(int(x))
    return {k: sorted(v) for k, v in table.items()}


def make_splits(dataset: Dataset, mapper_fn, max_train_samples=None) -> Tuple[Dataset, Dataset]:
    """
    Create train/val splits and apply mapping function
    """
    split = dataset.train_test_split(test_size=0.01, seed=SEED)
    
    if max_train_samples is not None and max_train_samples < len(split["train"]):
        split["train"] = split["train"].select(range(max_train_samples))
    
    if MAX_EVAL_SAMPLES is not None and MAX_EVAL_SAMPLES < len(split["test"]):
        split["test"] = split["test"].select(range(MAX_EVAL_SAMPLES))
    
    train = split["train"].map(mapper_fn, remove_columns=split["train"].column_names, num_proc=2)
    val = split["test"].map(mapper_fn, remove_columns=split["test"].column_names, num_proc=2)
    
    return train, val


def check_has_participant_id(dataset: Dataset) -> bool:
    """Check if dataset has participant_id column"""
    return "participant_id" in dataset.column_names