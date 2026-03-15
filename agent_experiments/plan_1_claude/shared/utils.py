"""
Shared utilities for agent experiments.

Provides common data loading, metric computation, and helper functions
used across all experiments.
"""

import os
import sys
import json
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# SMPL-X parameter layout (total 182 dims)
SMPLX_PARAM_DIMS = {
    'shape': (0, 10),
    'body_pose': (10, 73),
    'lhand_pose': (73, 118),
    'rhand_pose': (118, 163),
    'jaw_pose': (163, 166),
    'expression': (166, 176),
    'root_pose': (176, 179),
    'cam_trans': (179, 182),
}

SMPLX_TOTAL_DIM = 182

# Body part groups for Part-Aware experiments
BODY_PART_GROUPS = {
    'upper_body': ['body_pose', 'root_pose', 'cam_trans', 'shape'],
    'left_hand': ['lhand_pose'],
    'right_hand': ['rhand_pose'],
    'face': ['jaw_pose', 'expression'],
}

def get_part_indices(part_name: str) -> Tuple[int, int]:
    """Get start and end indices for a body part group."""
    if part_name in SMPLX_PARAM_DIMS:
        return SMPLX_PARAM_DIMS[part_name]
    
    if part_name in BODY_PART_GROUPS:
        indices = []
        for param in BODY_PART_GROUPS[part_name]:
            start, end = SMPLX_PARAM_DIMS[param]
            indices.extend(range(start, end))
        return sorted(indices)
    
    raise ValueError(f"Unknown part name: {part_name}")


def get_part_dim(part_name: str) -> int:
    """Get dimensionality for a body part group."""
    indices = get_part_indices(part_name)
    if isinstance(indices, tuple):
        return indices[1] - indices[0]
    return len(indices)


def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset_json(json_path: str) -> List[Dict[str, Any]]:
    """Load dataset from JSON file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset not found at: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        data = data.get("samples", data.get("data", []))
    
    return data


def filter_by_type(data: List[Dict[str, Any]], data_type: str = "sentence") -> List[Dict[str, Any]]:
    """Filter dataset by type (sentence, word, etc.)."""
    return [item for item in data if item.get("type", "").lower() == data_type]


def compute_reconstruction_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute reconstruction quality metrics between original and reconstructed motion.
    
    Args:
        original: (B, T, D) original motion features
        reconstructed: (B, T, D) reconstructed motion features
    
    Returns:
        Dictionary with per-param and overall metrics
    """
    metrics = {}
    
    # Overall L1 and L2
    metrics['l1_error'] = F.l1_loss(reconstructed, original).item()
    metrics['l2_error'] = F.mse_loss(reconstructed, original).item()
    
    # Per-parameter group errors
    for param_name, (start, end) in SMPLX_PARAM_DIMS.items():
        orig_param = original[:, :, start:end]
        recon_param = reconstructed[:, :, start:end]
        metrics[f'l1_{param_name}'] = F.l1_loss(recon_param, orig_param).item()
        metrics[f'l2_{param_name}'] = F.mse_loss(recon_param, orig_param).item()
    
    return metrics


def compute_codebook_utilization(
    code_indices: torch.Tensor,
    codebook_size: int,
) -> Dict[str, float]:
    """
    Compute codebook utilization metrics.
    
    Args:
        code_indices: (N,) tensor of codebook indices used
        codebook_size: Total number of codebook entries
    
    Returns:
        Dictionary with utilization metrics
    """
    unique_codes = torch.unique(code_indices)
    usage_counts = torch.bincount(code_indices.long(), minlength=codebook_size)
    
    total = code_indices.numel()
    probs = usage_counts.float() / total
    probs = probs[probs > 0]
    entropy = -torch.sum(probs * torch.log2(probs)).item()
    max_entropy = math.log2(codebook_size)
    
    return {
        'active_codes': len(unique_codes),
        'total_codes': codebook_size,
        'utilization_pct': len(unique_codes) / codebook_size * 100,
        'entropy': entropy,
        'max_entropy': max_entropy,
        'normalized_entropy': entropy / max_entropy if max_entropy > 0 else 0,
        'perplexity': 2 ** entropy,
    }


def compute_token_level_metrics(
    predicted_tokens: List[List[int]],
    ground_truth_tokens: List[List[int]],
) -> Dict[str, float]:
    """
    Compute token-level generation metrics.
    
    Args:
        predicted_tokens: List of predicted token sequences
        ground_truth_tokens: List of ground truth token sequences
    """
    from rapidfuzz.distance import Levenshtein
    
    edit_distances = []
    length_ratios = []
    exact_matches = 0
    
    for pred, gt in zip(predicted_tokens, ground_truth_tokens):
        ed = Levenshtein.distance(pred, gt)
        norm_ed = ed / max(len(gt), 1)
        edit_distances.append(norm_ed)
        
        lr = len(pred) / max(len(gt), 1)
        length_ratios.append(lr)
        
        if pred == gt:
            exact_matches += 1
    
    n = len(predicted_tokens)
    return {
        'normalized_edit_distance': np.mean(edit_distances) if edit_distances else 0,
        'edit_distance_std': np.std(edit_distances) if edit_distances else 0,
        'mean_length_ratio': np.mean(length_ratios) if length_ratios else 0,
        'exact_match_rate': exact_matches / n if n > 0 else 0,
        'num_samples': n,
    }


class EarlyStopping:
    """Early stopping with patience and delta threshold."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


def print_experiment_banner(experiment_name: str, description: str = ""):
    """Print a formatted experiment banner."""
    width = 80
    print("\n" + "=" * width)
    print(f"  EXPERIMENT: {experiment_name}")
    if description:
        print(f"  {description}")
    print("=" * width + "\n")


def save_experiment_results(
    output_dir: str,
    experiment_name: str,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    notes: str = "",
):
    """Save experiment configuration and results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'experiment': experiment_name,
        'config': config,
        'metrics': metrics,
        'notes': notes,
    }
    
    output_path = os.path.join(output_dir, f"{experiment_name}_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"Results saved to: {output_path}")
    return output_path
