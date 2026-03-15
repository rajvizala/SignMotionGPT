"""
Data loading utilities shared across experiments.

Provides common functions for loading the word-level (ASL Citizen) and
sentence-level (How2Sign) datasets in the project's JSON format.
"""

import json
import os
import random
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple
from torch.utils.data import Dataset


def load_json_data(json_path: str) -> List[Dict[str, Any]]:
    """Load the full dataset JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("samples", data.get("data", []))
    return data


def split_by_type(
    data: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split dataset into word-level and sentence-level subsets."""
    words, sentences = [], []
    for item in data:
        t = item.get("type", "").lower()
        if t == "sentence":
            sentences.append(item)
        else:
            words.append(item)
    return words, sentences


def extract_motion_vocab(data: List[Dict[str, Any]]) -> List[str]:
    """Collect sorted unique motion tokens (<M0>, <M1>, ...) from data."""
    tokens = set()
    for item in data:
        for tok in item.get("motion_tokens", "").split():
            if tok.strip():
                tokens.add(f"<M{tok.strip()}>")
    return sorted(tokens)


def train_val_split(
    data: List[Dict[str, Any]],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministic train / validation split."""
    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    n_val = max(1, int(len(data) * val_ratio))
    val_indices = set(indices[:n_val])
    train = [data[i] for i in range(len(data)) if i not in val_indices]
    val = [data[i] for i in val_indices]
    return train, val


class MotionSequenceDataset(Dataset):
    """
    Generic dataset that yields (motion_features, metadata) tuples from
    pre-extracted NPZ or JSON-encoded motion token sequences.

    This is used by VQ-VAE experiments that need raw motion features.
    For LLM experiments, use the specialized datasets in each experiment.
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        max_length: int = 512,
        pad_value: float = 0.0,
    ):
        self.samples = []
        self.max_length = max_length
        self.pad_value = pad_value
        for item in data:
            tokens_str = item.get("motion_tokens", "")
            if not tokens_str.strip():
                continue
            self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
