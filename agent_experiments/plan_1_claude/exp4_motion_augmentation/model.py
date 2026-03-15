"""
Experiment 4 -- Motion-Level Data Augmentation Pipeline

REFERENCES
----------
[1] EnsAug: "Augmentation-Driven Ensembles for Human Motion Sequence
    Analysis", arXiv:2603.06661, 2025.
    Key idea: diverse geometric augmentations foster model diversity.

[2] Cross-modality Data Augmentation (XmDA) for sign language translation,
    EMNLP 2023.
    Key idea: mix-up between modalities bridges domain gaps.

[3] "Using Sign Language Production as Data Augmentation to enhance Sign
    Language Translation", arXiv:2506.09643, 2025.
    Key idea: synthetic sign generation as data augmentation.

[4] MoMask: "Generative Masked Modeling of 3D Human Motions", CVPR 2024.
    Key idea: masked modeling forces bidirectional context learning.

[5] Perlin-noise label smoothing for motion capture (arXiv:2511.22288, 2025).
    Key idea: augmentation must preserve temporal smoothness and joint
    correlations.

CORE IDEA
---------
The current pipeline has a limited effective training set: How2Sign has
only ~2K unique sentence-level clips.  The LLM memorises these sequences
because it sees the same token patterns repeatedly for 40 epochs.

We introduce a comprehensive **motion token augmentation pipeline** that
operates at TWO levels:

(A) **VQ-VAE Feature-Space Augmentation** (applied during VQ-VAE training):
    - Temporal warping: stretch/compress sequences by random factors
    - Gaussian noise injection in feature space
    - Part-specific jittering: add different noise to hands vs body
    - Mirror augmentation: swap left/right hand features

(B) **Token-Level Augmentation** (applied during LLM training):
    - Token dropout: randomly mask motion tokens with <M_MASK>
    - Token substitution: replace tokens with codebook-nearest neighbours
    - Sequence cropping: use sub-sequences as training data
    - Token reordering within small windows (local permutation)

WHY THIS SHOULD HELP GENERALIZATION
------------------------------------
1. Feature-space augmentation creates *new* motion variants that get
   encoded to *new* token patterns, increasing codebook coverage.
2. Token dropout forces the LLM to reconstruct missing tokens from context,
   learning temporal dependencies rather than memorising sequences.
3. Nearest-neighbour substitution teaches the LLM that semantically similar
   tokens are interchangeable, directly fighting overfitting to exact codes.
4. Sequence cropping creates compositional sub-units that can be recombined,
   enabling the model to handle unseen sentence structures.
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


# =========================================================================
# (A) VQ-VAE Feature-Space Augmentations
# =========================================================================

class FeatureSpaceAugmentor:
    """
    Augmentation pipeline for raw SMPL-X features before VQ-VAE encoding.
    Each augmentation is applied independently with a configurable probability.
    """

    def __init__(
        self,
        temporal_warp_prob: float = 0.3,
        temporal_warp_range: Tuple[float, float] = (0.8, 1.2),
        noise_prob: float = 0.5,
        noise_std: float = 0.02,
        hand_jitter_prob: float = 0.4,
        hand_jitter_std: float = 0.03,
        mirror_prob: float = 0.3,
    ):
        self.temporal_warp_prob = temporal_warp_prob
        self.temporal_warp_range = temporal_warp_range
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.hand_jitter_prob = hand_jitter_prob
        self.hand_jitter_std = hand_jitter_std
        self.mirror_prob = mirror_prob

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: (T, 182) single motion sequence
        Returns: augmented (T', 182) sequence
        """
        x = features.clone()

        if random.random() < self.temporal_warp_prob:
            x = self._temporal_warp(x)

        if random.random() < self.noise_prob:
            x = self._add_noise(x)

        if random.random() < self.hand_jitter_prob:
            x = self._hand_jitter(x)

        if random.random() < self.mirror_prob:
            x = self._mirror_hands(x)

        return x

    def _temporal_warp(self, x: torch.Tensor) -> torch.Tensor:
        """Resample the temporal dimension by a random factor."""
        T, C = x.shape
        factor = random.uniform(*self.temporal_warp_range)
        new_T = max(8, int(T * factor))
        x_t = x.unsqueeze(0).permute(0, 2, 1)  # (1, C, T)
        x_resampled = torch.nn.functional.interpolate(
            x_t, size=new_T, mode='linear', align_corners=False
        )
        return x_resampled.permute(0, 2, 1).squeeze(0)  # (T', C)

    def _add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to all features."""
        return x + torch.randn_like(x) * self.noise_std

    def _hand_jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Add extra noise specifically to hand pose dimensions."""
        noise = torch.zeros_like(x)
        noise[:, 73:163] = torch.randn(x.shape[0], 90) * self.hand_jitter_std
        return x + noise

    def _mirror_hands(self, x: torch.Tensor) -> torch.Tensor:
        """Swap left and right hand pose parameters."""
        x_mirrored = x.clone()
        lhand = x[:, 73:118].clone()
        rhand = x[:, 118:163].clone()
        x_mirrored[:, 73:118] = rhand
        x_mirrored[:, 118:163] = lhand
        return x_mirrored


# =========================================================================
# (B) Token-Level Augmentations for LLM Training
# =========================================================================

class TokenAugmentor:
    """
    Augmentation pipeline for motion token sequences during LLM training.

    Operates on the string representation of motion tokens (e.g. "<M42> <M18> ...")
    before final tokenisation by the LLM tokenizer.
    """

    def __init__(
        self,
        codebook_size: int = 512,
        dropout_prob: float = 0.15,
        substitute_prob: float = 0.1,
        crop_prob: float = 0.2,
        crop_min_ratio: float = 0.5,
        local_permute_prob: float = 0.1,
        local_permute_window: int = 3,
        codebook_distances: Optional[torch.Tensor] = None,
    ):
        """
        codebook_distances: (K, K) pairwise L2 distances between codes.
            If provided, substitution picks from the nearest neighbours.
            If None, substitution picks a random code.
        """
        self.codebook_size = codebook_size
        self.dropout_prob = dropout_prob
        self.substitute_prob = substitute_prob
        self.crop_prob = crop_prob
        self.crop_min_ratio = crop_min_ratio
        self.local_permute_prob = local_permute_prob
        self.local_permute_window = local_permute_window

        self.nearest_neighbours = None
        if codebook_distances is not None:
            _, nn_idx = codebook_distances.topk(10, largest=False, dim=-1)
            self.nearest_neighbours = nn_idx[:, 1:]  # exclude self

    def __call__(self, token_ids: List[int]) -> List[int]:
        """
        token_ids: list of integer code indices (0-511)
        Returns: augmented list
        """
        tokens = list(token_ids)

        if random.random() < self.crop_prob:
            tokens = self._crop(tokens)

        if random.random() < self.local_permute_prob:
            tokens = self._local_permute(tokens)

        tokens = self._dropout_and_substitute(tokens)
        return tokens

    def _crop(self, tokens: List[int]) -> List[int]:
        """Take a random contiguous sub-sequence."""
        if len(tokens) < 4:
            return tokens
        min_len = max(2, int(len(tokens) * self.crop_min_ratio))
        crop_len = random.randint(min_len, len(tokens))
        start = random.randint(0, len(tokens) - crop_len)
        return tokens[start : start + crop_len]

    def _local_permute(self, tokens: List[int]) -> List[int]:
        """Permute tokens within small local windows."""
        result = list(tokens)
        w = self.local_permute_window
        for start in range(0, len(result) - w + 1, w):
            window = result[start : start + w]
            random.shuffle(window)
            result[start : start + w] = window
        return result

    def _dropout_and_substitute(self, tokens: List[int]) -> List[int]:
        """Apply per-token dropout or nearest-neighbour substitution."""
        result = []
        for t in tokens:
            r = random.random()
            if r < self.dropout_prob:
                continue  # drop this token
            elif r < self.dropout_prob + self.substitute_prob:
                result.append(self._substitute(t))
            else:
                result.append(t)
        return result if result else tokens[:1]  # never return empty

    def _substitute(self, code_id: int) -> int:
        """Replace with a codebook-nearest neighbour or random code."""
        if self.nearest_neighbours is not None and code_id < len(self.nearest_neighbours):
            nn = self.nearest_neighbours[code_id]
            return nn[random.randint(0, len(nn) - 1)].item()
        return random.randint(0, self.codebook_size - 1)


# =========================================================================
# Augmented Dataset Wrapper
# =========================================================================

class AugmentedMotionDataset(torch.utils.data.Dataset):
    """
    Wraps a sentence-level dataset and applies token augmentation on-the-fly.

    For each sample, it creates the original + N augmented variants per epoch.
    """

    def __init__(
        self,
        base_items: List[Dict],
        token_augmentor: TokenAugmentor,
        augmentations_per_sample: int = 2,
        m_start: str = "<M_START>",
        m_end: str = "<M_END>",
    ):
        self.base_items = base_items
        self.augmentor = token_augmentor
        self.n_aug = augmentations_per_sample
        self.m_start = m_start
        self.m_end = m_end

    def __len__(self):
        return len(self.base_items) * (1 + self.n_aug)

    def __getitem__(self, idx):
        base_idx = idx % len(self.base_items)
        is_augmented = idx >= len(self.base_items)

        item = self.base_items[base_idx]

        if not is_augmented:
            return item

        tokens_str = item.get("motion_tokens", "")
        token_ids = [int(t) for t in tokens_str.split() if t.strip()]

        aug_ids = self.augmentor(token_ids)

        wrapped = " ".join(f"<M{t}>" for t in aug_ids)
        target = f"{self.m_start} {wrapped} {self.m_end}"

        aug_item = dict(item)
        aug_item["full_text"] = item["prompt"] + target
        aug_item["motion_tokens"] = " ".join(str(t) for t in aug_ids)
        aug_item["is_augmented"] = True
        return aug_item
