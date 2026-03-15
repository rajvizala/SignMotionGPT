"""
Base configuration shared across all experiments.

All experiments inherit from ExperimentConfig and override only the fields
they need.  The data paths, SMPL-X feature layout, and codebook geometry
are kept in one place so that every experiment speaks the same "language".
"""

import os
import torch
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExperimentConfig:
    # -- Reproducibility ---------------------------------------------------
    seed: int = 42

    # -- Device ------------------------------------------------------------
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # -- SMPL-X Feature Layout (182-dim) -----------------------------------
    # Order: shape(10), body_pose(63), lhand_pose(45), rhand_pose(45),
    #        jaw_pose(3), expression(10), root_pose(3), cam_trans(3)
    smplx_dim: int = 182
    smplx_param_dims: List[int] = field(
        default_factory=lambda: [10, 63, 45, 45, 3, 10, 3, 3]
    )
    smplx_param_names: List[str] = field(
        default_factory=lambda: [
            "shape", "body_pose", "lhand_pose", "rhand_pose",
            "jaw_pose", "expression", "root_pose", "cam_trans",
        ]
    )

    # -- Base VQ-VAE geometry (current baseline) ---------------------------
    codebook_size: int = 512
    code_dim: int = 512
    vqvae_down_t: int = 3
    vqvae_stride_t: int = 2
    vqvae_width: int = 512
    vqvae_depth: int = 3
    vqvae_dilation_growth_rate: int = 3

    # -- LLM ---------------------------------------------------------------
    llm_model_name: str = "Qwen/Qwen3-0.6B"
    max_seq_len: int = 256

    # -- Paths (override via env or constructor) ---------------------------
    data_json_path: str = os.environ.get(
        "DATA_JSON_PATH", "./data/motion_llm_dataset.json"
    )
    sentence_data_json_path: str = os.environ.get(
        "SENTENCE_DATA_JSON_PATH", ""
    )
    vqvae_checkpoint_path: str = os.environ.get("VQVAE_CHECKPOINT", "")
    output_dir: str = "./agent_experiments/outputs"

    # -- Training defaults -------------------------------------------------
    batch_size: int = 16
    learning_rate: float = 2e-4
    num_epochs: int = 100
    grad_accum_steps: int = 1
    warmup_ratio: float = 0.05

    # -- Loss weights (for VQ-VAE recon loss) ------------------------------
    pose_weight: float = 10.0
    shape_weight: float = 5.0
    expression_weight: float = 8.0

    def get_loss_weights(self) -> torch.Tensor:
        """Build per-dimension loss weight tensor matching SMPL-X layout."""
        import numpy as np
        weights = torch.ones(self.smplx_dim)
        starts = list(np.cumsum([0] + self.smplx_param_dims[:-1]))
        weight_map = {
            "body_pose": self.pose_weight,
            "lhand_pose": self.pose_weight,
            "rhand_pose": self.pose_weight,
            "shape": self.shape_weight,
            "expression": self.expression_weight,
        }
        for name, start, dim in zip(
            self.smplx_param_names, starts, self.smplx_param_dims
        ):
            if name in weight_map:
                weights[start : start + dim] = weight_map[name]
        return weights
