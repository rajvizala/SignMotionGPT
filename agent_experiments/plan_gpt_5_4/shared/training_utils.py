from __future__ import annotations

import copy
import os
from dataclasses import asdict
from typing import Any, Dict, Optional

import torch


class EarlyStopper:
    def __init__(self, patience: int, mode: str = "min") -> None:
        self.patience = max(1, int(patience))
        self.mode = mode
        self.best_value: Optional[float] = None
        self.num_bad_epochs = 0

    def step(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
            self.num_bad_epochs = 0
            return False
        improved = value < self.best_value if self.mode == "min" else value > self.best_value
        if improved:
            self.best_value = value
            self.num_bad_epochs = 0
            return False
        self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_checkpoint(path: str, payload: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    torch.save(payload, path)


def load_checkpoint_if_exists(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def maybe_save_best(
    path: str,
    metric_value: float,
    best_value: Optional[float],
    payload: Dict[str, Any],
) -> float:
    if best_value is None or metric_value < best_value:
        save_checkpoint(path, payload)
        return metric_value
    return best_value


def state_dict_to_cpu(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in state_dict.items()}


def model_payload(
    model,
    optimizer,
    epoch: int,
    config,
    metrics: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "epoch": epoch,
        "model_state_dict": state_dict_to_cpu(model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "config": asdict(config) if hasattr(config, "__dataclass_fields__") else config,
        "metrics": metrics,
    }
    if extra:
        payload.update(extra)
    return payload


def diagnostic_warnings(metrics: Dict[str, Any], prefix: str = "") -> None:
    tag = f"[{prefix}] " if prefix else ""
    vq_loss = metrics.get("vq_loss")
    if vq_loss is not None and float(vq_loss) > 3.0:
        print(f"{tag}warning: VQ loss exceeded 3.0; suggest stopping and inspecting code collapse.")
    coverage = metrics.get("codebook_coverage_pct")
    if coverage is not None and float(coverage) < 60.0:
        print(f"{tag}warning: validation codebook coverage dropped below 60%; collapse likely.")
    hand_mse = metrics.get("hand_mse")
    body_mse = metrics.get("body_mse")
    prev_hand = metrics.get("prev_hand_mse")
    prev_body = metrics.get("prev_body_mse")
    if None not in (hand_mse, body_mse, prev_hand, prev_body):
        if float(hand_mse) >= float(prev_hand) and float(body_mse) < float(prev_body):
            print(f"{tag}diagnostic: hand MSE is not improving while body MSE is.")
