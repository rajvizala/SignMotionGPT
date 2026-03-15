from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class DataConfig:
    train_json: str
    dev_json: Optional[str] = None
    test_json: Optional[str] = None
    word_json: Optional[str] = None
    max_seq_len: int = 384
    min_motion_tokens: int = 8
    max_motion_tokens: int = 256


@dataclass
class RetrievalConfig:
    enabled: bool = False
    max_words: int = 6
    variants_per_word: int = 1
    anchor_tokens_per_variant: int = 6
    include_motion_anchors: bool = True
    include_lengths: bool = True


@dataclass
class ObjectiveConfig:
    word_replay_weight: float = 0.0
    contrastive_weight: float = 0.0
    denoise_weight: float = 0.0
    denoise_mask_ratio: float = 0.25
    randomized_position: bool = False
    max_position_offset: int = 2048
    contrastive_temperature: float = 0.07


@dataclass
class TrainingConfig:
    model_name: str = "Qwen/Qwen3-0.6B"
    output_dir: str = "./agent_experiment_runs/default"
    epochs: int = 10
    batch_size: int = 4
    eval_batch_size: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    grad_accum: int = 1
    max_steps: Optional[int] = None
    log_every: int = 10
    save_every: int = 200
    seed: int = 42
    num_workers: int = 0


@dataclass
class ExperimentConfig:
    name: str
    description: str
    data: DataConfig
    retrieval: RetrievalConfig
    objectives: ObjectiveConfig
    training: TrainingConfig

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ExperimentConfig":
        return cls(
            name=payload["name"],
            description=payload.get("description", ""),
            data=DataConfig(**payload["data"]),
            retrieval=RetrievalConfig(**payload.get("retrieval", {})),
            objectives=ObjectiveConfig(**payload.get("objectives", {})),
            training=TrainingConfig(**payload.get("training", {})),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def ensure_output_dir(self) -> Path:
        output_dir = Path(self.training.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def save_copy(self, output_dir: str | Path) -> Path:
        output_path = Path(output_dir) / "resolved_config.json"
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return output_path
