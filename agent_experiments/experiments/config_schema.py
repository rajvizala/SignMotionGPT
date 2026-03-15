from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import copy
import tomllib


@dataclass
class DataConfig:
    sentence_json: str = ""
    word_json: str = ""
    output_dir: str = "./agent_experiments/outputs/default"
    max_seq_len: int = 256
    seed: int = 42
    split_seed: int = 123
    val_ratio: float = 0.1


@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen3-0.6B"
    vqvae_ckpt: str = ""
    use_bfloat16: bool = True
    grad_checkpointing: bool = True


@dataclass
class TrainingConfig:
    epochs: int = 40
    batch_size: int = 16
    lr: float = 3e-5
    weight_decay: float = 0.01
    grad_accum: int = 1
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.05
    num_workers: int = 4
    eval_every_steps: int = 200
    save_every_steps: int = 500
    max_steps: int = -1
    replay_ratio: float = 0.2


@dataclass
class RegularizationConfig:
    label_smoothing: float = 0.05
    rdrop_alpha: float = 1.0
    use_rdrop: bool = True
    use_sam: bool = True
    sam_rho: float = 0.05
    use_swa: bool = True
    swa_start_ratio: float = 0.7
    ewc_lambda: float = 0.0
    ewc_samples: int = 2048
    use_ewc: bool = False


@dataclass
class ValidationConfig:
    group_by: List[str] = field(default_factory=lambda: ["motion_length_bucket", "text_length_bucket"])
    early_stop_patience: int = 6
    select_by: str = "worst_group_dtw_jpe_proxy"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    notes: str = ""

    @staticmethod
    def from_toml(path: str, overrides: Optional[List[str]] = None) -> "ExperimentConfig":
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        cfg = ExperimentConfig()
        cfg.apply_dict(raw)
        if overrides:
            cfg.apply_overrides(overrides)
        cfg.validate()
        return cfg

    def apply_dict(self, raw: Dict[str, Any]) -> None:
        for section in ("data", "model", "training", "regularization", "validation"):
            values = raw.get(section, {})
            target = getattr(self, section)
            for key, value in values.items():
                if hasattr(target, key):
                    setattr(target, key, value)
        if "notes" in raw:
            self.notes = str(raw["notes"])

    def apply_overrides(self, overrides: List[str]) -> None:
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"Invalid override '{item}'. Expected dotted.path=value.")
            dotted, value = item.split("=", 1)
            parts = dotted.split(".")
            if len(parts) != 2:
                raise ValueError(f"Invalid override key '{dotted}'. Use section.key form.")
            section, key = parts
            if not hasattr(self, section):
                raise ValueError(f"Unknown section '{section}' in override '{item}'.")
            target = getattr(self, section)
            if not hasattr(target, key):
                raise ValueError(f"Unknown key '{section}.{key}' in override '{item}'.")
            current = getattr(target, key)
            parsed = _coerce_value(value, current)
            setattr(target, key, parsed)

    def validate(self) -> None:
        if not self.data.sentence_json:
            raise ValueError("data.sentence_json is required.")
        if self.training.batch_size <= 0:
            raise ValueError("training.batch_size must be > 0.")
        if self.training.epochs <= 0:
            raise ValueError("training.epochs must be > 0.")
        if not (0.0 <= self.training.replay_ratio <= 1.0):
            raise ValueError("training.replay_ratio must be within [0, 1].")
        if not (0.0 <= self.regularization.label_smoothing < 1.0):
            raise ValueError("regularization.label_smoothing must be in [0, 1).")
        if self.regularization.use_swa and not (0.0 < self.regularization.swa_start_ratio < 1.0):
            raise ValueError("regularization.swa_start_ratio must be in (0, 1).")
        if self.validation.early_stop_patience < 1:
            raise ValueError("validation.early_stop_patience must be >= 1.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def clone(self) -> "ExperimentConfig":
        return copy.deepcopy(self)


def _coerce_value(raw: str, template: Any) -> Any:
    if isinstance(template, bool):
        s = raw.strip().lower()
        if s in {"1", "true", "yes", "y"}:
            return True
        if s in {"0", "false", "no", "n"}:
            return False
        raise ValueError(f"Cannot parse bool from '{raw}'.")
    if isinstance(template, int) and not isinstance(template, bool):
        return int(raw)
    if isinstance(template, float):
        return float(raw)
    if isinstance(template, list):
        return [x.strip() for x in raw.split(",") if x.strip()]
    return raw

