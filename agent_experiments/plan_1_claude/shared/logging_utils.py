"""
Experiment logging utilities.

Provides a lightweight logger that writes both to stdout and to a structured
JSON-Lines file so that results can be parsed programmatically later.
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional


class ExperimentLogger:
    """Write structured training / evaluation logs to a JSONL file."""

    def __init__(self, log_dir: str, experiment_name: str):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"{experiment_name}_{ts}.jsonl")
        self.experiment_name = experiment_name
        self._start_time = time.time()

    def log(self, payload: Dict[str, Any], step: Optional[int] = None):
        record = {
            "experiment": self.experiment_name,
            "wall_time_s": round(time.time() - self._start_time, 2),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        if step is not None:
            record["step"] = step
        record.update(payload)
        line = json.dumps(record, default=str)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        _pretty = {k: v for k, v in payload.items() if k != "raw"}
        print(f"[{self.experiment_name}] step={step}  {_pretty}")

    def log_hyperparameters(self, hparams: Dict[str, Any]):
        self.log({"event": "hyperparameters", **hparams})

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
        lr: Optional[float] = None,
    ):
        payload: Dict[str, Any] = {
            "event": "epoch_end",
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
        }
        if val_loss is not None:
            payload["val_loss"] = round(val_loss, 6)
        if metrics:
            payload["metrics"] = {
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in metrics.items()
            }
        if lr is not None:
            payload["lr"] = lr
        self.log(payload, step=epoch)

    def log_final(self, summary: Dict[str, Any]):
        self.log({"event": "experiment_complete", **summary})
