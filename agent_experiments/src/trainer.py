from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ExperimentConfig
from .data import M_END, M_MASK, M_START, PAD_TOKEN
from .objectives import contrastive_alignment_loss, maybe_position_ids


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model_and_tokenizer(
    model_name: str,
    motion_tokens: Iterable[str],
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    additional_specials = [token for token in [M_START, M_END, M_MASK] if token not in tokenizer.get_vocab()]
    tokenizer.add_special_tokens(
        {
            "pad_token": tokenizer.pad_token or PAD_TOKEN,
            "additional_special_tokens": additional_specials,
        }
    )
    tokenizer.add_tokens(list(motion_tokens), special_tokens=True)

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model, tokenizer


def move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    moved: Dict[str, object] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


class ExperimentTrainer:
    def __init__(
        self,
        config: ExperimentConfig,
        model,
        tokenizer,
        train_loader: DataLoader,
        dev_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        word_loader: Optional[DataLoader] = None,
    ) -> None:
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.word_loader = word_loader
        self.output_dir = Path(config.training.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.history = []
        self.model.to(self.device)

    def _forward_loss(self, batch: Dict[str, object], for_denoise: bool = False):
        attn_key = "denoise_attention_mask" if for_denoise else "attention_mask"
        input_key = "denoise_input_ids" if for_denoise else "input_ids"
        label_key = "denoise_labels" if for_denoise else "labels"
        attention_mask = batch[attn_key]
        position_ids = maybe_position_ids(
            attention_mask=attention_mask,
            enabled=self.config.objectives.randomized_position and not for_denoise,
            max_position_offset=self.config.objectives.max_position_offset,
        )
        return self.model(
            input_ids=batch[input_key],
            attention_mask=attention_mask,
            labels=batch[label_key],
            position_ids=position_ids,
            output_hidden_states=(not for_denoise and self.config.objectives.contrastive_weight > 0),
        )

    def dry_run(self) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            batch = next(iter(self.train_loader))
            batch = move_batch_to_device(batch, self.device)
            outputs = self._forward_loss(batch)
            metrics = {
                "main_loss": float(outputs.loss.item()),
                "batch_size": int(batch["input_ids"].shape[0]),
                "sequence_length": int(batch["input_ids"].shape[1]),
            }
            if self.config.objectives.contrastive_weight > 0 and outputs.hidden_states is not None:
                contrastive = contrastive_alignment_loss(
                    hidden_states=outputs.hidden_states[-1],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    temperature=self.config.objectives.contrastive_temperature,
                )
                metrics["contrastive_loss"] = float(contrastive.item())
            if self.config.objectives.denoise_weight > 0:
                denoise_outputs = self._forward_loss(batch, for_denoise=True)
                metrics["denoise_loss"] = float(denoise_outputs.loss.item())
            return metrics

    def _compute_train_step(
        self,
        sentence_batch: Dict[str, object],
        word_batch: Optional[Dict[str, object]],
    ) -> Dict[str, float]:
        outputs = self._forward_loss(sentence_batch)
        loss = outputs.loss
        metrics = {
            "sentence_loss": float(outputs.loss.item()),
        }

        if self.config.objectives.contrastive_weight > 0 and outputs.hidden_states is not None:
            contrastive = contrastive_alignment_loss(
                hidden_states=outputs.hidden_states[-1],
                attention_mask=sentence_batch["attention_mask"],
                labels=sentence_batch["labels"],
                temperature=self.config.objectives.contrastive_temperature,
            )
            loss = loss + self.config.objectives.contrastive_weight * contrastive
            metrics["contrastive_loss"] = float(contrastive.item())

        if self.config.objectives.denoise_weight > 0:
            denoise_outputs = self._forward_loss(sentence_batch, for_denoise=True)
            loss = loss + self.config.objectives.denoise_weight * denoise_outputs.loss
            metrics["denoise_loss"] = float(denoise_outputs.loss.item())

        if word_batch is not None and self.config.objectives.word_replay_weight > 0:
            replay_outputs = self._forward_loss(word_batch)
            loss = loss + self.config.objectives.word_replay_weight * replay_outputs.loss
            metrics["word_replay_loss"] = float(replay_outputs.loss.item())

        metrics["total_loss"] = float(loss.item())
        return {"loss_tensor": loss, **metrics}

    def _evaluate_loader(self, loader: Optional[DataLoader], split_name: str) -> Optional[Dict[str, float]]:
        if loader is None:
            return None
        self.model.eval()
        losses = []
        coverage_buckets: Dict[str, list[float]] = {}
        with torch.no_grad():
            for batch in loader:
                batch = move_batch_to_device(batch, self.device)
                outputs = self._forward_loss(batch)
                batch_loss = float(outputs.loss.item())
                losses.append(batch_loss)
                for bucket in batch["coverage_bucket"]:
                    coverage_buckets.setdefault(bucket, []).append(batch_loss)
        summary = {
            f"{split_name}_loss": float(sum(losses) / max(1, len(losses))),
        }
        for bucket, bucket_losses in sorted(coverage_buckets.items()):
            key = f"{split_name}_loss_{bucket.replace('.', '_').replace('-', '_')}"
            summary[key] = float(sum(bucket_losses) / max(1, len(bucket_losses)))
        return summary

    def _save_checkpoint(self, step: int) -> None:
        checkpoint_dir = self.output_dir / f"checkpoint_step_{step:06d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

    def train(self) -> Dict[str, object]:
        set_seed(self.config.training.seed)
        grad_accum = max(1, self.config.training.grad_accum)
        max_steps = self.config.training.max_steps
        word_iterator = iter(self.word_loader) if self.word_loader is not None else None
        global_step = 0

        for epoch in range(1, self.config.training.epochs + 1):
            self.model.train()
            running = []
            for batch_idx, sentence_batch in enumerate(self.train_loader, start=1):
                sentence_batch = move_batch_to_device(sentence_batch, self.device)
                word_batch = None
                if word_iterator is not None and self.config.objectives.word_replay_weight > 0:
                    try:
                        word_batch = next(word_iterator)
                    except StopIteration:
                        word_iterator = iter(self.word_loader)
                        word_batch = next(word_iterator)
                    word_batch = move_batch_to_device(word_batch, self.device)

                result = self._compute_train_step(sentence_batch, word_batch)
                (result["loss_tensor"] / grad_accum).backward()
                running.append(result["total_loss"])

                if batch_idx % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if global_step % self.config.training.log_every == 0:
                        avg_loss = sum(running[-self.config.training.log_every :]) / min(
                            len(running), self.config.training.log_every
                        )
                        print(
                            f"[train] epoch={epoch} step={global_step} "
                            f"loss={avg_loss:.4f}"
                        )

                    if global_step % self.config.training.save_every == 0:
                        self._save_checkpoint(global_step)

                    if max_steps is not None and global_step >= max_steps:
                        break

            epoch_summary = {
                "epoch": epoch,
                "train_loss": float(sum(running) / max(1, len(running))),
            }
            dev_summary = self._evaluate_loader(self.dev_loader, "dev")
            test_summary = self._evaluate_loader(self.test_loader, "test")
            if dev_summary:
                epoch_summary.update(dev_summary)
            if test_summary:
                epoch_summary.update(test_summary)
            self.history.append(epoch_summary)
            with open(self.output_dir / "metrics_history.json", "w", encoding="utf-8") as handle:
                json.dump(self.history, handle, indent=2)
            print(f"[epoch] {json.dumps(epoch_summary, indent=None)}")

            if max_steps is not None and global_step >= max_steps:
                break

        final_metrics = {
            "history": self.history,
            "config": asdict(self.config),
        }
        with open(self.output_dir / "final_metrics.json", "w", encoding="utf-8") as handle:
            json.dump(final_metrics, handle, indent=2)
        self._save_checkpoint(global_step or 1)
        return final_metrics
