from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from .config import DataConfig, RetrievalConfig
from .retrieval import (
    coverage_bucket,
    coverage_ratio,
    format_lexicon_memory,
    length_bucket_from_count,
)

M_START = "<M_START>"
M_END = "<M_END>"
M_MASK = "<M_MASK>"
PAD_TOKEN = "<PAD>"


def load_json_entries(path: str | Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        return data.get("samples", data.get("data", []))
    return data


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def extract_word_items(entries: Iterable[dict]) -> List[dict]:
    items: List[dict] = []
    for entry in entries:
        word = str(entry.get("word", "")).strip()
        motion = str(entry.get("motion_tokens", "")).strip()
        if word and motion:
            items.append(entry)
    return items


def extract_sentence_items(entries: Iterable[dict]) -> List[dict]:
    items: List[dict] = []
    for entry in entries:
        text = str(entry.get("text") or entry.get("sentence") or "").strip()
        motion = str(entry.get("motion_tokens", "")).strip()
        item_type = str(entry.get("type", "")).strip().lower()
        if motion and text and (item_type == "sentence" or "text" in entry or "sentence" in entry):
            items.append(entry)
    return items


def collect_motion_vocab(entries: Iterable[dict]) -> List[str]:
    vocab = set()
    for entry in entries:
        for token in str(entry.get("motion_tokens", "")).strip().split():
            vocab.add(f"<M{token}>")
    return sorted(vocab)


def wrap_motion_target(tokens: Sequence[str]) -> str:
    wrapped_parts = []
    for token in tokens:
        token = str(token).strip()
        if not token:
            continue
        if token.startswith("<") and token.endswith(">"):
            wrapped_parts.append(token)
        else:
            wrapped_parts.append(f"<M{token}>")
    wrapped = " ".join(wrapped_parts)
    return f"{M_START} {wrapped} {M_END}".strip()


def corrupt_motion_tokens(
    tokens: Sequence[str],
    mask_ratio: float,
    mask_token: str = M_MASK,
    span_min: int = 1,
    span_max: int = 3,
) -> List[str]:
    clean = [str(token) for token in tokens if str(token).strip()]
    if not clean:
        return []

    corrupted = list(clean)
    target_mask_count = max(1, int(round(len(clean) * max(0.0, min(1.0, mask_ratio)))))
    masked = 0
    attempts = 0
    while masked < target_mask_count and attempts < len(clean) * 4:
        attempts += 1
        start = random.randint(0, len(clean) - 1)
        span = random.randint(span_min, span_max)
        end = min(len(clean), start + span)
        for index in range(start, end):
            if corrupted[index] != mask_token:
                corrupted[index] = mask_token
                masked += 1
            if masked >= target_mask_count:
                break
    return corrupted


def build_sentence_prompt(
    text: str,
    motion_len: int,
    retrieval_cfg: RetrievalConfig,
    word_bank: Optional[Dict[str, object]] = None,
) -> str:
    length_bucket = length_bucket_from_count(motion_len)
    lines = [
        "Instruction: Generate ASL motion tokens for the sentence below.",
        f"Sentence: {text}",
        f"Expected motion length bucket: {length_bucket}",
    ]
    if retrieval_cfg.enabled and word_bank is not None:
        coverage = coverage_ratio(text, word_bank)
        lines.append(f"Lexical coverage bucket: {coverage_bucket(coverage)}")
        lines.append(
            format_lexicon_memory(
                sentence_text=text,
                word_bank=word_bank,
                max_words=retrieval_cfg.max_words,
                anchor_tokens_per_variant=retrieval_cfg.anchor_tokens_per_variant,
                include_motion_anchors=retrieval_cfg.include_motion_anchors,
                include_lengths=retrieval_cfg.include_lengths,
            )
        )
    lines.append("Motion:")
    return "\n".join(lines) + " "


def build_word_prompt(word: str) -> str:
    return f"Instruction: Generate ASL motion tokens for the word '{word}'.\nMotion: "


@dataclass
class PreparedSample:
    prompt: str
    full_text: str
    denoise_text: str
    coverage: float
    coverage_bucket: str
    motion_length: int
    text: str


class SentenceMotionDataset(Dataset):
    def __init__(
        self,
        sentence_items: Sequence[dict],
        data_cfg: DataConfig,
        retrieval_cfg: RetrievalConfig,
        word_bank: Optional[Dict[str, object]] = None,
        denoise_mask_ratio: float = 0.25,
    ) -> None:
        self.samples: List[PreparedSample] = []
        self.retrieval_cfg = retrieval_cfg
        for item in sentence_items:
            text = normalize_text(item.get("text") or item.get("sentence") or "")
            raw_tokens = str(item.get("motion_tokens", "")).strip().split()
            if not text or not raw_tokens:
                continue
            if len(raw_tokens) < data_cfg.min_motion_tokens or len(raw_tokens) > data_cfg.max_motion_tokens:
                continue

            prompt = build_sentence_prompt(
                text=text,
                motion_len=len(raw_tokens),
                retrieval_cfg=retrieval_cfg,
                word_bank=word_bank,
            )
            full_text = prompt + wrap_motion_target(raw_tokens)
            corrupted_tokens = corrupt_motion_tokens(raw_tokens, mask_ratio=denoise_mask_ratio)
            denoise_text = prompt + wrap_motion_target(corrupted_tokens)
            coverage = coverage_ratio(text, word_bank or {})
            self.samples.append(
                PreparedSample(
                    prompt=prompt,
                    full_text=full_text,
                    denoise_text=denoise_text,
                    coverage=coverage,
                    coverage_bucket=coverage_bucket(coverage),
                    motion_length=len(raw_tokens),
                    text=text,
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> PreparedSample:
        return self.samples[index]


class WordReplayDataset(Dataset):
    def __init__(self, word_items: Sequence[dict], data_cfg: DataConfig) -> None:
        self.samples: List[PreparedSample] = []
        for item in word_items:
            word = normalize_text(item.get("word", ""))
            raw_tokens = str(item.get("motion_tokens", "")).strip().split()
            if not word or not raw_tokens:
                continue
            if len(raw_tokens) < data_cfg.min_motion_tokens or len(raw_tokens) > data_cfg.max_motion_tokens:
                continue
            prompt = build_word_prompt(word)
            full_text = prompt + wrap_motion_target(raw_tokens)
            self.samples.append(
                PreparedSample(
                    prompt=prompt,
                    full_text=full_text,
                    denoise_text=full_text,
                    coverage=1.0,
                    coverage_bucket="word",
                    motion_length=len(raw_tokens),
                    text=word,
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> PreparedSample:
        return self.samples[index]


def _mask_prompt_and_padding(
    tokenized: Dict[str, torch.Tensor],
    prompt_lengths: List[int],
    pad_token_id: Optional[int],
) -> torch.Tensor:
    labels = tokenized["input_ids"].clone()
    for row_idx, prompt_len in enumerate(prompt_lengths):
        labels[row_idx, :prompt_len] = -100
        if pad_token_id is not None:
            labels[row_idx, tokenized["input_ids"][row_idx] == pad_token_id] = -100
    return labels


def create_collate_fn(tokenizer, max_length: int):
    pad_token_id = tokenizer.pad_token_id

    def collate_fn(batch: Sequence[PreparedSample]) -> Dict[str, object]:
        prompts = [item.prompt for item in batch]
        full_texts = [item.full_text for item in batch]
        denoise_texts = [item.denoise_text for item in batch]

        prompt_tokens = tokenizer(prompts, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        tokenized = tokenizer(full_texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        denoise_tokenized = tokenizer(
            denoise_texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )

        prompt_lengths = prompt_tokens["attention_mask"].sum(dim=1).tolist()
        labels = _mask_prompt_and_padding(tokenized, prompt_lengths, pad_token_id)
        denoise_labels = _mask_prompt_and_padding(denoise_tokenized, prompt_lengths, pad_token_id)

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
            "denoise_input_ids": denoise_tokenized["input_ids"],
            "denoise_attention_mask": denoise_tokenized["attention_mask"],
            "denoise_labels": denoise_labels,
            "coverage": torch.tensor([item.coverage for item in batch], dtype=torch.float32),
            "motion_length": torch.tensor([item.motion_length for item in batch], dtype=torch.long),
            "coverage_bucket": [item.coverage_bucket for item in batch],
            "text": [item.text for item in batch],
        }

    return collate_fn
