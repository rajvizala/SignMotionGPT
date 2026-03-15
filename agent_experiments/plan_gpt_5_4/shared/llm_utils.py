from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from training.sentence_level.pipeline import (
    CURRICULUM_MEDIUM_MAX,
    CURRICULUM_SHORT_MAX,
    LEN_LONG,
    LEN_MEDIUM,
    LEN_SHORT,
    initialize_embeddings_from_vqvae,
    setup_model_and_tokenizer,
)

from .data_utils import (
    collect_motion_vocab,
    coverage_bucket,
    coverage_ratio,
    extract_sentence_items,
    extract_word_items,
    load_json_entries,
    novelty_bucket,
    tokenize_text,
)


M_START = "<M_START>"
M_END = "<M_END>"
M_MASK = "<M_MASK>"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def length_token_for_motion_length(motion_length: int) -> str:
    if motion_length < CURRICULUM_SHORT_MAX:
        return LEN_SHORT
    if motion_length < CURRICULUM_MEDIUM_MAX:
        return LEN_MEDIUM
    return LEN_LONG


def length_bucket_name(motion_length: int) -> str:
    if motion_length < CURRICULUM_SHORT_MAX:
        return "length_short"
    if motion_length < CURRICULUM_MEDIUM_MAX:
        return "length_medium"
    return "length_long"


def build_word_bank(word_items: Iterable[dict]) -> Dict[str, List[List[str]]]:
    bank: Dict[str, List[List[str]]] = {}
    for item in word_items:
        word = str(item.get("word", "")).strip().lower()
        tokens = str(item.get("motion_tokens", "")).strip().split()
        if not word or not tokens:
            continue
        bank.setdefault(word, []).append(tokens)
    return bank


def compress_motion_anchor(tokens: Sequence[str], anchor_tokens: int = 6) -> str:
    clean = [str(token) for token in tokens if str(token).strip()]
    if not clean:
        return ""
    if len(clean) <= anchor_tokens:
        return " ".join(f"<M{token}>" for token in clean)
    indices = sorted({round(i * (len(clean) - 1) / max(1, anchor_tokens - 1)) for i in range(anchor_tokens)})
    return " ".join(f"<M{clean[index]}>" for index in indices)


def retrieved_lexicon_block(
    text: str,
    word_bank: Dict[str, List[List[str]]],
    max_words: int,
    anchor_tokens: int,
) -> str:
    lines = ["Lexicon memory:"]
    seen = set()
    matches = 0
    for token in tokenize_text(text):
        if token in seen:
            continue
        seen.add(token)
        if token not in word_bank:
            continue
        sequence = word_bank[token][0]
        lines.append(
            f"- word={token} approx_len={len(sequence)} anchors={compress_motion_anchor(sequence, anchor_tokens)}"
        )
        matches += 1
        if matches >= max(1, max_words):
            break
    if matches == 0:
        lines.append("- none")
    return "\n".join(lines)


def wrap_motion_target(tokens: Sequence[str]) -> str:
    wrapped = " ".join(f"<M{token}>" for token in tokens)
    return f"{M_START} {wrapped} {M_END}"


def corrupt_motion_tokens(tokens: Sequence[str], mask_rate: float) -> List[str]:
    clean = list(tokens)
    if not clean:
        return []
    target_masks = max(1, int(round(len(clean) * mask_rate)))
    chosen = set(random.sample(range(len(clean)), min(len(clean), target_masks)))
    return [M_MASK if index in chosen else f"<M{token}>" for index, token in enumerate(clean)]


@dataclass
class TokenSample:
    prompt: str
    full_text: str
    denoise_text: str
    coverage_key: str
    novelty_key: str
    length_key: str

    @property
    def bucket_names(self) -> List[str]:
        return [self.length_key, self.coverage_key, self.novelty_key]


class SentenceTokenDataset(Dataset):
    def __init__(
        self,
        sentence_items: Sequence[dict],
        word_bank: Dict[str, List[List[str]]],
        include_lexicon: bool,
        lexicon_context_prob: float,
        max_words: int,
        anchor_tokens: int,
        mask_rate: float,
        max_motion_tokens: int = 256,
    ) -> None:
        self.samples: List[TokenSample] = []
        word_lexicon = set(word_bank.keys())
        for item in sentence_items:
            text = str(item.get("text") or item.get("sentence") or "").strip()
            motion_tokens = str(item.get("motion_tokens", "")).strip().split()
            if not text or not motion_tokens or len(motion_tokens) > max_motion_tokens:
                continue
            motion_len = len(motion_tokens)
            prompt_lines = [
                "Instruction: Generate ASL motion tokens for the sentence below.",
                f"Sentence: {text}",
                f"Expected motion length bucket: {length_token_for_motion_length(motion_len)}",
            ]
            if include_lexicon and random.random() < lexicon_context_prob:
                prompt_lines.append(retrieved_lexicon_block(text, word_bank, max_words, anchor_tokens))
            prompt_lines.append("Motion:")
            prompt = "\n".join(prompt_lines) + " "
            full_text = prompt + wrap_motion_target(motion_tokens)
            denoise_text = prompt + f"{M_START} {' '.join(corrupt_motion_tokens(motion_tokens, mask_rate))} {M_END}"
            self.samples.append(
                TokenSample(
                    prompt=prompt,
                    full_text=full_text,
                    denoise_text=denoise_text,
                    coverage_key=coverage_bucket(text, word_lexicon),
                    novelty_key=novelty_bucket(text, word_lexicon),
                    length_key=length_bucket_name(motion_len),
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TokenSample:
        return self.samples[index]


class WordReplayDataset(Dataset):
    def __init__(self, word_items: Sequence[dict], max_motion_tokens: int = 256) -> None:
        self.samples: List[TokenSample] = []
        for item in word_items:
            word = str(item.get("word", "")).strip().lower()
            motion_tokens = str(item.get("motion_tokens", "")).strip().split()
            if not word or not motion_tokens or len(motion_tokens) > max_motion_tokens:
                continue
            prompt = f"Instruction: Generate ASL motion tokens for the word '{word}'.\nMotion: "
            full_text = prompt + wrap_motion_target(motion_tokens)
            self.samples.append(
                TokenSample(
                    prompt=prompt,
                    full_text=full_text,
                    denoise_text=full_text,
                    coverage_key="coverage_high",
                    novelty_key="novelty_seen_words",
                    length_key=length_bucket_name(len(motion_tokens)),
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TokenSample:
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


def create_token_collate_fn(tokenizer, max_length: int, denoise_mask_token_id: int):
    pad_token_id = tokenizer.pad_token_id

    def collate_fn(batch: Sequence[TokenSample]) -> Dict[str, object]:
        prompts = [sample.prompt for sample in batch]
        full_texts = [sample.full_text for sample in batch]
        denoise_texts = [sample.denoise_text for sample in batch]

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

        masked_positions = (denoise_tokenized["input_ids"] == denoise_mask_token_id) & (denoise_labels != -100)
        denoise_labels[~masked_positions] = -100

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
            "denoise_input_ids": denoise_tokenized["input_ids"],
            "denoise_attention_mask": denoise_tokenized["attention_mask"],
            "denoise_labels": denoise_labels,
            "bucket_names": [sample.bucket_names for sample in batch],
        }

    return collate_fn


def build_llm_model_and_tokenizer(
    model_name: str,
    json_path: str,
    vqvae_ckpt: str,
    device: torch.device,
):
    entries = load_json_entries(json_path)
    motion_tokens = collect_motion_vocab(entries)
    model, tokenizer = setup_model_and_tokenizer(model_name, motion_tokens)
    if M_MASK not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [M_MASK]})
        model.resize_token_embeddings(len(tokenizer))
    initialize_embeddings_from_vqvae(model, tokenizer, vqvae_ckpt, device)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        model.resize_token_embeddings(len(tokenizer))
    if device.type == "cpu":
        model = model.float()
    model.to(device)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model, tokenizer


class AlignmentProjector(nn.Module):
    def __init__(self, hidden_size: int, projection_dim: int = 256) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, projection_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(hidden_states), dim=-1)


def mean_pool(hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.float().unsqueeze(-1)
    total = (hidden_states * weights).sum(dim=1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return total / denom


def contrastive_alignment_loss(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    projector: AlignmentProjector,
    temperature: float = 0.07,
) -> torch.Tensor:
    prompt_mask = (labels == -100) & attention_mask.bool()
    motion_mask = (labels != -100) & attention_mask.bool()
    prompt_repr = projector(mean_pool(hidden_states, prompt_mask))
    motion_repr = projector(mean_pool(hidden_states, motion_mask))
    logits = prompt_repr @ motion_repr.transpose(0, 1)
    logits = logits / max(temperature, 1e-6)
    targets = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.transpose(0, 1), targets))


def randomized_position_ids(attention_mask: torch.Tensor, max_offset: int) -> torch.Tensor:
    batch_size, seq_len = attention_mask.shape
    offsets = torch.randint(0, max(1, max_offset), size=(batch_size, 1), device=attention_mask.device)
    base = torch.arange(seq_len, device=attention_mask.device).unsqueeze(0).expand(batch_size, -1)
    position_ids = base + offsets
    return position_ids * attention_mask.long()


def move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def evaluate_token_model(
    model,
    dataloader,
    device: torch.device,
    use_random_positions: bool,
    max_position_offset: int,
    max_batches: Optional[int] = None,
) -> Dict[str, object]:
    model.eval()
    losses = []
    bucket_losses: Dict[str, List[float]] = {}
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = move_batch_to_device(batch, device)
            position_ids = randomized_position_ids(batch["attention_mask"], max_position_offset) if use_random_positions else None
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                position_ids=position_ids,
            )
            batch_loss = float(outputs.loss.item())
            losses.append(batch_loss)
            for bucket_set in batch["bucket_names"]:
                for bucket_name in bucket_set:
                    bucket_losses.setdefault(bucket_name, []).append(batch_loss)
    average_loss = sum(losses) / max(1, len(losses))
    bucket_metrics = {bucket: sum(values) / len(values) for bucket, values in bucket_losses.items() if values}
    worst_group = max(bucket_metrics.values()) if bucket_metrics else average_loss
    return {
        "avg_loss": average_loss,
        "worst_group_score": worst_group,
        "bucket_metrics": bucket_metrics,
    }
