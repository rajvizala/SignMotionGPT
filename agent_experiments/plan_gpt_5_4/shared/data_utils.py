from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from mGPT.archs.mgpt_vq import VQVae


SMPL_DIM = 182
HAND_START = 73
HAND_END = 163
DOWNSAMPLE_FACTOR = 4
WORD_RE = re.compile(r"[a-zA-Z']+")

VQ_CONFIG = {
    "nfeats": SMPL_DIM,
    "code_num": 512,
    "code_dim": 512,
    "output_emb_width": 512,
    "down_t": 2,
    "stride_t": 2,
    "width": 512,
    "depth": 3,
    "dilation_growth_rate": 3,
    "activation": "relu",
    "norm": None,
    "quantizer": "ema_reset",
}

DEFAULT_VQVAE_CKPT_CANDIDATES = [
    os.environ.get("SIGNMOTION_VQVAE_CKPT"),
    "/content/drive/MyDrive/finetune_combine_checkpoint/vqvae_finetuned_epoch_450.pt",
    "/content/vqvae_finetuned_epoch_450.pt",
    "./vqvae_finetuned_epoch_450.pt",
    "./data/vqvae_model.pt",
]

DEFAULT_STATS_CANDIDATES = [
    os.environ.get("SIGNMOTION_STATS_PATH"),
    "./computed_stats.pt",
    "/content/combined_stats.pt",
    "./data/vqvae_stats.pt",
]

DEFAULT_JSON_CANDIDATES = [
    os.environ.get("SIGNMOTION_JSON_DATA"),
    "./data/motion_llm_dataset.json",
]


def resolve_default_path(candidates: Sequence[Optional[str]]) -> str:
    filtered = [candidate for candidate in candidates if candidate]
    for candidate in filtered:
        if os.path.exists(candidate):
            return candidate
    return filtered[0] if filtered else ""


def default_vqvae_ckpt() -> str:
    return resolve_default_path(DEFAULT_VQVAE_CKPT_CANDIDATES)


def default_stats_path() -> str:
    return resolve_default_path(DEFAULT_STATS_CANDIDATES)


def default_json_path() -> str:
    return resolve_default_path(DEFAULT_JSON_CANDIDATES)


def normalize_word(text: str) -> str:
    return re.sub(r"[^a-z0-9']+", "", str(text).lower().strip())


def tokenize_text(text: str) -> List[str]:
    return [normalize_word(token) for token in WORD_RE.findall(str(text).lower()) if normalize_word(token)]


def infer_text_from_path(path: str | Path) -> str:
    stem = Path(path).stem
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = re.sub(r"\b\d+\b", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem.lower()


def infer_text_from_npz(npz_path: str | Path) -> str:
    try:
        with np.load(npz_path, allow_pickle=True) as bundle:
            for key in ("text", "sentence", "word", "label", "prompt"):
                if key in bundle:
                    value = bundle[key]
                    if np.isscalar(value):
                        return str(value).strip().lower()
                    if isinstance(value, np.ndarray) and value.size > 0:
                        return str(value.reshape(-1)[0]).strip().lower()
    except Exception:
        pass
    return infer_text_from_path(npz_path)


def load_npz_motion(npz_path: str | Path) -> np.ndarray:
    with np.load(npz_path, allow_pickle=True) as bundle:
        if "motion" in bundle:
            motion = bundle["motion"]
        else:
            first_key = bundle.files[0]
            motion = bundle[first_key]
    motion = np.asarray(motion, dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] != SMPL_DIM:
        raise ValueError(f"Expected motion shape (T, {SMPL_DIM}) in {npz_path}, got {motion.shape}")
    return motion


def load_stats(stats_path: Optional[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    if stats_path and os.path.exists(stats_path):
        stats = torch.load(stats_path, map_location="cpu")
        mean = stats.get("mean", torch.zeros(SMPL_DIM))
        std = stats.get("std", torch.ones(SMPL_DIM))
    else:
        mean = torch.zeros(SMPL_DIM)
        std = torch.ones(SMPL_DIM)
    if not torch.is_tensor(mean):
        mean = torch.tensor(mean)
    if not torch.is_tensor(std):
        std = torch.tensor(std)
    return mean.float(), std.float()


def normalize_motion(motion: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (motion - mean) / (std + 1e-8)


def unnormalize_motion(motion: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return motion * (std + 1e-8) + mean


@dataclass
class MotionSample:
    path: str
    text: str
    motion: torch.Tensor

    @property
    def length(self) -> int:
        return int(self.motion.shape[0])


def discover_npz_files(root_dir: str) -> List[str]:
    if not root_dir or not os.path.exists(root_dir):
        return []
    return sorted(glob.glob(os.path.join(root_dir, "**", "*.npz"), recursive=True))


def load_motion_samples(root_dir: str, max_samples: Optional[int] = None) -> List[MotionSample]:
    files = discover_npz_files(root_dir)
    if max_samples is not None:
        files = files[: max(0, max_samples)]
    samples: List[MotionSample] = []
    for npz_path in files:
        try:
            motion = torch.tensor(load_npz_motion(npz_path), dtype=torch.float32)
            text = infer_text_from_npz(npz_path)
            samples.append(MotionSample(path=npz_path, text=text, motion=motion))
        except Exception:
            continue
    return samples


def build_word_lexicon_from_dir(word_data_dir: str, max_samples: Optional[int] = None) -> set[str]:
    lexicon = set()
    for sample in load_motion_samples(word_data_dir, max_samples=max_samples):
        tokens = tokenize_text(sample.text)
        if tokens:
            lexicon.add(tokens[0])
    return lexicon


def load_json_entries(json_path: str) -> List[dict]:
    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        return data.get("samples", data.get("data", []))
    return data


def extract_sentence_items(entries: Iterable[dict]) -> List[dict]:
    items: List[dict] = []
    for entry in entries:
        text = str(entry.get("text") or entry.get("sentence") or "").strip()
        motion = str(entry.get("motion_tokens", "")).strip()
        item_type = str(entry.get("type", "")).strip().lower()
        if text and motion and (item_type == "sentence" or "text" in entry or "sentence" in entry):
            items.append(entry)
    return items


def extract_word_items(entries: Iterable[dict]) -> List[dict]:
    items: List[dict] = []
    for entry in entries:
        word = str(entry.get("word", "")).strip()
        motion = str(entry.get("motion_tokens", "")).strip()
        if word and motion:
            items.append(entry)
    return items


def collect_motion_vocab(entries: Iterable[dict]) -> List[str]:
    vocab = set()
    for entry in entries:
        for token in str(entry.get("motion_tokens", "")).strip().split():
            vocab.add(f"<M{token}>")
    return sorted(vocab)


def wrap_motion_tokens(tokens: Sequence[str]) -> str:
    wrapped = []
    for token in tokens:
        token = str(token).strip()
        if not token:
            continue
        wrapped.append(token if token.startswith("<") and token.endswith(">") else f"<M{token}>")
    return " ".join(wrapped)


def coverage_ratio(text: str, word_lexicon: set[str]) -> float:
    tokens = tokenize_text(text)
    if not tokens:
        return 0.0
    covered = sum(1 for token in tokens if token in word_lexicon)
    return covered / max(1, len(tokens))


def length_bucket(length_value: int) -> str:
    if length_value < 30:
        return "length_short"
    if length_value < 80:
        return "length_medium"
    return "length_long"


def coverage_bucket(text: str, word_lexicon: set[str]) -> str:
    coverage = coverage_ratio(text, word_lexicon)
    if coverage < 0.34:
        return "coverage_low"
    if coverage < 0.67:
        return "coverage_medium"
    return "coverage_high"


def novelty_bucket(text: str, word_lexicon: set[str]) -> str:
    tokens = tokenize_text(text)
    if tokens and all(token in word_lexicon for token in tokens):
        return "novelty_seen_words"
    return "novelty_novel_vocab"


def sample_bucket_names(text: str, motion_length: int, word_lexicon: set[str]) -> List[str]:
    return [
        length_bucket(motion_length),
        coverage_bucket(text, word_lexicon),
        novelty_bucket(text, word_lexicon),
    ]


def pad_motion_batch(motions: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    if not motions:
        raise ValueError("Cannot pad an empty motion batch.")
    max_len = max(int(motion.shape[0]) for motion in motions)
    padded_len = ((max_len + DOWNSAMPLE_FACTOR - 1) // DOWNSAMPLE_FACTOR) * DOWNSAMPLE_FACTOR
    batch = torch.zeros(len(motions), padded_len, SMPL_DIM, dtype=torch.float32)
    lengths = torch.zeros(len(motions), dtype=torch.long)
    for index, motion in enumerate(motions):
        length = int(motion.shape[0])
        batch[index, :length] = motion
        lengths[index] = length
    return batch, lengths


def load_vqvae_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    config: Optional[Dict[str, int]] = None,
) -> VQVae:
    resolved_config = dict(VQ_CONFIG)
    if config:
        resolved_config.update(config)
    model = VQVae(**resolved_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    normalized_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("vqvae."):
            normalized_state_dict[key[6:]] = value
        else:
            normalized_state_dict[key] = value
    model.load_state_dict(normalized_state_dict, strict=False)
    model.eval()
    return model


def decode_codes_batch(model: VQVae, codes: torch.Tensor) -> torch.Tensor:
    decoded_rows = []
    for row in codes:
        decoded_rows.append(model.decode(row.unsqueeze(0)).detach())
    return torch.cat(decoded_rows, dim=0) if decoded_rows else torch.empty(0)


def compute_mse_breakdown(prediction: torch.Tensor, target: torch.Tensor, length: int) -> Dict[str, float]:
    pred = prediction[:length]
    tgt = target[:length]
    diff = (pred - tgt) ** 2
    overall = float(diff.mean().item())
    hand = float(diff[:, HAND_START:HAND_END].mean().item())
    body_mask = torch.cat([diff[:, :HAND_START], diff[:, HAND_END:]], dim=1)
    body = float(body_mask.mean().item())
    return {
        "overall_mse": overall,
        "hand_mse": hand,
        "body_mse": body,
    }


class MotionFolderDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        word_lexicon: set[str],
        max_samples: Optional[int] = None,
    ) -> None:
        self.samples = load_motion_samples(root_dir, max_samples=max_samples)
        self.word_lexicon = word_lexicon

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        return {
            "motion": sample.motion,
            "text": sample.text,
            "length": sample.length,
            "buckets": sample_bucket_names(sample.text, sample.length, self.word_lexicon),
            "path": sample.path,
        }


def collate_motion_items(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    motions = [item["motion"] for item in batch]
    texts = [str(item["text"]) for item in batch]
    buckets = [list(item["buckets"]) for item in batch]
    paths = [str(item["path"]) for item in batch]
    padded, lengths = pad_motion_batch(motions)
    return {
        "motion": padded,
        "lengths": lengths,
        "texts": texts,
        "buckets": buckets,
        "paths": paths,
    }
