from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from agent_experiments.src.analysis import build_bigram_set, summarize_sentence_split
from agent_experiments.src.config import DataConfig, RetrievalConfig
from agent_experiments.src.data import (
    SentenceMotionDataset,
    WordReplayDataset,
    collect_motion_vocab,
    create_collate_fn,
    extract_sentence_items,
    extract_word_items,
    load_json_entries,
)
from agent_experiments.src.retrieval import build_word_motion_bank


class DummyTokenizer:
    def __init__(self) -> None:
        self.vocab = {"<PAD>": 0}
        self.pad_token_id = 0

    def _encode(self, text: str) -> list[int]:
        ids = []
        for token in text.split():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
            ids.append(self.vocab[token])
        return ids

    def __call__(self, texts, truncation=True, max_length=None, padding=True, return_tensors="pt"):
        if isinstance(texts, str):
            texts = [texts]
        encoded = [self._encode(text) for text in texts]
        if max_length is not None:
            encoded = [ids[:max_length] for ids in encoded]
        max_len = max(len(ids) for ids in encoded) if padding else None
        if max_length is not None and padding:
            max_len = min(max_len, max_length)
        padded = []
        masks = []
        for ids in encoded:
            target_len = max_len if padding else len(ids)
            row = ids + [self.pad_token_id] * max(0, target_len - len(ids))
            row = row[:target_len]
            padded.append(row)
            masks.append([1 if token_id != self.pad_token_id else 0 for token_id in row])
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


def build_toy_entries() -> list[dict]:
    return [
        {"word": "hello", "participant_id": "P1", "motion_tokens": "1 2 3 4 5 6 7 8"},
        {"word": "library", "participant_id": "P2", "motion_tokens": "8 7 6 5 4 3 2 1"},
        {"type": "sentence", "text": "hello library", "motion_tokens": "1 2 3 4 8 7 6 5 4 3"},
        {"type": "sentence", "text": "library hello", "motion_tokens": "8 7 6 5 1 2 3 4 5 6"},
    ]


def main() -> None:
    entries = build_toy_entries()
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = Path(tmp_dir) / "toy.json"
        json_path.write_text(json.dumps(entries), encoding="utf-8")
        loaded = load_json_entries(json_path)
        word_items = extract_word_items(loaded)
        sentence_items = extract_sentence_items(loaded)
        word_bank = build_word_motion_bank(word_items, variants_per_word=1)

        data_cfg = DataConfig(
            train_json=str(json_path),
            word_json=str(json_path),
            max_seq_len=128,
            min_motion_tokens=4,
            max_motion_tokens=64,
        )
        retrieval_cfg = RetrievalConfig(enabled=True, max_words=4)

        sentence_dataset = SentenceMotionDataset(
            sentence_items=sentence_items,
            data_cfg=data_cfg,
            retrieval_cfg=retrieval_cfg,
            word_bank=word_bank,
            denoise_mask_ratio=0.25,
        )
        word_dataset = WordReplayDataset(word_items, data_cfg=data_cfg)
        tokenizer = DummyTokenizer()
        collate_fn = create_collate_fn(tokenizer, max_length=128)
        batch = collate_fn([sentence_dataset[0], sentence_dataset[1]])

        assert len(sentence_dataset) == 2
        assert len(word_dataset) == 2
        assert batch["input_ids"].shape[0] == 2
        assert "Lexicon memory:" in sentence_dataset[0].prompt
        assert len(collect_motion_vocab(loaded)) >= 8

        summary = summarize_sentence_split(
            sentence_items,
            word_bank=word_bank,
            train_bigram_set=build_bigram_set(sentence_items),
        )
        assert summary["num_samples"] == 2
        assert summary["avg_coverage"] > 0.9
        print("smoke_test_passed")


if __name__ == "__main__":
    main()
