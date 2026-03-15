from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader

from .analysis import build_bigram_set, save_report, summarize_sentence_split
from .config import ExperimentConfig
from .data import (
    collect_motion_vocab,
    create_collate_fn,
    extract_sentence_items,
    extract_word_items,
    load_json_entries,
    SentenceMotionDataset,
    WordReplayDataset,
)
from .retrieval import build_word_motion_bank
from .trainer import ExperimentTrainer, build_model_and_tokenizer, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ASL generalization experiments.")
    parser.add_argument("--config", required=True, help="Path to experiment JSON config.")
    parser.add_argument("--train-json", default=None, help="Optional override for train JSON.")
    parser.add_argument("--dev-json", default=None, help="Optional override for dev JSON.")
    parser.add_argument("--test-json", default=None, help="Optional override for test JSON.")
    parser.add_argument("--word-json", default=None, help="Optional override for word lexicon JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Run one forward pass and stop.")
    return parser.parse_args()


def maybe_override_paths(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    if args.train_json:
        config.data.train_json = args.train_json
    if args.dev_json:
        config.data.dev_json = args.dev_json
    if args.test_json:
        config.data.test_json = args.test_json
    if args.word_json:
        config.data.word_json = args.word_json
    return config


def load_split(path: Optional[str]) -> list[dict]:
    if not path:
        return []
    return load_json_entries(path)


def main() -> None:
    args = parse_args()
    config = maybe_override_paths(ExperimentConfig.from_json(args.config), args)
    output_dir = config.ensure_output_dir()
    config.save_copy(output_dir)
    set_seed(config.training.seed)

    train_entries = load_split(config.data.train_json)
    dev_entries = load_split(config.data.dev_json)
    test_entries = load_split(config.data.test_json)
    word_entries = load_split(config.data.word_json or config.data.train_json)

    train_sentence_items = extract_sentence_items(train_entries)
    dev_sentence_items = extract_sentence_items(dev_entries)
    test_sentence_items = extract_sentence_items(test_entries)
    word_items = extract_word_items(word_entries)
    word_bank = build_word_motion_bank(
        word_items,
        variants_per_word=config.retrieval.variants_per_word,
    )

    train_summary = summarize_sentence_split(
        train_sentence_items,
        word_bank=word_bank,
        train_bigram_set=build_bigram_set(train_sentence_items),
    )
    dev_summary = summarize_sentence_split(
        dev_sentence_items,
        word_bank=word_bank,
        train_bigram_set=build_bigram_set(train_sentence_items),
    ) if dev_sentence_items else None
    test_summary = summarize_sentence_split(
        test_sentence_items,
        word_bank=word_bank,
        train_bigram_set=build_bigram_set(train_sentence_items),
    ) if test_sentence_items else None
    save_report(output_dir / "analysis", train_summary, dev_summary, test_summary)

    motion_vocab = collect_motion_vocab(train_entries + word_entries)
    model, tokenizer = build_model_and_tokenizer(
        model_name=config.training.model_name,
        motion_tokens=motion_vocab,
    )

    train_dataset = SentenceMotionDataset(
        sentence_items=train_sentence_items,
        data_cfg=config.data,
        retrieval_cfg=config.retrieval,
        word_bank=word_bank,
        denoise_mask_ratio=config.objectives.denoise_mask_ratio,
    )
    if len(train_dataset) == 0:
        raise ValueError("No usable sentence-level training samples were found for the current config.")
    dev_dataset = SentenceMotionDataset(
        sentence_items=dev_sentence_items,
        data_cfg=config.data,
        retrieval_cfg=config.retrieval,
        word_bank=word_bank,
        denoise_mask_ratio=config.objectives.denoise_mask_ratio,
    ) if dev_sentence_items else None
    test_dataset = SentenceMotionDataset(
        sentence_items=test_sentence_items,
        data_cfg=config.data,
        retrieval_cfg=config.retrieval,
        word_bank=word_bank,
        denoise_mask_ratio=config.objectives.denoise_mask_ratio,
    ) if test_sentence_items else None
    word_dataset = WordReplayDataset(word_items=word_items, data_cfg=config.data) if word_items else None

    collate_fn = create_collate_fn(tokenizer, max_length=config.data.max_seq_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.training.num_workers,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.training.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.training.num_workers,
    ) if dev_dataset is not None else None
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.training.num_workers,
    ) if test_dataset is not None else None
    word_loader = DataLoader(
        word_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.training.num_workers,
    ) if word_dataset is not None and len(word_dataset) > 0 else None

    trainer = ExperimentTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        word_loader=word_loader,
    )

    if args.dry_run:
        metrics = trainer.dry_run()
        with open(output_dir / "dry_run_metrics.json", "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        print(json.dumps(metrics, indent=2))
        return

    metrics = trainer.train()
    print(json.dumps(metrics["history"][-1] if metrics["history"] else {}, indent=2))


if __name__ == "__main__":
    main()
