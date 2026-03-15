from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .retrieval import coverage_bucket, coverage_ratio, tokenize_for_lexicon


def sentence_bigrams(text: str) -> List[str]:
    tokens = tokenize_for_lexicon(text)
    return [f"{tokens[idx]}::{tokens[idx + 1]}" for idx in range(len(tokens) - 1)]


def build_bigram_set(sentence_items: Sequence[dict]) -> set[str]:
    bigrams = set()
    for item in sentence_items:
        text = str(item.get("text") or item.get("sentence") or "")
        bigrams.update(sentence_bigrams(text))
    return bigrams


def summarize_sentence_split(
    sentence_items: Sequence[dict],
    word_bank: Dict[str, object],
    train_bigram_set: Optional[set[str]] = None,
) -> Dict[str, object]:
    coverage_counter: Counter[str] = Counter()
    length_counter: Counter[str] = Counter()
    novel_bigram_count = 0
    total_bigrams = 0
    coverage_values: List[float] = []
    token_lengths: List[int] = []

    for item in sentence_items:
        text = str(item.get("text") or item.get("sentence") or "").strip()
        motion_tokens = str(item.get("motion_tokens", "")).strip().split()
        if not text or not motion_tokens:
            continue
        coverage = coverage_ratio(text, word_bank)
        coverage_counter[coverage_bucket(coverage)] += 1
        coverage_values.append(coverage)

        length_value = len(motion_tokens)
        token_lengths.append(length_value)
        if length_value < 30:
            length_counter["short"] += 1
        elif length_value < 80:
            length_counter["medium"] += 1
        else:
            length_counter["long"] += 1

        if train_bigram_set is not None:
            current_bigrams = sentence_bigrams(text)
            total_bigrams += len(current_bigrams)
            novel_bigram_count += sum(1 for bigram in current_bigrams if bigram not in train_bigram_set)

    sample_count = sum(coverage_counter.values())
    avg_coverage = sum(coverage_values) / max(1, len(coverage_values))
    avg_length = sum(token_lengths) / max(1, len(token_lengths))
    novel_bigram_rate = novel_bigram_count / max(1, total_bigrams)
    return {
        "num_samples": sample_count,
        "avg_coverage": avg_coverage,
        "avg_motion_token_length": avg_length,
        "coverage_buckets": dict(coverage_counter),
        "length_buckets": dict(length_counter),
        "novel_bigram_rate": novel_bigram_rate,
    }


def render_report_markdown(
    train_summary: Dict[str, object],
    dev_summary: Optional[Dict[str, object]],
    test_summary: Optional[Dict[str, object]],
) -> str:
    lines = [
        "# Generalization Analysis Report",
        "",
        "This report measures whether sentence splits are compositionally difficult with respect to the isolated-word lexicon.",
        "",
    ]
    for name, summary in [("train", train_summary), ("dev", dev_summary), ("test", test_summary)]:
        if summary is None:
            continue
        lines.extend(
            [
                f"## {name}",
                "",
                f"- num_samples: {summary['num_samples']}",
                f"- avg_coverage: {summary['avg_coverage']:.4f}",
                f"- avg_motion_token_length: {summary['avg_motion_token_length']:.2f}",
                f"- novel_bigram_rate: {summary['novel_bigram_rate']:.4f}",
                f"- coverage_buckets: {summary['coverage_buckets']}",
                f"- length_buckets: {summary['length_buckets']}",
                "",
            ]
        )
    return "\n".join(lines)


def save_report(
    output_dir: str | Path,
    train_summary: Dict[str, object],
    dev_summary: Optional[Dict[str, object]],
    test_summary: Optional[Dict[str, object]],
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report = {
        "train": train_summary,
        "dev": dev_summary,
        "test": test_summary,
    }
    with open(output_path / "generalization_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with open(output_path / "generalization_report.md", "w", encoding="utf-8") as handle:
        handle.write(render_report_markdown(train_summary, dev_summary, test_summary))
