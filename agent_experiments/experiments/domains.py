from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple


def _bucket(value: int, edges: List[int]) -> str:
    for e in edges:
        if value < e:
            return f"<{e}"
    return f">={edges[-1]}"


def assign_pseudo_domains(sample: Dict[str, Any], group_keys: List[str]) -> Dict[str, str]:
    text = (sample.get("text") or sample.get("sentence") or "").strip()
    motion_tokens = sample.get("motion_tokens", "").split()
    participant = str(sample.get("participant_id", "unknown"))

    text_len = len(text.split())
    motion_len = len(motion_tokens)

    groups: Dict[str, str] = {}
    for key in group_keys:
        if key == "participant":
            groups[key] = participant
        elif key == "text_length_bucket":
            groups[key] = _bucket(text_len, [4, 8, 16, 32])
        elif key == "motion_length_bucket":
            groups[key] = _bucket(motion_len, [32, 64, 96, 128, 192])
        elif key == "lexical_richness_bucket":
            uniq = len(set(text.lower().split()))
            groups[key] = _bucket(uniq, [3, 6, 10, 16])
        else:
            groups[key] = "unknown"
    return groups


def summarize_group_losses(
    samples: Iterable[Dict[str, Any]],
    per_sample_loss: Iterable[float],
    group_keys: List[str],
) -> Dict[str, Dict[str, float]]:
    totals: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for sample, loss in zip(samples, per_sample_loss):
        g = assign_pseudo_domains(sample, group_keys)
        for key, value in g.items():
            totals[key][value] += float(loss)
            counts[key][value] += 1

    out: Dict[str, Dict[str, float]] = {}
    for key in totals:
        out[key] = {}
        for bucket in totals[key]:
            out[key][bucket] = totals[key][bucket] / max(1, counts[key][bucket])
    return out


def worst_group_score(group_stats: Dict[str, Dict[str, float]]) -> float:
    worst = float("-inf")
    for _, buckets in group_stats.items():
        for _, value in buckets.items():
            worst = max(worst, float(value))
    if worst == float("-inf"):
        return 0.0
    return worst

