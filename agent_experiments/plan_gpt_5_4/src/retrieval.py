from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


WORD_RE = re.compile(r"[a-zA-Z']+")


def normalize_word(text: str) -> str:
    return re.sub(r"[^a-z0-9']+", "", str(text).lower().strip())


def tokenize_for_lexicon(text: str) -> List[str]:
    return [normalize_word(token) for token in WORD_RE.findall(str(text).lower()) if normalize_word(token)]


def length_bucket_from_count(token_count: int) -> str:
    if token_count < 30:
        return "short"
    if token_count < 80:
        return "medium"
    return "long"


def wrap_motion_tokens(tokens: Sequence[str]) -> str:
    return " ".join(f"<M{token}>" for token in tokens)


def compress_motion_sequence(tokens: Sequence[str], anchor_tokens: int = 6) -> str:
    clean = [str(token) for token in tokens if str(token).strip()]
    if not clean:
        return ""
    if len(clean) <= anchor_tokens:
        return wrap_motion_tokens(clean)
    if anchor_tokens <= 1:
        return f"<M{clean[0]}>"

    selected_indices = sorted(
        {
            round(i * (len(clean) - 1) / (anchor_tokens - 1))
            for i in range(anchor_tokens)
        }
    )
    selected = [clean[idx] for idx in selected_indices]
    return wrap_motion_tokens(selected)


@dataclass
class LexiconEntry:
    word: str
    representative_sequences: List[List[str]]
    median_length: int


def _select_representatives(sequences: List[List[str]], top_k: int) -> List[List[str]]:
    if not sequences:
        return []
    lengths = sorted(len(seq) for seq in sequences)
    median_length = lengths[len(lengths) // 2]
    ranked = sorted(sequences, key=lambda seq: (abs(len(seq) - median_length), len(seq)))
    unique: List[List[str]] = []
    seen = set()
    for seq in ranked:
        key = tuple(seq)
        if key in seen:
            continue
        seen.add(key)
        unique.append(seq)
        if len(unique) >= max(1, top_k):
            break
    return unique


def build_word_motion_bank(
    word_items: Iterable[dict],
    variants_per_word: int = 1,
) -> Dict[str, LexiconEntry]:
    grouped: Dict[str, List[List[str]]] = {}
    for item in word_items:
        word = normalize_word(item.get("word", ""))
        motion = str(item.get("motion_tokens", "")).strip().split()
        if not word or not motion:
            continue
        grouped.setdefault(word, []).append(motion)

    bank: Dict[str, LexiconEntry] = {}
    for word, sequences in grouped.items():
        representative_sequences = _select_representatives(sequences, variants_per_word)
        lengths = sorted(len(sequence) for sequence in sequences)
        bank[word] = LexiconEntry(
            word=word,
            representative_sequences=representative_sequences,
            median_length=lengths[len(lengths) // 2],
        )
    return bank


def retrieve_lexicon_entries(
    sentence_text: str,
    word_bank: Dict[str, LexiconEntry],
    max_words: int = 6,
) -> List[LexiconEntry]:
    seen = set()
    matches: List[LexiconEntry] = []
    for token in tokenize_for_lexicon(sentence_text):
        if token in seen:
            continue
        seen.add(token)
        if token in word_bank:
            matches.append(word_bank[token])
        if len(matches) >= max(1, max_words):
            break
    return matches


def coverage_ratio(sentence_text: str, word_bank: Dict[str, LexiconEntry]) -> float:
    tokens = tokenize_for_lexicon(sentence_text)
    if not tokens:
        return 0.0
    covered = sum(1 for token in tokens if token in word_bank)
    return covered / max(1, len(tokens))


def coverage_bucket(value: float) -> str:
    if value < 0.25:
        return "0.00-0.25"
    if value < 0.50:
        return "0.25-0.50"
    if value < 0.75:
        return "0.50-0.75"
    return "0.75-1.00"


def format_lexicon_memory(
    sentence_text: str,
    word_bank: Dict[str, LexiconEntry],
    max_words: int,
    anchor_tokens_per_variant: int,
    include_motion_anchors: bool,
    include_lengths: bool,
) -> str:
    entries = retrieve_lexicon_entries(sentence_text, word_bank, max_words=max_words)
    if not entries:
        return "Lexicon memory:\n- none"

    lines = ["Lexicon memory:"]
    for entry in entries:
        line = f"- word={entry.word}"
        if include_lengths:
            line += f" approx_len={entry.median_length}"
        if include_motion_anchors and entry.representative_sequences:
            anchor = compress_motion_sequence(
                entry.representative_sequences[0],
                anchor_tokens=anchor_tokens_per_variant,
            )
            if anchor:
                line += f" anchors={anchor}"
        lines.append(line)
    return "\n".join(lines)
