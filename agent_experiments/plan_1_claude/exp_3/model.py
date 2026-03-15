"""
LLM training utilities: masked denoising, word-sign dictionary, RAG.
Ref: MoMask (CVPR 2024), SOKE (ICCV 2025), RAG (NeurIPS 2020)
"""

import json
import os
import re
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F


class WordSignDictionary:
    """Maps words to representative motion token sequences from word-level data."""

    def __init__(self):
        self.word_to_motions = defaultdict(list)
        self.word_to_representative = {}

    def build_from_data(self, word_data):
        import numpy as np
        for item in word_data:
            word = str(item.get("word", "")).strip().lower()
            tokens = str(item.get("motion_tokens", "")).strip()
            if word and tokens:
                self.word_to_motions[word].append(tokens)
        for word, variants in self.word_to_motions.items():
            lengths = [len(v.split()) for v in variants]
            median_idx = sorted(range(len(lengths)), key=lambda i: lengths[i])[len(lengths) // 2]
            self.word_to_representative[word] = variants[median_idx]
        print(f"[Dictionary] {len(self.word_to_motions)} words")

    def lookup(self, word):
        return self.word_to_representative.get(word.strip().lower())

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"reps": self.word_to_representative}, f)

    def load(self, path):
        with open(path) as f:
            self.word_to_representative = json.load(f)["reps"]


STOP_WORDS = {"a", "an", "the", "is", "are", "was", "were", "be", "been",
              "to", "of", "in", "for", "on", "with", "at", "by", "from",
              "and", "or", "but", "not", "it", "i", "me", "my", "we",
              "you", "he", "she", "they", "this", "that", "do", "does"}


def retrieve_context(sentence, dictionary, max_words=5, max_tokens=15):
    words = re.findall(r'[a-zA-Z]+', sentence.lower())
    content = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    parts = []
    for w in content[:max_words]:
        m = dictionary.lookup(w)
        if m:
            toks = m.split()[:max_tokens]
            wrapped = " ".join(f"<M{t}>" for t in toks)
            parts.append(f"{w}: {wrapped}")
    if not parts:
        return ""
    return f"[SIGN_CONTEXT] {' | '.join(parts)} [/SIGN_CONTEXT]"


def apply_motion_mask(input_ids, tokenizer, mask_rate=0.15):
    """Randomly mask motion tokens for denoising objective."""
    m_start_id = tokenizer.convert_tokens_to_ids("<M_START>")
    m_end_id = tokenizer.convert_tokens_to_ids("<M_END>")
    mask_token_id = tokenizer.convert_tokens_to_ids("<M_MASK>")
    if mask_token_id == tokenizer.unk_token_id:
        return input_ids, torch.zeros_like(input_ids, dtype=torch.bool)

    masked = input_ids.clone()
    is_masked = torch.zeros_like(input_ids, dtype=torch.bool)

    for b in range(input_ids.shape[0]):
        in_motion = False
        for s in range(input_ids.shape[1]):
            tid = input_ids[b, s].item()
            if tid == m_start_id:
                in_motion = True
                continue
            if tid == m_end_id:
                in_motion = False
                continue
            if in_motion and random.random() < mask_rate:
                masked[b, s] = mask_token_id
                is_masked[b, s] = True
    return masked, is_masked
