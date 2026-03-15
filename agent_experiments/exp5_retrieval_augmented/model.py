"""
Experiment 5 -- Retrieval-Augmented Generation (RAG) for Sign Language

REFERENCES
----------
[1] Zuo et al., "Signs as Tokens: A Retrieval-Enhanced Multilingual Sign
    Language Generator" (SOKE), ICCV 2025.
    Key idea: retrieve word-level signs from a dictionary as auxiliary
    conditions to improve sentence-level generation precision.

[2] MOGO: "Textual Condition Alignment" uses LLM chain-of-thought to
    bridge real-world prompts to training data, arXiv:2506.05952, 2025.

[3] Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive
    NLP Tasks", NeurIPS 2020.
    Key idea: condition generation on retrieved relevant documents.

[4] wSignGen: "Word-Conditioned 3D American Sign Language Motion Generation",
    EMNLP Findings 2024.
    Key idea: word-level sign motion as compositional building blocks.

CORE IDEA
---------
The current pipeline trains the LLM on sentence-level data but has no
mechanism to leverage the rich word-level knowledge learned during earlier
training stages.  When the LLM encounters an unseen sentence, it cannot
decompose it into known word-level signs.

We introduce a **Retrieval-Augmented Generation** approach:

1. **Build a Word-Sign Dictionary**: from the word-level training data,
   create an index mapping each word to its representative motion token
   sequence(s).

2. **At Training Time**: for each sentence, extract key words, look up
   their word-level motion patterns, and inject these as *context tokens*
   in the LLM prompt.  The format is:
   "[CONTEXT] <word1>: <M42> <M18> ... | <word2>: <M91> <M205> ... [/CONTEXT]"

3. **At Inference Time**: the same retrieval mechanism provides compositional
   hints for unseen sentences, even if the exact sentence was never seen.

WHY THIS SHOULD HELP GENERALIZATION
------------------------------------
1. **Compositional priming**: the retrieved word-level signs give the LLM
   a strong prior for what motion tokens to expect, reducing the search
   space from K^L (full vocabulary ^ sequence length) to a neighbourhood
   of known word patterns.

2. **Knowledge transfer**: word-level data (ASL Citizen, ~83K samples) is
   much richer than sentence-level (How2Sign, ~2K).  RAG bridges this gap
   by making word-level knowledge available at sentence generation time.

3. **Robustness to unseen vocabulary**: even if a word in the test sentence
   was never seen in sentence-level training, the word-level dictionary
   may contain it, providing a valid motion pattern.

4. **Reduced memorisation**: the model learns to *condition* on retrieved
   context rather than memorise exact input->output mappings, naturally
   improving generalisation.
"""

import os
import sys
import json
import re
import random
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ---------------------------------------------------------------------------
# Word-Sign Dictionary
# ---------------------------------------------------------------------------

class WordSignDictionary:
    """
    Index of word -> [motion_token_sequence, ...] built from word-level data.

    Supports multiple variants per word (from different signers), enabling
    the retrieval of diverse motion patterns.
    """

    def __init__(self):
        self.word_to_motions: Dict[str, List[str]] = defaultdict(list)
        self.word_to_representative: Dict[str, str] = {}

    def build_from_data(self, word_data: List[Dict[str, Any]]):
        """Build dictionary from word-level dataset entries."""
        for item in word_data:
            word = str(item.get("word", "")).strip().lower()
            tokens = str(item.get("motion_tokens", "")).strip()
            if not word or not tokens:
                continue
            self.word_to_motions[word].append(tokens)

        for word, variants in self.word_to_motions.items():
            lengths = [len(v.split()) for v in variants]
            median_idx = np.argsort(lengths)[len(lengths) // 2]
            self.word_to_representative[word] = variants[median_idx]

        print(f"[Dictionary] Built with {len(self.word_to_motions)} words, "
              f"avg {np.mean([len(v) for v in self.word_to_motions.values()]):.1f} variants/word")

    def lookup(self, word: str, max_variants: int = 1) -> List[str]:
        """Look up motion sequences for a word."""
        word = word.strip().lower()
        motions = self.word_to_motions.get(word, [])
        if not motions:
            return []
        if max_variants >= len(motions):
            return motions
        return random.sample(motions, max_variants)

    def lookup_representative(self, word: str) -> Optional[str]:
        """Get the median-length representative sequence for a word."""
        return self.word_to_representative.get(word.strip().lower())

    def save(self, path: str):
        data = {
            "word_to_motions": dict(self.word_to_motions),
            "word_to_representative": self.word_to_representative,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.word_to_motions = defaultdict(list, data["word_to_motions"])
        self.word_to_representative = data["word_to_representative"]


# ---------------------------------------------------------------------------
# Retrieval Module
# ---------------------------------------------------------------------------

class SignRetriever:
    """
    Retrieves word-level motion patterns for a given sentence.

    Strategy:
    1. Tokenise the sentence into words
    2. Filter stop words and very short words
    3. Look up each remaining word in the dictionary
    4. Format as context tokens for the LLM prompt
    """

    STOP_WORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "and", "or", "but", "not", "no", "if", "then", "so",
        "it", "its", "i", "me", "my", "we", "our", "you", "your",
        "he", "him", "his", "she", "her", "they", "them", "their",
        "this", "that", "these", "those",
    }

    def __init__(
        self,
        dictionary: WordSignDictionary,
        max_context_words: int = 5,
        max_tokens_per_word: int = 15,
        context_start: str = "[SIGN_CONTEXT]",
        context_end: str = "[/SIGN_CONTEXT]",
    ):
        self.dictionary = dictionary
        self.max_context_words = max_context_words
        self.max_tokens_per_word = max_tokens_per_word
        self.context_start = context_start
        self.context_end = context_end

    def extract_keywords(self, sentence: str) -> List[str]:
        """Extract content words from a sentence."""
        words = re.findall(r'[a-zA-Z]+', sentence.lower())
        content = [w for w in words if w not in self.STOP_WORDS and len(w) > 2]
        return content

    def retrieve(self, sentence: str) -> str:
        """
        Retrieve word-level motion patterns and format as context string.

        Returns a string like:
        [SIGN_CONTEXT] hello: <M42> <M18> | world: <M91> <M205> [/SIGN_CONTEXT]
        """
        keywords = self.extract_keywords(sentence)

        context_parts = []
        for word in keywords[:self.max_context_words]:
            motion = self.dictionary.lookup_representative(word)
            if motion:
                motion_tokens = motion.split()[:self.max_tokens_per_word]
                wrapped = " ".join(f"<M{t}>" for t in motion_tokens)
                context_parts.append(f"{word}: {wrapped}")

        if not context_parts:
            return ""

        return f"{self.context_start} {' | '.join(context_parts)} {self.context_end}"


# ---------------------------------------------------------------------------
# RAG-Enhanced Dataset
# ---------------------------------------------------------------------------

class RAGSentenceDataset(torch.utils.data.Dataset):
    """
    Sentence-level dataset with retrieval-augmented context injection.

    Each training sample gets a context prefix containing retrieved
    word-level motion patterns, teaching the LLM to condition on these hints.
    """

    TEMPLATES = [
        "Instruction: Generate sign language motion for the sentence: '{text}'",
        "Instruction: Translate this text to sign language: '{text}'",
        "Instruction: How would you sign '{text}'?",
        "Instruction: Create a sign language animation for: '{text}'",
    ]

    def __init__(
        self,
        sentence_data: List[Dict[str, Any]],
        retriever: SignRetriever,
        include_context_prob: float = 0.8,
        m_start: str = "<M_START>",
        m_end: str = "<M_END>",
    ):
        self.items = []
        self.retriever = retriever
        self.include_context_prob = include_context_prob
        self.m_start = m_start
        self.m_end = m_end

        for item in sentence_data:
            text = (item.get("text") or item.get("sentence", "")).strip()
            tokens_str = item.get("motion_tokens", "")
            if not text or not tokens_str.strip():
                continue

            motion_len = len(tokens_str.split())
            if motion_len < 30:
                len_tok = "<|LEN_SHORT|>"
            elif motion_len < 80:
                len_tok = "<|LEN_MEDIUM|>"
            else:
                len_tok = "<|LEN_LONG|>"

            context = retriever.retrieve(text)

            for tmpl in self.TEMPLATES:
                instruction = tmpl.format(text=text)
                self.items.append({
                    "instruction": instruction,
                    "len_tok": len_tok,
                    "context": context,
                    "motion_tokens": tokens_str,
                    "text": text,
                    "motion_length": motion_len,
                })

        print(f"[RAGDataset] {len(self.items)} items from {len(sentence_data)} sentences")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        wrapped = " ".join(f"<M{t}>" for t in item["motion_tokens"].split())
        target = f"{self.m_start} {wrapped} {self.m_end}"

        if item["context"] and random.random() < self.include_context_prob:
            prompt = f"{item['context']}\n{item['instruction']} (Length: {item['len_tok']})\nMotion: "
        else:
            prompt = f"{item['instruction']} (Length: {item['len_tok']})\nMotion: "

        return {
            "prompt": prompt,
            "full_text": prompt + target,
            "text": item["text"],
            "motion_length": item["motion_length"],
        }


# ---------------------------------------------------------------------------
# RAG-Enhanced Inference
# ---------------------------------------------------------------------------

class RAGInference:
    """
    At inference time, retrieve word-level signs and use them as context
    for the LLM prompt, mirroring the training setup.
    """

    def __init__(
        self,
        model,
        tokenizer,
        retriever: SignRetriever,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.device = device

    def generate(
        self,
        sentence: str,
        len_token: str = "<|LEN_MEDIUM|>",
        max_new_tokens: int = 180,
        temperature: float = 0.3,
        use_context: bool = True,
    ) -> str:
        """Generate motion tokens for a sentence with retrieval context."""
        context = self.retriever.retrieve(sentence) if use_context else ""
        template = random.choice(RAGSentenceDataset.TEMPLATES)
        instruction = template.format(text=sentence)

        if context:
            prompt = f"{context}\n{instruction} (Length: {len_token})\nMotion: "
        else:
            prompt = f"{instruction} (Length: {len_token})\nMotion: "

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.convert_tokens_to_ids("<M_END>"),
            )

        generated = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                           skip_special_tokens=False)
        return generated
