"""
Training script for Experiment 5: Retrieval-Augmented Generation (RAG).

Usage
-----
    python -m agent_experiments.exp5_retrieval_augmented.train \
        --sentence-dataset /path/to/sentence_dataset.json \
        --word-dataset /path/to/word_level_dataset.json \
        --vqvae-ckpt /path/to/vqvae_checkpoint.pt \
        --output-dir ./agent_experiments/outputs/exp5 \
        [--context-prob 0.8] \
        [--max-context-words 5] \
        [--epochs 40]

WORKFLOW
-------
1. Build word-sign dictionary from word-level data
2. For each sentence, retrieve word-level motion patterns
3. Train LLM with context-augmented prompts
4. At inference, use same retrieval mechanism for unseen sentences
"""

import argparse
import json
import os
import random
import sys
import time
import warnings
from typing import Dict, List, Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from agent_experiments.shared.logging_utils import ExperimentLogger
from agent_experiments.shared.data_utils import load_json_data, split_by_type, extract_motion_vocab
from agent_experiments.exp5_retrieval_augmented.model import (
    WordSignDictionary, SignRetriever, RAGSentenceDataset, RAGInference,
)

warnings.filterwarnings("ignore")

M_START = "<M_START>"
M_END = "<M_END>"
PAD_TOKEN = "<PAD>"
LEN_SHORT = "<|LEN_SHORT|>"
LEN_MEDIUM = "<|LEN_MEDIUM|>"
LEN_LONG = "<|LEN_LONG|>"


def create_collate_fn(tokenizer, max_length=256):
    pad_id = tokenizer.pad_token_id
    m_start_id = tokenizer.convert_tokens_to_ids(M_START)

    def collate_fn(batch):
        full_texts = [item["full_text"] for item in batch]
        tokenized = tokenizer(full_texts, truncation=True, max_length=max_length,
                              padding=True, return_tensors="pt")
        labels = tokenized["input_ids"].clone()
        for i in range(len(full_texts)):
            ids = tokenized["input_ids"][i]
            positions = (ids == m_start_id).nonzero(as_tuple=True)[0]
            prompt_len = positions[0].item() if len(positions) > 0 else len(ids)
            labels[i, :prompt_len] = -100
            if pad_id is not None:
                labels[i, ids == pad_id] = -100
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }
    return collate_fn


def main():
    parser = argparse.ArgumentParser(description="Exp5: RAG for Sign Language")
    parser.add_argument("--sentence-dataset", required=True,
                        help="Path to sentence-level dataset JSON")
    parser.add_argument("--word-dataset", required=True,
                        help="Path to word-level dataset JSON")
    parser.add_argument("--vqvae-ckpt", required=True)
    parser.add_argument("--output-dir", default="./agent_experiments/outputs/exp5")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--context-prob", type=float, default=0.8,
                        help="Probability of including retrieval context (training)")
    parser.add_argument("--max-context-words", type=int, default=5)
    parser.add_argument("--max-tokens-per-word", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = ExperimentLogger(args.output_dir, "exp5_rag")
    logger.log_hyperparameters(vars(args))

    # -- Build word-sign dictionary ----------------------------------------
    print("\n[Step 1] Building word-sign dictionary...")
    word_data = load_json_data(args.word_dataset)
    word_data_filtered = [d for d in word_data if d.get("type", "word") != "sentence"]
    if not word_data_filtered:
        word_data_filtered = word_data

    dictionary = WordSignDictionary()
    dictionary.build_from_data(word_data_filtered)
    dict_path = os.path.join(args.output_dir, "word_sign_dictionary.json")
    dictionary.save(dict_path)
    print(f"  Dictionary saved to {dict_path}")

    retriever = SignRetriever(
        dictionary,
        max_context_words=args.max_context_words,
        max_tokens_per_word=args.max_tokens_per_word,
    )

    # -- Load sentence data -----------------------------------------------
    print("\n[Step 2] Loading sentence-level data...")
    raw_sentence = load_json_data(args.sentence_dataset)
    _, sentence_data = split_by_type(raw_sentence)
    if not sentence_data:
        sentence_data = raw_sentence

    n_val = max(1, int(len(sentence_data) * 0.1))
    rng = random.Random(args.seed)
    indices = list(range(len(sentence_data)))
    rng.shuffle(indices)
    val_data = [sentence_data[i] for i in indices[:n_val]]
    train_data = [sentence_data[i] for i in indices[n_val:]]

    # Collect all motion tokens from both datasets
    all_data = word_data + sentence_data
    motion_tokens = extract_motion_vocab(all_data)

    # -- Setup model -------------------------------------------------------
    print("\n[Step 3] Setting up model...")
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    context_tokens = ["[SIGN_CONTEXT]", "[/SIGN_CONTEXT]"]
    tokenizer.add_special_tokens({
        "pad_token": PAD_TOKEN,
        "additional_special_tokens": [
            M_START, M_END, LEN_SHORT, LEN_MEDIUM, LEN_LONG,
        ] + context_tokens,
    })
    tokenizer.add_tokens(motion_tokens, special_tokens=True)

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,
                                                  attn_implementation="sdpa")
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        model = model.to(torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.to(device)

    # -- Datasets ----------------------------------------------------------
    train_ds = RAGSentenceDataset(train_data, retriever,
                                   include_context_prob=args.context_prob)
    val_ds = RAGSentenceDataset(val_data, retriever,
                                 include_context_prob=1.0)

    collate_fn = create_collate_fn(tokenizer, max_length=384)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=2, pin_memory=True)

    # -- Training ----------------------------------------------------------
    print(f"\n[Step 4] Training for {args.epochs} epochs...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = len(train_loader)
        t0 = time.time()

        for i, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if i == 1 or i % max(1, n_batches // 10) == 0:
                print(f"\r  [Epoch {epoch}] {i}/{n_batches} loss={loss.item():.4f}",
                      end="", flush=True)
        print()
        scheduler.step()

        avg_loss = total_loss / n_batches

        model.eval()
        val_loss = 0.0
        n_val_b = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                n_val_b += 1
        val_loss /= max(1, n_val_b)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch}/{args.epochs}: train={avg_loss:.4f} val={val_loss:.4f} "
              f"lr={scheduler.get_last_lr()[0]:.2e} ({elapsed:.1f}s)")
        logger.log_epoch(epoch, avg_loss, val_loss, lr=scheduler.get_last_lr()[0])

        if val_loss < best_val:
            best_val = val_loss
            model.save_pretrained(os.path.join(args.output_dir, "best_model"))
            tokenizer.save_pretrained(os.path.join(args.output_dir, "best_model"))
            print(f"    -> New best val loss: {val_loss:.4f}")

    # -- Save inference helper ---------------------------------------------
    print("\n[Step 5] Inference example...")
    rag_inference = RAGInference(model, tokenizer, retriever, device)
    test_sentences = [
        "hello how are you",
        "thank you very much",
        "please help me",
    ]
    for sent in test_sentences:
        context = retriever.retrieve(sent)
        print(f"\n  Sentence: '{sent}'")
        print(f"  Retrieved context: {context}")
        generated = rag_inference.generate(sent, use_context=True)
        print(f"  Generated: {generated[:200]}")

    logger.log_final({"best_val_loss": best_val})
    print(f"\nExperiment 5 complete. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
