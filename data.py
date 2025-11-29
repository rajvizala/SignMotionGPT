"""
Dataset loading and vocabulary building utilities
"""
import json
import os
import random
from typing import List, Dict, Tuple, Any
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from config import M_START, M_END, PAD_TOKEN

# ======================================================================================
# Logic from test_overfit.py
# ======================================================================================

def read_json_data(json_path: str) -> List[Dict[str, Any]]:
    """Loads the dataset from the specified JSON file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset not found at: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def deduplicate_and_prepare_data(entries: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Cleans the entire dataset by ensuring each (word, participant_id) pair is unique.
    If a conflict is found (same pair, different motion), it keeps only the first one encountered.
    Then, it prepares the full list of motion tokens from the cleaned data.
    """
    print("\n---> Cleaning dataset by removing ambiguous (word, participant_id) pairs...")
    
    unique_samples = {}
    conflicts_found = 0
    
    for entry in entries:
        word = entry.get("word", "").lower()
        pid = entry.get("participant_id", "")
        key = (word, pid)
        
        if key not in unique_samples:
            unique_samples[key] = entry
        else:
            # A sample for this key already exists. We only care if it's a conflict.
            existing_tokens = unique_samples[key].get("motion_tokens")
            current_tokens = entry.get("motion_tokens")
            if existing_tokens != current_tokens:
                conflicts_found += 1
                # We do nothing, effectively discarding this new conflicting sample.
    
    cleaned_data = list(unique_samples.values())
    
    print(f"Original samples: {len(entries)}")
    print(f"Cleaned samples (unique (word, pid) pairs): {len(cleaned_data)}")
    print(f"Removed {len(entries) - len(cleaned_data)} total samples. ({conflicts_found} were direct conflicts).")

    print("\n---> Extracting motion tokens from the full cleaned dataset...")
    all_motion_tokens = set()
    for entry in cleaned_data:
        motion_tokens = entry.get("motion_tokens", "").strip().split()
        for token in motion_tokens:
            all_motion_tokens.add(f"<M{token}>")

    unique_tokens = sorted(list(all_motion_tokens))
    print(f"Found {len(unique_tokens)} unique motion tokens in the entire dataset.")
    
    return cleaned_data, unique_tokens

class MotionDataset(Dataset):
    """Dataset for Stage 1: Contains only motion token sequences."""
    def __init__(self, data: List[Dict[str, Any]], tokenizer: AutoTokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sequences = []

        for item in data:
            tokens_str = item.get("motion_tokens", "")
            wrapped_tokens = " ".join([f"<M{t}>" for t in tokens_str.split()])
            full_sequence = f"{M_START} {wrapped_tokens} {M_END}"
            self.sequences.append(full_sequence)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.tokenizer(
            self.sequences[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

class TextMotionDataset(Dataset):
    """Dataset for Stage 2: Contains (prompt, motion_sequence) pairs."""
    def __init__(self, data: List[Dict[str, Any]], tokenizer: AutoTokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.items = []

        for item in data:
            prompt = f"Instruction: Generate motion for word '{item['word']}' with variant '{item['participant_id']}'.\nMotion: "
            
            tokens_str = item.get("motion_tokens", "")
            wrapped_tokens = " ".join([f"<M{t}>" for t in tokens_str.split()])
            target_sequence = f"{M_START} {wrapped_tokens} {M_END}"
            
            full_text = prompt + target_sequence
            
            tokenized = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            prompt_tokenized = self.tokenizer(prompt, return_tensors="pt")
            prompt_len = prompt_tokenized.input_ids.shape[1]
            
            labels = tokenized['input_ids'].clone()
            labels[0, :prompt_len] = -100
            
            self.items.append({
                "input_ids": tokenized['input_ids'].squeeze(0),
                "attention_mask": tokenized['attention_mask'].squeeze(0),
                "labels": labels.squeeze(0)
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

# ======================================================================================
# Legacy utilities (kept for compatibility if needed, but mostly superseded)
# ======================================================================================

def build_motion_vocab(dataset):
    """
    Build motion vocabulary by finding max token ID
    Returns: (codebook_size, max_token_id)
    """
    def max_token_in_example(ex):
        return max(int(x) for x in ex["motion_tokens"].split())
    
    global_max_id = 0
    for ex in dataset:
        global_max_id = max(global_max_id, max_token_in_example(ex))
    
    codebook_size = global_max_id + 1
    return codebook_size, global_max_id

def motion_specials_to_ids(s: str) -> List[int]:
    """Extract motion IDs from special tokens"""
    toks = s.strip().split()
    ids = []
    for t in toks:
        if t.startswith("<motion_") or (t.startswith("<M") and t.endswith(">") and t[2:-1].isdigit()):
             # Handle both <motion_ID> and <MID> formats
            try:
                if t.startswith("<motion_"):
                    ids.append(int(t[8:-1]))
                else:
                    ids.append(int(t[2:-1]))
            except:
                pass
    return ids
