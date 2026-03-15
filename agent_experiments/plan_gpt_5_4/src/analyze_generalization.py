from __future__ import annotations

import argparse

from .analysis import build_bigram_set, save_report, summarize_sentence_split
from .data import extract_sentence_items, extract_word_items, load_json_entries
from .retrieval import build_word_motion_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze lexical coverage and compositional difficulty.")
    parser.add_argument("--train-json", required=True, help="Training split JSON.")
    parser.add_argument("--word-json", required=True, help="Word-level or mixed JSON used for lexicon memory.")
    parser.add_argument("--dev-json", default=None, help="Optional dev split JSON.")
    parser.add_argument("--test-json", default=None, help="Optional test split JSON.")
    parser.add_argument("--output-dir", required=True, help="Directory to write reports.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_entries = load_json_entries(args.train_json)
    word_entries = load_json_entries(args.word_json)
    dev_entries = load_json_entries(args.dev_json) if args.dev_json else []
    test_entries = load_json_entries(args.test_json) if args.test_json else []

    train_sentence_items = extract_sentence_items(train_entries)
    dev_sentence_items = extract_sentence_items(dev_entries)
    test_sentence_items = extract_sentence_items(test_entries)
    word_items = extract_word_items(word_entries)
    word_bank = build_word_motion_bank(word_items)
    train_bigrams = build_bigram_set(train_sentence_items)

    train_summary = summarize_sentence_split(train_sentence_items, word_bank, train_bigrams)
    dev_summary = summarize_sentence_split(dev_sentence_items, word_bank, train_bigrams) if dev_sentence_items else None
    test_summary = summarize_sentence_split(test_sentence_items, word_bank, train_bigrams) if test_sentence_items else None
    save_report(args.output_dir, train_summary, dev_summary, test_summary)


if __name__ == "__main__":
    main()
