"""
Main training pipeline for Motion LLM (Matched to test_overfit.py logic)
Run this script to execute the full training process matching the reference implementation.
"""
import os
import random
import torch
import json
import argparse
from types import SimpleNamespace
import warnings
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np

# Import updated modules
from config import (
    SEED, DATA_JSON_PATH, MODEL_NAME, PIPELINE_OUTPUT_DIR,
    HF_STAGE1_REPO_ID, HF_STAGE2_REPO_ID, HF_STAGE2_SAVE_SUBDIR,
    HF_STAGE3_REPO_ID, HF_STAGE3_SAVE_SUBDIR,
    FORCE_STAGE2_FROM_STAGE1, HF_USE_HUB, HF_TOKEN,
    EVALUATION_WORDS, EVAL_SAMPLE_LIMIT, RUN_EVALS_ONLY,
    TEST_EVAL_OUTPUT_DIR, TEST_EVAL_DOWNLOAD_DIR, TEST_EVAL_EXTRACT_DIR,
    TEST_EVAL_SAMPLE_LIMIT, TEST_EVAL_MAX_ZIPS, TEST_EVAL_HF_REPO, TEST_EVAL_HF_SUBFOLDER
)
from data import read_json_data, deduplicate_and_prepare_data, build_motion_vocab
from model import setup_model_and_tokenizer_raw, ensure_tokenizer_has_motion_tokens
from train import (
    train_stage1_raw, train_stage2_raw, train_stage3_instruct_raw, resolve_and_ensure_repo,
    repo_has_stage_latest, load_model_and_tokenizer_from_hf,
    download_training_state, repo_get_latest_epoch_subfolder,
    load_model_and_tokenizer_from_hf_subfolder, download_training_state_from_subfolder
)
from metrics import (
    evaluate_metrics_encoder_style, run_inference_on_all_samples,
    evaluate_metrics_motiongpt_style, save_side_by_side_visualizations,
    evaluate_stage3_multiref_encoder_style,  # <-- add this
)
import test_dataset_eval

# Suppress warnings
warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser(description="Train SignMotionGPT pipeline stages (1/2/3) with HF resume support.")
    p.add_argument(
        "--stage",
        type=str,
        default=os.environ.get("TRAIN_STAGE", "all"),
        choices=["1", "2", "3", "all"],
        help="Which stage(s) to run: 1, 2, 3, or all (default: all).",
    )
    p.add_argument("--skip-eval", action="store_true", help="Skip evaluation/metrics after training.")
    p.add_argument("--skip-test-eval", action="store_true", help="Skip held-out test dataset evaluation step.")
    # Held-out test dataset inputs (forwarded to test_dataset_eval.py)
    p.add_argument("--test-local-extracted-dir", type=str, default=None, help="Directory containing extracted `video_data.pkl` files.")
    p.add_argument("--test-drive-url", type=str, default=None, help="Google Drive folder URL containing the held-out test zips.")
    p.add_argument("--test-drive-id", type=str, default=None, help="Google Drive folder ID containing the held-out test zips.")
    p.add_argument("--test-hf-repo-id", type=str, default=TEST_EVAL_HF_REPO, help="HF repo for test eval model loading.")
    p.add_argument("--test-hf-subfolder", type=str, default=TEST_EVAL_HF_SUBFOLDER, help="HF subfolder checkpoint for test eval model loading.")
    p.add_argument("--test-sample-limit", type=int, default=TEST_EVAL_SAMPLE_LIMIT, help="Max held-out samples for test eval.")
    p.add_argument("--test-max-zips", type=int, default=TEST_EVAL_MAX_ZIPS, help="Max zip archives to extract if downloading from Drive.")
    return p.parse_args()


def _dir_contains_video_pkl(path: str) -> bool:
    try:
        for root, _dirs, files in os.walk(path):
            if "video_data.pkl" in files:
                return True
        return False
    except Exception:
        return False


def _dir_contains_zip_files(path: str) -> bool:
    try:
        for root, _dirs, files in os.walk(path):
            for f in files:
                if f.lower().endswith(".zip"):
                    return True
        return False
    except Exception:
        return False


def _load_stage_from_hf(resolved_repo: str, stage_subdir: str):
    """
    Best-effort load of (model, tokenizer, start_epoch) from HF for a given stage folder.
    Supports both {stage}/latest and fallback {stage}/epoch-XXX if latest is missing.
    """
    start_epoch = 0
    loaded = None
    if not resolved_repo:
        return None, 0

    if repo_has_stage_latest(resolved_repo, stage_subdir, HF_TOKEN):
        loaded = load_model_and_tokenizer_from_hf(resolved_repo, stage_subdir, HF_TOKEN)
        state = download_training_state(resolved_repo, stage_subdir, HF_TOKEN)
        if state and isinstance(state.get("epoch_completed"), int):
            start_epoch = state["epoch_completed"]
        return loaded, start_epoch

    latest_sub = repo_get_latest_epoch_subfolder(resolved_repo, stage_subdir, HF_TOKEN)
    if latest_sub:
        loaded = load_model_and_tokenizer_from_hf_subfolder(resolved_repo, latest_sub, HF_TOKEN)
        state = download_training_state_from_subfolder(resolved_repo, latest_sub, HF_TOKEN)
        if state and isinstance(state.get("epoch_completed"), int):
            start_epoch = state["epoch_completed"]
    return loaded, start_epoch

def main():
    """Main function to run the entire pipeline matching test_overfit.py."""
    args = parse_args()
    print("="*80)
    print("      Motion LLM Training Pipeline (Matches test_overfit.py)")
    print("="*80)

    # Set seeds
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load ALL data
    print(f"\n[1/6] Loading dataset from {DATA_JSON_PATH}...")
    all_entries = read_json_data(DATA_JSON_PATH)
    
    # 2. Clean the ENTIRE dataset and get all tokens
    print("\n[2/6] Cleaning dataset...")
    cleaned_data, all_motion_tokens = deduplicate_and_prepare_data(all_entries)
    unique_words = sorted({str(item.get("word", "")).lower().strip() for item in cleaned_data if str(item.get("word", "")).strip()})
    print(f"\n📌 Unique words in cleaned dataset: {len(unique_words)}")

    do_s1 = args.stage in ("1", "all")
    do_s2 = args.stage in ("2", "all")
    do_s3 = args.stage in ("3", "all")

    # We'll track the "current" model/tokenizer as we progress through stages
    current_model = None
    tokenizer = None

    # 3. Stage 1: Initialize or resume from HF, then train (optional)
    motion_model = None
    resolved_stage1_repo = resolve_and_ensure_repo(HF_STAGE1_REPO_ID, HF_TOKEN) if HF_USE_HUB else None
    if do_s1:
        print("\n[3/6] Stage 1 Setup & Training...")
        stage1_loaded, start_epoch_s1 = _load_stage_from_hf(resolved_stage1_repo, "stage1")
        if stage1_loaded:
            base_model, tokenizer = stage1_loaded
            added = ensure_tokenizer_has_motion_tokens(tokenizer, all_motion_tokens)
            if added > 0:
                base_model.resize_token_embeddings(len(tokenizer))
        else:
            base_model, tokenizer = setup_model_and_tokenizer_raw(MODEL_NAME, all_motion_tokens)

        print(f"\nStarting Stage 1 training on {len(cleaned_data)} samples (resume from epoch {start_epoch_s1}).")
        motion_model = train_stage1_raw(
            base_model,
            tokenizer,
            cleaned_data,
            device,
            start_epoch=start_epoch_s1,
            hf_repo_id=resolved_stage1_repo,
        )
        current_model = motion_model
    else:
        # If Stage 1 isn't being trained, set a safe baseline for downstream stages.
        # We'll prefer downstream checkpoints when available (Stage2/Stage3 loaders below).
        current_model, tokenizer = setup_model_and_tokenizer_raw(MODEL_NAME, all_motion_tokens)

    # 4. Stage 2: Initialize or resume from HF, then train
    final_model = current_model
    last_trained_stage = "stage1" if do_s1 else "base"

    resolved_stage2_repo = resolve_and_ensure_repo(HF_STAGE2_REPO_ID, HF_TOKEN) if HF_USE_HUB else None
    if do_s2:
        print("\n[4/6] Stage 2 Setup & Training...")
        print(f"Stage 2 resume policy: FORCE_STAGE2_FROM_STAGE1={FORCE_STAGE2_FROM_STAGE1}, save_subdir='{HF_STAGE2_SAVE_SUBDIR}'")

        stage2_loaded, start_epoch_s2 = (None, 0)
        if not FORCE_STAGE2_FROM_STAGE1 and resolved_stage2_repo:
            stage2_loaded, start_epoch_s2 = _load_stage_from_hf(resolved_stage2_repo, HF_STAGE2_SAVE_SUBDIR)

        if stage2_loaded:
            stage2_model, tokenizer = stage2_loaded
            added2 = ensure_tokenizer_has_motion_tokens(tokenizer, all_motion_tokens)
            if added2 > 0:
                stage2_model.resize_token_embeddings(len(tokenizer))
        else:
            stage2_model = motion_model if motion_model is not None else current_model

        print(f"\nStarting Stage 2 training on {len(cleaned_data)} samples (resume from epoch {start_epoch_s2}).")
        final_model = train_stage2_raw(
            stage2_model,
            tokenizer,
            cleaned_data,
            device,
            start_epoch=start_epoch_s2,
            hf_repo_id=resolved_stage2_repo,
            hf_stage_subdir=HF_STAGE2_SAVE_SUBDIR,
        )
        last_trained_stage = "stage2"

    # 4.5. Stage 3: Instruct tuning (optional)
    resolved_stage3_repo = resolve_and_ensure_repo(HF_STAGE3_REPO_ID, HF_TOKEN) if HF_USE_HUB else None
    if do_s3:
        print("\n[4.5/6] Stage 3 Setup & Training (Instruct)...")
        stage3_loaded, start_epoch_s3 = (None, 0)
        if resolved_stage3_repo:
            stage3_loaded, start_epoch_s3 = _load_stage_from_hf(resolved_stage3_repo, HF_STAGE3_SAVE_SUBDIR)

        if stage3_loaded:
            stage3_model, tokenizer = stage3_loaded
            added3 = ensure_tokenizer_has_motion_tokens(tokenizer, all_motion_tokens)
            if added3 > 0:
                stage3_model.resize_token_embeddings(len(tokenizer))
        else:
            # Fallback chain: Stage 2 (already trained in this run) -> Stage 2 from HF -> current_model
            stage3_model = final_model
            if (not do_s2) and resolved_stage2_repo:
                stage2_fallback_loaded, _ = _load_stage_from_hf(resolved_stage2_repo, HF_STAGE2_SAVE_SUBDIR)
                if stage2_fallback_loaded:
                    stage3_model, tokenizer = stage2_fallback_loaded
                    added3b = ensure_tokenizer_has_motion_tokens(tokenizer, all_motion_tokens)
                    if added3b > 0:
                        stage3_model.resize_token_embeddings(len(tokenizer))

        print(f"\nStarting Stage 3 training (instruct) (resume from epoch {start_epoch_s3}).")
        final_model = train_stage3_instruct_raw(
            stage3_model,
            tokenizer,
            cleaned_data,
            device,
            start_epoch=start_epoch_s3,
            hf_repo_id=resolved_stage3_repo,
            hf_stage_subdir=HF_STAGE3_SAVE_SUBDIR,
        )
        last_trained_stage = "stage3"
    
    # Save final model locally
    if not os.path.exists(PIPELINE_OUTPUT_DIR):
        os.makedirs(PIPELINE_OUTPUT_DIR)
    final_model.save_pretrained(PIPELINE_OUTPUT_DIR)
    tokenizer.save_pretrained(PIPELINE_OUTPUT_DIR)
    print(f"Model saved to {PIPELINE_OUTPUT_DIR}")

    # 5. Evaluation on Specific Words
    if args.skip_eval:
        print("\n[5/6] Evaluation skipped (--skip-eval).")
        evaluation_data = []
    else:
        print("\n[5/6] Evaluation on Specific Words...")
        print("--- Filtering data for evaluation on specific words ---")
        evaluation_data = [item for item in cleaned_data if item['word'].lower() in EVALUATION_WORDS]
        print(f"Found {len(evaluation_data)} samples for evaluation words: {EVALUATION_WORDS}")

    metrics_json_path = os.path.join(PIPELINE_OUTPUT_DIR, "metrics.json")

    # 6. Metrics-only mode or full flow
    include_participant_in_prompt = (last_trained_stage != "stage3")

    # Stage 3: use the new multi-ref encoder-only eval
    if (not args.skip_eval) and (last_trained_stage == "stage3"):
        stage3_metrics = evaluate_stage3_multiref_encoder_style(
            final_model,
            tokenizer,
            evaluation_data,
            device,
            k_samples=10,
            sample_limit=None,  # use all eval rows for those words
        )

        os.makedirs(PIPELINE_OUTPUT_DIR, exist_ok=True)
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(stage3_metrics, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Saved Stage 3 multi-ref metrics to {metrics_json_path}")

        if RUN_EVALS_ONLY:
            return

        # Visualization: one GT (closest ref) vs GEN (best-of-K) per word
        viz_dir = os.path.join(PIPELINE_OUTPUT_DIR, "html_visualizations")
        save_side_by_side_visualizations(stage3_metrics.get("pairs_closest", []), viz_dir, limit=4)

    # Stage 1/2: keep existing eval behavior
    elif (not args.skip_eval) and RUN_EVALS_ONLY:
        metrics_enc = evaluate_metrics_encoder_style(
            final_model, tokenizer, evaluation_data, device,
            sample_limit=EVAL_SAMPLE_LIMIT,
            include_participant=include_participant_in_prompt,
        )
        os.makedirs(PIPELINE_OUTPUT_DIR, exist_ok=True)
        metrics_payload = {
            "source": "vqvae_encoder",
            "fid": metrics_enc.get("fid"),
            "diversity": {
                "ground_truth": metrics_enc.get("diversity_gt"),
                "model": metrics_enc.get("diversity_gen"),
            },
            "multimodality": {
                "ground_truth": metrics_enc.get("mim_gt"),
                "model": metrics_enc.get("mim_gen"),
            },
            "num_pairs": len(metrics_enc.get("pairs", [])),
        }
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Saved metrics to {metrics_json_path}")
        return

    elif not args.skip_eval:
        run_inference_on_all_samples(final_model, tokenizer, evaluation_data, device, include_participant=include_participant_in_prompt)
        metrics_token = evaluate_metrics_motiongpt_style(
            final_model, tokenizer, evaluation_data, all_motion_tokens, device, include_participant=include_participant_in_prompt
        )
        metrics_enc = evaluate_metrics_encoder_style(
            final_model, tokenizer, evaluation_data, device, sample_limit=EVAL_SAMPLE_LIMIT, include_participant=include_participant_in_prompt
        )
        viz_dir = os.path.join(PIPELINE_OUTPUT_DIR, "html_visualizations")
        save_side_by_side_visualizations(metrics_token["pairs"], viz_dir, limit=4)

    # 7. Run Test Dataset Evaluation (test_dataset_eval.py)
    if args.skip_test_eval:
        print("\n[6/6] Held-out test dataset evaluation skipped (--skip-test-eval).")
    else:
        print("\n[6/6] Running Evaluation on Held-out Test Dataset...")
    if not args.skip_test_eval:
        try:
            # Construct args matching test_dataset_eval.parse_args requirements:
            # must provide exactly one of {drive_url, drive_id, local_extracted_dir}.
            local_dir = args.test_local_extracted_dir
            if (not local_dir) and os.path.isdir(TEST_EVAL_EXTRACT_DIR) and _dir_contains_video_pkl(TEST_EVAL_EXTRACT_DIR):
                local_dir = TEST_EVAL_EXTRACT_DIR

            drive_url = args.test_drive_url
            drive_id = args.test_drive_id

            # If user didn't provide anything, skip with a clear message
            if not local_dir and not drive_url and not drive_id:
                print("⚠️  Skipping test_dataset_eval: provide --test-local-extracted-dir OR --test-drive-url/--test-drive-id.")
            else:
                # Special case: user provided a "local dir" that contains zips (not extracted yet).
                # `test_dataset_eval.run_evaluation` expects local_extracted_dir to contain video_data.pkl somewhere underneath.
                if local_dir and os.path.isdir(local_dir) and (not _dir_contains_video_pkl(local_dir)) and _dir_contains_zip_files(local_dir):
                    extract_root = os.path.join(TEST_EVAL_EXTRACT_DIR, "from_local_zips")
                    os.makedirs(extract_root, exist_ok=True)
                    print(f"Detected zip archives in local dir (no video_data.pkl found). Extracting to: {extract_root}")
                    zips = test_dataset_eval.list_zip_files(local_dir)
                    if not zips:
                        print("⚠️  No zip files found under --test-local-extracted-dir.")
                    else:
                        # Extract and then point evaluation to the extraction root
                        test_dataset_eval.extract_zip_files(zips, extract_root, limit=args.test_max_zips)
                        local_dir = extract_root

                print("Calling test_dataset_eval.run_evaluation...")
                # If user didn't override the default test HF subfolder and we just ran Stage 3,
                # prefer evaluating the Stage 3 checkpoint.
                hf_subfolder = args.test_hf_subfolder
                if (last_trained_stage == "stage3") and (hf_subfolder == TEST_EVAL_HF_SUBFOLDER):
                    hf_subfolder = f"{HF_STAGE3_SAVE_SUBDIR}/latest"

                eval_args = SimpleNamespace(
                    drive_url=drive_url,
                    drive_id=drive_id,
                    local_extracted_dir=local_dir,
                    download_dir=TEST_EVAL_DOWNLOAD_DIR,
                    extract_dir=TEST_EVAL_EXTRACT_DIR,
                    max_zips=args.test_max_zips,
                    hf_repo_id=args.test_hf_repo_id,
                    hf_subfolder=hf_subfolder,
                    vqvae_ckpt=None,
                    stats_path=None,
                    output_dir=TEST_EVAL_OUTPUT_DIR,
                    sample_limit=args.test_sample_limit,
                    seed=SEED,
                    # Stage 3-specific test eval: word-only prompt + K samples per test item
                    include_participant_in_prompt=(last_trained_stage != "stage3"),
                    k_samples=(10 if last_trained_stage == "stage3" else 1),
                )
                test_dataset_eval.run_evaluation(eval_args)
        except Exception as e:
            print(f"⚠️  Test dataset evaluation failed: {e}")

    print("\n" + "="*60)
    print("Training pipeline complete!")
    print("="*60)
    print(f"Models saved to: {PIPELINE_OUTPUT_DIR}")

if __name__ == "__main__":
    main()
