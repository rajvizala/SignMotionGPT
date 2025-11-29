"""
Wrapper around the `test_overfit` training stack so the main training entrypoint
(`train_pipeline.py`) matches the behaviour of the proven end-to-end script.
"""

import argparse
import os
from typing import Any, Dict, List, Optional

from config import (
    DATA_JSON_PATH,
    WORK_DIR,
    PIPELINE_MODEL_NAME,
    PIPELINE_OUTPUT_DIR,
    PIPELINE_EVAL_WORDS,
    PIPELINE_EVAL_SAMPLE_LIMIT,
    PIPELINE_RUN_EVALS_ONLY,
    PIPELINE_S1_EPOCHS,
    PIPELINE_S2_EPOCHS,
    PIPELINE_S1_LR,
    PIPELINE_S2_LR,
    PIPELINE_S1_BATCH,
    PIPELINE_S2_BATCH,
    PIPELINE_HF_STAGE2_SUBDIR,
    PIPELINE_FORCE_STAGE2_FROM_STAGE1,
    HF_TOKEN,
    HUB_REPO_S1,
    HUB_REPO_S2,
)

import test_overfit

DEFAULT_OUTPUT_DIR = PIPELINE_OUTPUT_DIR or os.path.join(WORK_DIR, "motion_gpt_full_model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the SignMotionGPT pipeline using the tested two-stage training flow.",
    )
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to motion_llm_dataset.json")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for checkpoints, metrics, and final model")
    parser.add_argument("--model-name", type=str, default=None, help="Base HF model to fine-tune")
    parser.add_argument("--eval-words", nargs="+", default=None, help="Override evaluation word list")
    parser.add_argument("--eval-sample-limit", type=int, default=None, help="Max samples for encoder metrics")
    parser.add_argument("--evals-only", action="store_true", help="Skip inference logs/visuals and only compute metrics")
    parser.add_argument("--full-train", action="store_true", help="Force full training even if eval-only flag is set via env/config")
    parser.add_argument("--s1-epochs", type=int, default=None, help="Stage 1 epochs override")
    parser.add_argument("--s2-epochs", type=int, default=None, help="Stage 2 epochs override")
    parser.add_argument("--s1-lr", type=float, default=None, help="Stage 1 learning rate override")
    parser.add_argument("--s2-lr", type=float, default=None, help="Stage 2 learning rate override")
    parser.add_argument("--s1-batch-size", type=int, default=None, help="Stage 1 batch size override")
    parser.add_argument("--s2-batch-size", type=int, default=None, help="Stage 2 batch size override")
    parser.add_argument("--checkpoint-upload-interval", type=int, default=None, help="HF upload cadence in epochs")
    parser.add_argument("--hf-stage1-repo", type=str, default=None, help="HF repo id for Stage 1 checkpoints")
    parser.add_argument("--hf-stage2-repo", type=str, default=None, help="HF repo id for Stage 2 checkpoints")
    parser.add_argument("--stage2-subdir", type=str, default=None, help="Subfolder name for Stage 2 checkpoints on HF")
    parser.add_argument("--force-stage2-from-stage1", action="store_true", help="Always restart Stage 2 from Stage 1 weights")
    parser.add_argument("--allow-stage2-resume", action="store_true", help="Permit Stage 2 to resume from Hub checkpoints")
    parser.add_argument("--hf-token", type=str, default=None, help="Explicit HF token (otherwise env/config is used)")
    parser.add_argument("--no-hub", action="store_true", help="Disable Hugging Face Hub syncing entirely")
    parser.add_argument("--show-hub-progress", action="store_true", help="Re-enable HF progress bars")
    return parser.parse_args()


def _clean_eval_words(words: Optional[List[str]]) -> Optional[List[str]]:
    if not words:
        return None
    cleaned = [w.strip() for w in words if w and w.strip()]
    return cleaned or None


def _resolve_token(cli_token: Optional[str]) -> Optional[str]:
    env_token = os.environ.get("hf_auth_token") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    return cli_token or env_token or (HF_TOKEN if HF_TOKEN else None)


def build_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}

    overrides["DATASET_PATH"] = args.dataset_path or DATA_JSON_PATH
    overrides["OUTPUT_DIR"] = args.output_dir or DEFAULT_OUTPUT_DIR
    overrides["MODEL_NAME"] = args.model_name or PIPELINE_MODEL_NAME or test_overfit.MODEL_NAME

    eval_words = _clean_eval_words(args.eval_words) or PIPELINE_EVAL_WORDS
    if eval_words:
        overrides["EVALUATION_WORDS"] = eval_words

    overrides["EVAL_SAMPLE_LIMIT"] = args.eval_sample_limit or PIPELINE_EVAL_SAMPLE_LIMIT

    run_evals_only = PIPELINE_RUN_EVALS_ONLY
    if args.evals_only:
        run_evals_only = True
    if args.full_train:
        run_evals_only = False
    overrides["RUN_EVALS_ONLY"] = run_evals_only

    overrides["S1_EPOCHS"] = args.s1_epochs or PIPELINE_S1_EPOCHS
    overrides["S2_EPOCHS"] = args.s2_epochs or PIPELINE_S2_EPOCHS
    overrides["S1_LR"] = args.s1_lr or PIPELINE_S1_LR
    overrides["S2_LR"] = args.s2_lr or PIPELINE_S2_LR
    overrides["S1_BATCH_SIZE"] = args.s1_batch_size or PIPELINE_S1_BATCH
    overrides["S2_BATCH_SIZE"] = args.s2_batch_size or PIPELINE_S2_BATCH

    if args.checkpoint_upload_interval:
        overrides["CHECKPOINT_UPLOAD_INTERVAL_EPOCHS"] = args.checkpoint_upload_interval

    if args.stage2_subdir or PIPELINE_HF_STAGE2_SUBDIR:
        overrides["HF_STAGE2_SAVE_SUBDIR"] = args.stage2_subdir or PIPELINE_HF_STAGE2_SUBDIR

    force_stage2 = PIPELINE_FORCE_STAGE2_FROM_STAGE1
    if args.force_stage2_from_stage1:
        force_stage2 = True
    if args.allow_stage2_resume:
        force_stage2 = False
    overrides["FORCE_STAGE2_FROM_STAGE1"] = force_stage2

    if args.hf_stage1_repo or HUB_REPO_S1:
        overrides["HF_STAGE1_REPO_ID"] = args.hf_stage1_repo or HUB_REPO_S1
    if args.hf_stage2_repo or HUB_REPO_S2:
        overrides["HF_STAGE2_REPO_ID"] = args.hf_stage2_repo or HUB_REPO_S2

    token = _resolve_token(args.hf_token)
    if token:
        overrides["hf_auth_token"] = token

    use_hub = not args.no_hub
    if use_hub and not token:
        # Without a token, skip Hub sync to avoid repeated warnings
        use_hub = False
    overrides["HF_USE_HUB"] = use_hub

    if args.show_hub_progress:
        overrides["HF_DISABLE_PROGRESS"] = False

    return overrides


def main():
    args = parse_args()
    overrides = build_overrides(args)
    print("Running SignMotionGPT pipeline with the test_overfit configuration...")
    test_overfit.main(overrides)


if __name__ == "__main__":
    main()
