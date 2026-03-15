from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Dict, Optional

import torch
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader

from .data_utils import default_json_path, extract_sentence_items, extract_word_items, load_json_entries
from .eval_harness import evaluate_vqvae_checkpoint
from .llm_utils import (
    AlignmentProjector,
    SentenceTokenDataset,
    WordReplayDataset,
    build_llm_model_and_tokenizer,
    build_word_bank,
    contrastive_alignment_loss,
    create_token_collate_fn,
    evaluate_token_model,
    move_batch_to_device,
    randomized_position_ids,
    set_seed,
)
from .training_utils import EarlyStopper, diagnostic_warnings, load_checkpoint_if_exists, model_payload, save_checkpoint


def _resume_model_state(model, checkpoint: Dict[str, Any]) -> None:
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)


def _select_eval_model(model, swa_model: Optional[AveragedModel], epoch: int, start_swa_epoch: int):
    if swa_model is not None and epoch >= start_swa_epoch:
        return swa_model
    return model


def run_llm_experiment(config, recipe: Dict[str, Any]) -> None:
    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_json = config.train_json or default_json_path()
    val_json = config.val_json or train_json
    train_entries = load_json_entries(train_json)
    val_entries = load_json_entries(val_json)
    word_entries = load_json_entries(config.word_json or train_json)
    word_bank = build_word_bank(extract_word_items(word_entries))

    model, tokenizer = build_llm_model_and_tokenizer(
        model_name=config.model_name,
        json_path=train_json,
        vqvae_ckpt=config.vqvae_ckpt,
        device=device,
    )

    alignment_head = None
    if recipe.get("use_alignment", False):
        alignment_head = AlignmentProjector(model.config.hidden_size, projection_dim=config.alignment_dim).to(device)

    collate_fn = create_token_collate_fn(
        tokenizer=tokenizer,
        max_length=config.max_seq_len,
        denoise_mask_token_id=tokenizer.convert_tokens_to_ids("<M_MASK>"),
    )
    train_dataset = SentenceTokenDataset(
        sentence_items=extract_sentence_items(train_entries),
        word_bank=word_bank,
        include_lexicon=recipe.get("use_lexicon", False),
        lexicon_context_prob=config.lexicon_context_prob,
        max_words=config.max_context_words,
        anchor_tokens=config.anchor_tokens,
        mask_rate=config.mask_rate,
        max_motion_tokens=config.max_motion_tokens,
    )
    val_dataset = SentenceTokenDataset(
        sentence_items=extract_sentence_items(val_entries),
        word_bank=word_bank,
        include_lexicon=recipe.get("use_lexicon", False),
        lexicon_context_prob=1.0,
        max_words=config.max_context_words,
        anchor_tokens=config.anchor_tokens,
        mask_rate=config.mask_rate,
        max_motion_tokens=config.max_motion_tokens,
    )
    replay_dataset = WordReplayDataset(extract_word_items(word_entries), max_motion_tokens=config.max_motion_tokens)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
    )
    replay_loader = DataLoader(
        replay_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
    ) if len(replay_dataset) > 0 else None

    params = list(model.parameters()) + (list(alignment_head.parameters()) if alignment_head is not None else [])
    optimizer = AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)

    start_epoch = 1
    best_worst = None
    best_avg = None
    resume_path = os.path.join(config.output_dir, "resume_state.pt")
    resume_state = load_checkpoint_if_exists(resume_path)
    swa_model = AveragedModel(model).to(device)
    start_swa_epoch = max(1, int(config.epochs * 0.8))

    if resume_state is not None:
        _resume_model_state(model, resume_state)
        if alignment_head is not None and resume_state.get("alignment_state_dict"):
            alignment_head.load_state_dict(resume_state["alignment_state_dict"], strict=False)
        if resume_state.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        if resume_state.get("swa_state_dict") is not None:
            swa_model.load_state_dict(resume_state["swa_state_dict"], strict=False)
        start_epoch = int(resume_state.get("epoch", 0)) + 1
        best_worst = resume_state.get("best_worst")
        best_avg = resume_state.get("best_avg")
        print(f"Resuming from epoch {start_epoch}.")

    early_stopper = EarlyStopper(config.patience, mode="min")
    if best_worst is not None:
        early_stopper.best_value = best_worst

    replay_iter = iter(replay_loader) if replay_loader is not None else None
    vq_eval_prev = None

    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        if alignment_head is not None:
            alignment_head.train()
        train_losses = []

        max_train_batches = 2 if config.dry_run else None
        for batch_idx, batch in enumerate(train_loader):
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break
            batch = move_batch_to_device(batch, device)
            position_ids = randomized_position_ids(batch["attention_mask"], config.max_position_offset) if recipe.get("use_random_positions", False) else None
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                position_ids=position_ids,
                output_hidden_states=alignment_head is not None,
            )
            loss = outputs.loss

            denoise_outputs = model(
                input_ids=batch["denoise_input_ids"],
                attention_mask=batch["denoise_attention_mask"],
                labels=batch["denoise_labels"],
            )
            loss = loss + config.denoise_weight * denoise_outputs.loss

            if alignment_head is not None and outputs.hidden_states is not None:
                align_loss = contrastive_alignment_loss(
                    hidden_states=outputs.hidden_states[-1],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    projector=alignment_head,
                    temperature=config.alignment_temperature,
                )
                loss = loss + config.alignment_weight * align_loss

            if replay_iter is not None and batch_idx % max(1, int(round(1.0 / config.replay_ratio))) == 0:
                try:
                    replay_batch = next(replay_iter)
                except StopIteration:
                    replay_iter = iter(replay_loader)
                    replay_batch = next(replay_iter)
                replay_batch = move_batch_to_device(replay_batch, device)
                replay_outputs = model(
                    input_ids=replay_batch["input_ids"],
                    attention_mask=replay_batch["attention_mask"],
                    labels=replay_batch["labels"],
                )
                loss = loss + replay_outputs.loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, config.max_grad_norm)
            optimizer.step()
            train_losses.append(float(loss.item()))

        if epoch >= start_swa_epoch:
            swa_model.update_parameters(model)

        eval_model = _select_eval_model(model, swa_model, epoch, start_swa_epoch)
        val_metrics = evaluate_token_model(
            model=eval_model,
            dataloader=val_loader,
            device=device,
            use_random_positions=recipe.get("use_random_positions", False),
            max_position_offset=config.max_position_offset,
            max_batches=2 if config.dry_run else None,
        )

        log_payload = {
            "epoch": epoch,
            "train_loss": sum(train_losses) / max(1, len(train_losses)),
            "val_avg_loss": val_metrics["avg_loss"],
            "val_worst_group": val_metrics["worst_group_score"],
        }

        if epoch % 5 == 0 or config.dry_run:
            print(json.dumps({"epoch": epoch, "bucket_metrics": val_metrics["bucket_metrics"]}, indent=2))

        if config.val_motion_dir and config.word_data_dir and (epoch % 5 == 0 or config.dry_run):
            vq_eval = evaluate_vqvae_checkpoint(
                vqvae_ckpt=config.vqvae_ckpt,
                val_dir=config.val_motion_dir,
                word_data_dir=config.word_data_dir,
                stats_path=config.stats_path,
                batch_size=config.eval_batch_size,
                max_samples=16 if config.dry_run else config.eval_max_motion_samples,
            )
            if vq_eval_prev is not None:
                vq_eval["prev_hand_mse"] = vq_eval_prev["hand_mse"]
                vq_eval["prev_body_mse"] = vq_eval_prev["body_mse"]
            diagnostic_warnings(vq_eval, prefix=f"epoch={epoch}")
            vq_eval_prev = vq_eval

        checkpoint_extra = {
            "alignment_state_dict": alignment_head.state_dict() if alignment_head is not None else None,
            "best_worst": best_worst,
            "best_avg": best_avg,
            "swa_state_dict": swa_model.state_dict(),
            "recipe": recipe,
        }
        if best_worst is None or val_metrics["worst_group_score"] < best_worst:
            best_worst = val_metrics["worst_group_score"]
            save_checkpoint(
                os.path.join(config.output_dir, "best_worst_group.pt"),
                model_payload(
                    eval_model,
                    optimizer,
                    epoch,
                    config,
                    {**log_payload, "bucket_metrics": val_metrics["bucket_metrics"]},
                    extra=checkpoint_extra,
                ),
            )
        if best_avg is None or val_metrics["avg_loss"] < best_avg:
            best_avg = val_metrics["avg_loss"]
            save_checkpoint(
                os.path.join(config.output_dir, "best_avg.pt"),
                model_payload(
                    eval_model,
                    optimizer,
                    epoch,
                    config,
                    {**log_payload, "bucket_metrics": val_metrics["bucket_metrics"]},
                    extra=checkpoint_extra,
                ),
            )

        save_checkpoint(
            resume_path,
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "alignment_state_dict": alignment_head.state_dict() if alignment_head is not None else None,
                "best_worst": best_worst,
                "best_avg": best_avg,
                "swa_state_dict": swa_model.state_dict(),
                "recipe": recipe,
                "config": asdict(config),
            },
        )

        print(json.dumps(log_payload, indent=2))
        should_stop = early_stopper.step(val_metrics["worst_group_score"])
        if should_stop:
            print("Early stopping triggered on worst-group validation score.")
            break
        if config.dry_run:
            print("Dry run finished after two batches.")
            break
