"""Train and compare the five educational models in ``learning_assets/model/``.

Example:
    uv run python -m train_architectures --architecture gqa \
        --config configs/architecture_tiny.json --text data/sample.txt \
        --output runs/gqa_tiny
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

from config import TrainingConfig, validate_resume_config
from benchmarking.compute import active_parameter_count, estimate_flops, estimate_training_flops
from data import load_token_artifacts
from data.datasets import make_loaders
from data.readers import read_documents
from data.splits import deduplicate_documents, split_documents_three
from data.tokenizer import build_tokenizer, tokenizer_from_state
from evaluation.loss import token_weighted_loss
from models.registry import (
    available_architectures,
    build_model,
    model_metadata,
    parameter_count,
)
from training.loop import restore_rng_state
from training.optim import build_adamw
from training.schedule import cosine_lr, set_optimizer_lr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one standalone educational architecture")
    parser.add_argument("--architecture", choices=available_architectures(), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--text", type=Path, default=None)
    parser.add_argument("--artifact-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-vocab-size", type=int, default=None)
    parser.add_argument("--tokenizer-min-frequency", type=int, default=None)
    parser.add_argument("--tokenizer", choices=("bpe", "tiktoken"), default=None)
    parser.add_argument("--tokenizer-name", choices=("gpt2", "r50k_base"), default="r50k_base")
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--device", default=None, help="cpu, cuda, or auto")
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def evaluate(model, loader, device, max_batches: int | None) -> float:
    """Return token-weighted next-token cross entropy."""
    return token_weighted_loss(model, loader, device, max_batches)


def sample_text(model, tokenizer, device, token_ids, max_tokens=32) -> str:
    """Greedy sample used as a qualitative checkpoint sanity check."""
    model.eval()
    context_length = model.training_cfg["context_length"]
    current = torch.tensor(token_ids[-context_length:], dtype=torch.long, device=device)[None]
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(current, use_cache=False)
            current = torch.cat([current, logits[:, -1:].argmax(dim=-1)], dim=1)
            current = current[:, -context_length:]
    return tokenizer.decode(current[0].tolist())


def _next_batch(loader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _validate_global_tokens_per_update(
    data_cfg: dict[str, object], training_cfg: TrainingConfig, context_length: int
) -> None:
    configured = data_cfg.get("global_tokens_per_update")
    if configured is None:
        return
    expected = training_cfg.batch_size * context_length * training_cfg.gradient_accumulation_steps
    if int(configured) != expected:
        raise ValueError(
            "global_tokens_per_update does not match batch_size * context_length * "
            f"gradient_accumulation_steps: {configured} != {expected}"
        )


def main() -> None:
    args = parse_args()
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    data_cfg = dict(raw.get("data", {}))
    model_cfg = dict(raw["model"])
    model_cfg.update(dict(raw.get("model_overrides", {}).get(args.architecture, {})))
    training_cfg = TrainingConfig(**raw["training"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(training_cfg.seed)
    checkpoint = (
        torch.load(args.resume, map_location=device, weights_only=False)
        if args.resume is not None
        else None
    )
    if checkpoint is not None and checkpoint["architecture"] != args.architecture:
        raise ValueError("Resume checkpoint architecture does not match --architecture")

    artifact_manifest = args.artifact_manifest or (
        Path(data_cfg["artifact_manifest"]) if data_cfg.get("artifact_manifest") else None
    )
    data_manifest = None
    data_contract = None
    if artifact_manifest is not None:
        artifacts = load_token_artifacts(artifact_manifest)
        train_fraction = float(artifacts.manifest["train_fraction"])
        validation_fraction = float(
            artifacts.manifest.get(
                "validation_fraction",
                artifacts.manifest["validation_document_count"]
                / artifacts.manifest["document_count"],
            )
        )
        tokenizer = (
            tokenizer_from_state(checkpoint["tokenizer"])
            if checkpoint is not None
            else artifacts.tokenizer
        )
        train_tokens = artifacts.train_tokens
        validation_tokens = artifacts.validation_tokens
        test_tokens = artifacts.test_tokens
        data_manifest = artifacts.manifest
        data_contract = artifacts.contract
    else:
        if args.text is None:
            raise ValueError("Provide --text or --artifact-manifest")
        documents = deduplicate_documents(read_documents(args.text))
        train_fraction = float(
            args.train_fraction if args.train_fraction is not None else raw.get("train_fraction", 0.8)
        )
        validation_fraction = float(raw.get("validation_fraction", 0.1))
        train_documents, validation_documents, test_documents = split_documents_three(
            documents, train_fraction, validation_fraction, training_cfg.seed
        )
        tokenizer_kind = args.tokenizer or str(data_cfg.get("tokenizer", "bpe"))
        tokenizer_vocab_size = int(args.tokenizer_vocab_size or data_cfg.get("tokenizer_vocab_size", 512))
        tokenizer_min_frequency = int(args.tokenizer_min_frequency or data_cfg.get("tokenizer_min_frequency", 2))
        tokenizer_name = args.tokenizer_name
        tokenizer = (
            tokenizer_from_state(checkpoint["tokenizer"])
            if checkpoint is not None
            else build_tokenizer(
                tokenizer_kind,
                train_documents,
                tokenizer_vocab_size,
                tokenizer_min_frequency,
                tokenizer_name,
            )
        )
        model_cfg["vocab_size"] = tokenizer.vocab_size
        train_tokens = tokenizer.encode_documents(train_documents)
        validation_tokens = tokenizer.encode_documents(validation_documents)
        test_tokens = tokenizer.encode_documents(test_documents)
    model_cfg = (
        dict(checkpoint["model_config"]) if checkpoint is not None else model_cfg
    )
    model_cfg["vocab_size"] = tokenizer.vocab_size
    if checkpoint is not None:
        if checkpoint.get("data_contract") != data_contract:
            raise ValueError("Resume data contract does not match the current data")
        if float(checkpoint.get("train_fraction", train_fraction)) != train_fraction:
            raise ValueError("Resume train_fraction does not match the current config")
        if float(checkpoint.get("validation_fraction", validation_fraction)) != validation_fraction:
            raise ValueError("Resume validation_fraction does not match the current config")
    context_length = model_cfg["context_length"]
    _validate_global_tokens_per_update(data_cfg, training_cfg, context_length)
    train_loader, val_loader = make_loaders(
        train_tokens, validation_tokens, context_length, training_cfg.batch_size, training_cfg.seed
    )
    _, test_loader = make_loaders(
        train_tokens, test_tokens, context_length, training_cfg.batch_size, training_cfg.seed
    )

    model = build_model(args.architecture, model_cfg).to(device)
    model_info = asdict(model_metadata(args.architecture, model))
    flop_estimate = estimate_flops(model, args.architecture)
    optimizer = build_adamw(model, training_cfg)
    start_step = 0
    history = []
    tokens_seen = 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = int(checkpoint["step"])
        validate_resume_config(checkpoint["training_config"], training_cfg, start_step)
        history = list(checkpoint.get("history", []))
        tokens_seen = int(checkpoint.get("tokens_seen", 0))
    use_amp = device.type == "cuda" and training_cfg.precision != "fp32"
    if training_cfg.precision == "bf16":
        amp_dtype = torch.bfloat16
    elif training_cfg.precision == "fp16":
        amp_dtype = torch.float16
    elif use_amp and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    if checkpoint is not None and checkpoint.get("scaler_state"):
        scaler.load_state_dict(checkpoint["scaler_state"])
    args.output.mkdir(parents=True, exist_ok=True)
    if checkpoint is not None and checkpoint.get("loader_state"):
        train_loader.load_state_dict(checkpoint["loader_state"]["train"])
        val_loader.load_state_dict(checkpoint["loader_state"]["validation"])
        test_loader.load_state_dict(checkpoint["loader_state"]["test"])
        restore_rng_state(checkpoint.get("rng_state", checkpoint))
    train_iterator = iter(train_loader)
    best_validation_loss = min(
        (float(row["validation_loss"]) for row in history),
        default=float("inf"),
    )
    best_model_step = max(
        (int(row["step"]) for row in history if float(row["validation_loss"]) == best_validation_loss),
        default=None,
    )
    start_time = time.perf_counter()

    def checkpoint_payload(step: int, test_loss: float | None = None) -> dict[str, object]:
        return {
            "architecture": args.architecture,
            "model_config": model_cfg,
            "training_config": asdict(training_cfg),
            "step": step,
            "tokens_seen": tokens_seen,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if use_scaler else None,
            "tokenizer": tokenizer.to_state(),
            "train_fraction": train_fraction,
            "validation_fraction": validation_fraction,
            "test_loss": test_loss,
            "test_model_selection": "best_validation" if test_loss is not None else None,
            "best_validation_step": best_model_step,
            "data_manifest": data_manifest,
            "data_contract": data_contract,
            "artifact_manifest": str(artifacts.manifest_path) if artifact_manifest is not None else None,
            "flops": flop_estimate.as_dict(),
            "estimated_training_flops": estimate_training_flops(
                model, args.architecture, tokens_seen
            ),
            "history": history,
            "loader_state": {
                "train": train_loader.state_dict(),
                "validation": val_loader.state_dict(),
                "test": test_loader.state_dict(),
            },
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "python": random.getstate(),
                "numpy": np.random.get_state(),
            },
            "parameters": parameter_count(model),
            "active_parameters": active_parameter_count(model, args.architecture),
            "model_metadata": model_info,
            "device": str(device),
        }

    for step in range(start_step + 1, training_cfg.max_steps + 1):
        model.train()
        learning_rate = cosine_lr(
            step,
            warmup_steps=training_cfg.warmup_steps,
            max_steps=training_cfg.max_steps,
            lr=training_cfg.learning_rate,
            min_lr=training_cfg.learning_rate * training_cfg.min_lr_ratio,
        )
        set_optimizer_lr(optimizer, learning_rate)
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        microbatches = []
        for _ in range(training_cfg.gradient_accumulation_steps):
            batch, train_iterator = _next_batch(train_loader, train_iterator)
            microbatches.append(batch)
        total_tokens = sum(targets.numel() for _, targets in microbatches)
        for inputs, targets in microbatches:
            inputs, targets = inputs.to(device), targets.to(device)
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if use_amp
                else nullcontext()
            )
            with autocast_context:
                logits = model(inputs, use_cache=False)
                token_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction="sum",
                )
            step_loss += token_loss.item()
            scaled_loss = token_loss / total_tokens
            if use_scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            tokens_seen += inputs.numel()
        if use_scaler:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training_cfg.grad_clip_norm)
        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if step == 1 or step % training_cfg.eval_every == 0 or step == training_cfg.max_steps:
            train_loss = evaluate(model, train_loader, device, training_cfg.eval_batches)
            validation_loss = evaluate(model, val_loader, device, training_cfg.eval_batches)
            record = {
                "step": step,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_perplexity": math.exp(min(validation_loss, 20.0)),
                "tokens_seen": tokens_seen,
                "estimated_training_flops": estimate_training_flops(
                    model, args.architecture, tokens_seen
                ),
                "estimated_training_flops_per_token": flop_estimate.training_flops_per_token,
                "learning_rate": learning_rate,
                "micro_train_loss": step_loss / total_tokens,
                "grad_norm": float(grad_norm),
                "elapsed_seconds": time.perf_counter() - start_time,
            }
            history.append(record)
            print(json.dumps(record))
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_model_step = step
                torch.save(model.state_dict(), args.output / "best_model.pt")
            print(json.dumps({"sample": sample_text(model, tokenizer, device, train_tokens[:context_length])}))

        if step % training_cfg.save_every == 0 or step == training_cfg.max_steps:
            torch.save(checkpoint_payload(step), args.output / "checkpoint.pt")

    # Report test quality for the checkpoint selected by validation, while
    # keeping the resumable checkpoint on the final training state.
    final_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    best_state_path = args.output / "best_model.pt"
    if best_state_path.exists():
        model.load_state_dict(torch.load(best_state_path, map_location=device, weights_only=True))
    test_loss = evaluate(model, test_loader, device, None)
    model.load_state_dict(final_state)
    torch.save(checkpoint_payload(training_cfg.max_steps, test_loss), args.output / "checkpoint.pt")
    tokenizer.save(args.output / "tokenizer.json")
    (args.output / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.output / "config.json").write_text(json.dumps({"architecture": args.architecture, "model": model_cfg, "training": raw["training"], "data": data_cfg, "model_metadata": model_info}, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "parameters": parameter_count(model),
        "model_metadata": model_info,
        "flops": flop_estimate.as_dict(),
        "estimated_training_flops": estimate_training_flops(model, args.architecture, tokens_seen),
        "device": str(device),
    }))


if __name__ == "__main__":
    main()
