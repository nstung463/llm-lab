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
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from config import TrainingConfig
from benchmarking.compute import active_parameter_count, estimate_flops, estimate_training_flops
from data import load_token_artifacts
from data.datasets import make_loaders
from data.readers import read_documents
from data.splits import split_documents_three
from data.tokenizer import build_tokenizer
from evaluation.loss import token_weighted_loss
from models.registry import (
    available_architectures,
    build_model,
    model_metadata,
    parameter_count,
)


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


def main() -> None:
    args = parse_args()
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    data_cfg = dict(raw.get("data", {}))
    model_cfg = dict(raw["model"])
    model_cfg.update(dict(raw.get("model_overrides", {}).get(args.architecture, {})))
    training_cfg = TrainingConfig(**raw["training"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(training_cfg.seed)

    artifact_manifest = args.artifact_manifest or (
        Path(data_cfg["artifact_manifest"]) if data_cfg.get("artifact_manifest") else None
    )
    data_manifest = None
    data_contract = None
    if artifact_manifest is not None:
        artifacts = load_token_artifacts(artifact_manifest)
        train_fraction = float(artifacts.manifest["train_fraction"])
        validation_fraction = float(
            artifacts.manifest["validation_document_count"]
            / artifacts.manifest["document_count"]
        )
        tokenizer = artifacts.tokenizer
        train_tokens = artifacts.train_tokens
        validation_tokens = artifacts.validation_tokens
        test_tokens = artifacts.test_tokens
        data_manifest = artifacts.manifest
        data_contract = artifacts.contract
    else:
        if args.text is None:
            raise ValueError("Provide --text or --artifact-manifest")
        documents = read_documents(args.text)
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
        tokenizer = build_tokenizer(
            tokenizer_kind,
            train_documents,
            tokenizer_vocab_size,
            tokenizer_min_frequency,
            tokenizer_name,
        )
        model_cfg["vocab_size"] = tokenizer.vocab_size
        train_tokens = tokenizer.encode_documents(train_documents)
        validation_tokens = tokenizer.encode_documents(validation_documents)
        test_tokens = tokenizer.encode_documents(test_documents)
    model_cfg["vocab_size"] = tokenizer.vocab_size
    context_length = model_cfg["context_length"]
    train_loader, val_loader = make_loaders(
        train_tokens, validation_tokens, context_length, training_cfg.batch_size, training_cfg.seed
    )
    _, test_loader = make_loaders(
        train_tokens, test_tokens, context_length, training_cfg.batch_size, training_cfg.seed
    )

    model = build_model(args.architecture, model_cfg).to(device)
    model_info = asdict(model_metadata(args.architecture, model))
    flop_estimate = estimate_flops(model, args.architecture)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg.learning_rate,
        betas=(training_cfg.beta1, training_cfg.beta2),
        eps=training_cfg.adam_eps,
        weight_decay=training_cfg.weight_decay,
    )
    def lr_lambda(step: int) -> float:
        if training_cfg.warmup_steps and step <= training_cfg.warmup_steps:
            return step / training_cfg.warmup_steps
        remaining = max(training_cfg.max_steps - training_cfg.warmup_steps, 1)
        progress = min(max(step - training_cfg.warmup_steps, 0) / remaining, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return training_cfg.min_lr_ratio + (1.0 - training_cfg.min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
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
    args.output.mkdir(parents=True, exist_ok=True)
    history = []
    train_iterator = iter(train_loader)
    tokens_seen = 0
    best_validation_loss = float("inf")
    start_time = time.perf_counter()

    for step in range(1, training_cfg.max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(training_cfg.gradient_accumulation_steps):
            (inputs, targets), train_iterator = _next_batch(train_loader, train_iterator)
            inputs, targets = inputs.to(device), targets.to(device)
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if use_amp
                else nullcontext()
            )
            with autocast_context:
                logits = model(inputs, use_cache=False)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            step_loss += loss.item()
            scaled_loss = loss / training_cfg.gradient_accumulation_steps
            if use_scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            tokens_seen += inputs.numel()
        if use_scaler:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), training_cfg.grad_clip_norm)
        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()

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
                "learning_rate": scheduler.get_last_lr()[0],
                "micro_train_loss": step_loss / training_cfg.gradient_accumulation_steps,
                "elapsed_seconds": time.perf_counter() - start_time,
            }
            history.append(record)
            print(json.dumps(record))
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                torch.save(model.state_dict(), args.output / "best_model.pt")
            print(json.dumps({"sample": sample_text(model, tokenizer, device, train_tokens[:context_length])}))

    checkpoint = {
        "architecture": args.architecture,
        "model_config": model_cfg,
        "training_config": raw["training"],
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "tokenizer": tokenizer.to_state(),
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "test_loss": evaluate(model, test_loader, device, None),
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
        "parameters": parameter_count(model),
        "active_parameters": active_parameter_count(model, args.architecture),
        "model_metadata": model_info,
        "device": str(device),
    }
    torch.save(checkpoint, args.output / "checkpoint.pt")
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
