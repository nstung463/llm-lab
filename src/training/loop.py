"""Baseline training loop and checkpoint API."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict
import inspect
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from config import TrainingConfig
from evaluation.loss import token_weighted_loss
from models.baseline import GPTModel
from training.optim import build_adamw
from training.schedule import cosine_lr, set_optimizer_lr


def loss_for_batch(model: GPTModel, inputs: Tensor, targets: Tensor) -> Tensor:
    """Training deliberately never creates a KV cache."""
    logits, cache = model(inputs, use_cache=False)
    assert cache is None
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())


def evaluate(
    model: GPTModel,
    batches: Iterable[tuple[Tensor, Tensor]],
    device: torch.device,
    max_batches: int | None,
) -> float:
    """Return token-weighted next-token cross entropy."""
    return token_weighted_loss(model, batches, device, max_batches)


def train(
    model: GPTModel,
    train_loader: Iterable[tuple[Tensor, Tensor]],
    val_loader: Iterable[tuple[Tensor, Tensor]],
    cfg: TrainingConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    start_step: int = 0,
    history: list[dict[str, float]] | None = None,
    checkpoint_callback: Callable[[int, torch.optim.Optimizer, list[dict[str, float]], int], None] | None = None,
    scaler: Any | None = None,
    tokens_seen: int = 0,
) -> list[dict[str, float]]:
    model.to(device)
    optimizer = optimizer or build_adamw(model, cfg)
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, Tensor):
                state[key] = value.to(device)
    use_amp = device.type == "cuda" and cfg.precision != "fp32"
    if cfg.precision == "bf16":
        amp_dtype = torch.bfloat16
    elif cfg.precision == "fp16":
        amp_dtype = torch.float16
    elif use_amp and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = scaler or torch.amp.GradScaler("cuda", enabled=use_scaler)
    history = history or []
    iterator = iter(train_loader)
    model.train()
    for step in range(start_step + 1, cfg.max_steps + 1):
        learning_rate = cosine_lr(
            step,
            warmup_steps=cfg.warmup_steps,
            max_steps=cfg.max_steps,
            lr=cfg.learning_rate,
            min_lr=cfg.learning_rate * cfg.min_lr_ratio,
        )
        set_optimizer_lr(optimizer, learning_rate)
        microbatches: list[tuple[Tensor, Tensor]] = []
        for _ in range(cfg.gradient_accumulation_steps):
            try:
                microbatches.append(next(iterator))
            except StopIteration:
                iterator = iter(train_loader)
                microbatches.append(next(iterator))
        total_tokens = sum(targets.numel() for _, targets in microbatches)
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for inputs, targets in microbatches:
            inputs, targets = inputs.to(device), targets.to(device)
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if use_amp
                else torch.autocast(device_type="cpu", enabled=False)
            )
            with autocast_context:
                logits, cache = model(inputs, use_cache=False)
                assert cache is None
                token_loss = F.cross_entropy(
                    logits.flatten(0, 1), targets.flatten(), reduction="sum"
                )
            loss_sum += float(token_loss.detach())
            scaled_loss = token_loss / total_tokens
            if use_scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
        if use_scaler:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        tokens_seen += total_tokens
        if step == 1 or step % cfg.eval_every == 0 or step == cfg.max_steps:
            train_loss = evaluate(model, train_loader, device, cfg.eval_batches)
            val_loss = evaluate(model, val_loader, device, cfg.eval_batches)
            history.append({
                "step": float(step),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": learning_rate,
                "micro_train_loss": loss_sum / total_tokens,
                "grad_norm": float(grad_norm),
                "tokens_seen": float(tokens_seen),
            })
            model.train()
        if step % cfg.save_every == 0 or step == cfg.max_steps:
            if checkpoint_callback is not None:
                # Keep compatibility with older user callbacks that predate
                # the token-budget argument.
                parameter_count = len(inspect.signature(checkpoint_callback).parameters)
                if parameter_count >= 4:
                    checkpoint_callback(step, optimizer, history, tokens_seen)
                else:
                    checkpoint_callback(step, optimizer, history)  # type: ignore[misc]
    return history


def save_checkpoint(
    path: Path,
    model: GPTModel,
    training_cfg: TrainingConfig,
    tokenizer_state: dict[str, Any],
    history: list[dict[str, float]],
    optimizer: torch.optim.Optimizer,
    step: int,
    data_manifest: dict[str, Any],
    loader_state: dict[str, Any] | None = None,
    scaler: Any | None = None,
    tokens_seen: int = 0,
    test_loss: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": asdict(model.cfg),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "training_config": asdict(training_cfg),
            "tokenizer": tokenizer_state,
            "history": history,
            "step": step,
            "tokens_seen": tokens_seen,
            "test_loss": test_loss,
            "data_manifest": data_manifest,
            "loader_state": loader_state,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "python": random.getstate(),
                "numpy": np.random.get_state(),
            },
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


def restore_rng_state(state: dict[str, Any] | None) -> None:
    """Restore all available host/device RNG streams from a checkpoint."""
    if not state:
        return
    torch_state = state.get("torch", state.get("torch_rng_state"))
    if torch_state is not None:
        torch.set_rng_state(torch.as_tensor(torch_state, dtype=torch.uint8))
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([torch.as_tensor(item, dtype=torch.uint8).cpu() for item in state["cuda"]])
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])


__all__ = [
    "evaluate",
    "load_checkpoint",
    "loss_for_batch",
    "restore_rng_state",
    "save_checkpoint",
    "train",
]
