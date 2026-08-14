"""Baseline training loop and checkpoint API."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..config import TrainingConfig
from ..evaluation.loss import token_weighted_loss
from ..models.baseline import GPTModel


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
    checkpoint_callback: Callable[[int, torch.optim.Optimizer, list[dict[str, float]]], None] | None = None,
) -> list[dict[str, float]]:
    model.to(device)
    optimizer = optimizer or torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, Tensor):
                state[key] = value.to(device)
    history = history or []
    iterator = iter(train_loader)
    model.train()
    for step in range(start_step + 1, cfg.max_steps + 1):
        try:
            inputs, targets = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            inputs, targets = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for_batch(model, inputs.to(device), targets.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        optimizer.step()
        if step == 1 or step % cfg.eval_every == 0 or step == cfg.max_steps:
            train_loss = evaluate(model, train_loader, device, cfg.eval_batches)
            val_loss = evaluate(model, val_loader, device, cfg.eval_batches)
            history.append({"step": float(step), "train_loss": train_loss, "val_loss": val_loss})
            model.train()
        if step % cfg.save_every == 0 or step == cfg.max_steps:
            if checkpoint_callback is not None:
                checkpoint_callback(step, optimizer, history)
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
            "data_manifest": data_manifest,
            "loader_state": loader_state,
            "torch_rng_state": torch.get_rng_state(),
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


__all__ = ["evaluate", "load_checkpoint", "loss_for_batch", "save_checkpoint", "train"]
