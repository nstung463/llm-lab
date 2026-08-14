"""Learning-rate schedules shared by every training entrypoint."""

from __future__ import annotations

import math

import torch


def cosine_lr(
    step: int,
    *,
    warmup_steps: int,
    max_steps: int,
    lr: float,
    min_lr: float,
) -> float:
    """Return the LR for a 1-based optimizer update.

    Update ``1`` starts warmup, update ``warmup_steps`` reaches ``lr``,
    and update ``max_steps`` reaches ``min_lr``. Values after the run stay
    at ``min_lr`` so resume and boundary behavior are deterministic.
    """
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if warmup_steps < 0 or warmup_steps >= max_steps:
        raise ValueError("warmup_steps must satisfy 0 <= warmup_steps < max_steps")
    if lr <= 0:
        raise ValueError("lr must be positive")
    if not 0 <= min_lr <= lr:
        raise ValueError("min_lr must satisfy 0 <= min_lr <= lr")

    if step <= 0:
        return 0.0 if warmup_steps else lr
    if not warmup_steps:
        if step >= max_steps:
            return min_lr
        progress = (step - 1) / (max_steps - 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (lr - min_lr) * cosine
    if warmup_steps and step <= warmup_steps:
        return lr * step / warmup_steps
    if step >= max_steps:
        return min_lr

    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (lr - min_lr) * cosine


def set_optimizer_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    """Set one absolute LR on all optimizer parameter groups."""
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


__all__ = ["cosine_lr", "set_optimizer_lr"]
