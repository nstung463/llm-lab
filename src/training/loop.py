"""Baseline training loop and checkpoint API."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
import inspect
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm

from config import TrainingConfig
from evaluation.loss import token_weighted_loss
from models.baseline import GPTModel
from training.optim import build_adamw
from training.schedule import cosine_lr, set_optimizer_lr


@dataclass
class TrainingState:
    """Minimal OLMo-style state shared by the loop and checkpoints.

    ``global_step`` counts optimizer updates; ``tokens_seen`` is the primary
    data-progress counter. ``epoch`` is a one-based loader/reporting value.
    """

    global_step: int = 0
    tokens_seen: int = 0
    epoch: int = 1
    tokens_per_epoch: int | None = None

    @property
    def fractional_epoch(self) -> float | None:
        if not self.tokens_per_epoch:
            return None
        return self.tokens_seen / self.tokens_per_epoch

    def state_dict(self) -> dict[str, int | float | None]:
        return {
            "global_step": self.global_step,
            "step": self.global_step,
            "tokens_seen": self.tokens_seen,
            "global_train_tokens_seen": self.tokens_seen,
            "epoch": self.epoch,
            "tokens_per_epoch": self.tokens_per_epoch,
            "fractional_epoch": self.fractional_epoch,
        }


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
    checkpoint_callback: Callable[..., None] | None = None,
    scaler: Any | None = None,
    tokens_seen: int = 0,
    state: TrainingState | None = None,
) -> list[dict[str, float]]:
    model.to(device)
    optimizer = optimizer or build_adamw(model, cfg)
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if isinstance(value, Tensor):
                optimizer_state[key] = value.to(device)
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
    state = state or TrainingState(
        global_step=start_step,
        tokens_seen=tokens_seen,
        epoch=int(getattr(train_loader, "epoch", 1)),
        tokens_per_epoch=getattr(train_loader, "tokens_per_epoch", None),
    )
    start_step = state.global_step
    tokens_seen = state.tokens_seen
    iterator = iter(train_loader)
    model.train()
    last_val_loss = float(history[-1]["val_loss"]) if history else None
    start_time = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    progress = tqdm(
        range(start_step + 1, cfg.max_steps + 1),
        total=max(0, cfg.max_steps - start_step),
        desc="Training",
        unit="step",
        dynamic_ncols=True,
    )
    for step in progress:
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
                if not hasattr(train_loader, "epoch"):
                    state.epoch += 1
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
        state.global_step = step
        state.tokens_seen = tokens_seen
        state.epoch = int(getattr(train_loader, "epoch", state.epoch))
        if step == 1 or step % cfg.eval_every == 0 or step == cfg.max_steps:
            train_loss = evaluate(model, train_loader, device, cfg.eval_batches)
            val_loss = evaluate(model, val_loader, device, cfg.eval_batches)
            last_val_loss = float(val_loss)
            history.append({
                "step": float(step),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": learning_rate,
                "micro_train_loss": loss_sum / total_tokens,
                "grad_norm": float(grad_norm),
                "tokens_seen": float(tokens_seen),
                "global_step": float(step),
                "epoch": float(state.epoch),
                "fractional_epoch": (
                    float(state.fractional_epoch)
                    if state.fractional_epoch is not None
                    else float("nan")
                ),
            })
            model.train()
        elapsed = max(time.perf_counter() - start_time, 1e-9)
        postfix = {
            "loss": f"{loss_sum / total_tokens:.4f}",
            "lr": f"{learning_rate:.2e}",
            "grad": f"{float(grad_norm):.2f}",
            "tok/s": f"{tokens_seen / elapsed:.0f}",
            "epoch": (
                f"{state.fractional_epoch:.2f}"
                if state.fractional_epoch is not None
                else str(state.epoch)
            ),
        }
        if last_val_loss is not None:
            postfix["val"] = f"{last_val_loss:.4f}"
        if device.type == "cuda":
            peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            postfix["vram"] = f"{peak_vram:.1f}G"
        progress.set_postfix(postfix)
        if step % cfg.save_every == 0 or step == cfg.max_steps:
            if checkpoint_callback is not None:
                # Keep compatibility with older user callbacks that predate
                # the token-budget argument.
                parameter_count = len(inspect.signature(checkpoint_callback).parameters)
                if parameter_count >= 5:
                    checkpoint_callback(step, optimizer, history, tokens_seen, state.state_dict())
                elif parameter_count >= 4:
                    checkpoint_callback(step, optimizer, history, tokens_seen)
                else:
                    checkpoint_callback(step, optimizer, history)  # type: ignore[misc]
    progress.close()
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
    epoch: int | None = None,
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
            "global_step": step,
            "tokens_seen": tokens_seen,
            "global_train_tokens_seen": tokens_seen,
            "epoch": epoch,
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
