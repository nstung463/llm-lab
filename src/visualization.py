"""Paper-friendly plots generated from a single training run history."""

from __future__ import annotations

from collections.abc import Sequence
import math
from pathlib import Path
from typing import Any


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _points(
    rows: Sequence[dict[str, Any]], x_key: str, y_key: str
) -> tuple[list[float], list[float]]:
    points = [(_number(row.get(x_key)), _number(row.get(y_key))) for row in rows]
    valid = [(x, y) for x, y in points if x is not None and y is not None]
    return [x for x, _ in valid], [y for _, y in valid]


def _validation_key(rows: Sequence[dict[str, Any]]) -> str | None:
    if any("validation_loss" in row for row in rows):
        return "validation_loss"
    if any("val_loss" in row for row in rows):
        return "val_loss"
    return None


def save_training_plots(
    history: Sequence[dict[str, Any]],
    output_dir: Path,
    *,
    title: str = "Training",
) -> list[Path]:
    """Save standard diagnostics and return the generated image paths.

    Plotting is intentionally best-effort: a training run should remain usable
    on a minimal environment where matplotlib is not installed.
    """
    if not history:
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    rows = list(history)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    validation_key = _validation_key(rows)
    x_key = "tokens_seen" if any("tokens_seen" in row for row in rows) else "step"
    x_label = "tokens seen" if x_key == "tokens_seen" else "optimizer step"

    def save(fig: Any, filename: str) -> None:
        path = output_dir / filename
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    x_train, train_loss = _points(rows, x_key, "train_loss")
    x_val, val_loss = _points(rows, x_key, validation_key) if validation_key else ([], [])
    fig, axis = plt.subplots(figsize=(8, 5))
    if x_train:
        axis.plot(x_train, train_loss, marker="o", label="train loss")
    if x_val:
        axis.plot(x_val, val_loss, marker="o", label="validation loss")
    axis.set(xlabel=x_label, ylabel="cross-entropy loss", title=f"{title}: loss")
    axis.grid(True, alpha=0.3)
    axis.legend()
    save(fig, "loss_vs_tokens.png" if x_key == "tokens_seen" else "loss_vs_step.png")

    if x_val:
        perplexity = [math.exp(min(20.0, loss)) for loss in val_loss]
        fig, axis = plt.subplots(figsize=(8, 5))
        axis.plot(x_val, perplexity, marker="o", color="tab:orange")
        axis.set(xlabel=x_label, ylabel="perplexity", title=f"{title}: validation perplexity")
        axis.grid(True, alpha=0.3)
        save(fig, "perplexity_vs_tokens.png" if x_key == "tokens_seen" else "perplexity_vs_step.png")

    lr_steps, learning_rates = _points(rows, "step", "learning_rate")
    grad_steps, grad_norms = _points(rows, "step", "grad_norm")
    throughput_steps, throughputs = _points(rows, "step", "tokens_per_second")
    vram_steps, peak_vram = _points(rows, "step", "peak_vram_gb")
    if not throughputs:
        throughput_steps, tokens = _points(rows, "step", "tokens_seen")
        _, elapsed = _points(rows, "step", "elapsed_seconds")
        throughputs = [token / seconds for token, seconds in zip(tokens, elapsed) if seconds > 0]
        throughput_steps = throughput_steps[: len(throughputs)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(lr_steps, learning_rates, marker="o")
    axes[0, 0].set(title="learning rate", xlabel="optimizer step", ylabel="lr")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(grad_steps, grad_norms, marker="o", color="tab:red")
    axes[0, 1].set(title="gradient norm", xlabel="optimizer step", ylabel="norm")
    axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].plot(throughput_steps, throughputs, marker="o", color="tab:green")
    axes[1, 0].set(title="throughput", xlabel="optimizer step", ylabel="tokens/sec")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].plot(vram_steps, peak_vram, marker="o", color="tab:purple")
    axes[1, 1].set(title="peak VRAM", xlabel="optimizer step", ylabel="GB")
    axes[1, 1].grid(True, alpha=0.3)
    fig.suptitle(f"{title}: training diagnostics")
    fig.tight_layout()
    save(fig, "training_diagnostics.png")

    flops_key = "estimated_training_flops"
    x_flops, y_flops = _points(rows, flops_key, validation_key) if validation_key else ([], [])
    if x_flops:
        fig, axis = plt.subplots(figsize=(8, 5))
        axis.plot(x_flops, y_flops, marker="o", color="tab:blue")
        axis.set(xlabel="estimated training FLOPs", ylabel="validation loss", title=f"{title}: loss vs compute")
        axis.grid(True, alpha=0.3)
        save(fig, "loss_vs_flops.png")

    return paths


__all__ = ["save_training_plots"]
