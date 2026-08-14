from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_layers: int
    dropout: float = 0.0
    qkv_bias: bool = False
    rope_dim: int | None = None
    rope_base: float = 10_000.0

    def __post_init__(self) -> None:
        if self.vocab_size <= 1:
            raise ValueError("vocab_size must be greater than one")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.emb_dim <= 0:
            raise ValueError("emb_dim must be positive")
        if self.emb_dim % self.n_heads:
            raise ValueError("emb_dim must be divisible by n_heads")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.rope_dim is not None:
            head_dim = self.emb_dim // self.n_heads
            if self.rope_dim <= 0 or self.rope_dim % 2:
                raise ValueError("rope_dim must be a positive even number")
            if self.rope_dim > head_dim:
                raise ValueError("rope_dim cannot exceed head dimension")
        if self.rope_base <= 0:
            raise ValueError("rope_base must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    precision: str = "auto"
    learning_rate: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 0.1
    max_steps: int = 500
    warmup_steps: int = 0
    min_lr_ratio: float = 0.1
    eval_every: int = 50
    eval_batches: int = 8
    grad_clip_norm: float = 1.0
    seed: int = 42
    save_every: int = 250

    def __post_init__(self) -> None:
        for name in (
            "batch_size",
            "gradient_accumulation_steps",
            "max_steps",
            "warmup_steps",
            "eval_every",
            "eval_batches",
            "seed",
            "save_every",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
        for name in (
            "learning_rate",
            "beta1",
            "beta2",
            "adam_eps",
            "weight_decay",
            "min_lr_ratio",
            "grad_clip_norm",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.precision not in {"auto", "fp32", "fp16", "bf16"}:
            raise ValueError("precision must be one of auto, fp32, fp16, or bf16")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("beta1 and beta2 must be in [0, 1)")
        if self.adam_eps <= 0:
            raise ValueError("adam_eps must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.warmup_steps >= self.max_steps:
            raise ValueError("warmup_steps must be smaller than max_steps")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if self.eval_every <= 0 or self.eval_batches <= 0:
            raise ValueError("eval_every and eval_batches must be positive")
        if self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive")
        if self.save_every <= 0:
            raise ValueError("save_every must be positive")


def validate_resume_config(
    saved_config: Mapping[str, object], current: TrainingConfig, saved_step: int
) -> None:
    """Reject resume settings that would silently change training dynamics."""
    saved = TrainingConfig(**dict(saved_config))
    for name in (
        "batch_size",
        "gradient_accumulation_steps",
        "precision",
        "learning_rate",
        "beta1",
        "beta2",
        "adam_eps",
        "weight_decay",
        "warmup_steps",
        "min_lr_ratio",
        "grad_clip_norm",
        "seed",
    ):
        if getattr(saved, name) != getattr(current, name):
            raise ValueError(f"Resume config mismatch for {name}")
    if current.max_steps <= saved_step:
        raise ValueError("Resume max_steps must be greater than the checkpoint step")
