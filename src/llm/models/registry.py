"""Registry and factory for the architecture comparison models."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from torch import nn

from .base import DecoderModel, ModelMetadata, as_module


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "learning_assets" / "model"


@dataclass(frozen=True)
class ModelSpec:
    """Registration record for one architecture implementation."""

    name: str
    module_file: str
    family: str
    description: str


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "mha": ModelSpec(
        name="mha",
        module_file="standard_mha.py",
        family="dense-attention",
        description="Standard causal multi-head attention with SwiGLU.",
    ),
    "gqa": ModelSpec(
        name="gqa",
        module_file="gqa.py",
        family="grouped-attention",
        description="Grouped-query attention with SwiGLU.",
    ),
    "mla": ModelSpec(
        name="mla",
        module_file="mla.py",
        family="latent-attention",
        description="Multi-head latent attention with dense SwiGLU when MoE is disabled.",
    ),
    "moe": ModelSpec(
        name="moe",
        module_file="moe.py",
        family="sparse-mixture",
        description="Grouped-query attention with routed and shared experts.",
    ),
    "v4": ModelSpec(
        name="v4",
        module_file="v4.py",
        family="compressed-sparse",
        description="Readable V4-style compressed sparse attention and MoE decoder.",
    ),
}


def available_architectures() -> tuple[str, ...]:
    """Return architecture names in stable registry order."""
    return tuple(MODEL_REGISTRY)


def _load_module(name: str, path: Path) -> ModuleType:
    # Standalone learning assets import tiktoken only inside their CLI main.
    # Keep the shared registry usable when optional tokenizer dependencies are
    # absent in a minimal environment.
    if "tiktoken" not in sys.modules:
        try:
            import tiktoken  # type: ignore[import-not-found,unused-ignore]
        except ModuleNotFoundError:
            sys.modules["tiktoken"] = ModuleType("tiktoken")
    if not path.exists():
        raise FileNotFoundError(f"Model implementation does not exist: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load model module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_config(architecture: str, model_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(model_cfg)
    cfg["dtype"] = torch.float32
    cfg.setdefault("drop_rate", cfg.get("dropout", 0.0))
    cfg.setdefault("qkv_bias", False)
    if architecture == "gqa":
        cfg.setdefault("n_kv_groups", cfg["n_heads"])
    elif architecture == "moe":
        cfg.setdefault("n_kv_groups", cfg["n_heads"])
        cfg.setdefault("num_experts", 4)
        cfg.setdefault("num_experts_per_tok", 2)
        cfg.setdefault("shared_expert_hidden_dim", cfg["hidden_dim"])
    elif architecture == "mla":
        cfg.setdefault("latent_dim", max(16, cfg["emb_dim"] // 4))
        cfg.setdefault("num_experts", 0)
        cfg.setdefault("num_experts_per_tok", 0)
    return cfg


def _tie_embeddings(model: nn.Module, architecture: str) -> None:
    input_embedding = getattr(model, "tok_emb", None)
    output_head = getattr(model, "out_head", None)
    if output_head is None:
        output_head = getattr(model, "lm_head", None)
    if input_embedding is None or output_head is None:
        raise ValueError(f"{architecture} does not expose compatible embedding layers")
    if input_embedding.weight.shape != output_head.weight.shape:
        raise ValueError(f"{architecture} input/output embedding shapes do not match")
    # nn.Embedding defaults to unit-scale initialization, while decoder LM
    # heads are normally initialized near zero.  Normalize before tying.
    nn.init.normal_(input_embedding.weight, mean=0.0, std=0.02)
    output_head.weight = input_embedding.weight


def build_model(architecture: str, model_cfg: dict[str, Any]) -> nn.Module:
    """Build a registered decoder with the common training-facing API."""
    try:
        model_spec = MODEL_REGISTRY[architecture]
    except KeyError as exc:
        known = ", ".join(available_architectures())
        raise ValueError(f"Unknown architecture {architecture!r}; choose from {known}") from exc

    cfg = _normalize_config(architecture, model_cfg)
    module = _load_module(
        f"llm_registered_{architecture}", MODEL_DIR / model_spec.module_file
    )
    model = module.GPTModel(cfg)
    if cfg.get("tie_embeddings", False):
        _tie_embeddings(model, architecture)
    model.training_cfg = cfg
    return model


def parameter_count(model: DecoderModel) -> int:
    """Count trainable parameters without counting tied weights twice."""
    return sum(parameter.numel() for parameter in as_module(model).parameters() if parameter.requires_grad)


def model_metadata(architecture: str, model: DecoderModel) -> ModelMetadata:
    """Return normalized metadata for reports and checkpoint validation."""
    if architecture not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture: {architecture}")
    cfg = model.training_cfg
    spec = MODEL_REGISTRY[architecture]
    return ModelMetadata(
        architecture=architecture,
        family=spec.family,
        module=spec.module_file,
        parameters=parameter_count(model),
        context_length=int(cfg["context_length"]),
        layers=int(cfg["n_layers"]),
        embedding_dim=int(cfg["emb_dim"]),
        vocabulary_size=int(cfg["vocab_size"]),
    )
