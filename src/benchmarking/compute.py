"""Analytical FLOPs estimates for the educational decoder architectures.

These estimates count matrix-multiply FLOPs and the dominant attention
matmuls. They are useful for comparing runs with the same convention, but are
not a replacement for a hardware profiler. In particular, kernels, masking,
normalization, routing, and sparse/compressed attention overhead are only
approximated.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import torch
from torch import nn


_MOE_EXPERT_LINEAR = re.compile(r"^(?P<layer>.+\.ff)\.(?:gate_proj|up_proj|out_proj)\.(?P<expert>\d+)$")
_MOE_EXPERT_PARAMETER = re.compile(r"^(?P<layer>.+\.ff)\.(?:gate_proj|up_proj|out_proj)\.(?P<expert>\d+)\.")


@dataclass(frozen=True)
class FlopEstimate:
    """Estimated compute cost under one explicit, reproducible convention."""

    architecture: str
    sequence_length: int
    forward_flops_per_token: int
    training_flops_per_token: int
    method: str = "linear_matmuls_plus_attention; training=3x-forward"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class KVCacheEstimate:
    """Analytical KV-cache footprint for one batch element."""

    architecture: str
    sequence_length: int
    dtype_bytes: int
    cache_kind: str
    bytes_per_token: float
    total_bytes: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _model_dtype_bytes(model: nn.Module) -> int:
    try:
        return next(model.parameters()).element_size()
    except StopIteration:
        return 4


def estimate_kv_cache(
    model: nn.Module,
    architecture: str,
    sequence_length: int | None = None,
) -> KVCacheEstimate:
    """Estimate accumulated KV-cache bytes for one batch element.

    MHA/GQA/MoE cache K and V tensors, while this project's MLA caches only
    the compressed latent. V4 is estimated as local full-resolution K/V plus
    compressed historical latents. Runtime cache statistics should be used to
    validate these estimates for a concrete generation path.
    """
    cfg = model.training_cfg
    seq_len = int(sequence_length or cfg["context_length"])
    if seq_len <= 0:
        raise ValueError("sequence_length must be positive")
    dtype_bytes = _model_dtype_bytes(model)
    emb_dim = int(cfg["emb_dim"])
    layers = int(cfg["n_layers"])

    if architecture == "mha":
        elements = seq_len * layers * 2 * emb_dim
        cache_kind = "full_kv"
    elif architecture in {"gqa", "moe"}:
        head_dim = emb_dim // int(cfg["n_heads"])
        kv_groups = int(cfg["n_kv_groups"])
        elements = seq_len * layers * 2 * kv_groups * head_dim
        cache_kind = "grouped_kv"
    elif architecture == "mla":
        elements = seq_len * layers * int(cfg["latent_dim"])
        cache_kind = "latent_kv"
    elif architecture == "v4":
        window = min(seq_len, int(cfg["window_size"]))
        heads = int(cfg["n_heads"])
        head_dim = int(cfg["head_dim"])
        ratios = cfg.get("compress_ratios", [0] * layers)
        if not isinstance(ratios, (list, tuple)):
            ratios = [int(ratios)] * layers
        elements = 0
        for layer in range(layers):
            ratio = int(ratios[layer]) if layer < len(ratios) else 0
            local_elements = window * 2 * heads * head_dim
            compressed_tokens = max(seq_len - window, 0) // ratio if ratio > 0 else 0
            compressed_elements = compressed_tokens * int(cfg["latent_dim"])
            elements += local_elements + compressed_elements
        cache_kind = "local_kv_plus_compressed_latent"
    else:
        raise ValueError(f"Unknown architecture {architecture!r}")

    total_bytes = elements * dtype_bytes
    return KVCacheEstimate(
        architecture=architecture,
        sequence_length=seq_len,
        dtype_bytes=dtype_bytes,
        cache_kind=cache_kind,
        bytes_per_token=total_bytes / seq_len,
        total_bytes=total_bytes,
    )


def _cache_tensors(value: object):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _cache_tensors(item)


def collect_kv_cache_stats(model: nn.Module) -> dict[str, int | float]:
    """Collect actual cache tensors left by a cached generation pass."""
    total_bytes = 0
    tensor_count = 0
    cached_positions = 0
    cache_attributes = (
        "cache_k",
        "cache_v",
        "cache_latent",
        "local_kv_cache",
        "compressed_kv_cache",
    )
    for module in model.modules():
        for attribute in cache_attributes:
            value = getattr(module, attribute, None)
            for tensor in _cache_tensors(value):
                total_bytes += tensor.numel() * tensor.element_size()
                tensor_count += 1
                if tensor.ndim >= 2:
                    cached_positions = max(cached_positions, int(tensor.shape[-2]))
    return {
        "bytes": total_bytes,
        "tensors": tensor_count,
        "tokens": cached_positions,
        "bytes_per_token": total_bytes / cached_positions if cached_positions else 0.0,
    }


def _linear_flops_per_token(model: nn.Module, architecture: str) -> int:
    """Count 2*in*out for linear layers used by one token.

    MoE registers all experts but routes each token to only top-k experts. The
    routed expert contribution is therefore adjusted from all experts to the
    configured active expert count.
    """
    total = 0
    expert_flops_by_layer: dict[str, list[int]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        flops = 2 * module.in_features * module.out_features
        total += flops
        if architecture == "moe":
            match = _MOE_EXPERT_LINEAR.match(name)
            if match:
                expert_flops_by_layer.setdefault(match.group("layer"), []).append(flops)

    if architecture == "moe" and expert_flops_by_layer:
        num_experts = int(model.training_cfg["num_experts"])
        active_experts = int(model.training_cfg["num_experts_per_tok"])
        all_expert_flops = sum(sum(values) for values in expert_flops_by_layer.values())
        one_expert_flops = sum(sum(values) / num_experts for values in expert_flops_by_layer.values())
        total = total - all_expert_flops + int(one_expert_flops * active_experts)
    return total


def active_parameter_count(model: nn.Module, architecture: str) -> int:
    """Estimate parameters touched by one token, keeping MoE total params separate.

    For dense models this equals total trainable parameters. For MoE, all
    expert weights remain resident in memory, but only the configured top-k
    routed experts plus the shared path are counted as active per token.
    """
    total = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if architecture != "moe":
        return total
    expert_parameters: dict[str, list[int]] = {}
    for name, parameter in model.named_parameters():
        match = _MOE_EXPERT_PARAMETER.match(name)
        if match:
            expert_parameters.setdefault(match.group("layer"), []).append(parameter.numel())
    if not expert_parameters:
        return total
    num_experts = int(model.training_cfg["num_experts"])
    active_experts = int(model.training_cfg["num_experts_per_tok"])
    all_expert_parameters = sum(sum(values) for values in expert_parameters.values())
    one_expert_parameters = sum(sum(values) / num_experts for values in expert_parameters.values())
    return total - all_expert_parameters + int(one_expert_parameters * active_experts)


def estimate_flops(
    model: nn.Module,
    architecture: str,
    sequence_length: int | None = None,
) -> FlopEstimate:
    """Estimate forward and training FLOPs per token.

    ``training_flops_per_token`` uses the common 3x-forward approximation for
    forward plus backward computation. Optimizer and non-matmul kernel costs
    are intentionally excluded.
    """
    cfg = model.training_cfg
    seq_len = int(sequence_length or cfg["context_length"])
    if seq_len <= 0:
        raise ValueError("sequence_length must be positive")

    linear_flops = _linear_flops_per_token(model, architecture)
    emb_dim = int(cfg["emb_dim"])
    layers = int(cfg["n_layers"])
    if architecture == "v4":
        attention_context = min(seq_len, int(cfg.get("window_size", seq_len)))
    else:
        attention_context = seq_len
    attention_flops = 4 * attention_context * emb_dim * layers
    forward_flops = linear_flops + attention_flops
    return FlopEstimate(
        architecture=architecture,
        sequence_length=seq_len,
        forward_flops_per_token=forward_flops,
        training_flops_per_token=3 * forward_flops,
    )


def estimate_training_flops(
    model: nn.Module,
    architecture: str,
    tokens_seen: int,
    sequence_length: int | None = None,
) -> int:
    """Estimate total training FLOPs for the tokens processed so far."""
    if tokens_seen < 0:
        raise ValueError("tokens_seen must be non-negative")
    estimate = estimate_flops(model, architecture, sequence_length)
    return estimate.training_flops_per_token * tokens_seen
