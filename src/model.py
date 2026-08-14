from __future__ import annotations

import math
from contextlib import contextmanager
from typing import TypeAlias

import torch
from torch import Tensor, nn

from config import ModelConfig

PastKeyValue: TypeAlias = tuple[Tensor, Tensor]
PastKeyValues: TypeAlias = tuple[PastKeyValue | None, ...]


class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(variance + self.eps) + self.shift


class MultiHeadAttention(nn.Module):
    """Causal MHA with an optional *functional* KV cache.

    `past_key_value` is never stored on the module. This avoids stale cache state
    between requests and makes the training path exactly the ordinary MHA path.
    Cache tensors have shape (batch, heads, sequence, head_dim).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.emb_dim // cfg.n_heads
        self.q_proj = nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=cfg.qkv_bias)
        self.k_proj = nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=cfg.qkv_bias)
        self.v_proj = nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=cfg.qkv_bias)
        self.out_proj = nn.Linear(cfg.emb_dim, cfg.emb_dim)
        self.attn_dropout = nn.Dropout(cfg.dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, tokens, _ = x.shape
        return x.view(batch, tokens, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: Tensor,
        past_key_value: PastKeyValue | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, PastKeyValue | None]:
        batch, query_tokens, _ = x.shape
        queries = self._split_heads(self.q_proj(x))
        keys = self._split_heads(self.k_proj(x))
        values = self._split_heads(self.v_proj(x))
        past_length = 0

        if past_key_value is not None:
            past_keys, past_values = past_key_value
            expected_prefix = (batch, self.n_heads)
            if past_keys.ndim != 4 or past_values.ndim != 4:
                raise ValueError("KV cache tensors must have shape (batch, heads, tokens, head_dim)")
            if past_keys.shape != past_values.shape:
                raise ValueError("KV cache keys and values must have identical shapes")
            if past_keys.shape[:2] != expected_prefix or past_keys.shape[-1] != self.head_dim:
                raise ValueError("KV cache batch size, head count, or head dimension does not match attention")
            if past_keys.device != x.device or past_values.device != x.device:
                raise ValueError("KV cache and input must be on the same device")
            if past_keys.dtype != past_values.dtype:
                raise ValueError("KV cache keys and values must have the same dtype")
            past_length = past_keys.shape[2]
            keys = torch.cat((past_keys, keys), dim=2)
            values = torch.cat((past_values, values), dim=2)

        key_tokens = keys.shape[2]
        query_positions = torch.arange(past_length, past_length + query_tokens, device=x.device)
        key_positions = torch.arange(key_tokens, device=x.device)
        future_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)

        scores = (queries @ keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(future_mask.unsqueeze(0).unsqueeze(0), torch.finfo(scores.dtype).min)
        weights = self.attn_dropout(torch.softmax(scores, dim=-1))
        context = weights @ values
        context = context.transpose(1, 2).contiguous().view(batch, query_tokens, -1)
        present = (keys, values) if use_cache else None
        return self.out_proj(context), present


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * cfg.emb_dim, cfg.emb_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.attention = MultiHeadAttention(cfg)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.ffn = FeedForward(cfg)
        self.residual_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: Tensor,
        past_key_value: PastKeyValue | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, PastKeyValue | None]:
        attention_out, present = self.attention(self.norm1(x), past_key_value, use_cache)
        x = x + self.residual_dropout(attention_out)
        x = x + self.residual_dropout(self.ffn(self.norm2(x)))
        return x, present


class GPTModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.position_embedding = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.embedding_dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = LayerNorm(cfg.emb_dim)
        self.lm_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Tensor,
        past_key_values: PastKeyValues | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, PastKeyValues | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, tokens)")
        batch, tokens = input_ids.shape
        past_length = self._validate_past_key_values(past_key_values, batch, input_ids.device)
        if tokens + past_length > self.cfg.context_length:
            raise ValueError("input plus cache exceeds context_length; rebuild from the sliding window")

        positions = torch.arange(past_length, past_length + tokens, device=input_ids.device)
        x = self.embedding_dropout(self.token_embedding(input_ids) + self.position_embedding(positions).unsqueeze(0))
        present_key_values: list[PastKeyValue | None] = []
        for layer_index, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[layer_index]
            x, present = block(x, past, use_cache)
            present_key_values.append(present)
        logits = self.lm_head(self.final_norm(x))
        return logits, tuple(present_key_values) if use_cache else None

    def _validate_past_key_values(
        self, past_key_values: PastKeyValues | None, batch: int, device: torch.device
    ) -> int:
        if past_key_values is None:
            return 0
        if len(past_key_values) != len(self.blocks):
            raise ValueError("One past key/value pair is required for every transformer block")
        if any(cache is None for cache in past_key_values):
            raise ValueError("KV cache must be present for every layer or omitted for every layer")
        first_keys, first_values = past_key_values[0]  # type: ignore[misc]
        if first_keys.ndim != 4 or first_values.ndim != 4 or first_keys.shape != first_values.shape:
            raise ValueError("Each layer KV cache must be matching rank-4 key/value tensors")
        expected_shape = (batch, self.cfg.n_heads, first_keys.shape[2], self.cfg.emb_dim // self.cfg.n_heads)
        if first_keys.shape != expected_shape:
            raise ValueError("KV cache shape does not match model batch, heads, or head dimension")
        if first_keys.device != device or first_values.device != device:
            raise ValueError("KV cache and input_ids must be on the same device")
        if first_keys.dtype != first_values.dtype:
            raise ValueError("KV cache keys and values must have the same dtype")
        for layer_index, cache in enumerate(past_key_values):
            keys, values = cache  # type: ignore[misc]
            if keys.shape != expected_shape or values.shape != expected_shape:
                raise ValueError(f"KV cache shape differs at transformer layer {layer_index}")
            if keys.device != device or values.device != device:
                raise ValueError(f"KV cache device differs at transformer layer {layer_index}")
            if keys.dtype != first_keys.dtype or values.dtype != first_keys.dtype:
                raise ValueError(f"KV cache dtype differs at transformer layer {layer_index}")
        return first_keys.shape[2]

    @contextmanager
    def _generation_mode(self):
        was_training = self.training
        self.eval()
        try:
            yield
        finally:
            self.train(was_training)

    @torch.inference_mode()
    def generate_uncached(self, input_ids: Tensor, max_new_tokens: int) -> Tensor:
        """Reference decoder: recomputes the entire context at every new token."""
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("A non-empty prompt with shape (batch, tokens) is required")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        output = input_ids.clone()
        with self._generation_mode():
            for _ in range(max_new_tokens):
                logits, _ = self(output[:, -self.cfg.context_length :])
                next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
                output = torch.cat((output, next_token), dim=1)
        return output

    @torch.inference_mode()
    def generate_cached(self, input_ids: Tensor, max_new_tokens: int) -> Tensor:
        """Greedy decoder that reuses K/V tensors between decode steps.

        Once the window is full, it rebuilds the cache from the last window so
        absolute positional embeddings retain uncached-generation semantics.
        """
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("A non-empty prompt with shape (batch, tokens) is required")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        output = input_ids.clone()
        with self._generation_mode():
            logits, cache = self(output[:, -self.cfg.context_length :], use_cache=True)
            for step in range(max_new_tokens):
                next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
                output = torch.cat((output, next_token), dim=1)
                if step == max_new_tokens - 1:
                    break
                cache_length = cache[0][0].shape[2] if cache and cache[0] is not None else 0
                if cache_length >= self.cfg.context_length:
                    logits, cache = self(output[:, -self.cfg.context_length :], use_cache=True)
                else:
                    logits, cache = self(next_token, past_key_values=cache, use_cache=True)
        return output


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
