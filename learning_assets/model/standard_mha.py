"""Small standard causal multi-head attention decoder.

This module is intentionally separate from the historical ``mha.py`` learning
asset, which contains a latent-attention experiment.  It provides the actual
MHA baseline used by the architecture comparison runner.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from rope import RotaryEmbedding


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
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        emb_dim = cfg["emb_dim"]
        self.n_heads = cfg["n_heads"]
        self.head_dim = emb_dim // self.n_heads
        if emb_dim % self.n_heads:
            raise ValueError("emb_dim must be divisible by n_heads")
        dtype = cfg.get("dtype")
        bias = cfg.get("qkv_bias", False)
        self.q_proj = nn.Linear(emb_dim, emb_dim, bias=bias, dtype=dtype)
        self.k_proj = nn.Linear(emb_dim, emb_dim, bias=bias, dtype=dtype)
        self.v_proj = nn.Linear(emb_dim, emb_dim, bias=bias, dtype=dtype)
        self.out_proj = nn.Linear(emb_dim, emb_dim, bias=False, dtype=dtype)
        self.dropout = nn.Dropout(cfg.get("drop_rate", 0.0))
        rope_dim = cfg.get("rope_dim", self.head_dim)
        if rope_dim is None:
            rope_dim = self.head_dim
        if rope_dim > self.head_dim:
            raise ValueError("rope_dim cannot exceed head_dim")
        self.rope = RotaryEmbedding(
            rope_dim=int(rope_dim),
            max_seq_len=cfg["context_length"],
            base=float(cfg.get("rope_base", 10_000.0)),
        )
        self.context_length = int(cfg["context_length"])
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.current_pos = 0

    def _split(self, x: Tensor) -> Tensor:
        batch, tokens, _ = x.shape
        return x.view(batch, tokens, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: Tensor, use_cache: bool = False) -> Tensor:
        batch, tokens, _ = x.shape
        queries = self._split(self.q_proj(x))
        keys = self._split(self.k_proj(x))
        values = self._split(self.v_proj(x))
        start = self.current_pos if use_cache else 0
        if start + tokens > self.context_length:
            raise ValueError("MHA input exceeds context_length; reset or truncate the cache")
        positions = torch.arange(start, start + tokens, device=x.device)
        queries, keys = self.rope(queries, keys, positions)

        if use_cache:
            if self.cache_k is not None:
                keys = torch.cat((self.cache_k, keys), dim=2)
                values = torch.cat((self.cache_v, values), dim=2)
            self.cache_k = keys
            self.cache_v = values
            self.current_pos += tokens
            query_positions = torch.arange(start, start + tokens, device=x.device)
            key_positions = torch.arange(keys.shape[2], device=x.device)
            future = key_positions[None, :] > query_positions[:, None]
        else:
            self.reset_cache()
            future = torch.triu(
                torch.ones(tokens, tokens, dtype=torch.bool, device=x.device), diagonal=1
            )

        scores = (queries @ keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(future[None, None, :, :], torch.finfo(scores.dtype).min)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        output = weights @ values
        output = output.transpose(1, 2).contiguous().view(batch, tokens, -1)
        return self.out_proj(output)

    def reset_cache(self) -> None:
        self.cache_k = None
        self.cache_v = None
        self.current_pos = 0


class SwiGLU(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        emb_dim = cfg["emb_dim"]
        hidden_dim = cfg["hidden_dim"]
        dtype = cfg.get("dtype")
        self.gate = nn.Linear(emb_dim, hidden_dim, bias=False, dtype=dtype)
        self.value = nn.Linear(emb_dim, hidden_dim, bias=False, dtype=dtype)
        self.output = nn.Linear(hidden_dim, emb_dim, bias=False, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.output(torch.nn.functional.silu(self.gate(x)) * self.value(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.att = MultiHeadAttention(cfg)
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.ff = SwiGLU(cfg)
        self.dropout = nn.Dropout(cfg.get("drop_rate", 0.0))

    def forward(self, x: Tensor, use_cache: bool = False) -> Tensor:
        x = x + self.dropout(self.att(self.norm1(x), use_cache=use_cache))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x

    def reset_cache(self) -> None:
        self.att.reset_cache()


class GPTModel(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        dtype = cfg.get("dtype")
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=dtype)
        self.drop_emb = nn.Dropout(cfg.get("drop_rate", 0.0))
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=dtype)

    def forward(self, input_ids: Tensor, use_cache: bool = False) -> Tensor:
        _, tokens = input_ids.shape
        if not use_cache:
            self.reset_kv_cache()
        x = self.tok_emb(input_ids)
        x = self.drop_emb(x)
        for block in self.blocks:
            x = block(x, use_cache=use_cache)
        return self.out_head(self.final_norm(x))

    def reset_kv_cache(self) -> None:
        for block in self.blocks:
            block.reset_cache()
