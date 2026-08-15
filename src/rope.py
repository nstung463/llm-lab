"""Shared rotary position embedding implementation."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RotaryEmbedding(nn.Module):
    """Apply split-halves RoPE to Q/K tensors shaped ``(B, H, T, D)``."""

    def __init__(self, rope_dim: int, max_seq_len: int, base: float = 10_000.0) -> None:
        super().__init__()
        if rope_dim <= 0 or rope_dim % 2:
            raise ValueError("rope_dim must be a positive even number")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if base <= 0:
            raise ValueError("RoPE base must be positive")
        inv_freq = 1.0 / (
            base ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.cat((torch.outer(positions, inv_freq),) * 2, dim=-1)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)
        self.rope_dim = rope_dim
        self.max_seq_len = max_seq_len

    def apply_rotary(self, x: Tensor, positions: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError("RoPE input must have shape (batch, heads, tokens, head_dim)")
        if positions.ndim != 1 or positions.numel() != x.shape[-2]:
            raise ValueError("RoPE positions must be a 1D tensor matching token length")
        if positions.numel() and (
            positions.min().item() < 0 or positions.max().item() >= self.max_seq_len
        ):
            raise ValueError("RoPE position exceeds context_length")
        cos = self.cos.index_select(0, positions).to(device=x.device, dtype=x.dtype)
        sin = self.sin.index_select(0, positions).to(device=x.device, dtype=x.dtype)
        cos = cos.view(1, 1, x.shape[-2], self.rope_dim)
        sin = sin.view(1, 1, x.shape[-2], self.rope_dim)
        x_rot, x_pass = x[..., : self.rope_dim], x[..., self.rope_dim :]
        x1, x2 = x_rot.chunk(2, dim=-1)
        rotated = torch.cat((-x2, x1), dim=-1)
        x_rot = (x_rot * cos + rotated * sin).to(dtype=x.dtype)
        return torch.cat((x_rot, x_pass), dim=-1)

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        query_positions: Tensor,
        key_positions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if key_positions is None:
            key_positions = query_positions
        return self.apply_rotary(queries, query_positions), self.apply_rotary(keys, key_positions)

