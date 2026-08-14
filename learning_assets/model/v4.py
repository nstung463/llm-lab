"""Readable, educational DeepSeek-V4-style decoder.

This file intentionally uses ordinary PyTorch layers. It keeps the main ideas
of the V4 inference code—low-rank Q/KV projections, compressed KV caching,
local-plus-historical attention, and shared-expert MoE—without custom FP4/FP8
kernels, distributed sharding, mHC, or DSpark speculative decoding.
"""

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMSNorm used by modern decoder-only language models."""

    def __init__(self, emb_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms * self.weight).to(dtype=x.dtype)


class RotaryEmbedding(nn.Module):
    """Simple RoPE implementation for the first ``rope_dim`` dimensions."""

    def __init__(self, rope_dim, max_seq_len, base=10000.0):
        super().__init__()
        assert rope_dim % 2 == 0, "rope_dim must be even"
        inv_freq = 1.0 / (base ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x, positions):
        # x: (batch, heads, seq_len, rope_dim)
        cos = self.cos[positions].to(dtype=x.dtype)[None, None, :, :]
        sin = self.sin[positions].to(dtype=x.dtype)[None, None, :, :]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated = torch.stack(
            [x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1
        )
        return rotated.flatten(-2)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block."""

    def __init__(self, emb_dim, hidden_dim, dtype=None):
        super().__init__()
        self.gate_proj = nn.Linear(emb_dim, hidden_dim, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(emb_dim, hidden_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(hidden_dim, emb_dim, bias=False, dtype=dtype)

    def forward(self, x):
        hidden = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.out_proj(hidden)


class Compressor(nn.Module):
    """Compress consecutive latent KV tokens into one learned summary."""

    def __init__(self, latent_dim, compression_ratio):
        super().__init__()
        self.latent_dim = latent_dim
        self.ratio = compression_ratio
        self.score_proj = nn.Linear(latent_dim, 1, bias=False)
        self.position_bias = nn.Parameter(torch.zeros(compression_ratio))
        self.pending_latent = None
        self.pending_positions = None

    def reset(self):
        self.pending_latent = None
        self.pending_positions = None

    def _compress(self, latent, positions):
        batch, num_blocks, ratio, dim = latent.shape
        scores = self.score_proj(latent.float()).squeeze(-1)
        scores = scores + self.position_bias.float()
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        compressed = (latent.float() * weights).sum(dim=2).to(dtype=latent.dtype)
        compressed_positions = positions[:, :, -1]
        return compressed, compressed_positions

    def forward(self, latent_new, positions):
        """Return newly completed compressed blocks and their end positions."""
        if self.pending_latent is not None:
            latent_new = torch.cat([self.pending_latent, latent_new], dim=1)
            positions = torch.cat([self.pending_positions, positions], dim=1)

        num_complete = latent_new.size(1) // self.ratio
        cutoff = num_complete * self.ratio
        if num_complete:
            blocks = latent_new[:, :cutoff].view(
                latent_new.size(0), num_complete, self.ratio, self.latent_dim
            )
            block_positions = positions[:, :cutoff].view(
                positions.size(0), num_complete, self.ratio
            )
            compressed, compressed_positions = self._compress(blocks, block_positions)
        else:
            compressed = latent_new[:, :0]
            compressed_positions = positions[:, :0]

        self.pending_latent = latent_new[:, cutoff:]
        self.pending_positions = positions[:, cutoff:]
        return compressed, compressed_positions


class Indexer(nn.Module):
    """Select top-k compressed KV positions for each query token."""

    def __init__(self, top_k):
        super().__init__()
        self.top_k = top_k

    def forward(self, queries, compressed_keys, query_positions, compressed_positions):
        if compressed_keys.size(-2) == 0:
            return torch.zeros(
                queries.size(0), queries.size(2), 0, dtype=torch.bool, device=queries.device
            )

        # Average head scores to produce one routing score per query/position.
        scores = (queries.unsqueeze(-2) * compressed_keys.unsqueeze(2)).sum(dim=-1)
        scores = scores.mean(dim=1)
        causal = compressed_positions[:, None, :] <= query_positions[:, :, None]
        scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)

        k = min(self.top_k, compressed_keys.size(-2))
        topk = scores.topk(k, dim=-1).indices
        selected = torch.zeros_like(scores, dtype=torch.bool)
        selected.scatter_(-1, topk, True)
        return selected & causal


class CompressedSparseAttention(nn.Module):
    """Hybrid local + compressed sparse attention in a readable form."""

    def __init__(self, cfg, layer_id):
        super().__init__()
        self.emb_dim = cfg["emb_dim"]
        self.num_heads = cfg["n_heads"]
        self.head_dim = cfg["head_dim"]
        self.q_lora_rank = cfg["q_lora_rank"]
        self.latent_dim = cfg["latent_dim"]
        self.rope_dim = cfg["rope_dim"]
        self.window_size = cfg["window_size"]
        self.compression_ratio = cfg["compress_ratios"][layer_id]
        self.index_topk = cfg["index_topk"]
        dtype = cfg.get("dtype")

        assert self.head_dim % 2 == 0
        assert self.rope_dim % 2 == 0
        assert self.rope_dim <= self.head_dim

        # Low-rank query path: x -> q_latent -> query heads.
        self.q_down = nn.Linear(self.emb_dim, self.q_lora_rank, bias=False, dtype=dtype)
        self.q_norm = RMSNorm(self.q_lora_rank)
        self.q_up = nn.Linear(
            self.q_lora_rank,
            self.num_heads * self.head_dim,
            bias=False,
            dtype=dtype,
        )

        # KV path: x -> compressed latent -> reconstructed K and V.
        self.kv_down = nn.Linear(self.emb_dim, self.latent_dim, bias=False, dtype=dtype)
        self.key_up = nn.Linear(
            self.latent_dim,
            self.num_heads * self.head_dim,
            bias=False,
            dtype=dtype,
        )
        self.value_up = nn.Linear(
            self.latent_dim,
            self.num_heads * self.head_dim,
            bias=False,
            dtype=dtype,
        )
        self.out_proj = nn.Linear(
            self.num_heads * self.head_dim, self.emb_dim, bias=False, dtype=dtype
        )
        self.rope = RotaryEmbedding(
            self.rope_dim, cfg["context_length"], cfg["rope_base"]
        )
        self.dropout = nn.Dropout(cfg["drop_rate"])

        # Local cache stores full K/V for the recent window.
        # Compressed cache stores one latent per completed compression block.
        self.local_kv_cache = None
        self.compressed_kv_cache = None
        self.compressed_positions = None
        self.compressor = (
            Compressor(self.latent_dim, self.compression_ratio)
            if self.compression_ratio > 0
            else None
        )
        self.indexer = Indexer(self.index_topk) if self.compressor else None
        self.cache_start_pos = 0

    def reset_cache(self):
        self.local_kv_cache = None
        self.compressed_kv_cache = None
        self.compressed_positions = None
        if self.compressor:
            self.compressor.reset()
        self.cache_start_pos = 0

    def _split_heads(self, x, num_tokens):
        return x.view(x.size(0), num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, use_cache=False):
        batch, num_tokens, _ = x.shape
        query_positions = torch.arange(
            self.cache_start_pos,
            self.cache_start_pos + num_tokens,
            device=x.device,
        )

        # Query projection is computed only for the current input tokens.
        queries = self._split_heads(
            self.q_up(self.q_norm(self.q_down(x))), num_tokens
        )

        # Build K/V for current tokens and append them to the local cache.
        latent_new = self.kv_down(x)
        current_keys = self._split_heads(self.key_up(latent_new), num_tokens)
        current_values = self._split_heads(self.value_up(latent_new), num_tokens)
        if self.local_kv_cache is None:
            local_keys, local_values = current_keys, current_values
        else:
            local_keys = torch.cat([self.local_kv_cache[0], current_keys], dim=2)
            local_values = torch.cat([self.local_kv_cache[1], current_values], dim=2)
        self.cache_start_pos += num_tokens

        # Keep only recent full-resolution K/V; older history is compressed.
        if local_keys.size(2) > self.window_size:
            local_keys = local_keys[:, :, -self.window_size :]
            local_values = local_values[:, :, -self.window_size :]
        self.local_kv_cache = (local_keys, local_values)

        # Compress completed latent blocks and append them to compressed cache.
        if self.compressor:
            new_compressed, new_positions = self.compressor(
                latent_new, query_positions[None, :].expand(batch, num_tokens)
            )
            if self.compressed_kv_cache is None:
                self.compressed_kv_cache = new_compressed
                self.compressed_positions = new_positions
            else:
                self.compressed_kv_cache = torch.cat(
                    [self.compressed_kv_cache, new_compressed], dim=1
                )
                self.compressed_positions = torch.cat(
                    [self.compressed_positions, new_positions], dim=1
                )

        local_len = local_keys.size(2)
        local_start = self.cache_start_pos - local_len
        local_positions = torch.arange(
            local_start, self.cache_start_pos, device=x.device
        )

        # Apply RoPE to local keys and queries.
        queries_rope = self.rope(queries[..., -self.rope_dim :], query_positions)
        local_rope = self.rope(
            local_keys[..., -self.rope_dim :], local_positions
        )
        queries = torch.cat([queries[..., :-self.rope_dim], queries_rope], dim=-1)
        local_keys = torch.cat([local_keys[..., :-self.rope_dim], local_rope], dim=-1)

        # Historical compressed KV is selected by the learned Indexer.
        if self.compressor and self.compressed_kv_cache.size(1):
            compressed_keys = self._split_heads(
                self.key_up(self.compressed_kv_cache), self.compressed_kv_cache.size(1)
            )
            compressed_values = self._split_heads(
                self.value_up(self.compressed_kv_cache), self.compressed_kv_cache.size(1)
            )
            compressed_rope = self.rope(
                compressed_keys[..., -self.rope_dim :], self.compressed_positions[0]
            )
            compressed_keys = torch.cat(
                [compressed_keys[..., :-self.rope_dim], compressed_rope], dim=-1
            )
            historical_mask = self.indexer(
                queries,
                compressed_keys,
                query_positions[None].expand(batch, -1),
                self.compressed_positions,
            )
        else:
            compressed_keys = local_keys[:, :, :0]
            compressed_values = local_values[:, :, :0]
            historical_mask = torch.zeros(
                batch, num_tokens, 0, dtype=torch.bool, device=x.device
            )

        keys = torch.cat([local_keys, compressed_keys], dim=2)
        values = torch.cat([local_values, compressed_values], dim=2)
        key_positions = torch.cat(
            [local_positions, self.compressed_positions[0] if self.compressor and self.compressed_positions is not None else local_positions[:0]]
        )

        scores = queries @ keys.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        local_allowed = local_positions[None, None, :] <= query_positions[None, :, None]
        allowed = torch.cat([local_allowed.expand(batch, -1, -1), historical_mask], dim=-1)
        scores = scores.masked_fill(~allowed[:, None], torch.finfo(scores.dtype).min)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        context = weights @ values
        context = context.transpose(1, 2).contiguous().view(batch, num_tokens, -1)
        output = self.out_proj(context)
        if not use_cache:
            self.reset_cache()
        return output


class MoEFeedForward(nn.Module):
    """Shared expert plus top-k routed SwiGLU experts."""

    def __init__(self, cfg):
        super().__init__()
        self.emb_dim = cfg["emb_dim"]
        self.hidden_dim = cfg["moe_hidden_dim"]
        self.num_experts = cfg["num_experts"]
        self.top_k = cfg["num_experts_per_tok"]
        dtype = cfg.get("dtype")
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError("num_experts_per_tok must be between 1 and num_experts")

        self.router = nn.Linear(self.emb_dim, self.num_experts, bias=False, dtype=dtype)
        self.router_bias = nn.Parameter(torch.zeros(self.num_experts))
        self.experts = nn.ModuleList(
            [SwiGLU(self.emb_dim, self.hidden_dim, dtype) for _ in range(self.num_experts)]
        )
        self.shared_expert = SwiGLU(
            self.emb_dim, cfg.get("shared_hidden_dim", self.hidden_dim), dtype
        )

    def forward(self, x):
        original_shape = x.shape
        x_flat = x.reshape(-1, self.emb_dim)

        # Bias selects experts; original scores determine contribution weights.
        raw_scores = self.router(x_flat).float()
        selection_scores = F.softplus(raw_scores).sqrt() + self.router_bias.float()
        topk_indices = selection_scores.topk(self.top_k, dim=-1).indices
        topk_weights = torch.gather(F.softplus(raw_scores).sqrt(), -1, topk_indices)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(dtype=x.dtype)

        routed = torch.zeros_like(x_flat)
        for expert_id in torch.unique(topk_indices).tolist():
            selected = (topk_indices == expert_id).any(dim=-1).nonzero().squeeze(-1)
            expert_output = self.experts[expert_id](x_flat.index_select(0, selected))
            expert_slots = (topk_indices[selected] == expert_id).int().argmax(dim=-1)
            weights = topk_weights[selected, expert_slots].unsqueeze(-1)
            routed.index_add_(0, selected, expert_output * weights)

        shared = self.shared_expert(x_flat)
        return (routed + shared).view(original_shape)


class TransformerBlock(nn.Module):
    """Pre-norm attention + MoE block with ordinary residual connections."""

    def __init__(self, cfg, layer_id):
        super().__init__()
        self.attn_norm = RMSNorm(cfg["emb_dim"], cfg["norm_eps"])
        self.ffn_norm = RMSNorm(cfg["emb_dim"], cfg["norm_eps"])
        self.attn = CompressedSparseAttention(cfg, layer_id)
        self.ffn = MoEFeedForward(cfg)
        self.dropout = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, use_cache=False):
        x = x + self.dropout(self.attn(self.attn_norm(x), use_cache=use_cache))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x

    def reset_cache(self):
        self.attn.reset_cache()


class GPTModel(nn.Module):
    """Readable V4-style decoder model."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        dtype = cfg.get("dtype")
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=dtype)
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, layer_id) for layer_id in range(cfg["n_layers"])]
        )
        self.final_norm = RMSNorm(cfg["emb_dim"], cfg["norm_eps"])
        self.lm_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=dtype)

    def forward(self, input_ids, use_cache=False):
        x = self.drop_emb(self.tok_emb(input_ids))
        for block in self.blocks:
            x = block(x, use_cache=use_cache)
        return self.lm_head(self.final_norm(x))

    def reset_kv_cache(self):
        for block in self.blocks:
            block.reset_cache()


@torch.no_grad()
def generate_text(model, input_ids, max_new_tokens, use_cache=True):
    """Greedy generation helper; cached and uncached paths should agree."""
    model.eval()
    generated = input_ids.clone()
    if use_cache:
        model.reset_kv_cache()
        logits = model(generated, use_cache=True)
        for _ in range(max_new_tokens):
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            logits = model(next_token, use_cache=True)
    else:
        for _ in range(max_new_tokens):
            logits = model(generated, use_cache=False)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
    return generated


def build_config(args):
    return {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "head_dim": args.head_dim,
        "q_lora_rank": args.q_lora_rank,
        "latent_dim": args.latent_dim,
        "rope_dim": args.rope_dim,
        "rope_base": args.rope_base,
        "window_size": args.window_size,
        "compress_ratios": args.compress_ratios,
        "index_topk": args.index_topk,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "moe_hidden_dim": args.moe_hidden_dim,
        "shared_hidden_dim": args.shared_hidden_dim,
        "num_experts": args.num_experts,
        "num_experts_per_tok": args.num_experts_per_tok,
        "drop_rate": 0.0,
        "norm_eps": 1e-6,
        "dtype": torch.float32,
    }


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--emb_dim", type=int, default=768)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--q_lora_rank", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=192)
    parser.add_argument("--rope_dim", type=int, default=32)
    parser.add_argument("--rope_base", type=float, default=10000.0)
    parser.add_argument("--window_size", type=int, default=128)
    parser.add_argument(
        "--compress_ratios",
        type=int,
        nargs="+",
        default=[0, 0, 4, 128, 4, 128, 4, 0, 0, 0, 0, 0],
        help="Compression ratio for each transformer layer; 0 means local-only attention.",
    )
    parser.add_argument("--index_topk", type=int, default=16, help="Top-k compressed positions per query.")
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--moe_hidden_dim", type=int, default=1024)
    parser.add_argument("--shared_hidden_dim", type=int, default=1024)
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--num_experts_per_tok", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    args = parser.parse_args()
    if len(args.compress_ratios) != args.n_layers:
        raise ValueError("compress_ratios must contain exactly n_layers values")

    torch.manual_seed(123)
    model = GPTModel(build_config(args))
    prompt = torch.randint(0, args.vocab_size, (1, 8))
    output = generate_text(model, prompt, args.max_new_tokens)
    print("Input shape:", tuple(prompt.shape))
    print("Output shape:", tuple(output.shape))
    print("Output tokens:", output.tolist())


if __name__ == "__main__":
    main()
