"""Benchmark cached and uncached generation for architecture checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from benchmarking.compute import collect_kv_cache_stats, estimate_flops, estimate_kv_cache
from benchmarking.inference import run_generation, synchronize
from data.tokenizer import tokenizer_from_state
from models.registry import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="The small model learns")
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tokenizer = tokenizer_from_state(checkpoint["tokenizer"])
    prompt = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    model = build_model(checkpoint["architecture"], checkpoint["model_config"]).to(device).eval()
    best_model_path = args.checkpoint.parent / "best_model.pt"
    model.load_state_dict(
        torch.load(best_model_path, map_location=device, weights_only=True)
        if best_model_path.exists() else checkpoint["model_state"]
    )
    flop_estimate = estimate_flops(model, checkpoint["architecture"])
    cache_sequence_length = min(
        prompt.shape[1] + args.new_tokens,
        int(model.training_cfg["context_length"]),
    )
    kv_estimate = estimate_kv_cache(
        model,
        checkpoint["architecture"],
        sequence_length=cache_sequence_length,
    )

    for use_cache in (True, False):
        for _ in range(args.warmup):
            run_generation(model, prompt, args.new_tokens, use_cache)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        synchronize(device)
        start = time.perf_counter()
        output = run_generation(model, prompt, args.new_tokens, use_cache)
        synchronize(device)
        elapsed = time.perf_counter() - start
        peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        actual_cache = collect_kv_cache_stats(model) if use_cache else {
            "bytes": 0,
            "tensors": 0,
            "tokens": 0,
            "bytes_per_token": 0.0,
        }
        print(json.dumps({
            "architecture": checkpoint["architecture"],
            "use_cache": use_cache,
            "new_tokens": args.new_tokens,
            "seconds": elapsed,
            "tokens_per_second": args.new_tokens / elapsed,
            "estimated_forward_flops_per_token": flop_estimate.forward_flops_per_token,
            "estimated_flops": flop_estimate.forward_flops_per_token * args.new_tokens,
            "estimated_kv_cache": kv_estimate.as_dict(),
            "actual_kv_cache": actual_cache,
            "peak_memory_bytes": peak_memory,
            "output_tokens": output.tolist(),
        }))

    cached = run_generation(model, prompt, args.new_tokens, True)
    uncached = run_generation(model, prompt, args.new_tokens, False)
    print(json.dumps({"cached_uncached_equal": bool(torch.equal(cached, uncached))}))


if __name__ == "__main__":
    main()
