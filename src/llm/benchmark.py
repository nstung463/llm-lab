from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .config import ModelConfig
from .models.baseline import GPTModel, count_parameters
from .training.loop import load_checkpoint


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
    return {"median_seconds": statistics.median(values), "p95_seconds": ordered[p95_index], "trials": float(len(values))}


def cache_bytes(model: GPTModel, batch_size: int, sequence_length: int, dtype: torch.dtype) -> int:
    element_bytes = torch.empty((), dtype=dtype).element_size()
    head_dim = model.cfg.emb_dim // model.cfg.n_heads
    return 2 * model.cfg.n_layers * batch_size * model.cfg.n_heads * sequence_length * head_dim * element_bytes


def timed(fn, trials: int, warmup: int, device: torch.device) -> list[float]:
    for _ in range(warmup):
        fn()
    values: list[float] = []
    for _ in range(trials):
        synchronize(device)
        started = time.perf_counter()
        fn()
        synchronize(device)
        values.append(time.perf_counter() - started)
    return values


@torch.inference_mode()
def uncached_decode(model: GPTModel, prompt: torch.Tensor, steps: int) -> None:
    output = prompt.clone()
    for _ in range(steps):
        logits, _ = model(output[:, -model.cfg.context_length :])
        output = torch.cat((output, logits[:, -1].argmax(dim=-1, keepdim=True)), dim=1)


@torch.inference_mode()
def cached_decode(model: GPTModel, prompt: torch.Tensor, steps: int) -> None:
    output = prompt.clone()
    logits, cache = model(output[:, -model.cfg.context_length :], use_cache=True)
    for _ in range(steps):
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        output = torch.cat((output, next_token), dim=1)
        cache_length = cache[0][0].shape[2] if cache and cache[0] is not None else 0
        if cache_length >= model.cfg.context_length:
            logits, cache = model(output[:, -model.cfg.context_length :], use_cache=True)
        else:
            logits, cache = model(next_token, past_key_values=cache, use_cache=True)


def benchmark_scenario(model: GPTModel, prompt: torch.Tensor, steps: int, trials: int, warmup: int, device: torch.device) -> dict[str, object]:
    prefill_uncached = timed(lambda: model(prompt), trials, warmup, device)
    prefill_cached = timed(lambda: model(prompt, use_cache=True), trials, warmup, device)
    uncached = timed(lambda: uncached_decode(model, prompt, steps), trials, warmup, device)
    cached = timed(lambda: cached_decode(model, prompt, steps), trials, warmup, device)
    retained_tokens = min(model.cfg.context_length, prompt.shape[1] + steps)
    return {
        "prompt_tokens": prompt.shape[1],
        "decode_forward_calls": steps,
        "prefill_uncached": summarize(prefill_uncached),
        "prefill_cached": summarize(prefill_cached),
        "uncached_full_context_decode": summarize(uncached),
        "cached_incremental_decode": summarize(cached),
        "uncached_decode_forward_tokens_per_second": steps / statistics.median(uncached),
        "cached_decode_forward_tokens_per_second": steps / statistics.median(cached),
        "decode_speedup": statistics.median(uncached) / statistics.median(cached),
        "retained_kv_cache_bytes": cache_bytes(model, 1, retained_tokens, next(model.parameters()).dtype),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark prefill separately from steady-state KV-cache decode")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--prompt-tokens", type=int, default=8)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("runs/benchmark.json"))
    args = parser.parse_args()
    if args.new_tokens <= 0 or args.prompt_tokens <= 0 or args.trials <= 0 or args.warmup < 0:
        raise ValueError("new-tokens, prompt-tokens, and trials must be positive; warmup cannot be negative")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint, device)
    cfg = ModelConfig(**checkpoint["model_config"])
    model = GPTModel(cfg).to(device).eval()
    model.load_state_dict(checkpoint["model_state"])
    torch.manual_seed(42)
    prompt_length = min(args.prompt_tokens, cfg.context_length)
    prompt = torch.randint(cfg.vocab_size, (1, prompt_length), device=device)
    uncached_output = model.generate_uncached(prompt, args.new_tokens)
    cached_output = model.generate_cached(prompt, args.new_tokens)
    if not torch.equal(uncached_output, cached_output):
        raise AssertionError("Cached and uncached generation outputs differ")

    within_prompt = torch.randint(cfg.vocab_size, (1, min(prompt_length, max(1, cfg.context_length // 4))), device=device)
    within_steps = min(args.new_tokens, max(1, cfg.context_length - within_prompt.shape[1] - 1))
    cross_prompt = torch.randint(cfg.vocab_size, (1, max(1, cfg.context_length - 4)), device=device)
    cross_steps = max(args.new_tokens, cfg.context_length - cross_prompt.shape[1] + 2)
    result = {
        "model": asdict(cfg),
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint["step"],
        "parameters": count_parameters(model),
        "device": str(device),
        "within_window": benchmark_scenario(model, within_prompt, within_steps, args.trials, args.warmup, device),
        "cross_window": benchmark_scenario(model, cross_prompt, cross_steps, args.trials, args.warmup, device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
