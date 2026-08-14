"""Shared cached and uncached autoregressive generation helpers."""

from __future__ import annotations

import torch


def run_generation(model, prompt, max_new_tokens: int, use_cache: bool):
    """Generate greedily through the common model contract."""
    model.eval()
    output = prompt.clone()
    if hasattr(model, "reset_kv_cache"):
        model.reset_kv_cache()
    with torch.no_grad():
        if use_cache:
            context_length = int(model.training_cfg["context_length"])
            cached_tokens = min(output.shape[1], context_length)
            logits = model(output[:, -cached_tokens:], use_cache=True)
            for _ in range(max_new_tokens):
                next_token = logits[:, -1:].argmax(dim=-1)
                output = torch.cat([output, next_token], dim=1)
                if cached_tokens >= context_length:
                    model.reset_kv_cache()
                    cached_tokens = context_length
                    logits = model(output[:, -cached_tokens:], use_cache=True)
                else:
                    logits = model(next_token, use_cache=True)
                    cached_tokens += 1
        else:
            for _ in range(max_new_tokens):
                logits = model(
                    output[:, -model.training_cfg["context_length"] :],
                    use_cache=False,
                )
                next_token = logits[:, -1:].argmax(dim=-1)
                output = torch.cat([output, next_token], dim=1)
    return output


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


__all__ = ["run_generation", "synchronize"]
