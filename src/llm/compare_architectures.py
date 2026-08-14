"""Compare trained architecture checkpoints on the same held-out test split."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .data import load_token_artifacts
from .data.datasets import make_loaders
from .data.readers import read_documents
from .data.splits import split_documents_three
from .data.tokenizer import tokenizer_from_state
from .benchmarking.compute import active_parameter_count, estimate_flops
from .evaluation.loss import evaluate_loss_stats
from .models.registry import available_architectures, build_model, model_metadata, parameter_count


def evaluate(model, loader, device, max_batches: int | None):
    """Return token-weighted next-token cross entropy and perplexity."""
    stats = evaluate_loss_stats(model, loader, device, max_batches)
    loss = float(stats["loss"])
    return {"test_loss": loss, "perplexity": math.exp(min(loss, 20.0)), "tokens": int(stats["tokens"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--run-prefix", default="architecture_compare_", help="Checkpoint directory prefix.")
    parser.add_argument("--text", type=Path, default=None)
    parser.add_argument("--artifact-manifest", type=Path, default=None)
    parser.add_argument("--architectures", nargs="+", default=list(available_architectures()))
    parser.add_argument("--eval-batches", type=int, default=None, help="Optional limit; default evaluates the full test artifact.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    artifacts = load_token_artifacts(args.artifact_manifest) if args.artifact_manifest else None
    if artifacts is None and args.text is None:
        raise ValueError("Provide --text or --artifact-manifest")
    documents = read_documents(args.text) if args.text is not None else None
    results = []
    for architecture in args.architectures:
        checkpoint_path = args.checkpoint_root / f"{args.run_prefix}{architecture}" / "checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        seed = int(checkpoint["training_config"]["seed"])
        if artifacts is not None:
            if checkpoint.get("data_contract") != artifacts.contract:
                raise ValueError(
                    f"Checkpoint {checkpoint_path} does not match artifact manifest contract"
                )
            test_tokens = artifacts.test_tokens
        else:
            train_fraction = float(checkpoint.get("train_fraction", 0.8))
            validation_fraction = float(checkpoint.get("validation_fraction", 0.1))
            _, _, test_documents = split_documents_three(documents, train_fraction, validation_fraction, seed)
            tokenizer = tokenizer_from_state(checkpoint["tokenizer"])
            test_tokens = tokenizer.encode_documents(test_documents)
        _, loader = make_loaders(
            test_tokens,
            test_tokens,
            checkpoint["model_config"]["context_length"],
            checkpoint["training_config"]["batch_size"],
            seed,
        )
        model = build_model(architecture, checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model_state"])
        metrics = evaluate(model, loader, device, args.eval_batches)
        metadata = model_metadata(architecture, model)
        results.append({
            "architecture": architecture,
            "family": metadata.family,
            "module": metadata.module,
            "parameters": parameter_count(model),
            "active_parameters": active_parameter_count(model, architecture),
            "flops": estimate_flops(model, architecture).as_dict(),
            **metrics,
        })

    results.sort(key=lambda item: item["test_loss"])
    print(json.dumps(results, indent=2))
    (args.checkpoint_root / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
