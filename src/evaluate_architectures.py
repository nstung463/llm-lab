"""Evaluate a checkpoint produced by ``train_architectures``."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from data import load_token_artifacts
from data.datasets import make_loaders
from data.readers import read_documents
from data.splits import split_documents_three
from data.tokenizer import tokenizer_from_state
from evaluation.loss import evaluate_loss_stats
from models.registry import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text", type=Path, default=None)
    parser.add_argument("--artifact-manifest", type=Path, default=None)
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tokenizer = tokenizer_from_state(checkpoint["tokenizer"])
    seed = int(checkpoint["training_config"]["seed"])
    artifacts = load_token_artifacts(args.artifact_manifest) if args.artifact_manifest else None
    if artifacts is not None:
        if checkpoint.get("data_contract") != artifacts.contract:
            raise ValueError("Checkpoint does not match artifact manifest contract")
        tokens = artifacts.validation_tokens
    else:
        if args.text is None:
            raise ValueError("Provide --text or --artifact-manifest")
        documents = read_documents(args.text)
        train_documents, validation_documents, _ = split_documents_three(
            documents,
            float(checkpoint.get("train_fraction", 0.8)),
            float(checkpoint.get("validation_fraction", 0.1)),
            seed,
        )
        del train_documents
        tokens = tokenizer.encode_documents(validation_documents)
    model_cfg = checkpoint["model_config"]
    _, loader = make_loaders(tokens, tokens, model_cfg["context_length"], checkpoint["training_config"]["batch_size"], seed)
    model = build_model(checkpoint["architecture"], model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    stats = evaluate_loss_stats(model, loader, device, args.eval_batches)
    loss = float(stats["loss"])
    print(json.dumps({"architecture": checkpoint["architecture"], "validation_loss": loss, "perplexity": math.exp(min(loss, 20.0)), "batches": stats["batches"], "tokens": stats["tokens"], "device": str(device)}, indent=2))


if __name__ == "__main__":
    main()
