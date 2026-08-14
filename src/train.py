from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from config import ModelConfig, TrainingConfig, validate_resume_config
from data.datasets import make_loaders
from data.manifest import build_manifest
from data.readers import read_documents
from data.splits import deduplicate_documents, split_documents_three
from data.tokenizer import build_tokenizer, tokenizer_from_state
from models.baseline import GPTModel, count_parameters
from training.loop import evaluate, load_checkpoint, restore_rng_state, save_checkpoint, train
from training.optim import build_adamw


TINYSTORIES_SOURCE = {
    "id": "roneneldan/TinyStories",
    "url": "https://huggingface.co/datasets/roneneldan/TinyStories",
    "license": "CDLA-Sharing-1.0",
    "license_url": "https://cdla.dev/sharing-1-0/",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the BPE GPT baseline")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True, help="Plain text or JSONL file with one document per record")
    parser.add_argument("--output", type=Path, default=Path("runs/baseline"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--source-id", default=None)
    parser.add_argument("--source-url", default=None)
    parser.add_argument("--license", dest="license_name", default=None)
    parser.add_argument("--license-url", default=None)
    parser.add_argument("--tokenizer-vocab-size", type=int, default=512)
    parser.add_argument("--tokenizer-min-frequency", type=int, default=2)
    parser.add_argument("--tokenizer", choices=("bpe", "tiktoken"), default="bpe")
    parser.add_argument("--tokenizer-name", choices=("gpt2", "r50k_base"), default="r50k_base")
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    training_cfg = TrainingConfig(**raw["training"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(training_cfg.seed)
    documents = deduplicate_documents(read_documents(args.text))
    source_metadata_path = args.text.with_suffix(".manifest.json")
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8")) if source_metadata_path.exists() else {}
    source = dict(source_metadata.get("source", {}))
    source.update({
        "id": args.source_id or source.get("id", "local-corpus"),
        "url": args.source_url or source.get("url", "local"),
        "license": args.license_name or source.get("license", "verify-before-use"),
        "license_url": args.license_url or source.get("license_url", ""),
    })
    validation_fraction = args.validation_fraction
    train_documents, validation_documents, test_documents = split_documents_three(
        documents, args.train_fraction, validation_fraction, training_cfg.seed
    )
    manifest = build_manifest(
        documents, train_documents, validation_documents, source, training_cfg.seed,
        args.train_fraction, test_documents, validation_fraction=validation_fraction
    )
    checkpoint = None

    if args.resume is None:
        tokenizer = build_tokenizer(
            args.tokenizer,
            train_documents,
            args.tokenizer_vocab_size,
            args.tokenizer_min_frequency,
            args.tokenizer_name,
        )
        model_cfg = ModelConfig(**{**raw["model"], "vocab_size": tokenizer.vocab_size})
        train_tokens = tokenizer.encode_documents(train_documents)
        validation_tokens = tokenizer.encode_documents(validation_documents)
        test_tokens = tokenizer.encode_documents(test_documents)
        model = GPTModel(model_cfg)
        optimizer = build_adamw(model, training_cfg)
        start_step = 0
        tokens_seen = 0
        history: list[dict[str, float]] = []
    else:
        checkpoint = load_checkpoint(args.resume, device)
        saved_manifest = checkpoint["data_manifest"]
        if not saved_manifest.get("test_document_count"):
            raise ValueError("Resume checkpoint predates the three-way train/validation/test split")
        if saved_manifest["document_sha256"] != manifest.document_sha256:
            raise ValueError("Resume data does not match checkpoint document_sha256")
        if (
            saved_manifest["split_seed"] != manifest.split_seed
            or saved_manifest["train_fraction"] != manifest.train_fraction
            or saved_manifest.get("validation_fraction") != manifest.validation_fraction
        ):
            raise ValueError("Resume split settings do not match checkpoint")
        tokenizer = tokenizer_from_state(checkpoint["tokenizer"])
        model_cfg = ModelConfig(**checkpoint["model_config"])
        train_tokens = tokenizer.encode_documents(train_documents)
        validation_tokens = tokenizer.encode_documents(validation_documents)
        test_tokens = tokenizer.encode_documents(test_documents)
        model = GPTModel(model_cfg)
        model.load_state_dict(checkpoint["model_state"])
        optimizer = build_adamw(model, training_cfg)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = int(checkpoint["step"])
        tokens_seen = int(checkpoint.get("tokens_seen", 0))
        validate_resume_config(checkpoint["training_config"], training_cfg, start_step)
        history = list(checkpoint["history"])
        restore_rng_state(checkpoint.get("rng_state", checkpoint))

    use_amp = device.type == "cuda" and training_cfg.precision != "fp32"
    if training_cfg.precision == "bf16":
        amp_dtype = torch.bfloat16
    elif training_cfg.precision == "fp16":
        amp_dtype = torch.float16
    elif use_amp and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    if checkpoint is not None and checkpoint.get("scaler_state"):
        scaler.load_state_dict(checkpoint["scaler_state"])

    train_loader, val_loader = make_loaders(
        train_tokens, validation_tokens, model_cfg.context_length,
        training_cfg.batch_size, training_cfg.seed
    )
    _, test_loader = make_loaders(
        train_tokens, test_tokens, model_cfg.context_length,
        training_cfg.batch_size, training_cfg.seed
    )
    if checkpoint is not None and checkpoint.get("loader_state"):
        train_loader.load_state_dict(checkpoint["loader_state"]["train"])
        val_loader.load_state_dict(checkpoint["loader_state"]["validation"])
        if checkpoint["loader_state"].get("test") is not None:
            test_loader.load_state_dict(checkpoint["loader_state"]["test"])
    args.output.mkdir(parents=True, exist_ok=True)

    def checkpoint_callback(
        step: int,
        current_optimizer: torch.optim.Optimizer,
        current_history: list[dict[str, float]],
        current_tokens_seen: int,
    ) -> None:
        save_checkpoint(
            args.output / "checkpoint.pt",
            model,
            training_cfg,
            tokenizer.to_state(),
            current_history,
            current_optimizer,
            step,
            manifest.to_dict(),
            loader_state={
                "train": train_loader.state_dict(),
                "validation": val_loader.state_dict(),
                "test": test_loader.state_dict(),
            },
            scaler=scaler,
            tokens_seen=current_tokens_seen,
        )

    history = train(
        model,
        train_loader,
        val_loader,
        training_cfg,
        device,
        optimizer,
        start_step,
        history,
        checkpoint_callback,
        scaler=scaler,
        tokens_seen=tokens_seen,
    )
    tokens_seen = int(history[-1].get("tokens_seen", tokens_seen)) if history else tokens_seen
    test_loss = evaluate(model, test_loader, device, None)
    save_checkpoint(
        args.output / "checkpoint.pt", model, training_cfg, tokenizer.to_state(),
        history, optimizer, training_cfg.max_steps, manifest.to_dict(),
        loader_state={
            "train": train_loader.state_dict(),
            "validation": val_loader.state_dict(),
            "test": test_loader.state_dict(),
        },
        scaler=scaler, tokens_seen=tokens_seen, test_loss=test_loss,
    )
    tokenizer.save(args.output / "tokenizer.json")
    (args.output / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.output / "manifest.json").write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    (args.output / "test_metrics.json").write_text(
        json.dumps({"test_loss": test_loss, "perplexity": float(torch.exp(torch.tensor(min(test_loss, 20.0))))}, indent=2),
        encoding="utf-8",
    )
    (args.output / "run.json").write_text(json.dumps({"model": asdict(model_cfg), "training": asdict(training_cfg), "device": str(device), "parameters": count_parameters(model), "test_loss": test_loss, "manifest": manifest.to_dict()}, indent=2), encoding="utf-8")
    print(json.dumps({"parameters": count_parameters(model), "device": str(device), "last": history[-1], "test_loss": test_loss, "output": str(args.output), "manifest": manifest.to_dict()}, indent=2))


if __name__ == "__main__":
    main()
