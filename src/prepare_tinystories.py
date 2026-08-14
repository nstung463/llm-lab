from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from data.datasets import take_token_budget
from data.manifest import build_manifest
from data.splits import deduplicate_documents, split_documents_three
from data.tokenizer import build_tokenizer
from .train import TINYSTORIES_SOURCE


def _token_sha256(values: list[int]) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(values, dtype=np.uint32)).view(np.uint8)).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream a licensed TinyStories subset into JSONL and optional fixed-token artifacts"
    )
    parser.add_argument("--output", type=Path, default=Path("data/tinystories_sample.jsonl"))
    parser.add_argument("--max-examples", type=int, default=10_000)
    parser.add_argument("--target-train-tokens", type=int, default=None)
    parser.add_argument("--validation-tokens", type=int, default=None)
    parser.add_argument("--test-tokens", type=int, default=None)
    parser.add_argument("--tokenizer-vocab-size", type=int, default=16_000)
    parser.add_argument("--tokenizer-min-frequency", type=int, default=2)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_examples <= 1:
        raise ValueError("max-examples must be greater than one")
    if args.target_train_tokens is not None and args.target_train_tokens <= 0:
        raise ValueError("target-train-tokens must be positive")
    if args.validation_tokens is not None and args.validation_tokens <= 0:
        raise ValueError("validation-tokens must be positive")
    if args.test_tokens is not None and args.test_tokens <= 0:
        raise ValueError("test-tokens must be positive")
    if args.target_train_tokens is None and (args.validation_tokens or args.test_tokens):
        raise ValueError("target-train-tokens is required when validation/test token budgets are set")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the data extra first: uv sync --extra dev --extra data") from exc

    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    documents: list[str] = []
    count = 0
    for row in dataset:
        text = str(row["text"]).strip()
        if not text:
            continue
        documents.append(text)
        count += 1
        if count >= args.max_examples:
            break

    documents = deduplicate_documents(documents)
    if len(documents) < 3:
        raise ValueError("TinyStories stream returned fewer than three non-empty documents")

    with args.output.open("w", encoding="utf-8") as handle:
        for text in documents:
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    if args.target_train_tokens is None:
        metadata = {
            "source": TINYSTORIES_SOURCE,
            "document_count": count,
            "data_file": str(args.output),
        }
        args.output.with_suffix(".manifest.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print(json.dumps(metadata, indent=2))
        return

    train_documents, validation_documents, test_documents = split_documents_three(
        documents, args.train_fraction, args.validation_fraction, args.seed
    )
    tokenizer = build_tokenizer(
        "bpe",
        train_documents,
        args.tokenizer_vocab_size,
        args.tokenizer_min_frequency,
    )
    train_ids_full = tokenizer.encode_documents(train_documents)
    validation_ids_full = tokenizer.encode_documents(validation_documents)
    test_ids_full = tokenizer.encode_documents(test_documents)
    validation_budget = args.validation_tokens or max(args.target_train_tokens // 20, 1)
    test_budget = args.test_tokens or max(args.target_train_tokens // 20, 1)
    train_ids = take_token_budget(train_ids_full, args.target_train_tokens)
    validation_ids = take_token_budget(validation_ids_full, validation_budget)
    test_ids = take_token_budget(test_ids_full, test_budget)

    tokenizer_path = args.output.with_suffix(".tokenizer.json")
    tokenizer.save(tokenizer_path)
    tokenizer_sha256 = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    train_path = args.output.with_suffix(".train.npy")
    validation_path = args.output.with_suffix(".validation.npy")
    test_path = args.output.with_suffix(".test.npy")
    np.save(train_path, np.asarray(train_ids, dtype=np.uint32))
    np.save(validation_path, np.asarray(validation_ids, dtype=np.uint32))
    np.save(test_path, np.asarray(test_ids, dtype=np.uint32))

    manifest = build_manifest(
        documents,
        train_documents,
        validation_documents,
        TINYSTORIES_SOURCE,
        args.seed,
        args.train_fraction,
        test_documents,
        tokenizer_kind="byte_level_bpe",
        tokenizer_vocab_size=tokenizer.vocab_size,
        tokenizer_sha256=tokenizer_sha256,
        train_token_count=len(train_ids),
        validation_token_count=len(validation_ids),
        test_token_count=len(test_ids),
        target_train_tokens=args.target_train_tokens,
        target_validation_tokens=validation_budget,
        target_test_tokens=test_budget,
        token_id_dtype="uint32",
        train_token_sha256=_token_sha256(train_ids),
        validation_token_sha256=_token_sha256(validation_ids),
        test_token_sha256=_token_sha256(test_ids),
        validation_fraction=args.validation_fraction,
    )
    manifest_data = manifest.to_dict()
    manifest_data["data_file"] = str(args.output)
    manifest_data["token_artifacts"] = {
        "train": str(train_path),
        "validation": str(validation_path),
        "test": str(test_path),
        "tokenizer": str(tokenizer_path),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    print(json.dumps(manifest_data, indent=2))


if __name__ == "__main__":
    main()
