import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import BPETokenizer, build_manifest, load_token_artifacts, split_documents
from data.datasets import NextTokenDataset, StatefulBatchLoader, take_token_budget
from data.readers import read_documents
from data.splits import deduplicate_documents, split_documents_three


def test_duplicate_documents_are_removed_before_splitting() -> None:
    documents = ["same", "other", "same", "third"]
    assert deduplicate_documents(documents) == ["same", "other", "third"]
    train, validation, test = split_documents_three(documents, 0.6, 0.2, seed=2)
    assert sum(len(part) for part in (train, validation, test)) == 3
    assert set(train).isdisjoint(validation)
    assert set(train).isdisjoint(test)
    assert set(validation).isdisjoint(test)


def test_bpe_and_split_are_deterministic() -> None:
    documents = [f"A small story about a fox number {index}." for index in range(12)]
    train_a, val_a = split_documents(documents, 0.75, seed=7)
    train_b, val_b = split_documents(documents, 0.75, seed=7)
    assert train_a == train_b and val_a == val_b
    tokenizer = BPETokenizer.fit(train_a, vocab_size=300, min_frequency=1)
    ids = tokenizer.encode_documents(train_a)
    assert tokenizer.vocab_size >= 258
    assert tokenizer.eos_id in ids
    assert tokenizer.decode(ids)
    assert "validation-only-secret" not in tokenizer.tokenizer.get_vocab()


def test_plain_text_reader_and_split_have_no_document_overlap(tmp_path: Path) -> None:
    path = tmp_path / "stories.txt"
    path.write_text("first story\n\nsecond story\n\nthird story", encoding="utf-8")
    documents = read_documents(path)
    train, validation, test = split_documents_three(documents, 0.6, 0.2, seed=4)

    assert documents == ["first story", "second story", "third story"]
    assert set(train).isdisjoint(validation)
    assert set(train).isdisjoint(test)
    assert set(validation).isdisjoint(test)
    assert sorted(train + validation + test) == sorted(documents)


def test_manifest_records_source_and_counts() -> None:
    documents = ["one", "two", "three", "four"]
    train_documents, validation_documents = split_documents(documents, 0.75, seed=1)
    manifest = build_manifest(
        documents,
        train_documents,
        validation_documents,
        {"id": "example", "url": "https://example.invalid", "license": "MIT", "license_url": "https://opensource.org/license/mit"},
        1,
        0.75,
    )
    assert manifest.document_count == 4
    assert manifest.train_document_count == 3
    assert manifest.validation_document_count == 1
    assert manifest.train_token_count is None
    assert manifest.tokenizer_kind is None


def test_token_budget_is_exact_and_rejects_short_streams() -> None:
    assert take_token_budget(range(10), 4) == [0, 1, 2, 3]
    try:
        take_token_budget(range(3), 4)
    except ValueError as error:
        assert "fewer than target budget" in str(error)
    else:
        raise AssertionError("short token streams must be rejected")


def test_jsonl_to_next_token_batch(tmp_path: Path) -> None:
    path = tmp_path / "stories.jsonl"
    path.write_text(
        "\n".join(f'{{"text": "Story number {index} about a fox."}}' for index in range(12)),
        encoding="utf-8",
    )
    documents = read_documents(path)
    train, validation, test = split_documents_three(documents, 0.75, 0.1, seed=3)
    tokenizer = BPETokenizer.fit(train, vocab_size=300, min_frequency=1)
    dataset = NextTokenDataset(tokenizer.encode_documents(train), context_length=8, stride=8)
    inputs, labels = next(iter(DataLoader(dataset, batch_size=2)))

    assert len(train) + len(validation) + len(test) == len(documents)
    assert inputs.shape == labels.shape == (2, 8)
    assert torch.equal(inputs[:, 1:], labels[:, :-1])
    assert dataset.starts[-1] + dataset.context_length < len(dataset.tokens)


def test_batch_loader_resume_keeps_order_and_cursor() -> None:
    dataset = NextTokenDataset(list(range(100)), context_length=8, stride=8)
    loader = StatefulBatchLoader(dataset, batch_size=3, shuffle=True, seed=11)
    next(iter(loader))
    state = loader.state_dict()

    restored = StatefulBatchLoader(dataset, batch_size=3, shuffle=True, seed=999)
    restored.load_state_dict(state)
    expected = next(iter(loader))
    actual = next(iter(restored))

    torch.testing.assert_close(expected[0], actual[0])
    torch.testing.assert_close(expected[1], actual[1])


def test_fixed_token_artifact_loader_validates_manifest_contract(tmp_path: Path) -> None:
    tokenizer = BPETokenizer.fit(["a tiny story", "another tiny story"], vocab_size=300, min_frequency=1)
    artifact_dir = tmp_path / "data"
    artifact_dir.mkdir()
    tokenizer_path = artifact_dir / "tokens.tokenizer.json"
    tokenizer.save(tokenizer_path)
    streams = {
        "train": np.asarray(tokenizer.encode_documents(["a tiny story"]), dtype=np.uint32),
        "validation": np.asarray(tokenizer.encode_documents(["another tiny story"]), dtype=np.uint32),
        "test": np.asarray(tokenizer.encode_documents(["a tiny story"]), dtype=np.uint32),
    }
    paths = {}
    for name, values in streams.items():
        path = artifact_dir / f"tokens.{name}.npy"
        np.save(path, values)
        paths[name] = f"data\\tokens.{name}.npy"
    paths["tokenizer"] = "data\\tokens.tokenizer.json"
    manifest_path = tmp_path / "tokens.manifest.json"
    manifest_path.write_text(
        json.dumps({
            "document_count": 4,
            "document_sha256": "source",
            "split_seed": 42,
            "train_fraction": 0.5,
            "train_document_count": 2,
            "validation_document_count": 1,
            "test_document_count": 1,
            "train_token_count": len(streams["train"]),
            "validation_token_count": len(streams["validation"]),
            "test_token_count": len(streams["test"]),
            "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
            "token_artifacts": {**paths, "tokenizer": str(tokenizer_path)},
        }),
        encoding="utf-8",
    )

    artifacts = load_token_artifacts(manifest_path)
    assert artifacts.contract["train_token_count"] == len(streams["train"])
    assert artifacts.tokenizer.vocab_size == tokenizer.vocab_size
    np.testing.assert_array_equal(artifacts.test_tokens, streams["test"])
