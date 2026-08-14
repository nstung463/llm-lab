"""Validated fixed-token artifacts for controlled architecture experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .tokenizer import ByteLevelBPE


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_artifact_path(manifest_path: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    candidates = [manifest_path.parent / candidate, Path.cwd() / candidate]
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"Artifact path does not exist: {value}")


def _hash_token_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    view = np.ascontiguousarray(array).view(np.uint8)
    digest.update(view)
    return digest.hexdigest()


@dataclass(frozen=True)
class TokenArtifacts:
    """Three fixed token streams, tokenizer and their reproducibility contract."""

    manifest_path: Path
    manifest: dict[str, object]
    tokenizer: ByteLevelBPE
    train_tokens: np.ndarray
    validation_tokens: np.ndarray
    test_tokens: np.ndarray
    contract: dict[str, object]


def load_token_artifacts(manifest_path: Path) -> TokenArtifacts:
    """Load and validate a manifest plus its fixed token arrays."""
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_paths = manifest.get("token_artifacts")
    if not isinstance(artifact_paths, dict):
        raise ValueError("Manifest is missing token_artifacts")
    required = ("train", "validation", "test", "tokenizer")
    if any(key not in artifact_paths for key in required):
        raise ValueError("Manifest token_artifacts must include train, validation, test, tokenizer")

    train_path = _resolve_artifact_path(manifest_path, str(artifact_paths["train"]))
    validation_path = _resolve_artifact_path(manifest_path, str(artifact_paths["validation"]))
    test_path = _resolve_artifact_path(manifest_path, str(artifact_paths["test"]))
    tokenizer_path = _resolve_artifact_path(manifest_path, str(artifact_paths["tokenizer"]))
    tokenizer_sha256 = manifest.get("tokenizer_sha256")
    if tokenizer_sha256 and _sha256_file(tokenizer_path) != tokenizer_sha256:
        raise ValueError("Tokenizer artifact hash does not match manifest")

    arrays = {
        "train": np.load(train_path, mmap_mode="r"),
        "validation": np.load(validation_path, mmap_mode="r"),
        "test": np.load(test_path, mmap_mode="r"),
    }
    expected_counts = {
        "train": manifest.get("train_token_count"),
        "validation": manifest.get("validation_token_count"),
        "test": manifest.get("test_token_count"),
    }
    for name, array in arrays.items():
        if array.ndim != 1:
            raise ValueError(f"{name} token artifact must be one-dimensional")
        expected = expected_counts[name]
        if expected is not None and len(array) != int(expected):
            raise ValueError(f"{name} token count does not match manifest")
        if len(array) == 0:
            raise ValueError(f"{name} token artifact is empty")

    tokenizer = ByteLevelBPE.from_state(
        {"tokenizer_json": tokenizer_path.read_text(encoding="utf-8")}
    )
    for name, array in arrays.items():
        if int(array.min()) < 0 or int(array.max()) >= tokenizer.vocab_size:
            raise ValueError(f"{name} token artifact contains an ID outside tokenizer vocabulary")

    contract = {
        "manifest_sha256": _sha256_file(manifest_path),
        "tokenizer_sha256": _sha256_file(tokenizer_path),
        "train_token_sha256": _hash_token_array(arrays["train"]),
        "validation_token_sha256": _hash_token_array(arrays["validation"]),
        "test_token_sha256": _hash_token_array(arrays["test"]),
        "train_token_count": len(arrays["train"]),
        "validation_token_count": len(arrays["validation"]),
        "test_token_count": len(arrays["test"]),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "token_id_dtype": str(arrays["train"].dtype),
    }
    return TokenArtifacts(
        manifest_path=manifest_path,
        manifest=manifest,
        tokenizer=tokenizer,
        train_tokens=arrays["train"],
        validation_tokens=arrays["validation"],
        test_tokens=arrays["test"],
        contract=contract,
    )
