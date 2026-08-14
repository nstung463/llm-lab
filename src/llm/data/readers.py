"""Read raw text into a list of documents.

The reader deliberately does not tokenize or split data. Those operations belong
to the next stages of the data pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path


def read_documents(path: Path) -> list[str]:
    """Read JSONL records with a ``text`` field or blank-line text documents."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        documents: list[str] = []
        for line_number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise ValueError(f"JSONL record at {path}:{line_number} must contain string 'text'")
            text = record["text"].strip()
            if text:
                documents.append(text)
    else:
        documents = [chunk.strip() for chunk in raw.split("\n\n") if chunk.strip()]

    if not documents:
        raise ValueError(f"No documents found in {path}")
    return documents
