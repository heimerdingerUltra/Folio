"""Validate an exported chunk dataset against the schema.

Usage:
    python -m pdf_chunking.dataset_validation_cli results.json [--max-tokens 400]

Useful as a CI gate, or before loading a dataset produced by someone else
into a vector store. Exits with 0 if the dataset is valid, 1 otherwise.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from pdf_chunking.chunk_record_schema import ChunkRecord, validate_chunk_dataset

_INTEGER_FIELDS = tuple(
    name for name, info in ChunkRecord.model_fields.items() if info.annotation is int
)


def load_records(path: Path) -> list[dict[str, Any]]:
    """Read chunk records from a ``.json``, ``.jsonl`` or ``.csv`` file.

    Args:
        path: Dataset file, read as UTF-8. Its extension (case-insensitive) selects the format.

    Returns:
        One dictionary per record, not yet validated. For CSV, the integer fields of the schema
        are converted back from text when they hold digits; blank JSON Lines are skipped.

    Raises:
        ValueError: If the extension is unsupported or the JSON is not a list.
    """
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Expected a JSON array of chunk records.")
        return data
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    if suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        # CSV has no types: restore integers so the schema can check them.
        for row in rows:
            for name in _INTEGER_FIELDS:
                if (row.get(name) or "").lstrip("-").isdigit():
                    row[name] = int(row[name])
        return rows
    raise ValueError(f"Unsupported file type: {suffix}")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit code.

    Args:
        argv: Command-line arguments without the program name. None reads ``sys.argv``.

    Returns:
        0 if the dataset is valid, 1 if it is not or if the file cannot be read.
    """
    parser = argparse.ArgumentParser(description="Validate a chunk dataset.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Also flag chunks above this token count.")  # fmt: skip
    args = parser.parse_args(argv)
    try:
        records = load_records(args.dataset)
    except (OSError, ValueError) as exc:
        print(f"Cannot read {args.dataset}: {exc}", file=sys.stderr)
        return 1
    summary = validate_chunk_dataset(records, max_tokens=args.max_tokens)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["is_valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
