"""Writers for the chunk dataset (JSON, JSON Lines, CSV)."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

from pdf_chunking.chunk_record_schema import ChunkRecord

SUPPORTED_FORMATS = (".json", ".jsonl", ".csv")


def _write_atomically(path: Path, write: Any, newline: str | None = None) -> None:
    """Write through a temporary file, then rename over ``path``.

    A crash or a full disk mid-write therefore never leaves a truncated,
    unparseable dataset where a previous good one used to be.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline=newline) as stream:
            write(stream)
        os.replace(temp_name, path)
    except BaseException:
        if os.path.exists(temp_name):
            os.remove(temp_name)
        raise


def write_chunk_dataset(chunks: Sequence[ChunkRecord], path: Path | str) -> Path:
    """Write chunks as ``.json`` (array), ``.jsonl`` or ``.csv``, by extension.

    Raises:
        ValueError: If the extension is not in ``SUPPORTED_FORMATS``.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported output format {suffix!r}; use one of {', '.join(SUPPORTED_FORMATS)}."
        )
    rows = [chunk.model_dump() for chunk in chunks]

    if suffix == ".json":
        _write_atomically(path, lambda stream: json.dump(rows, stream, ensure_ascii=False, indent=2))
    elif suffix == ".jsonl":

        def write_lines(stream: Any) -> None:
            """Write one JSON object per line."""
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

        _write_atomically(path, write_lines)
    else:

        def write_csv(stream: Any) -> None:
            """Write a header row and one row per chunk."""
            writer = csv.DictWriter(stream, fieldnames=list(ChunkRecord.model_fields))
            writer.writeheader()
            writer.writerows(rows)

        # newline="" lets the csv module control line endings (RFC 4180).
        _write_atomically(path, write_csv, newline="")
    return path.resolve()


def write_json_file(data: dict[str, Any], path: Path | str) -> Path:
    """Write a dictionary (e.g. the run report) as pretty-printed JSON."""
    path = Path(path)
    _write_atomically(path, lambda stream: json.dump(data, stream, ensure_ascii=False, indent=2))
    return path.resolve()
