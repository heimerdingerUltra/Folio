"""Tests for the Pydantic schemas and dataset-level validation."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from pdf_chunking.chunk_record_schema import ChunkRecord, validate_chunk_dataset


def make_record(**overrides: Any) -> dict[str, Any]:
    """Return a valid chunk record as a dict, with optional overrides."""
    text = overrides.pop("chunk_text", "A valid chunk of text.")
    record: dict[str, Any] = {
        "chunk_id": "paper-0000",
        "source_document": "paper.pdf",
        "document_title": "A Paper",
        "page_number": 1,
        "page_end": 1,
        "section": "1 Introduction",
        "content_type": "text",
        "chunk_index": 0,
        "token_count": 6,
        "char_count": len(text),
        "chunk_text": text,
        "embedding_text": f"A Paper - 1 Introduction\n\n{text}",
    }
    record.update(overrides)
    return record


class TestChunkRecord:
    def test_accepts_valid_record(self) -> None:
        assert ChunkRecord.model_validate(make_record()).chunk_id == "paper-0000"

    @pytest.mark.parametrize(
        "overrides",
        [
            {"chunk_id": "Paper 0"},  # bad id format
            {"page_number": 0},  # pages are 1-based
            {"page_number": 3, "page_end": 2},  # range reversed
            {"content_type": "image"},  # unknown type
            {"chunk_text": "   ", "char_count": 3},  # whitespace only
            {"chunk_text": "-- ** --", "char_count": 8},  # no alphanumeric character
            {"chunk_text": "bad\x00text", "char_count": 8},  # control character
            {"char_count": 999},  # inconsistent with the text
            {"token_count": 0},
            {"unexpected_field": 1},  # extra fields are rejected
        ],
    )
    def test_rejects_invalid_record(self, overrides: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            ChunkRecord.model_validate(make_record(**overrides))

    def test_is_immutable(self) -> None:
        record = ChunkRecord.model_validate(make_record())
        with pytest.raises(ValidationError):
            record.page_number = 5  # type: ignore[misc]


class TestValidateDataset:
    def test_valid_dataset(self) -> None:
        records = [
            make_record(),
            make_record(chunk_id="paper-0001", chunk_index=1),
        ]
        summary = validate_chunk_dataset(records, max_tokens=400)
        assert summary["is_valid"] is True
        assert summary["records_checked"] == 2

    def test_detects_duplicate_ids(self) -> None:
        summary = validate_chunk_dataset([make_record(), make_record(chunk_index=1)])
        assert summary["is_valid"] is False
        assert summary["duplicate_ids"] == 1

    def test_detects_gap_in_indices(self) -> None:
        records = [make_record(), make_record(chunk_id="paper-0002", chunk_index=2)]
        summary = validate_chunk_dataset(records)
        assert summary["documents_with_non_contiguous_indices"] == 1

    def test_detects_chunks_over_budget(self) -> None:
        summary = validate_chunk_dataset([make_record(token_count=900)], max_tokens=400)
        assert summary["chunks_over_token_budget"] == 1

    def test_reports_schema_errors_without_raising(self) -> None:
        summary = validate_chunk_dataset([make_record(page_number="first")])
        assert summary["schema_errors"] == 1
        assert "page_number" in summary["errors"][0]
