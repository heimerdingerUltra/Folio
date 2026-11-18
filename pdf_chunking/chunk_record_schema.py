"""Schema of an exported chunk, and dataset-level validation.

Every chunk is built through :class:`ChunkRecord`, so a malformed one is
caught before it reaches an embedding model or a vector store.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

ContentType = Literal["text", "table", "references"]

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class ChunkRecord(BaseModel):
    """One chunk of a source document, ready for embedding.

    Attributes:
        chunk_id: Stable id ``<document-slug>-<index>``; identical across re-runs.
        source_document: File name of the PDF.
        document_title: Extracted title (best effort).
        page_number: 1-based page where the chunk starts.
        page_end: 1-based page where it ends.
        section: Enclosing headings, outermost first, joined by `` > ``.
        content_type: ``text``, ``table`` or ``references``.
        chunk_index: 0-based position within the document.
        token_count: Tokens in ``chunk_text``.
        char_count: Characters in ``chunk_text``.
        chunk_text: The chunk itself.
        embedding_text: Text to index: ``chunk_text`` prefixed with the document
            title and section path, which measurably improves dense retrieval.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(pattern=r"^[a-z0-9_]+-\d{4,}$")
    source_document: str = Field(min_length=1)
    document_title: str
    page_number: int = Field(ge=1)
    page_end: int = Field(ge=1)
    section: str
    content_type: ContentType
    chunk_index: int = Field(ge=0)
    token_count: int = Field(ge=1)
    char_count: int = Field(ge=1)
    chunk_text: str = Field(min_length=1)
    embedding_text: str = Field(min_length=1)

    @field_validator("chunk_text")
    @classmethod
    def _text_is_meaningful(cls, value: str) -> str:
        """Reject whitespace-only, symbol-only or binary-looking text."""
        if value != value.strip():
            raise ValueError("chunk_text must not start or end with whitespace")
        if _CONTROL_CHARS_RE.search(value):
            raise ValueError("chunk_text contains control characters")
        if not any(char.isalnum() for char in value):
            raise ValueError("chunk_text contains no alphanumeric character")
        return value

    @model_validator(mode="after")
    def _fields_are_consistent(self) -> ChunkRecord:
        """Cross-field checks: page range and character count."""
        if self.page_end < self.page_number:
            raise ValueError("page_end must be >= page_number")
        if self.char_count != len(self.chunk_text):
            raise ValueError("char_count does not match len(chunk_text)")
        if not self.embedding_text.endswith(self.chunk_text):
            raise ValueError("embedding_text must end with chunk_text")
        return self


def validate_chunk_dataset(
    records: Iterable[ChunkRecord | dict[str, Any]],
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Check records (models or raw dicts) individually and as a dataset.

    Dataset checks: unique ids, contiguous ``chunk_index`` per document, and
    the token budget when ``max_tokens`` is given.

    Returns:
        JSON-serialisable summary with ``is_valid`` and up to 20 ``errors``.
    """
    errors: list[str] = []
    valid: list[ChunkRecord] = []
    n_records = 0
    for position, record in enumerate(records):
        n_records += 1
        if isinstance(record, ChunkRecord):
            valid.append(record)
            continue
        try:
            valid.append(ChunkRecord.model_validate(record))
        except ValidationError as exc:
            first = exc.errors()[0]
            location = ".".join(str(part) for part in first["loc"]) or "record"
            errors.append(f"record {position}: {location}: {first['msg']}")
    schema_errors = len(errors)

    # Chunk ids must be globally unique: they are the primary key downstream.
    id_counts = Counter(record.chunk_id for record in valid)
    duplicate_ids = sorted(chunk_id for chunk_id, count in id_counts.items() if count > 1)
    errors.extend(f"duplicate chunk_id: {chunk_id}" for chunk_id in duplicate_ids)

    # Within a document, chunk_index must be 0..n-1 with no gap, otherwise a
    # chunk was lost between chunking and export.
    indices: dict[str, list[int]] = defaultdict(list)
    for record in valid:
        indices[record.source_document].append(record.chunk_index)
    non_contiguous = sorted(
        document
        for document, values in indices.items()
        if sorted(values) != list(range(len(values)))
    )
    errors.extend(f"non-contiguous chunk_index in {document}" for document in non_contiguous)

    over_budget = (
        [record.chunk_id for record in valid if record.token_count > max_tokens]
        if max_tokens is not None
        else []
    )
    errors.extend(f"chunk over token budget: {chunk_id}" for chunk_id in over_budget)

    return {
        "is_valid": not errors,
        "records_checked": n_records,
        "schema_errors": schema_errors,
        "duplicate_ids": len(duplicate_ids),
        "documents_with_non_contiguous_indices": len(non_contiguous),
        "chunks_over_token_budget": len(over_budget),
        "errors": errors[:20],
    }
