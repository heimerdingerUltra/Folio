"""Schema of the JSON object the model must produce, and its parser."""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

AnswerStatus = Literal["ok", "no_json", "invalid_json", "schema_error"]

_JSON_OBJECT_RE = re.compile(r"\{.*\}", flags=re.DOTALL)


class PaperMetadata(BaseModel):
    """Metadata supported by one chunk; unsupported fields are ``None`` / ``[]``."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    main_authors: list[str] = Field(default_factory=list)
    core_problem_statement: str | None = None
    key_metrics: list[str] = Field(default_factory=list)

    @field_validator("title", "core_problem_statement")
    @classmethod
    def _blank_string_means_absent(cls, value: str | None) -> str | None:
        """Normalise an empty or blank string to ``None``."""
        return (value or "").strip() or None

    @field_validator("main_authors", "key_metrics")
    @classmethod
    def _strip_and_deduplicate(cls, values: list[str]) -> list[str]:
        """Strip items, drop empty ones, remove duplicates in order."""
        return list(dict.fromkeys(item.strip() for item in values if item.strip()))

    @property
    def is_empty(self) -> bool:
        """True when the chunk supports no field at all."""
        return not (self.title or self.main_authors or self.core_problem_statement or self.key_metrics)


def parse_model_answer(raw_answer: str) -> tuple[PaperMetadata | None, AnswerStatus]:
    """Parse a model answer, tolerating text around the JSON object.

    Args:
        raw_answer: Model output as text. The span from its first ``{`` to its last ``}`` is
            parsed.

    Returns:
        ``(metadata, status)``. ``metadata`` is ``None`` unless ``status`` is ``"ok"``. The other
        statuses are ``"no_json"`` (no braces found), ``"invalid_json"`` (the span is not JSON)
        and ``"schema_error"`` (unknown key or wrong type).
    """
    match = _JSON_OBJECT_RE.search(raw_answer)
    if not match:
        return None, "no_json"
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, "invalid_json"
    try:
        return PaperMetadata.model_validate(payload), "ok"
    except ValidationError:
        return None, "schema_error"
