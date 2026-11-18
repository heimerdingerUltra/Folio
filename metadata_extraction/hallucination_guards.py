"""Deterministic guards applied to a model answer before it is trusted.

A fine-tuned model lowers the hallucination rate; it cannot bring it to
zero. These guards are cheap, model-independent and verifiable:

* **JSON repair** - recover the object from code fences or surrounding prose
  and drop keys outside the schema, instead of losing the whole answer.
* **Grounding filter** - title, authors and metric figures are *extractive*:
  if a predicted value does not occur in the chunk, it was invented, so it
  is removed. The free-text problem statement is abstractive and is left
  untouched.
"""

from __future__ import annotations

import json
import re
from typing import Callable

from metadata_extraction.grounded_training_examples import comparison_key
from metadata_extraction.paper_metadata_schema import PaperMetadata, parse_model_answer

_JSON_OBJECT_RE = re.compile(r"\{.*\}", flags=re.DOTALL)
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
_SCHEMA_FIELDS = tuple(PaperMetadata.model_fields)

#: Turns (raw answer, chunk text) into the answer that is finally scored.
AnswerGuard = Callable[[str, str], str]


def repair_json_answer(raw_answer: str) -> str:
    """Return a schema-conforming JSON string when the answer can be salvaged.

    Handles prose or code fences around the object, unknown keys, a string
    where a list is expected, and ``null`` lists. Unsalvageable answers are
    returned unchanged (and will be scored as invalid).

    Args:
        raw_answer: Model output as text.

    Returns:
        A JSON object with exactly the schema keys, or ``raw_answer`` itself when no JSON object
        can be read from it.
    """
    match = _JSON_OBJECT_RE.search(raw_answer)
    if not match:
        return raw_answer
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return raw_answer
    if not isinstance(payload, dict):
        return raw_answer
    cleaned = {field: payload.get(field) for field in _SCHEMA_FIELDS}
    for field in ("main_authors", "key_metrics"):
        value = cleaned[field]
        cleaned[field] = [value] if isinstance(value, str) else [str(item) for item in value or []]
    for field in ("title", "core_problem_statement"):
        cleaned[field] = cleaned[field] if isinstance(cleaned[field], str) else None
    return json.dumps(cleaned, ensure_ascii=False)


def remove_ungrounded_fields(metadata: PaperMetadata, chunk_text: str) -> PaperMetadata:
    """Drop extractive values that do not occur in the chunk.

    A title or author must appear verbatim (case- and whitespace-insensitive).
    A metric is kept when every number it quotes appears in the chunk; a
    metric without any number cannot be verified and is kept.

    Args:
        metadata: Parsed model answer.
        chunk_text: Text of the chunk the answer was produced from.

    Returns:
        A new record without the ungrounded values; the problem statement is copied as is.
    """
    text = comparison_key(chunk_text)

    def metric_is_grounded(metric: str) -> bool:
        """Does every number of ``metric`` occur in the chunk?"""
        numbers = _NUMBER_RE.findall(metric)
        return all(number in text for number in numbers) if numbers else True

    return PaperMetadata(
        title=metadata.title if metadata.title and comparison_key(metadata.title) in text else None,
        main_authors=[name for name in metadata.main_authors if comparison_key(name) in text],
        core_problem_statement=metadata.core_problem_statement,
        key_metrics=[metric for metric in metadata.key_metrics if metric_is_grounded(metric)],
    )


def keep_answer_unchanged(raw_answer: str, chunk_text: str) -> str:
    """No guard: the raw model answer.

    Args:
        raw_answer: Model output as text.
        chunk_text: Unused; present so that the function matches :data:`AnswerGuard`.

    Returns:
        ``raw_answer`` as given.
    """
    return raw_answer


def repair_json_only(raw_answer: str, chunk_text: str) -> str:
    """JSON repair, no content check.

    Args:
        raw_answer: Model output as text.
        chunk_text: Unused; present so that the function matches :data:`AnswerGuard`.

    Returns:
        The repaired JSON string, or ``raw_answer`` when it cannot be salvaged.
    """
    return repair_json_answer(raw_answer)


def repair_json_and_remove_ungrounded_fields(raw_answer: str, chunk_text: str) -> str:
    """JSON repair, then the grounding filter.

    Args:
        raw_answer: Model output as text.
        chunk_text: Text of the chunk, against which the extractive values are checked.

    Returns:
        JSON of the filtered metadata, or the repaired answer as is when it still does not
        validate against the schema.
    """
    repaired = repair_json_answer(raw_answer)
    metadata, status = parse_model_answer(repaired)
    if status != "ok" or metadata is None:
        return repaired
    return remove_ungrounded_fields(metadata, chunk_text).model_dump_json()


ANSWER_GUARDS: dict[str, AnswerGuard] = {
    "no guard": keep_answer_unchanged,
    "JSON repair": repair_json_only,
    "JSON repair + grounding filter": repair_json_and_remove_ungrounded_fields,
}
