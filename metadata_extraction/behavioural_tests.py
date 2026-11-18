"""Behavioural tests: properties the extractor must keep, beyond average metrics.

* **Invariance** - reflowing the whitespace of a chunk and appending an
  irrelevant sentence must not change the extracted metadata.
* **Abstention under temptation** - a bibliography chunk is full of author
  names and titles, none of them the paper's own. The right answer is empty.
  The grounding filter cannot help here (the names *are* in the text), which
  is exactly why this case is tested separately.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from metadata_extraction.grounded_training_examples import TrainingExample
from metadata_extraction.paper_metadata_schema import parse_model_answer
from metadata_extraction.statistical_tests import wilson_interval

DISTRACTOR_SENTENCE = "This page intentionally contains no further information."
_WHITESPACE_RE = re.compile(r"\s+")


def perturb_chunk_without_changing_its_meaning(chunk_text: str) -> str:
    """Flatten all whitespace and append a sentence carrying no metadata.

    Args:
        chunk_text: Original text of the chunk.

    Returns:
        The text with every run of whitespace replaced by one space, followed by
        :data:`DISTRACTOR_SENTENCE`.
    """
    return f"{_WHITESPACE_RE.sub(' ', chunk_text).strip()} {DISTRACTOR_SENTENCE}"


def select_invariance_cases(examples: Sequence[TrainingExample], limit: int = 4) -> list[TrainingExample]:
    """Informative examples first: invariance of an empty answer proves little.

    Args:
        examples: Labelled examples to choose from.
        limit: Maximum number of cases returned.

    Returns:
        At most ``limit`` examples, those with a non-empty target first; the input order is
        otherwise kept.
    """
    return sorted(examples, key=lambda example: example.target.is_empty)[:limit]


def select_bibliography_chunks(chunks: Sequence[dict[str, Any]], document: str, limit: int = 6) -> list[dict[str, Any]]:
    """Bibliography chunks of ``document`` (never used for training).

    Args:
        chunks: Chunk records with ``source_document`` and ``content_type``.
        document: File name of the paper, compared with ``source_document``.
        limit: Maximum number of chunks returned.

    Returns:
        The first ``limit`` chunks of the document whose ``content_type`` is ``"references"``.
    """
    return [c for c in chunks if c["source_document"] == document and c["content_type"] == "references"][:limit]


def score_invariance(original_answers: Sequence[str], perturbed_answers: Sequence[str]) -> dict[str, Any]:
    """Share of cases whose parsed metadata is identical before and after perturbation.

    Args:
        original_answers: Raw model answers on the unmodified chunks.
        perturbed_answers: Raw model answers on the perturbed chunks, in the same order. Two
            unparsable answers count as identical.

    Returns:
        ``test`` (name of the test), ``cases``, ``passed``, ``pass_rate`` (``None`` without cases)
        and ``pass_rate_ci95`` (Wilson interval).
    """
    unchanged = [
        parse_model_answer(original)[0] == parse_model_answer(perturbed)[0]
        for original, perturbed in zip(original_answers, perturbed_answers)
    ]
    return {
        "test": "invariance to whitespace and a distractor sentence",
        "cases": len(unchanged),
        "passed": sum(unchanged),
        "pass_rate": round(sum(unchanged) / len(unchanged), 3) if unchanged else None,
        "pass_rate_ci95": wilson_interval(sum(unchanged), len(unchanged)),
    }


def score_abstention_on_bibliography(answers: Sequence[str], test_name: str) -> dict[str, Any]:
    """Share of bibliography chunks answered with an empty, valid object.

    Args:
        answers: Raw model answers, one per bibliography chunk.
        test_name: Label stored under ``test`` in the result.

    Returns:
        ``test`` (name of the test), ``cases``, ``passed``, ``pass_rate`` (``None`` without cases)
        and ``pass_rate_ci95`` (Wilson interval).
    """
    abstained = []
    for answer in answers:
        metadata, status = parse_model_answer(answer)
        abstained.append(status == "ok" and metadata is not None and metadata.is_empty)
    return {
        "test": test_name,
        "cases": len(abstained),
        "passed": sum(abstained),
        "pass_rate": round(sum(abstained) / len(abstained), 3) if abstained else None,
        "pass_rate_ci95": wilson_interval(sum(abstained), len(abstained)),
    }
