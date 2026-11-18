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
    """Flatten all whitespace and append a sentence carrying no metadata."""
    return f"{_WHITESPACE_RE.sub(' ', chunk_text).strip()} {DISTRACTOR_SENTENCE}"


def select_invariance_cases(examples: Sequence[TrainingExample], limit: int = 4) -> list[TrainingExample]:
    """Informative examples first: invariance of an empty answer proves little."""
    return sorted(examples, key=lambda example: example.target.is_empty)[:limit]


def select_bibliography_chunks(chunks: Sequence[dict[str, Any]], document: str, limit: int = 6) -> list[dict[str, Any]]:
    """Bibliography chunks of ``document`` (never used for training)."""
    return [c for c in chunks if c["source_document"] == document and c["content_type"] == "references"][:limit]


def score_invariance(original_answers: Sequence[str], perturbed_answers: Sequence[str]) -> dict[str, Any]:
    """Share of cases whose parsed metadata is identical before and after perturbation."""
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
    """Share of bibliography chunks answered with an empty, valid object."""
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
