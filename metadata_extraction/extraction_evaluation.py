"""Score model answers against grounded targets, per example and in aggregate.

Two kinds of error are kept apart, because they call for different fixes:

* an **invented value** is a title, author or metric figure that does not
  occur in the chunk. It is checked against the text, not against the
  labels, so it is the objective measure of hallucination;
* an **over-extraction** is a non-empty answer for a chunk whose label is
  empty. It depends on the labelling policy: a model that summarises the
  aim of a "Related work" paragraph is over-extracting, not inventing.

Key metrics are matched on the figures they quote, not on wording, and a
predicted metric that is grounded in the chunk is not counted as an error:
the gold list is not exhaustive.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Sequence

from metadata_extraction.grounded_training_examples import TrainingExample, comparison_key
from metadata_extraction.paper_metadata_schema import PaperMetadata, parse_model_answer

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
_NON_ALPHANUMERIC_RE = re.compile(r"[^a-z0-9]+")


def _numbers_in(text: str) -> frozenset[str]:
    """The set of numbers written in ``text``."""
    return frozenset(_NUMBER_RE.findall(text))


def _wording_key(text: str) -> str:
    """Letters and digits only: "State-of-the-art" equals "state of the art"."""
    return _NON_ALPHANUMERIC_RE.sub("", text.lower())


def gold_metric_is_recovered(gold_metric: str, predicted_metrics: Sequence[str]) -> bool:
    """A gold metric is found when a prediction quotes all of its figures.

    A gold metric without any figure is compared on wording, ignoring
    punctuation and case.
    """
    gold_numbers = _numbers_in(gold_metric)
    if gold_numbers:
        return any(gold_numbers <= _numbers_in(predicted) for predicted in predicted_metrics)
    gold_wording = _wording_key(gold_metric)
    return any(
        gold_wording in _wording_key(predicted) or _wording_key(predicted) in gold_wording
        for predicted in predicted_metrics
        if _wording_key(predicted)
    )


def count_invented_values(predicted: PaperMetadata, chunk_text: str) -> int:
    """Number of extractive values absent from the chunk (title, authors, metric figures)."""
    text = comparison_key(chunk_text)
    invented = int(bool(predicted.title) and comparison_key(predicted.title or "") not in text)
    invented += sum(comparison_key(name) not in text for name in predicted.main_authors)
    invented += sum(
        any(number not in text for number in _numbers_in(metric)) for metric in predicted.key_metrics
    )
    return invented


def compare_answer_with_target(example: TrainingExample, raw_answer: str) -> dict[str, Any]:
    """Field-by-field comparison of one model answer with its target.

    An unparsable answer is scored as an empty prediction (it extracts
    nothing usable) and is flagged by ``answer_status``.
    """
    parsed, status = parse_model_answer(raw_answer)
    predicted = parsed or PaperMetadata()
    gold = example.target

    gold_authors = {comparison_key(name) for name in gold.main_authors}
    predicted_authors = {comparison_key(name) for name in predicted.main_authors}
    metrics_recovered = sum(gold_metric_is_recovered(metric, predicted.key_metrics) for metric in gold.key_metrics)
    invented_values = count_invented_values(predicted, example.chunk_text)

    row: dict[str, Any] = {
        "chunk_id": example.chunk_id,
        "answer_status": status,
        "target_is_empty": gold.is_empty,
        "title_correct": (
            comparison_key(predicted.title or "") == comparison_key(gold.title) if gold.title else None
        ),
        "wrong_title": bool(predicted.title) and comparison_key(predicted.title or "") != comparison_key(gold.title or ""),
        "authors_found": len(gold_authors & predicted_authors),
        "authors_missed": len(gold_authors - predicted_authors),
        "authors_wrong": len(predicted_authors - gold_authors),
        "metrics_found": metrics_recovered,
        "metrics_missed": len(gold.key_metrics) - metrics_recovered,
        "invented_values": invented_values,
        "problem_statement_agrees": bool(predicted.core_problem_statement) == bool(gold.core_problem_statement),
        "over_extracted": gold.is_empty and not predicted.is_empty,
        "raw_answer": raw_answer,
    }
    row["verifiable_fields_correct"] = (
        status == "ok"
        and not row["wrong_title"]
        and row["title_correct"] is not False
        and not row["authors_missed"]
        and not row["authors_wrong"]
        and not row["metrics_missed"]
        and not invented_values
    )
    row["fully_correct"] = row["verifiable_fields_correct"] and row["problem_statement_agrees"]
    return row


def _rate(flags: Sequence[bool]) -> float | None:
    """Share of true flags, or ``None`` for an empty list."""
    return round(sum(flags) / len(flags), 3) if flags else None


def _f1(found: int, missed: int, wrong: int) -> float | None:
    """F1 from counts, or ``None`` when nothing was expected or produced."""
    denominator = 2 * found + missed + wrong
    return round(2 * found / denominator, 3) if denominator else None


def score_extraction_answers(examples: Sequence[TrainingExample], raw_answers: Sequence[str]) -> dict[str, Any]:
    """Aggregate metrics over paired examples and raw answers.

    Returns:
        ``valid_json_rate``; ``title_accuracy``; ``author_f1``;
        ``key_metric_recall`` (gold figures recovered);
        ``problem_statement_agreement`` (present / absent, against weak labels);
        ``verifiable_fields_correct_rate`` and ``fully_correct_rate``;
        ``invented_value_rate`` (answers containing a value absent from the
        chunk) and ``over_extraction_rate`` (non-empty answers on chunks
        labelled empty); ``answer_status_counts``.
        A metric no example exercises is ``None``.
    """
    rows = [compare_answer_with_target(example, answer) for example, answer in zip(examples, raw_answers)]
    metrics_found = sum(row["metrics_found"] for row in rows)
    metrics_total = metrics_found + sum(row["metrics_missed"] for row in rows)
    return {
        "examples": len(rows),
        "valid_json_rate": _rate([row["answer_status"] == "ok" for row in rows]),
        "title_accuracy": _rate([row["title_correct"] for row in rows if row["title_correct"] is not None]),
        "author_f1": _f1(
            sum(row["authors_found"] for row in rows),
            sum(row["authors_missed"] for row in rows),
            sum(row["authors_wrong"] for row in rows),
        ),
        "key_metric_recall": round(metrics_found / metrics_total, 3) if metrics_total else None,
        "problem_statement_agreement": _rate([row["problem_statement_agrees"] for row in rows]),
        "verifiable_fields_correct_rate": _rate([row["verifiable_fields_correct"] for row in rows]),
        "fully_correct_rate": _rate([row["fully_correct"] for row in rows]),
        "invented_value_rate": _rate([row["invented_values"] > 0 for row in rows]),
        "over_extraction_rate": _rate([row["over_extracted"] for row in rows if row["target_is_empty"]]),
        "answer_status_counts": dict(Counter(row["answer_status"] for row in rows)),
    }


def build_error_analysis(examples: Sequence[TrainingExample], raw_answers: Sequence[str]) -> list[dict[str, Any]]:
    """One readable row per example: what was expected, what went wrong."""
    table = []
    for example, answer in zip(examples, raw_answers):
        row = compare_answer_with_target(example, answer)
        problems = [
            None if row["answer_status"] == "ok" else f"answer {row['answer_status']}",
            "wrong title" if row["wrong_title"] or row["title_correct"] is False else None,
            f"{row['authors_missed']} author(s) missed" if row["authors_missed"] else None,
            f"{row['authors_wrong']} wrong author(s)" if row["authors_wrong"] else None,
            f"{row['metrics_missed']} metric(s) missed" if row["metrics_missed"] else None,
            f"{row['invented_values']} invented value(s)" if row["invented_values"] else None,
            None if row["problem_statement_agrees"] else "problem statement disagrees with the label",
        ]
        issues = [problem for problem in problems if problem]
        table.append(
            {
                "chunk_id": row["chunk_id"],
                "target": "empty" if row["target_is_empty"] else "informative",
                "answer_status": row["answer_status"],
                "verifiable_fields": "correct" if row["verifiable_fields_correct"] else "wrong",
                "verdict": "correct" if not issues else "; ".join(issues),
            }
        )
    return table
