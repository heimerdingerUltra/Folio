"""Compare extraction approaches on the test paper and select one by rule.

An approach is a way of producing answers (rules, zero-shot, few-shot,
fine-tuned) combined with an answer guard. All are scored on the same
chunks, with confidence intervals and a paired significance test, and the
winner is picked by an explicit, ordered criterion.
"""

from __future__ import annotations

from typing import Any, Sequence

from metadata_extraction.extraction_evaluation import compare_answer_with_target, score_extraction_answers
from metadata_extraction.grounded_training_examples import TrainingExample
from metadata_extraction.hallucination_guards import ANSWER_GUARDS
from metadata_extraction.statistical_tests import exact_mcnemar_test, wilson_interval

SELECTION_RULE = (
    "lowest invented-value rate, then highest share of chunks with all verifiable fields correct, "
    "then lowest over-extraction rate, then highest valid-JSON rate, then the simplest system"
)


def _verifiable_flags(examples: Sequence[TrainingExample], answers: Sequence[str]) -> list[bool]:
    """Per example: are all verifiable fields of the answer correct?"""
    return [
        compare_answer_with_target(example, answer)["verifiable_fields_correct"]
        for example, answer in zip(examples, answers)
    ]


def build_approach_leaderboard(
    examples: Sequence[TrainingExample],
    answers_by_approach: dict[str, Sequence[str]],
    reference_approach: str,
) -> list[dict[str, Any]]:
    """Score every approach under every answer guard.

    Args:
        examples: Labelled chunks every approach was run on.
        answers_by_approach: Approach name -> raw model answers, one per example and in the same
            order as ``examples``.
        reference_approach: Key of ``answers_by_approach`` whose unguarded answers are the baseline
            of the McNemar test.

    Returns:
        One row per (approach, guard). ``verifiable_correct`` counts chunks
        whose title, authors and metric figures are all right, with a Wilson
        interval; ``p_value_vs_reference`` is an exact McNemar test on those
        paired outcomes against the unguarded reference approach.
    """
    reference_flags = _verifiable_flags(examples, answers_by_approach[reference_approach])
    rows = []
    for approach, raw_answers in answers_by_approach.items():
        for guard_name, guard in ANSWER_GUARDS.items():
            answers = [guard(answer, example.chunk_text) for answer, example in zip(raw_answers, examples)]
            metrics = score_extraction_answers(examples, answers)
            flags = _verifiable_flags(examples, answers)
            is_reference = approach == reference_approach and guard_name == "no guard"
            rows.append(
                {
                    "approach": approach,
                    "answer_guard": guard_name,
                    "verifiable_correct": sum(flags),
                    "examples": len(flags),
                    "verifiable_correct_rate": round(sum(flags) / len(flags), 3),
                    "verifiable_correct_ci95": wilson_interval(sum(flags), len(flags)),
                    "fully_correct_rate": metrics["fully_correct_rate"],
                    "invented_value_rate": metrics["invented_value_rate"],
                    "over_extraction_rate": metrics["over_extraction_rate"],
                    "valid_json_rate": metrics["valid_json_rate"],
                    "author_f1": metrics["author_f1"],
                    "key_metric_recall": metrics["key_metric_recall"],
                    "p_value_vs_reference": None if is_reference else exact_mcnemar_test(reference_flags, flags)["p_value"],
                }
            )
    return rows


def select_best_approach(leaderboard: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Apply :data:`SELECTION_RULE` and return the winning row.

    Args:
        leaderboard: Non-empty list of rows from :func:`build_approach_leaderboard`. A rate that
            is ``None`` is ranked as 0.

    Returns:
        The row ranked first; ties on all rates go to the guard listed first in ``ANSWER_GUARDS``.
    """
    guard_order = list(ANSWER_GUARDS)
    return min(
        leaderboard,
        key=lambda row: (
            row["invented_value_rate"] or 0.0,
            -row["verifiable_correct_rate"],
            row["over_extraction_rate"] or 0.0,
            -(row["valid_json_rate"] or 0.0),
            guard_order.index(row["answer_guard"]),
        ),
    )
