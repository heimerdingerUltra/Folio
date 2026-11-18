"""Checks a training set must pass before any GPU time is spent on it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from metadata_extraction.gold_paper_metadata import GOLD_PAPER_METADATA
from metadata_extraction.grounded_training_examples import TrainingExample, comparison_key
from metadata_extraction.paper_metadata_schema import PaperMetadata


@dataclass(frozen=True)
class ValidationCheck:
    """Outcome of one check."""

    name: str
    passed: bool
    detail: str


class TrainingDataValidationError(ValueError):
    """Raised when the training set fails a blocking check."""


def _check(name: str, passed: bool, detail: str) -> ValidationCheck:
    """Build one check result."""
    return ValidationCheck(name=name, passed=bool(passed), detail=detail)


def validate_training_data(
    all_examples: Sequence[TrainingExample],
    train_examples: Sequence[TrainingExample],
    validation_examples: Sequence[TrainingExample],
    gold_by_document: dict[str, dict[str, Any]] | None = None,
    test_examples: Sequence[TrainingExample] = (),
) -> list[ValidationCheck]:
    """Run every data check and return the results (never raises).

    ``validation_examples`` and ``test_examples`` are both held out: neither
    may share a document, a chunk or near-identical text with the training set.

    Args:
        all_examples: Full labelled set, used for the grounding and gold-coverage checks.
        train_examples: Examples the model is fitted on.
        validation_examples: Held-out examples used for model selection.
        gold_by_document: PDF file name -> gold metadata (``authors``, ``metrics`` with ``text``).
            ``None`` or an empty dict selects ``GOLD_PAPER_METADATA``.
        test_examples: Held-out examples used for the final report; none by default.

    Returns:
        One result per check, always in the same order, whether it passed or not.
    """
    gold_by_document = gold_by_document or GOLD_PAPER_METADATA
    held_out_examples = [*validation_examples, *test_examples]
    train_documents = {example.document for example in train_examples}
    validation_documents = {example.document for example in held_out_examples}
    train_ids = {example.chunk_id for example in train_examples}
    validation_ids = {example.chunk_id for example in held_out_examples}
    selected = [*train_examples, *held_out_examples]
    select_and_report_overlap = {e.document for e in validation_examples} & {e.document for e in test_examples}

    def word_set(example: TrainingExample) -> frozenset[str]:
        """The set of normalised words of an example's chunk."""
        return frozenset(comparison_key(example.chunk_text).split())

    train_words = [word_set(example) for example in train_examples]
    highest_overlap = max(
        (len(words & other) / len(words | other) for words in map(word_set, held_out_examples) for other in train_words),
        default=0.0,
    )

    def schema_is_valid(example: TrainingExample) -> bool:
        """Does the target survive a round trip through the schema unchanged?"""
        return PaperMetadata.model_validate(example.target.model_dump()) == example.target

    def is_grounded(example: TrainingExample) -> bool:
        """Are the target's title and author names present in the chunk?"""
        text = comparison_key(example.chunk_text)
        names_present = all(comparison_key(name) in text for name in example.target.main_authors)
        title_present = not example.target.title or comparison_key(example.target.title) in text
        return names_present and title_present

    uncovered = []
    for document, gold in gold_by_document.items():
        targets = [example.target for example in all_examples if example.document == document]
        if targets and (
            {name for target in targets for name in target.main_authors} != set(gold["authors"])
            or {metric for target in targets for metric in target.key_metrics}
            != {metric["text"] for metric in gold["metrics"]}
            or not any(target.title for target in targets)
            or not any(target.core_problem_statement for target in targets)
        ):
            uncovered.append(document)

    def informative_share(examples: Sequence[TrainingExample]) -> float:
        """Share of examples whose target is not empty."""
        return sum(not example.target.is_empty for example in examples) / max(len(examples), 1)

    return [
        _check("targets satisfy the schema", all(map(schema_is_valid, selected)), f"{len(selected)} targets checked"),
        _check(
            "no document shared by train and validation",
            not train_documents & validation_documents,
            f"train={sorted(train_documents)}, validation={sorted(validation_documents)}",
        ),
        _check("no chunk shared by train and validation", not train_ids & validation_ids, "by chunk_id"),
        _check(
            "no duplicated chunk inside a split",
            len(train_ids) == len(train_examples) and len(validation_ids) == len(held_out_examples),
            f"train={len(train_examples)}, held out={len(held_out_examples)}",
        ),
        _check(
            "validation and test papers are different",
            not select_and_report_overlap,
            "model selection and final report use separate papers",
        ),
        _check(
            "no near-duplicate text between train and held-out chunks",
            highest_overlap < 0.8,
            f"highest word-set Jaccard similarity = {highest_overlap:.2f} (threshold 0.80)",
        ),
        _check("no empty chunk text", all(example.chunk_text.strip() for example in selected), "non-blank inputs"),
        _check(
            "titles and authors are grounded in their chunk",
            all(map(is_grounded, all_examples)),
            "each labelled string occurs in the chunk text",
        ),
        _check(
            "every gold field is supported by some chunk",
            not uncovered,
            "all documents covered" if not uncovered else f"missing coverage: {uncovered}",
        ),
        _check(
            "both classes present in each split",
            all(0.25 <= informative_share(split) <= 0.75 for split in (train_examples, held_out_examples)),
            f"informative share: train={informative_share(train_examples):.2f}, "
            f"held out={informative_share(held_out_examples):.2f}",
        ),
    ]


def raise_if_any_check_failed(checks: Sequence[ValidationCheck]) -> None:
    """Stop the experiment when the data is not fit for training.

    Args:
        checks: Results of :func:`validate_training_data`.

    Raises:
        TrainingDataValidationError: If at least one check failed. The message lists every
            failed check with its detail.
    """
    failed = [check for check in checks if not check.passed]
    if failed:
        raise TrainingDataValidationError("; ".join(f"{check.name} ({check.detail})" for check in failed))
