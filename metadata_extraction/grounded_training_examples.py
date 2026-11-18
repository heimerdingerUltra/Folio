"""Turn document-level gold metadata into chunk-level training examples.

The model sees one chunk at a time. Labelling every chunk with the full
metadata would teach it to invent authors from a methods paragraph, so a
chunk is given a field only when its own text supports it.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from metadata_extraction.gold_paper_metadata import GOLD_PAPER_METADATA
from metadata_extraction.paper_metadata_schema import PaperMetadata

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class TrainingExample:
    """One chunk and the metadata it supports."""

    chunk_id: str
    document: str
    chunk_text: str
    target: PaperMetadata


def comparison_key(text: str) -> str:
    """Case- and whitespace-insensitive form used for substring tests."""
    return _WHITESPACE_RE.sub(" ", text).strip().lower()


def load_chunk_dataset(path: Path | str) -> list[dict[str, Any]]:
    """Read the chunk dataset written by ``main.py``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def label_chunk_with_supported_metadata(chunk: dict[str, Any], gold: dict[str, Any]) -> PaperMetadata:
    """Keep only the gold fields that this chunk's text supports.

    * title / authors: first-page text chunk containing the string;
    * problem statement: opening chunk (page 1, Abstract or Introduction)
      containing every anchor term;
    * a metric: chunk containing all of its evidence strings.
    """
    text = comparison_key(chunk["chunk_text"])
    is_prose = chunk["content_type"] == "text"
    on_first_page = is_prose and chunk["page_number"] == 1
    section = chunk["section"].lower()
    in_opening = is_prose and (on_first_page or "abstract" in section or "introduction" in section)

    def contains(*needles: str) -> bool:
        """Are all ``needles`` present in the chunk text?"""
        return all(comparison_key(needle) in text for needle in needles)

    return PaperMetadata(
        title=gold["title"] if on_first_page and contains(gold["title"]) else None,
        main_authors=[name for name in gold["authors"] if on_first_page and contains(name)],
        core_problem_statement=gold["problem"] if in_opening and contains(*gold["problem_anchors"]) else None,
        key_metrics=[metric["text"] for metric in gold["metrics"] if contains(*metric["evidence"])],
    )


def build_grounded_examples(
    chunks: list[dict[str, Any]], gold_by_document: dict[str, dict[str, Any]] | None = None
) -> list[TrainingExample]:
    """Label every non-bibliography chunk of the documents that have gold metadata."""
    gold_by_document = gold_by_document or GOLD_PAPER_METADATA
    return [
        TrainingExample(
            chunk_id=chunk["chunk_id"],
            document=chunk["source_document"],
            chunk_text=chunk["chunk_text"],
            target=label_chunk_with_supported_metadata(chunk, gold_by_document[chunk["source_document"]]),
        )
        for chunk in chunks
        if chunk["source_document"] in gold_by_document and chunk["content_type"] != "references"
    ]


def downsample_empty_examples(
    examples: list[TrainingExample], empty_per_informative: float, seed: int
) -> list[TrainingExample]:
    """Keep every informative example and a seeded sample of the empty ones.

    Empty targets dominate a paper; left as is, "always answer empty" would
    be the cheapest strategy to learn.
    """
    informative = [example for example in examples if not example.target.is_empty]
    empty = [example for example in examples if example.target.is_empty]
    rng = random.Random(seed)
    kept = rng.sample(empty, min(len(empty), max(1, round(len(informative) * empty_per_informative))))
    selected = informative + kept
    rng.shuffle(selected)
    return selected


def split_by_document(
    examples: list[TrainingExample], validation_document: str
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """Hold out one whole paper.

    Chunks of a paper share its title, authors and vocabulary: a chunk-level
    split would leak the answers into validation.
    """
    train = [example for example in examples if example.document != validation_document]
    validation = [example for example in examples if example.document == validation_document]
    if not train or not validation:
        raise ValueError(f"Cannot hold out {validation_document!r}: one side of the split is empty.")
    return train, validation


def split_train_validation_test_by_document(
    examples: list[TrainingExample], validation_document: str, test_document: str
) -> tuple[list[TrainingExample], list[TrainingExample], list[TrainingExample]]:
    """Three-way split by paper: fit / select / report.

    Hyperparameters are chosen on the validation paper; the test paper is
    read once, at the end, so its score is not inflated by model selection.
    """
    if validation_document == test_document:
        raise ValueError("validation_document and test_document must differ.")
    held_out = {validation_document, test_document}
    train = [example for example in examples if example.document not in held_out]
    validation = [example for example in examples if example.document == validation_document]
    test = [example for example in examples if example.document == test_document]
    if not (train and validation and test):
        raise ValueError(f"Empty split for validation={validation_document!r}, test={test_document!r}.")
    return train, validation, test


def summarize_label_coverage(examples: list[TrainingExample]) -> list[dict[str, Any]]:
    """Per document: how many gold fields are supported by at least one chunk."""
    rows = []
    for document, gold in GOLD_PAPER_METADATA.items():
        targets = [example.target for example in examples if example.document == document]
        rows.append(
            {
                "document": document,
                "chunks": len(targets),
                "informative": sum(not target.is_empty for target in targets),
                "authors_found": f"{len({a for t in targets for a in t.main_authors})}/{len(gold['authors'])}",
                "metrics_found": f"{len({m for t in targets for m in t.key_metrics})}/{len(gold['metrics'])}",
                "title_chunks": sum(bool(target.title) for target in targets),
                "problem_chunks": sum(bool(target.core_problem_statement) for target in targets),
            }
        )
    return rows
