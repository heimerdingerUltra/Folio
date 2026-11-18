"""Training examples labelled from arXiv metadata instead of by hand.

The arXiv API gives the title, the authors and the abstract of a paper. That
is enough to label three of the four fields without annotation:

* title and authors, on the first-page chunks that contain them;
* the problem statement, taken as the opening sentences of the abstract, on
  the chunk that contains them. This is a proxy: an abstract usually opens
  with the problem, not always.

Key metrics are not in the API. A chunk that quotes a result-like figure is
therefore left out: labelling it "no metric" would be a wrong label.

These examples are used for training only. Evaluation stays on the
hand-labelled papers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from metadata_extraction.grounded_training_examples import TrainingExample, comparison_key
from metadata_extraction.paper_metadata_schema import PaperMetadata

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
# A percentage or a decimal number: what a reported result looks like.
_RESULT_FIGURE_RE = re.compile(r"\d\s?%|\d\.\d")
_MAX_PROBLEM_SENTENCES = 2
_MIN_PROBLEM_CHARS = 40


def load_arxiv_metadata(path: Path | str) -> dict[str, dict[str, Any]]:
    """Read ``arxiv_metadata.json``: PDF file name -> title, authors, abstract."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def quotes_a_result_figure(text: str) -> bool:
    """Does ``text`` contain a percentage or a decimal number?"""
    return bool(_RESULT_FIGURE_RE.search(text))


def opening_sentences_found_in(abstract: str, chunk_key: str) -> str | None:
    """Leading sentences of ``abstract`` (at most two) that the chunk contains verbatim."""
    found: list[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(abstract)[:_MAX_PROBLEM_SENTENCES]:
        if comparison_key(sentence) not in chunk_key:
            break
        found.append(sentence)
    statement = " ".join(found)
    return statement if len(statement) >= _MIN_PROBLEM_CHARS else None


def label_chunk_with_arxiv_metadata(chunk: dict[str, Any], paper: dict[str, Any]) -> PaperMetadata | None:
    """Label one chunk from the arXiv record of its paper.

    Returns:
        The supported fields, or ``None`` when the chunk must be left out
        (bibliography, table, or prose quoting a figure whose metric label is
        unknown).
    """
    if chunk["content_type"] != "text" or quotes_a_result_figure(chunk["chunk_text"]):
        return None
    text = comparison_key(chunk["chunk_text"])
    on_first_page = chunk["page_number"] == 1
    return PaperMetadata(
        title=paper["title"] if on_first_page and comparison_key(paper["title"]) in text else None,
        main_authors=[name for name in paper["authors"] if on_first_page and comparison_key(name) in text],
        core_problem_statement=opening_sentences_found_in(paper["abstract"], text),
    )


def build_arxiv_training_examples(
    chunks: list[dict[str, Any]], metadata_by_document: dict[str, dict[str, Any]]
) -> list[TrainingExample]:
    """Label the chunks of every paper that has an arXiv record."""
    examples = []
    for chunk in chunks:
        paper = metadata_by_document.get(chunk["source_document"])
        if paper is None or (target := label_chunk_with_arxiv_metadata(chunk, paper)) is None:
            continue
        examples.append(
            TrainingExample(
                chunk_id=chunk["chunk_id"],
                document=chunk["source_document"],
                chunk_text=chunk["chunk_text"],
                target=target,
            )
        )
    return examples


def summarize_arxiv_label_coverage(
    examples: list[TrainingExample], metadata_by_document: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """How many papers got each field, and how many examples carry a label."""
    by_document: dict[str, list[PaperMetadata]] = {}
    for example in examples:
        by_document.setdefault(example.document, []).append(example.target)
    authors_expected = sum(len(metadata_by_document[document]["authors"]) for document in by_document)
    authors_found = sum(len({name for target in targets for name in target.main_authors}) for targets in by_document.values())
    return {
        "papers": len(by_document),
        "examples": len(examples),
        "informative_examples": sum(not example.target.is_empty for example in examples),
        "papers_with_title": sum(any(t.title for t in targets) for targets in by_document.values()),
        "papers_with_problem_statement": sum(
            any(t.core_problem_statement for t in targets) for targets in by_document.values()
        ),
        "authors_found": authors_found,
        "authors_expected": authors_expected,
    }
