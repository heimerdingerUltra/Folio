"""Document-level metadata extraction, driven by the document structure.

The chunk-level extractor reads every chunk and finds nothing in most of
them, because a paper states its title, authors and problem once. This
module starts from the parsed document instead:

* the title is the block the layout analysis typed as ``title``;
* the authors are parsed from the first-page blocks between the title and
  the abstract;
* the problem statement is the opening of the abstract;
* key metrics are the abstract sentences that quote a result figure.

Every value comes with its evidence (page and block), so it can be checked
against the PDF. For a model that reads chunks, :func:`select_chunks_worth_reading`
keeps the chunks where a field can appear, and
:func:`merge_chunk_metadata` folds the per-chunk answers into one record.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from metadata_extraction.arxiv_training_examples import quotes_a_result_figure
from metadata_extraction.paper_metadata_schema import PaperMetadata
from pdf_chunking.pdf_layout_extraction import ParsedDocument, TextBlock

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_NAME_SEPARATOR_RE = re.compile(r",|;|&|\band\b|[∗*†‡§¶⋆★♠♣♦♥^]|\d+")
# E-mail addresses, "{a, b}@host" groups and "<address>": removed before a line is judged.
_EMAIL_RE = re.compile(r"\{[^}]*\}?\S*|<[^>]*>|\S*@\S*")
_NAME_TOKEN_RE = re.compile(r"^(?:[A-ZÀ-Þ][\w'’.-]*|de|del|van|von|der|den|la|le|di|da|dos|bin|al)$")
_INITIAL_RE = re.compile(r"^[A-Z]\.$")
_NOT_AN_AUTHOR_LINE_RE = re.compile(
    r"http|\b(Universit|Institut|Laborato|Research|College|School|Department|Corporation|Google|Microsoft"
    r"|Facebook|Meta AI|OpenAI|DeepMind|Brain|Inc\b|Lab\b|Labs\b|France|USA|Published|Equal|arXiv|Abstract)",
    flags=re.IGNORECASE,
)
_PARTICLES = frozenset({"de", "del", "van", "von", "der", "den", "la", "le", "di", "da", "dos", "bin", "al"})
_MIN_ABSTRACT_CHARS = 300
_MAX_KEY_METRICS = 3
_SECTIONS_WORTH_READING = ("abstract", "introduction", "conclusion", "result", "experiment", "evaluation")


@dataclass(frozen=True)
class FieldEvidence:
    """Where a value was read.

    Attributes:
        field: Name of the metadata field.
        value: The extracted value.
        page_number: 1-based page of the supporting block.
        reading_order: Index of that block in ``ParsedDocument.blocks``.
    """

    field: str
    value: str
    page_number: int
    reading_order: int


@dataclass
class DocumentMetadata:
    """Metadata of one document and the evidence for each value."""

    source_document: str
    metadata: PaperMetadata
    evidence: list[FieldEvidence] = field(default_factory=list)


def _split_unseparated_names(tokens: list[str]) -> list[str]:
    """Split "Jacob Devlin Ming-Wei Chang" into names of two words.

    Author lines often separate names by spacing only. An initial ("P.")
    or a particle ("van") attaches to the name it belongs to; a line that
    does not then divide into pairs is kept whole.
    """
    units: list[str] = []
    for token in tokens:
        attaches = units and (_INITIAL_RE.match(token) or units[-1].split()[-1].lower() in _PARTICLES)
        if attaches or (token.lower() in _PARTICLES and units):
            units[-1] += f" {token}"
        else:
            units.append(token)
    if len(units) <= 3 or len(units) % 2:
        return [" ".join(tokens)]
    return [f"{units[index]} {units[index + 1]}" for index in range(0, len(units), 2)]


def parse_author_names(lines: list[str]) -> list[str]:
    """Person names found in the lines of an author block.

    E-mail addresses are removed and affiliation lines skipped; the rest is
    split on commas, "and" and footnote markers, and a segment is kept when
    it reads as two to four capitalised words.

    Args:
        lines: Text lines of one author block, in reading order.

    Returns:
        The distinct names, in order of first appearance.
    """
    names: list[str] = []
    for line in lines:
        line = _EMAIL_RE.sub(" ", line)
        if _NOT_AN_AUTHOR_LINE_RE.search(line):
            continue
        for segment in _NAME_SEPARATOR_RE.split(line):
            tokens = segment.split()
            if len(tokens) < 2 or not all(_NAME_TOKEN_RE.match(token) for token in tokens):
                continue
            names.extend(name for name in _split_unseparated_names(tokens) if 2 <= len(name.split()) <= 4)
    return list(dict.fromkeys(names))


def _is_abstract_heading(block: TextBlock) -> bool:
    """Is the block the "Abstract" heading?"""
    return block.block_type == "heading" and block.text.lower().strip(" .:") == "abstract"


def locate_abstract(document: ParsedDocument) -> list[tuple[int, TextBlock]]:
    """Paragraphs of the abstract, with their reading order.

    They follow the "Abstract" heading up to the next heading. Without such
    a heading, the abstract is the first long paragraph of the first page.

    Args:
        document: Parsed document; its blocks carry ``block_type``, ``page_number`` and ``text``.

    Returns:
        ``(reading_order, block)`` pairs, ``reading_order`` being the index of the block in
        ``document.blocks``. Empty when no abstract is found.
    """
    blocks = list(enumerate(document.blocks))
    for position, (_, block) in enumerate(blocks):
        if _is_abstract_heading(block):
            abstract = []
            for index, following in blocks[position + 1 :]:
                if following.block_type == "heading":
                    break
                if following.block_type == "paragraph":
                    abstract.append((index, following))
            return abstract
    for index, block in blocks:
        if block.page_number == 1 and block.block_type == "paragraph" and len(block.text) >= _MIN_ABSTRACT_CHARS:
            return [(index, block)]
    return []


def locate_author_blocks(document: ParsedDocument) -> list[tuple[int, TextBlock]]:
    """First-page paragraphs between the title and the abstract.

    Args:
        document: Parsed document with typed blocks.

    Returns:
        ``(reading_order, block)`` pairs for the paragraphs after the last ``title`` block (from
        the first block when there is none) and before the first abstract paragraph.
    """
    abstract = locate_abstract(document)
    end = abstract[0][0] if abstract else len(document.blocks)
    title_positions = [index for index, block in enumerate(document.blocks) if block.block_type == "title"]
    start = title_positions[-1] + 1 if title_positions else 0
    return [
        (index, block)
        for index, block in enumerate(document.blocks[:end])
        if index >= start and block.page_number == 1 and block.block_type == "paragraph"
    ]


def extract_document_metadata(document: ParsedDocument) -> DocumentMetadata:
    """Read title, authors, problem statement and key metrics from the structure.

    Args:
        document: Parsed document, with typed blocks and the ``title`` set by the layout analysis.

    Returns:
        The metadata and one evidence entry per extracted value. The problem statement is the
        first two sentences of the abstract; at most three key metrics are kept.
    """
    evidence: list[FieldEvidence] = []

    title = None
    for index, block in enumerate(document.blocks):
        if block.block_type == "title":
            title = document.title
            evidence.append(FieldEvidence("title", title, block.page_number, index))
            break

    authors: list[str] = []
    for index, block in locate_author_blocks(document):
        for name in parse_author_names(block.lines):
            if name not in authors:
                authors.append(name)
                evidence.append(FieldEvidence("main_authors", name, block.page_number, index))

    problem = None
    metrics: list[str] = []
    abstract = locate_abstract(document)
    if abstract:
        index, block = abstract[0]
        problem = " ".join(_SENTENCE_SPLIT_RE.split(block.text)[:2])
        evidence.append(FieldEvidence("core_problem_statement", problem, block.page_number, index))
    for index, block in abstract:
        for sentence in _SENTENCE_SPLIT_RE.split(block.text):
            if quotes_a_result_figure(sentence) and len(metrics) < _MAX_KEY_METRICS:
                metrics.append(sentence)
                evidence.append(FieldEvidence("key_metrics", sentence, block.page_number, index))

    return DocumentMetadata(
        source_document=document.source_document,
        metadata=PaperMetadata(
            title=title, main_authors=authors, core_problem_statement=problem, key_metrics=metrics
        ),
        evidence=evidence,
    )


def select_chunks_worth_reading(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the chunks in which a metadata field can appear.

    A model that reads chunks does not need the whole paper: first page,
    abstract, introduction, results and conclusion, plus any table or
    prose that quotes a result figure. The bibliography is never read.

    Args:
        chunks: Chunk records with ``content_type``, ``page_number`` (1-based), ``section`` and
            ``chunk_text``.

    Returns:
        The selected chunks, in input order.
    """
    return [
        chunk
        for chunk in chunks
        if chunk["content_type"] != "references"
        and (
            chunk["page_number"] == 1
            or any(name in chunk["section"].lower() for name in _SECTIONS_WORTH_READING)
            or quotes_a_result_figure(chunk["chunk_text"])
        )
    ]


def merge_chunk_metadata(answers: list[PaperMetadata]) -> PaperMetadata:
    """Fold per-chunk answers, given in reading order, into one record.

    The title is the value most chunks agree on (the earliest wins a tie),
    the problem statement is the first one stated, authors and metrics are
    united in order of appearance.

    Args:
        answers: Metadata extracted from each chunk of one document.

    Returns:
        The merged record; duplicate authors and metrics are removed by the schema.
    """
    titles = [answer.title for answer in answers if answer.title]
    return PaperMetadata(
        title=max(titles, key=titles.count) if titles else None,
        main_authors=[name for answer in answers for name in answer.main_authors],
        core_problem_statement=next((a.core_problem_statement for a in answers if a.core_problem_statement), None),
        key_metrics=[metric for answer in answers for metric in answer.key_metrics],
    )
