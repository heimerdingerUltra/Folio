"""Compare PDF parsers end to end: same chunker, same questions, same metrics.

Every parser is reduced to one function ``paths -> list[ParsedDocument]``.
The documents then go through the *same* chunker with the same settings, so
every difference in the result comes from extraction:

* extraction    wall time, characters, block types, tables returned with text;
* structure     document title against the hand-written gold title, headings
                against the PDF outline, column-flow probes on two-column pages;
* chunks        count, boundary integrity, tables kept in one chunk;
* retrieval     the benchmark questions, one row per retriever.

Two measures favour the PyMuPDF pipeline by construction and are marked with
``*``: it reads the PDF outline to classify headings, and the benchmark
evidence strings were written against its text.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import pymupdf

from pdf_chunking.chunk_quality_metrics import compute_chunk_quality_report
from pdf_chunking.chunking_strategies import ChunkingConfig, chunk_parsed_document
from pdf_chunking.opendataloader_extraction import extract_pdf_documents_with_opendataloader
from pdf_chunking.pdf_layout_extraction import ParsedDocument, extract_pdf_document
from pdf_chunking.retrieval_benchmark import (
    Question,
    SimilarityScorer,
    find_relevant_passages,
    paired_bootstrap_interval,
    reciprocal_ranks,
    score_retrieval,
)

PdfParser = Callable[[Sequence[Path]], list[ParsedDocument]]

PYMUPDF_PIPELINE = "this pipeline (PyMuPDF)"
OPENDATALOADER = "OpenDataLoader PDF"
OPENDATALOADER_CLUSTER_TABLES = "OpenDataLoader PDF (cluster tables)"

# (file, 0-based page, sentence fragment). Each fragment runs over a line
# break inside ONE column of a two-column page. It is found verbatim only if
# the parser reads a column top to bottom; a parser that reads straight
# across the page inserts the other column's line in the middle of it.
COLUMN_PROBES: list[tuple[str, int, str]] = [
    ("ChunkNorris.pdf", 0, "that combines the strengths of Information Retrieval (IR) and generative"),
    ("ChunkNorris.pdf", 0, "This retrieval step ensures the generated responses are coherent as well"),
    ("LLaMA.pdf", 1, "Our training dataset is a mixture of several sources, reported in Table 1"),
    ("LLaMA.pdf", 1, "that have been leveraged to train other LLMs, with the restriction"),
]

_NON_ALPHANUMERIC_RE = re.compile(r"[^a-z0-9]+")
_SECTION_NUMBER_RE = re.compile(r"^(?:\d{1,2}|[A-Z])(?:\.\d{1,2}){0,3}\.?\s+")
_SPACES_RE = re.compile(r"\s+")


def comparison_key(text: str) -> str:
    """Lower-case alphanumerics only, section number removed.

    Args:
        text: A title or heading, possibly starting with a section number such as ``3.2`` or
            ``A.1``.

    Returns:
        The text without its leading section number, lower-cased, keeping only ``a-z`` and
        ``0-9``. Two headings that differ only in numbering, case, spacing or punctuation get
        the same key.
    """
    return _NON_ALPHANUMERIC_RE.sub("", _SECTION_NUMBER_RE.sub("", text).lower())


# ------------------------------------------------------------------ parsers


def extract_with_pymupdf_pipeline(paths: Sequence[Path]) -> list[ParsedDocument]:
    """This repository's extractor, default configuration.

    Args:
        paths: PDFs to parse, one after the other.

    Returns:
        One parsed document per path, in the same order.
    """
    return [extract_pdf_document(path) for path in paths]


def extract_with_opendataloader(paths: Sequence[Path], table_method: str = "default") -> list[ParsedDocument]:
    """OpenDataLoader PDF; a file it cannot parse raises.

    Args:
        paths: PDFs to parse in one batch.
        table_method: ``default`` finds tables from ruling lines; ``cluster`` also groups
            aligned text.

    Returns:
        One parsed document per path, in the same order.

    Raises:
        PdfExtractionError: If the backend is unavailable or any of the files cannot be parsed.
    """
    results = extract_pdf_documents_with_opendataloader(paths, table_method)
    for result in results.values():
        if isinstance(result, Exception):
            raise result
    return [results[path.name] for path in paths]


def extract_with_opendataloader_cluster_tables(paths: Sequence[Path]) -> list[ParsedDocument]:
    """OpenDataLoader PDF with ``table_method="cluster"``.

    Args:
        paths: PDFs to parse in one batch.

    Returns:
        One parsed document per path, in the same order.
    """
    return extract_with_opendataloader(paths, "cluster")


DEFAULT_PARSERS: dict[str, PdfParser] = {
    PYMUPDF_PIPELINE: extract_with_pymupdf_pipeline,
    OPENDATALOADER: extract_with_opendataloader,
    OPENDATALOADER_CLUSTER_TABLES: extract_with_opendataloader_cluster_tables,
}


# ---------------------------------------------------------------- structure


def count_correct_titles(documents: Sequence[ParsedDocument], gold_titles: dict[str, str]) -> str:
    """``"k/n"`` documents whose detected title equals the gold title.

    Args:
        documents: Parsed documents to score.
        gold_titles: Expected title per source file name. Documents without an entry are not
            scored.

    Returns:
        ``"k/n"``, with ``n`` the documents that have a gold title and ``k`` those whose title
        has the same :func:`comparison_key` as the gold one.
    """
    scored = [document for document in documents if document.source_document in gold_titles]
    correct = sum(
        comparison_key(document.title) == comparison_key(gold_titles[document.source_document]) for document in scored
    )
    return f"{correct}/{len(scored)}"


def score_headings_against_outline(documents: Sequence[ParsedDocument], paths: Sequence[Path]) -> dict[str, Any]:
    """Recall and precision of detected headings against the PDF outline.

    Only PDFs that carry an outline are scored.

    Args:
        documents: Parsed documents to score.
        paths: The source PDFs, in the same order as ``documents``; their outline is the
            reference.

    Returns:
        ``outline_entries`` and ``detected_headings`` (totals over the scored PDFs), ``recall``
        (share of outline entries found among the headings) and ``precision`` (share of
        headings present in the outline), rounded to 3 decimals. A rate is None when its
        denominator is zero.
    """
    expected_total = found_total = detected_total = matching_total = 0
    for document, path in zip(documents, paths):
        with pymupdf.open(path) as pdf:
            outline = {comparison_key(title) for _, title, _ in pdf.get_toc()} - {""}
        if not outline:
            continue
        detected = [comparison_key(block.text) for block in document.blocks if block.block_type == "heading"]
        expected_total += len(outline)
        found_total += len(outline & set(detected))
        detected_total += len(detected)
        matching_total += sum(key in outline for key in detected)
    return {
        "outline_entries": expected_total,
        "recall": round(found_total / expected_total, 3) if expected_total else None,
        "detected_headings": detected_total,
        "precision": round(matching_total / detected_total, 3) if detected_total else None,
    }


def score_column_flow(documents: Sequence[ParsedDocument]) -> str:
    """``"k/n"`` probes found verbatim: a column was read without mixing in the other.

    Args:
        documents: Parsed documents. Only those named in ``COLUMN_PROBES`` are probed.

    Returns:
        ``"k/n"``, with ``n`` the probes whose document is present and ``k`` those whose
        fragment occurs in the text of its page, whitespace ignored.
    """
    found = total = 0
    by_name = {document.source_document: document for document in documents}
    for filename, page_index, fragment in COLUMN_PROBES:
        if filename not in by_name:
            continue
        page_text = "".join(b.text for b in by_name[filename].blocks if b.page_number == page_index + 1)
        total += 1
        found += _SPACES_RE.sub("", fragment) in _SPACES_RE.sub("", page_text)
    return f"{found}/{total}"


# --------------------------------------------------------------- comparison


def measure_parser(
    name: str,
    parser: PdfParser,
    paths: Sequence[Path],
    chunking: ChunkingConfig,
    gold_titles: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Extract, chunk and measure; returns ``(metrics, exported chunks)``.

    Args:
        name: Label of the parser, stored under ``parser`` in the metrics.
        parser: Function mapping the PDF paths to parsed documents, in the same order.
        paths: PDFs to parse.
        chunking: Chunking settings applied to the documents of this parser.
        gold_titles: Expected title per source file name, for the title score.

    Returns:
        The extraction, structure and chunk metrics of the parser, and its chunks as
        dictionaries, usable as passages by the retrieval benchmark.
    """
    started = time.perf_counter()
    documents = parser(paths)
    seconds = time.perf_counter() - started

    chunks = [chunk for document in documents for chunk in chunk_parsed_document(document, chunking)[0]]
    quality = compute_chunk_quality_report(documents, chunks, chunking.min_tokens, chunking.max_tokens)
    block_types: dict[str, int] = {}
    for document in documents:
        for block in document.blocks:
            block_types[block.block_type] = block_types.get(block.block_type, 0) + 1
    metrics = {
        "parser": name,
        "extraction_seconds": round(seconds, 1),
        "pages": sum(document.page_count for document in documents),
        "characters": sum(len(block.text) for document in documents for block in document.blocks),
        "blocks_by_type": dict(sorted(block_types.items())),
        "titles_correct": count_correct_titles(documents, gold_titles),
        "headings_vs_outline": score_headings_against_outline(documents, paths),
        "column_flow": score_column_flow(documents),
        "chunks": len(chunks),
        "tokens": sum(chunk.token_count for chunk in chunks),
        "boundary_integrity": quality["boundary_integrity"]["clean_both_rate"],
        "table_integrity": quality["table_integrity"],
    }
    return metrics, [chunk.model_dump() for chunk in chunks]


def score_retrieval_per_parser(
    passages_by_parser: dict[str, list[dict[str, Any]]],
    questions: Sequence[Question],
    scorers: dict[str, SimilarityScorer],
) -> dict[str, Any]:
    """Retrieval metrics per retriever and parser; gains are paired against the first parser.

    Args:
        passages_by_parser: Chunk dictionaries per parser label. The first entry is the
            reference parser.
        questions: Benchmark questions, each with ``question``, ``document`` and ``evidence``.
        scorers: Similarity scorer per retriever label.

    Returns:
        ``{retriever: {parser: metrics}}``. The metrics are those of :func:`score_retrieval`
        plus ``answerable_questions`` (``"k/n"`` questions with at least one relevant passage)
        and, except for the reference parser, ``mrr_gain_vs_reference`` and ``mrr_gain_ci95``.
    """
    reference_parser = next(iter(passages_by_parser))
    results: dict[str, Any] = {}
    for retriever, scorer in scorers.items():
        results[retriever] = {}
        for parser, passages in passages_by_parser.items():
            answerable = sum(bool(find_relevant_passages(passages, question).any()) for question in questions)
            metrics = score_retrieval(passages, [p["embedding_text"] for p in passages], questions, scorer)
            results[retriever][parser] = {**metrics, "answerable_questions": f"{answerable}/{len(questions)}"}
        reference = reciprocal_ranks(results[retriever][reference_parser]["ranks"])
        for parser, metrics in results[retriever].items():
            if parser != reference_parser:
                gain = reciprocal_ranks(metrics["ranks"]) - reference
                metrics["mrr_gain_vs_reference"] = round(float(gain.mean()), 3)
                metrics["mrr_gain_ci95"] = list(paired_bootstrap_interval(gain))
    return results


def compare_pdf_parsers(
    paths: Sequence[Path],
    questions: Sequence[Question],
    scorers: dict[str, SimilarityScorer],
    gold_titles: dict[str, str],
    parsers: dict[str, PdfParser] | None = None,
    chunking: ChunkingConfig | None = None,
) -> dict[str, Any]:
    """Run every parser through the same chunker and the same benchmark.

    The first parser is the reference for the paired retrieval gains.

    Args:
        paths: PDFs given to every parser.
        questions: Benchmark questions, each with ``question``, ``document`` and ``evidence``.
        scorers: Similarity scorer per retriever label.
        gold_titles: Expected title per source file name.
        parsers: Parser function per label. None or empty uses ``DEFAULT_PARSERS``.
        chunking: Chunking settings shared by all parsers. None uses the defaults of
            ``ChunkingConfig``.

    Returns:
        ``n_pdfs``, ``parsers`` (the metrics of :func:`measure_parser`, one entry per parser)
        and ``retrieval`` (the output of :func:`score_retrieval_per_parser`).
    """
    chunking = chunking or ChunkingConfig()
    measured, passages_by_parser = [], {}
    for name, parser in (parsers or DEFAULT_PARSERS).items():
        metrics, passages_by_parser[name] = measure_parser(name, parser, paths, chunking, gold_titles)
        measured.append(metrics)
    return {
        "n_pdfs": len(paths),
        "parsers": measured,
        "retrieval": score_retrieval_per_parser(passages_by_parser, questions, scorers),
    }


def tabulate_parser_comparison(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per measure, one column per parser (for a DataFrame or Markdown).

    Args:
        comparison: Output of :func:`compare_pdf_parsers`.

    Returns:
        One dictionary per measure: ``measure`` holds its label and each parser label maps to
        the value for that parser. Extraction, structure and chunk measures come first, then
        four rows per retriever.
    """
    parsers = comparison["parsers"]
    measures: list[tuple[str, Callable[[dict[str, Any]], Any]]] = [
        ("extraction seconds", lambda m: m["extraction_seconds"]),
        ("characters", lambda m: f"{m['characters']:,}"),
        ("titles correct", lambda m: m["titles_correct"]),
        ("headings detected", lambda m: m["blocks_by_type"].get("heading", 0)),
        ("heading recall vs outline *", lambda m: m["headings_vs_outline"]["recall"]),
        ("heading precision vs outline *", lambda m: m["headings_vs_outline"]["precision"]),
        ("column flow probes", lambda m: m["column_flow"]),
        ("tables with text", lambda m: m["blocks_by_type"].get("table", 0)),
        ("captions / equations", lambda m: f"{m['blocks_by_type'].get('caption', 0)} / {m['blocks_by_type'].get('equation', 0)}"),
        ("chunks", lambda m: m["chunks"]),
        ("boundary integrity", lambda m: m["boundary_integrity"]),
        ("tables intact after chunking", lambda m: f"{m['table_integrity']['intact']}/{m['table_integrity']['tables']}"),
    ]
    rows = [{"measure": label, **{m["parser"]: value(m) for m in parsers}} for label, value in measures]
    for retriever, systems in comparison["retrieval"].items():
        for label, key in (("answerable questions *", "answerable_questions"), ("hit@5", "hit@5"), ("MRR@10", "mrr@10")):
            rows.append({"measure": f"{retriever}: {label}", **{m["parser"]: systems[m["parser"]][key] for m in parsers}})
        rows.append(
            {
                "measure": f"{retriever}: MRR gain vs reference (95 % CI)",
                **{
                    m["parser"]: (
                        f"{s['mrr_gain_vs_reference']:+.3f} [{s['mrr_gain_ci95'][0]:+.3f}, {s['mrr_gain_ci95'][1]:+.3f}]"
                        if "mrr_gain_vs_reference" in (s := systems[m["parser"]])
                        else "reference"
                    )
                    for m in parsers
                },
            }
        )
    return rows
