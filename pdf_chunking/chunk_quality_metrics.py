"""Chunk-quality metrics, comparable across chunking strategies.

* size: token-length distribution and share within the budget;
* boundary integrity: chunks that start and end on a sentence boundary;
* table integrity: tables kept in a single chunk;
* lexical cohesion: TF-IDF similarity inside vs. between chunks.

These are proxies; ``retrieval_benchmark`` measures retrieval itself.
"""

from __future__ import annotations

import logging
import re
import statistics
from collections import Counter
from typing import Any, Sequence

from pdf_chunking.text_normalization import split_sentences
from pdf_chunking.pdf_layout_extraction import ParsedDocument
from pdf_chunking.chunk_record_schema import ChunkRecord

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")
# A clean start: uppercase letter, digit, or an opening quote/bracket/bullet.
_CLEAN_START_RE = re.compile(r"^[\"'“‘(\[•\-]?[A-Z0-9Α-Ω]")
# A clean end: terminal punctuation, optionally closed by a quote/bracket
# or followed by a citation such as "[12]".
_CLEAN_END_RE = re.compile(r"[.!?:;][\"'”’)\]]*(\s*\[[\d,\s–-]+\])?$")


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """Linear-interpolation percentile of an already sorted sequence."""
    if not sorted_values:
        return 0.0
    position = fraction * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def summarize_token_lengths(token_counts: Sequence[int]) -> dict[str, float]:
    """Summarise a token-length distribution.

    Args:
        token_counts: Token count of each chunk.

    Returns:
        Count, mean, median, standard deviation, coefficient of variation
        (std / mean; lower means more uniform chunks), min, max, 5th and
        95th percentiles.
    """
    if not token_counts:
        return {"count": 0}
    values = sorted(token_counts)
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean": round(mean, 1),
        "median": round(statistics.median(values), 1),
        "std": round(std, 1),
        "cv": round(std / mean, 3) if mean else 0.0,
        "min": values[0],
        "max": values[-1],
        "p05": round(_percentile(values, 0.05), 1),
        "p95": round(_percentile(values, 0.95), 1),
    }


def bin_token_lengths(token_counts: Sequence[int], bin_width: int = 50) -> dict[str, int]:
    """Bucket token counts into fixed-width bins, e.g. ``{"0-49": 3, ...}``."""
    bins = Counter((count // bin_width) * bin_width for count in token_counts)
    return {f"{start}-{start + bin_width - 1}": bins[start] for start in sorted(bins)}


def measure_sentence_boundary_integrity(chunks: Sequence[ChunkRecord]) -> dict[str, Any]:
    """Share of prose chunks that start and end on a sentence boundary.

    Only ``text`` chunks are scored: table rows and bibliography entries do
    not follow sentence punctuation. The check is syntactic (capital/digit
    at the start, terminal punctuation at the end), so it slightly
    under-counts breaks: a chunk cut right before a proper noun still looks
    clean. It is a lower bound on the damage, which is enough to compare
    strategies.
    """
    prose = [chunk for chunk in chunks if chunk.content_type == "text"]
    if not prose:
        return {"chunks_scored": 0}
    clean_start = sum(1 for chunk in prose if _CLEAN_START_RE.match(chunk.chunk_text))
    clean_end = sum(1 for chunk in prose if _CLEAN_END_RE.search(chunk.chunk_text))
    clean_both = sum(
        1
        for chunk in prose
        if _CLEAN_START_RE.match(chunk.chunk_text) and _CLEAN_END_RE.search(chunk.chunk_text)
    )
    return {
        "chunks_scored": len(prose),
        "clean_start_rate": round(clean_start / len(prose), 3),
        "clean_end_rate": round(clean_end / len(prose), 3),
        "clean_both_rate": round(clean_both / len(prose), 3),
    }


def measure_table_integrity(document: ParsedDocument, chunks: Sequence[ChunkRecord]) -> dict[str, Any]:
    """Share of detected tables that survive chunking in one piece.

    A table is intact when all of its rows are found in one single chunk. Whitespace is normalised first, so the test does not depend on
    how a strategy joins lines. ``1 - intact_rate`` is the *table degradation
    rate* caused by chunking (parsing errors inside cells are a separate
    issue, discussed in the README).
    """
    tables = [block for block in document.blocks if block.block_type == "table"]
    if not tables:
        return {"tables": 0}
    normalized = [_WHITESPACE_RE.sub(" ", chunk.chunk_text) for chunk in chunks]
    intact = 0
    for table in tables:
        rows = [_WHITESPACE_RE.sub(" ", row).strip() for row in table.text.split("\n") if row.strip()]
        # Every row must be in the same chunk. (Checking only the first and
        # last rows would be fooled by the header repeated in each part of
        # a table that was split.)
        if any(all(row in text for row in rows) for text in normalized):
            intact += 1
    return {"tables": len(tables), "intact": intact, "intact_rate": round(intact / len(tables), 3)}


def measure_lexical_cohesion(chunks: Sequence[ChunkRecord]) -> dict[str, Any]:
    """TF-IDF cosine inside chunks (first vs. second half) and between neighbours.

    A chunker that cuts at topic boundaries yields chunks more similar to
    themselves than to their neighbours (positive ``cohesion_gap``). Overlap
    mechanically raises the adjacent similarity. Skipped without scikit-learn.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        logger.info("scikit-learn not installed; skipping lexical cohesion metrics.")
        return {"available": False}

    prose = [chunk for chunk in chunks if chunk.content_type == "text"]
    halves: list[tuple[str, str]] = []
    for chunk in prose:
        sentences = split_sentences(chunk.chunk_text)
        if len(sentences) >= 4:
            middle = len(sentences) // 2
            halves.append((" ".join(sentences[:middle]), " ".join(sentences[middle:])))
    if len(prose) < 3 or not halves:
        return {"available": True, "chunks_scored": len(prose)}

    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2)
    try:
        vectorizer.fit([chunk.chunk_text for chunk in prose])
    except ValueError:  # vocabulary empty after pruning (tiny corpus)
        return {"available": True, "chunks_scored": len(prose)}

    def row_cosine(left: list[str], right: list[str]) -> float:
        """Mean cosine similarity between paired texts."""
        # Rows are L2-normalised by TfidfVectorizer, so the row-wise dot
        # product is the cosine similarity.
        a, b = vectorizer.transform(left), vectorizer.transform(right)
        return float(a.multiply(b).sum(axis=1).mean())

    intra = row_cosine([first for first, _ in halves], [second for _, second in halves])
    pairs = [
        (previous.chunk_text, current.chunk_text)
        for previous, current in zip(prose, prose[1:])
        if previous.source_document == current.source_document
    ]
    adjacent = row_cosine([a for a, _ in pairs], [b for _, b in pairs]) if pairs else 0.0
    return {
        "available": True,
        "chunks_scored": len(prose),
        "intra_chunk_similarity": round(intra, 3),
        "adjacent_chunk_similarity": round(adjacent, 3),
        "cohesion_gap": round(intra - adjacent, 3),
    }


def compute_chunk_quality_report(
    documents: Sequence[ParsedDocument],
    chunks: Sequence[ChunkRecord],
    min_tokens: int,
    max_tokens: int,
) -> dict[str, Any]:
    """Compute every quality metric for a set of documents and their chunks.

    Args:
        documents: Parsed documents the chunks were built from.
        chunks: All chunk records.
        min_tokens: Lower bound of the target size range.
        max_tokens: Upper bound of the target size range.

    Returns:
        A JSON-serialisable dictionary of metrics.
    """
    token_counts = [chunk.token_count for chunk in chunks]
    in_range = sum(1 for count in token_counts if min_tokens <= count <= max_tokens)

    tables = intact = 0
    for document in documents:
        own = [chunk for chunk in chunks if chunk.source_document == document.source_document]
        result = measure_table_integrity(document, own)
        tables += result.get("tables", 0)
        intact += result.get("intact", 0)

    texts = Counter(chunk.chunk_text for chunk in chunks)
    duplicates = sum(count - 1 for count in texts.values() if count > 1)

    return {
        "token_length": summarize_token_lengths(token_counts),
        "token_histogram": bin_token_lengths(token_counts),
        "within_size_range_rate": round(in_range / len(token_counts), 3) if token_counts else 0.0,
        "content_types": dict(Counter(chunk.content_type for chunk in chunks)),
        "boundary_integrity": measure_sentence_boundary_integrity(chunks),
        "table_integrity": {
            "tables": tables,
            "intact": intact,
            "intact_rate": round(intact / tables, 3) if tables else None,
        },
        "lexical_cohesion": measure_lexical_cohesion(chunks),
        "exact_duplicate_chunks": duplicates,
    }
