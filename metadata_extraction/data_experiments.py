"""Experiments on the training data, run before any model is trained.

Each one varies a single decision about the data and measures what it
changes in the training set. None of them needs a GPU.

* chunk size: how many chunks carry a label when chunks get smaller or larger;
* labelling rule: what a looser rule would add, and how many of the added
  labels are absent from the chunk they are attached to;
* training corpus: what each batch of extra arXiv papers adds;
* class ratio: the balance of the training set for each sampling ratio.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from metadata_extraction.dataset_exploration import SERIES_BLUE, SERIES_ORANGE, style_figure
from metadata_extraction.gold_paper_metadata import GOLD_PAPER_METADATA
from metadata_extraction.grounded_training_examples import (
    TrainingExample,
    build_grounded_examples,
    comparison_key,
    downsample_empty_examples,
    label_chunk_with_supported_metadata,
)
from metadata_extraction.paper_metadata_schema import PaperMetadata
from pdf_chunking.chunking_strategies import ChunkingConfig, chunk_parsed_document
from pdf_chunking.pdf_layout_extraction import ParsedDocument, PdfExtractionConfig, extract_pdf_document

MAX_TOKENS_GRID = (150, 250, 400, 600, 800)
EMPTY_PER_INFORMATIVE_GRID = (0.5, 1.0, 2.0, 4.0)
EXTRA_PAPERS_GRID = (0, 10, 20, 40)

LabellingRule = Callable[[dict[str, Any], dict[str, Any]], PaperMetadata]


def parse_papers(data_dir: Path | str, table_detection: str = "heuristic") -> list[ParsedDocument]:
    """Parse the hand-labelled papers found in ``data_dir``.

    The heuristic table mode is used by default: it takes seconds instead of
    minutes, and tables carry few of the labels studied here.
    """
    config = PdfExtractionConfig(table_detection=table_detection)  # type: ignore[arg-type]
    paths = [Path(data_dir) / name for name in GOLD_PAPER_METADATA if (Path(data_dir) / name).is_file()]
    return [extract_pdf_document(path, config) for path in paths]


def _field_counts(examples: Sequence[TrainingExample]) -> dict[str, int]:
    """Number of examples supporting each field."""
    return {
        "title": sum(bool(example.target.title) for example in examples),
        "authors": sum(bool(example.target.main_authors) for example in examples),
        "problem": sum(bool(example.target.core_problem_statement) for example in examples),
        "metrics": sum(bool(example.target.key_metrics) for example in examples),
    }


def run_chunk_size_experiment(
    documents: Sequence[ParsedDocument], max_tokens_grid: Sequence[int] = MAX_TOKENS_GRID
) -> pd.DataFrame:
    """Re-chunk the papers at several sizes and count what the labels become.

    A gold metric is "found" when one chunk contains all of its evidence
    strings: evidence split over two chunks is lost, which is how small
    chunks hurt.
    """
    default = ChunkingConfig()
    gold_metrics = sum(len(gold["metrics"]) for gold in GOLD_PAPER_METADATA.values())
    gold_authors = sum(len(gold["authors"]) for gold in GOLD_PAPER_METADATA.values())
    rows = []
    for max_tokens in max_tokens_grid:
        config = replace(
            default,
            max_tokens=max_tokens,
            overlap_tokens=min(default.overlap_tokens, max_tokens // 8),
            min_tokens=min(default.min_tokens, max_tokens // 4),
        )
        chunks = [chunk.model_dump() for document in documents for chunk in chunk_parsed_document(document, config)[0]]
        examples = build_grounded_examples(chunks)
        informative = [example for example in examples if not example.target.is_empty]
        rows.append(
            {
                "max_tokens": max_tokens,
                "chunks": len(chunks),
                "labelled_chunks": len(examples),
                "informative": len(informative),
                "informative_share": round(len(informative) / len(examples), 3),
                **_field_counts(examples),
                "gold_authors_found": f"{len({(e.document, a) for e in examples for a in e.target.main_authors})}/{gold_authors}",
                "gold_metrics_found": f"{len({(e.document, m) for e in examples for m in e.target.key_metrics})}/{gold_metrics}",
            }
        )
    return pd.DataFrame(rows)


def _label_every_chunk_with_the_document_metadata(chunk: dict[str, Any], gold: dict[str, Any]) -> PaperMetadata:
    """The naive rule: every chunk of a paper gets the paper's metadata."""
    return PaperMetadata(
        title=gold["title"],
        main_authors=gold["authors"],
        core_problem_statement=gold["problem"],
        key_metrics=[metric["text"] for metric in gold["metrics"]],
    )


def _label_first_page_chunks_with_the_document_metadata(chunk: dict[str, Any], gold: dict[str, Any]) -> PaperMetadata:
    """A middle rule: the paper's metadata on first-page chunks, nothing elsewhere."""
    if chunk["page_number"] != 1:
        return PaperMetadata()
    return _label_every_chunk_with_the_document_metadata(chunk, gold)


LABELLING_RULES: dict[str, LabellingRule] = {
    "grounded: a field only where the chunk states it (used)": label_chunk_with_supported_metadata,
    "document metadata on first-page chunks": _label_first_page_chunks_with_the_document_metadata,
    "document metadata on every chunk": _label_every_chunk_with_the_document_metadata,
}


def run_labelling_rule_experiment(chunks: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Compare labelling rules by the labels they produce and by how many are unsupported.

    A title or author label is unsupported when the string does not occur
    in the chunk: training on it teaches the model to produce values it
    cannot read.
    """
    rows = []
    for name, rule in LABELLING_RULES.items():
        labelled = [
            (chunk, rule(chunk, GOLD_PAPER_METADATA[chunk["source_document"]]))
            for chunk in chunks
            if chunk["source_document"] in GOLD_PAPER_METADATA and chunk["content_type"] != "references"
        ]
        titles = [(chunk, target.title) for chunk, target in labelled if target.title]
        authors = [(chunk, author) for chunk, target in labelled for author in target.main_authors]

        def unsupported_share(pairs: list[tuple[dict[str, Any], str]]) -> float | None:
            """Share of labels whose string is absent from their chunk."""
            if not pairs:
                return None
            absent = sum(comparison_key(value) not in comparison_key(chunk["chunk_text"]) for chunk, value in pairs)
            return round(absent / len(pairs), 3)

        rows.append(
            {
                "labelling_rule": name,
                "informative": sum(not target.is_empty for _, target in labelled),
                "title_labels": len(titles),
                "title_labels_unsupported": unsupported_share(titles),
                "author_labels": len(authors),
                "author_labels_unsupported": unsupported_share(authors),
            }
        )
    return pd.DataFrame(rows)


def run_training_corpus_experiment(
    hand_labelled_train: Sequence[TrainingExample],
    arxiv_examples: Sequence[TrainingExample],
    extra_papers_grid: Sequence[int] = EXTRA_PAPERS_GRID,
) -> pd.DataFrame:
    """Size of the training set when the first ``n`` extra arXiv papers are added."""
    papers = sorted({example.document for example in arxiv_examples})
    rows = []
    for count in extra_papers_grid:
        if count > len(papers):
            continue
        kept = set(papers[:count])
        examples = [*hand_labelled_train, *(example for example in arxiv_examples if example.document in kept)]
        rows.append(
            {
                "extra_papers": count,
                "papers": len({example.document for example in examples}),
                "informative": sum(not example.target.is_empty for example in examples),
                **_field_counts(examples),
            }
        )
    return pd.DataFrame(rows)


def run_class_ratio_experiment(
    train_examples: Sequence[TrainingExample], seed: int, ratios: Sequence[float] = EMPTY_PER_INFORMATIVE_GRID
) -> pd.DataFrame:
    """Size and balance of the training set for each ratio of empty to informative examples."""
    rows = []
    for ratio in ratios:
        selected = downsample_empty_examples(list(train_examples), ratio, seed)
        informative = sum(not example.target.is_empty for example in selected)
        rows.append(
            {
                "empty_per_informative": ratio,
                "examples": len(selected),
                "informative": informative,
                "empty": len(selected) - informative,
                "informative_share": round(informative / len(selected), 3),
                # 8 examples per optimiser step: batch 2, gradient accumulation 4.
                "epochs_in_20_steps": round(20 * 8 / len(selected), 1),
            }
        )
    return pd.DataFrame(rows)


def plot_chunk_size_experiment(results: pd.DataFrame) -> Any:
    """Informative chunks (bars, by field) and their share of all chunks, per chunk size."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    sizes = [str(value) for value in results["max_tokens"]]
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Chunks supporting each field", "Share of chunks that carry a label"),
        horizontal_spacing=0.12,
    )
    colours = {"title": SERIES_BLUE, "authors": "#7aaee8", "problem": SERIES_ORANGE, "metrics": "#1baf7a"}
    for field, colour in colours.items():
        figure.add_trace(
            go.Bar(x=sizes, y=results[field], name=field, marker={"color": colour, "cornerradius": 3},
                   offsetgroup=field, text=results[field], textposition="outside", textfont={"size": 10},
                   cliponaxis=False),
            row=1, col=1,
        )  # fmt: skip
    figure.add_trace(
        go.Scatter(x=sizes, y=results["informative_share"], mode="lines+markers+text", showlegend=False,
                   line={"color": SERIES_BLUE, "width": 2.5},
                   marker={"size": 9, "color": SERIES_BLUE, "line": {"color": "white", "width": 1.5}},
                   text=[f"{share:.0%}" for share in results["informative_share"]], textposition="top center",
                   cliponaxis=False),
        row=1, col=2,
    )  # fmt: skip
    figure.update_xaxes(title_text="maximum chunk size (tokens)", type="category")
    figure.update_yaxes(title_text="chunks", row=1, col=1)
    figure.update_yaxes(title_text="informative / labelled chunks", tickformat=".0%", rangemode="tozero", row=1, col=2)
    figure.update_layout(legend={"orientation": "h", "x": 0.0, "y": -0.22}, bargroupgap=0.1)
    return style_figure(figure, "Chunk size and the labels it leaves", height=440, width=1150)
