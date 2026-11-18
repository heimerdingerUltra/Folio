"""Exploratory analysis of the chunk dataset and of the training labels."""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from metadata_extraction.grounded_training_examples import TrainingExample

SERIES_BLUE, SERIES_ORANGE, SERIES_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
METADATA_FIELDS = ("title", "main_authors", "core_problem_statement", "key_metrics")


def summarize_chunks_per_document(chunks: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Per document: pages, chunks by content type, token statistics."""
    frame = pd.DataFrame(chunks)
    summary = frame.groupby("source_document").agg(
        pages=("page_end", "max"),
        chunks=("chunk_id", "count"),
        tokens=("token_count", "sum"),
        median_tokens=("token_count", "median"),
        max_tokens=("token_count", "max"),
    )
    by_type = frame.pivot_table(index="source_document", columns="content_type", values="chunk_id", aggfunc="count")
    return summary.join(by_type.fillna(0).astype(int)).reset_index()


def summarize_label_distribution(splits: dict[str, Sequence[TrainingExample]]) -> pd.DataFrame:
    """Per split: number of examples supporting each metadata field."""
    rows = []
    for name, examples in splits.items():
        targets = [example.target for example in examples]
        rows.append(
            {
                "split": name,
                "examples": len(targets),
                "documents": len({example.document for example in examples}),
                "empty": sum(target.is_empty for target in targets),
                "title": sum(bool(target.title) for target in targets),
                "main_authors": sum(bool(target.main_authors) for target in targets),
                "core_problem_statement": sum(bool(target.core_problem_statement) for target in targets),
                "key_metrics": sum(bool(target.key_metrics) for target in targets),
            }
        )
    return pd.DataFrame(rows)


def plot_dataset_overview(chunks: Sequence[dict[str, Any]], splits: dict[str, Sequence[TrainingExample]]) -> Any:
    """Three panels: chunk length by content type, chunks per document, labels per split."""
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(chunks)
    figure, (lengths, documents, labels) = plt.subplots(1, 3, figsize=(15, 3.8))

    content_types = [name for name in ("text", "table", "references") if name in set(frame["content_type"])]
    lengths.hist(
        [frame.loc[frame["content_type"] == name, "token_count"] for name in content_types],
        bins=16,
        stacked=True,
        label=content_types,
        color=[SERIES_BLUE, SERIES_ORANGE, SERIES_AQUA][: len(content_types)],
        edgecolor="white",
    )
    lengths.set(title="Chunk length by content type", xlabel="tokens", ylabel="chunks")
    lengths.legend(frameon=False)

    per_document = frame.groupby("source_document")["chunk_id"].count().sort_values()
    documents.barh([name.removesuffix(".pdf") for name in per_document.index], per_document.values, color=SERIES_BLUE)
    for position, value in enumerate(per_document.values):
        documents.text(value + 1, position, str(value), va="center", fontsize=9)
    documents.set(title="Chunks per document", xlabel="chunks")

    distribution = summarize_label_distribution(splits).set_index("split")
    columns = ["empty", *METADATA_FIELDS]
    width = 0.8 / len(distribution)
    for offset, (split, row) in enumerate(distribution.iterrows()):
        positions = [index + offset * width for index in range(len(columns))]
        labels.bar(positions, row[columns], width=width, label=split, color=[SERIES_BLUE, SERIES_ORANGE][offset % 2])
    labels.set_xticks([index + width * (len(distribution) - 1) / 2 for index in range(len(columns))])
    labels.set_xticklabels(["empty", "title", "authors", "problem", "metrics"])
    labels.set(title="Examples supporting each field", ylabel="examples")
    labels.legend(frameon=False)

    for axis in (lengths, documents, labels):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y" if axis is not documents else "x", alpha=0.25)
    figure.tight_layout()
    return figure
