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


def style_figure(figure: Any, height: int, width: int | None = None) -> Any:
    """Shared look of every chart: white background, light grid, legend without frame."""
    figure.update_layout(
        template="plotly_white",
        height=height,
        width=width,
        margin={"l": 60, "r": 30, "t": 70, "b": 50},
        font={"size": 12},
        legend={"bgcolor": "rgba(0,0,0,0)"},
    )
    return figure


def plot_dataset_overview(chunks: Sequence[dict[str, Any]], splits: dict[str, Sequence[TrainingExample]]) -> Any:
    """Three panels: chunk length by content type, chunks per document, labels per split."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    frame = pd.DataFrame(chunks)
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Chunk length by content type", "Chunks per document", "Examples supporting each field"),
        horizontal_spacing=0.09,
    )

    # One layout-wide bar mode: "stack" stacks the bars of a same offset group
    # (the histogram) and puts different groups side by side (the splits).
    colours = {"text": SERIES_BLUE, "table": SERIES_ORANGE, "references": SERIES_AQUA}
    for name, colour in colours.items():
        lengths = frame.loc[frame["content_type"] == name, "token_count"]
        if len(lengths):
            figure.add_trace(
                go.Histogram(x=lengths, name=name, nbinsx=16, marker_color=colour, marker_line={"color": "white", "width": 1},
                             offsetgroup="length", legendgroup="content"),
                row=1, col=1,
            )  # fmt: skip

    per_document = frame.groupby("source_document")["chunk_id"].count().sort_values()
    figure.add_trace(
        go.Bar(x=per_document.values, y=[name.removesuffix(".pdf") for name in per_document.index], orientation="h",
               marker_color=SERIES_BLUE, text=per_document.values, textposition="outside", showlegend=False,
               offsetgroup="documents", cliponaxis=False),
        row=1, col=2,
    )  # fmt: skip

    distribution = summarize_label_distribution(splits).set_index("split")
    columns = ["empty", *METADATA_FIELDS]
    for offset, (split, row) in enumerate(distribution.iterrows()):
        figure.add_trace(
            go.Bar(x=["empty", "title", "authors", "problem", "metrics"], y=[row[column] for column in columns],
                   name=str(split), marker_color=[SERIES_BLUE, SERIES_ORANGE, SERIES_AQUA][offset % 3],
                   offsetgroup=str(split), legendgroup="split"),
            row=1, col=3,
        )  # fmt: skip

    figure.update_xaxes(title_text="tokens", row=1, col=1)
    figure.update_yaxes(title_text="chunks", row=1, col=1)
    figure.update_xaxes(title_text="chunks", row=1, col=2)
    figure.update_yaxes(title_text="examples", row=1, col=3)
    figure.update_layout(barmode="stack", legend={"orientation": "h", "y": -0.2})
    return style_figure(figure, height=400)
