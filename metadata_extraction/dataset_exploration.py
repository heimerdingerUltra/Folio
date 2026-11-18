"""Exploratory analysis of the chunk dataset and of the training labels."""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from metadata_extraction.grounded_training_examples import TrainingExample

SERIES_BLUE, SERIES_ORANGE, SERIES_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_MUTED, GRID = "#1f1e1c", "#6b6963", "#ebe9e4"
METADATA_FIELDS = ("title", "main_authors", "core_problem_statement", "key_metrics")


def summarize_chunks_per_document(chunks: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Per document: pages, chunks by content type, token statistics.

    Args:
        chunks: Chunk records with ``source_document``, ``chunk_id``, ``page_end``,
            ``token_count`` and ``content_type``.

    Returns:
        One row per document: ``pages`` (highest ``page_end``), ``chunks``, ``tokens``,
        ``median_tokens``, ``max_tokens`` and one chunk-count column per content type.
    """
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
    """Per split: number of examples supporting each metadata field.

    Args:
        splits: Split name (for instance ``train``) to its examples.

    Returns:
        One row per split: ``examples``, ``documents``, ``empty`` (examples with an empty
        target) and, for each metadata field, the number of examples whose target fills it.
    """
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


def style_figure(figure: Any, title: str, height: int, width: int | None = None) -> Any:
    """Shared look of every chart: white background, recessive grid, titles aligned left.

    Subplot titles (the annotations present when this is called) are moved
    from the centre of their panel to its left edge.

    Args:
        figure: Plotly figure, modified in place.
        title: Figure title, set in bold at the top left.
        height: Figure height in pixels.
        width: Figure width in pixels; None lets Plotly size the figure to its container.

    Returns:
        The same figure.
    """
    panel_width = None
    if "xaxis" in figure.layout and figure.layout.xaxis.domain:
        panel_width = figure.layout.xaxis.domain[1] - figure.layout.xaxis.domain[0]
    for annotation in figure.layout.annotations:
        if panel_width and annotation.xref == "paper" and annotation.xanchor == "center" and annotation.yanchor == "bottom":
            annotation.update(x=annotation.x - panel_width / 2, xanchor="left", font={"size": 14, "color": INK})
            annotation.text = f"<b>{annotation.text}</b>"
            annotation.yshift = 14
    figure.update_layout(
        template="plotly_white",
        title={"text": f"<b>{title}</b>", "x": 0.012, "xanchor": "left", "font": {"size": 18, "color": INK}},
        height=height,
        width=width,
        margin={"l": 70, "r": 40, "t": 110, "b": 60},
        font={"family": "Inter, Segoe UI, Helvetica, Arial, sans-serif", "size": 12, "color": INK_MUTED},
        paper_bgcolor="white",
        plot_bgcolor="white",
        hoverlabel={"font_size": 12},
    )
    figure.update_xaxes(gridcolor=GRID, linecolor="#c9c7c2", zeroline=False, ticks="outside", tickcolor="#c9c7c2", ticklen=4)
    figure.update_yaxes(gridcolor=GRID, linecolor="#c9c7c2", zeroline=False)
    return figure


def legend_inside(figure: Any, row: int, col: int, above: bool = False, left: bool = False) -> dict[str, Any]:
    """Legend settings that place a legend in a top corner of one panel.

    The corner is the right one unless ``left``. With ``above``, the legend
    is laid out horizontally on the title line of the panel instead of
    inside the plot area.

    Args:
        figure: Plotly figure built with ``make_subplots``.
        row: Row of the panel, starting at 1.
        col: Column of the panel, starting at 1.
        above: Place the legend horizontally just above the panel instead of inside it.
        left: Anchor the legend to the left edge of the panel instead of the right.

    Returns:
        A dict for a ``legend`` entry of ``figure.update_layout``, positioned in paper
        coordinates.
    """
    panel = figure.get_subplot(row, col)
    (start, end), top = panel.xaxis.domain, panel.yaxis.domain[1]
    legend = {
        "x": start + 0.01 if left else end,
        "xanchor": "left" if left else "right",
        "bgcolor": "rgba(255,255,255,0.85)",
        "font": {"size": 11},
        "traceorder": "normal",
    }
    if above:
        return {**legend, "y": top + 0.012, "yanchor": "bottom", "orientation": "h"}
    return {**legend, "y": top, "yanchor": "top"}


def _wrap_label(label: str, width: int = 14) -> str:
    """Break a long axis label in two lines at the space nearest to its middle."""
    if len(label) <= width or " " not in label:
        return label
    spaces = [index for index, char in enumerate(label) if char == " "]
    cut = min(spaces, key=lambda index: abs(index - len(label) / 2))
    return label[:cut] + "<br>" + label[cut + 1 :]


def plot_dataset_overview(chunks: Sequence[dict[str, Any]], splits: dict[str, Sequence[TrainingExample]]) -> Any:
    """Three panels: chunk length by content type, chunks per document, labels per split.

    Args:
        chunks: Chunk records with ``source_document``, ``chunk_id``, ``token_count`` and
            ``content_type`` (``text``, ``table`` or ``references``).
        splits: Split name to its examples, as for ``summarize_label_distribution``.

    Returns:
        A Plotly figure.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    frame = pd.DataFrame(chunks)
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Chunk length by content type", "Chunks per document", "Examples supporting each field"),
        horizontal_spacing=0.1,
    )

    # One layout-wide bar mode: "stack" stacks the bars of a same offset group
    # (the histogram) and puts different groups side by side (the splits).
    colours = {"text": SERIES_BLUE, "table": SERIES_ORANGE, "references": SERIES_AQUA}
    bins = {"start": 0, "end": int(frame["token_count"].max()) + 25, "size": 25}
    for name, colour in colours.items():
        lengths = frame.loc[frame["content_type"] == name, "token_count"]
        if len(lengths):
            figure.add_trace(
                go.Histogram(x=lengths, name=name, xbins=bins, marker_color=colour,
                             marker_line={"color": "white", "width": 1}, offsetgroup="length",
                             hovertemplate="%{x} tokens<br>%{y} chunks<extra>" + name + "</extra>"),
                row=1, col=1,
            )  # fmt: skip

    per_document = frame.groupby("source_document")["chunk_id"].count().sort_values()
    figure.add_trace(
        go.Bar(x=per_document.values, y=[_wrap_label(name.removesuffix(".pdf").replace("_", " ")) for name in per_document.index],
               orientation="h", marker={"color": SERIES_BLUE, "cornerradius": 4}, text=per_document.values,
               textposition="outside", showlegend=False, offsetgroup="documents", cliponaxis=False, width=0.6,
               hovertemplate="%{y}<br>%{x} chunks<extra></extra>"),
        row=1, col=2,
    )  # fmt: skip

    distribution = summarize_label_distribution(splits).set_index("split")
    columns = ["empty", *METADATA_FIELDS]
    for offset, (split, row) in enumerate(distribution.iterrows()):
        counts = [int(row[column]) for column in columns]
        figure.add_trace(
            go.Bar(x=["empty", "title", "authors", "problem", "metrics"], y=counts, name=str(split),
                   marker={"color": [SERIES_BLUE, SERIES_ORANGE, SERIES_AQUA][offset % 3], "cornerradius": 3},
                   offsetgroup=str(split), legend="legend2", text=counts, textposition="outside",
                   textfont={"size": 10}, cliponaxis=False),
            row=1, col=3,
        )  # fmt: skip

    figure.update_xaxes(title_text="tokens per chunk", row=1, col=1)
    figure.update_yaxes(title_text="chunks", row=1, col=1)
    figure.update_xaxes(title_text="chunks", range=[0, per_document.max() * 1.15], row=1, col=2)
    figure.update_yaxes(showgrid=False, row=1, col=2)
    figure.update_yaxes(title_text="examples", row=1, col=3)
    figure.update_layout(
        barmode="stack",
        bargroupgap=0.1,
        legend=legend_inside(figure, row=1, col=1, left=True),
        legend2=legend_inside(figure, row=1, col=3),
    )
    return style_figure(figure, "The chunk dataset and its labels", height=440, width=1400)
