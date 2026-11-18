"""One-figure summary of a fine-tuning experiment."""

from __future__ import annotations

from typing import Any

from metadata_extraction.dataset_exploration import SERIES_AQUA, SERIES_BLUE, SERIES_ORANGE, style_figure

DASHBOARD_METRICS = {
    "valid_json_rate": "valid JSON",
    "title_accuracy": "title",
    "author_f1": "authors (F1)",
    "key_metric_recall": "metrics<br>(recall)",
    "verifiable_fields_correct_rate": "verifiable<br>fields correct",
    "fully_correct_rate": "fully<br>correct",
}
ANSWER_STATUS_COLOURS = {"ok": SERIES_AQUA, "schema_error": "#eda100", "invalid_json": SERIES_ORANGE, "no_json": "#e34948"}


def plot_experiment_dashboard(
    loss_history: list[dict[str, Any]],
    evaluations: dict[str, dict[str, Any]],
    training_metrics: dict[str, Any],
    title: str,
) -> Any:
    """Loss curves, metrics per evaluation, answer validity, run facts."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "table"}]],
        subplot_titles=("Loss", "Extraction quality on the held-out paper", "Answer validity", "Run facts"),
        vertical_spacing=0.16,
        horizontal_spacing=0.1,
    )

    train = [(entry["step"], entry["loss"]) for entry in loss_history if "loss" in entry]
    validation = [(entry["step"], entry["eval_loss"]) for entry in loss_history if "eval_loss" in entry]
    if train:
        steps, losses = zip(*train)
        figure.add_trace(
            go.Scatter(x=steps, y=losses, mode="lines", name="train (per step)", line={"color": SERIES_BLUE, "width": 2},
                       legendgroup="loss"),
            row=1, col=1,
        )  # fmt: skip
    if validation:
        steps, losses = zip(*validation)
        figure.add_trace(
            go.Scatter(x=steps, y=losses, mode="lines+markers+text", name="validation (held-out paper)",
                       line={"color": SERIES_ORANGE, "width": 2.5}, legendgroup="loss",
                       text=[""] * (len(losses) - 1) + [f"{losses[-1]:.3f}"], textposition="top right"),
            row=1, col=1,
        )  # fmt: skip

    labels = list(evaluations)
    for offset, label in enumerate(labels):
        values = [evaluations[label].get(metric) for metric in DASHBOARD_METRICS]
        figure.add_trace(
            go.Bar(x=list(DASHBOARD_METRICS.values()), y=[value or 0.0 for value in values], name=label,
                   marker_color=[SERIES_ORANGE, SERIES_BLUE][offset % 2], offsetgroup=label, legendgroup="evaluation",
                   text=["n/a" if value is None else f"{value:.2f}" for value in values], textposition="outside",
                   cliponaxis=False),
            row=1, col=2,
        )  # fmt: skip

    # Same offset group: the statuses of one evaluation are stacked.
    for status, colour in ANSWER_STATUS_COLOURS.items():
        counts = [evaluations[label].get("answer_status_counts", {}).get(status, 0) for label in labels]
        if any(counts):
            figure.add_trace(
                go.Bar(x=counts, y=labels, orientation="h", name=status, marker_color=colour,
                       marker_line={"color": "white", "width": 1}, offsetgroup="status", legendgroup="status", width=0.45),
                row=2, col=1,
            )  # fmt: skip

    figure.add_trace(
        go.Table(
            header={"values": ["", ""], "height": 0, "fill_color": "white", "line_color": "white"},
            cells={
                "values": [
                    [name.replace("_", " ") for name in training_metrics],
                    [str(value) for value in training_metrics.values()],
                ],
                "align": "left",
                "fill_color": "white",
                "line_color": "#e6e4df",
                "height": 26,
            },
        ),
        row=2, col=2,
    )  # fmt: skip

    figure.update_xaxes(title_text="step", row=1, col=1)
    figure.update_yaxes(title_text="cross-entropy", row=1, col=1)
    figure.update_yaxes(range=[0, 1.2], tickvals=[0, 0.25, 0.5, 0.75, 1.0], row=1, col=2)
    figure.update_xaxes(title_text="evaluation chunks", row=2, col=1)
    figure.update_layout(barmode="stack", title={"text": f"<b>{title}</b>"}, legend={"orientation": "h", "y": -0.08})
    return style_figure(figure, height=780)


def plot_approach_leaderboard(leaderboard: list[dict[str, Any]], selected: dict[str, Any]) -> Any:
    """Forest plot: share of fully correct answers per approach, with 95 % intervals."""
    import plotly.graph_objects as go

    rows = sorted(
        leaderboard, key=lambda row: (row["verifiable_correct_rate"], -(row["invented_value_rate"] or 0.0))
    )
    labels = [f"{row['approach']}  ·  {row['answer_guard']}" for row in rows]
    figure = go.Figure()
    for label, row in zip(labels, rows):
        is_selected = (row["approach"], row["answer_guard"]) == (selected.get("approach"), selected.get("answer_guard"))
        colour = SERIES_BLUE if is_selected else "#898781"
        rate = row["verifiable_correct_rate"]
        low, high = row["verifiable_correct_ci95"]
        note = f"{row['verifiable_correct']}/{row['examples']} · invented {row['invented_value_rate'] or 0:.0%}"
        figure.add_trace(
            go.Scatter(
                x=[rate],
                y=[label],
                mode="markers",
                marker={"color": colour, "size": 11},
                error_x={"type": "data", "symmetric": False, "array": [high - rate], "arrayminus": [rate - low],
                         "color": colour, "thickness": 2.5, "width": 0},
                showlegend=False,
                hovertemplate=f"{label}<br>{rate:.0%} [{low:.0%}, {high:.0%}]<br>{note}<extra></extra>",
            )
        )  # fmt: skip
        figure.add_annotation(
            x=1.02, xref="paper", y=label, xanchor="left", showarrow=False,
            text=f"<b>{note}   ← selected</b>" if is_selected else note, font={"size": 11},
        )  # fmt: skip
    figure.update_xaxes(
        range=[0, 1.0],
        title_text="share of test chunks with all verifiable fields correct (95 % Wilson interval)",
    )
    figure.update_yaxes(categoryorder="array", categoryarray=labels, showgrid=False)
    figure.update_layout(title={"text": "<b>Approaches compared on the test paper</b>", "x": 0})
    style_figure(figure, height=40 * len(rows) + 150)
    figure.update_layout(margin={"l": 60, "r": 230, "t": 70, "b": 60})
    return figure
