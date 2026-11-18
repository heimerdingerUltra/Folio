"""Charts of a fine-tuning experiment: loss curves, approach comparison, dashboard."""

from __future__ import annotations

from typing import Any

from metadata_extraction.dataset_exploration import (
    INK,
    INK_MUTED,
    SERIES_AQUA,
    SERIES_BLUE,
    SERIES_ORANGE,
    legend_inside,
    style_figure,
)

DASHBOARD_METRICS = {
    "valid_json_rate": "valid JSON",
    "title_accuracy": "title",
    "author_f1": "authors (F1)",
    "key_metric_recall": "metrics<br>(recall)",
    "verifiable_fields_correct_rate": "verifiable<br>fields correct",
    "fully_correct_rate": "fully<br>correct",
}
ANSWER_STATUS_COLOURS = {"ok": SERIES_AQUA, "schema_error": "#eda100", "invalid_json": SERIES_ORANGE, "no_json": "#e34948"}
NOT_SELECTED_GREY = "#8a8882"


def _add_loss_traces(figure: Any, loss_history: list[dict[str, Any]], **subplot: int) -> None:
    """Training loss per logged step and validation loss per evaluation, last value labelled."""
    import plotly.graph_objects as go

    train = [(entry["step"], entry["loss"]) for entry in loss_history if "loss" in entry]
    validation = [(entry["step"], entry["eval_loss"]) for entry in loss_history if "eval_loss" in entry]
    for points, name, colour, width, size in (
        (train, "train", SERIES_BLUE, 2, 6),
        (validation, "validation (held-out paper)", SERIES_ORANGE, 2.5, 9),
    ):
        if not points:
            continue
        steps, losses = zip(*points)
        figure.add_trace(
            go.Scatter(
                x=steps,
                y=losses,
                mode="lines+markers+text",
                name=name,
                line={"color": colour, "width": width},
                marker={"size": size, "color": colour, "line": {"color": "white", "width": 1.5}},
                text=[""] * (len(losses) - 1) + [f"{losses[-1]:.3f}"],
                textposition="middle right",
                textfont={"color": colour, "size": 12},
                cliponaxis=False,
                hovertemplate="step %{x}<br>loss %{y:.3f}<extra>" + name + "</extra>",
            ),
            **subplot,
        )
    last_step = max((step for step, _ in train + validation), default=1)
    figure.update_xaxes(title_text="training step", range=[0, last_step * 1.12], **subplot)
    figure.update_yaxes(title_text="cross-entropy loss", rangemode="tozero", **subplot)


def plot_loss_curves(loss_history: list[dict[str, Any]], title: str) -> Any:
    """Training and validation loss against the training step.

    Args:
        loss_history: Trainer log entries; an entry with ``loss`` is a training step,
            one with ``eval_loss`` an evaluation. Both carry ``step``.
        title: Chart title.

    Returns:
        A Plotly figure.
    """
    import plotly.graph_objects as go

    figure = go.Figure()
    _add_loss_traces(figure, loss_history)
    figure.update_layout(legend={"x": 0.99, "y": 0.98, "xanchor": "right", "yanchor": "top"})
    return style_figure(figure, title, height=400, width=780)


def plot_search_trials(trials: list[dict[str, Any]], title: str = "Hyperparameter search: one point per trial") -> Any:
    """Validation loss of each completed trial, best at the top, labelled with its configuration.

    Rows are trials of a search (``lora_rank``, ``lora_alpha``,
    ``learning_rate``) or variants of an ablation (``variant``). Points are
    used instead of bars because the losses differ in the second decimal and
    the axis does not start at zero.

    Args:
        trials: One dict per trial with ``validation_loss``; a trial whose loss is None or
            absent is left out. A row is labelled with ``variant`` when it has one, otherwise
            with ``trial``, ``lora_rank``, ``lora_alpha`` and ``learning_rate``.
        title: Chart title.

    Returns:
        A Plotly figure with one point per trial, the lowest loss highlighted.
    """
    import plotly.graph_objects as go

    done = sorted(
        (trial for trial in trials if trial.get("validation_loss") is not None),
        key=lambda trial: trial["validation_loss"],
        reverse=True,
    )
    labels = [
        trial["variant"]
        if "variant" in trial
        else f"trial {trial['trial']}  ·  r={trial['lora_rank']}, alpha={trial['lora_alpha']}, lr={trial['learning_rate']:.1e}"
        for trial in done
    ]
    losses = [trial["validation_loss"] for trial in done]
    best = min(losses, default=None)
    colours = [SERIES_BLUE if loss == best else NOT_SELECTED_GREY for loss in losses]
    figure = go.Figure(
        go.Scatter(
            x=losses,
            y=labels,
            mode="markers+text",
            marker={"color": colours, "size": 14, "line": {"color": "white", "width": 2}},
            text=[f"{loss:.4f}" + ("   ← lowest" if loss == best else "") for loss in losses],
            textposition="middle right",
            textfont={"color": colours, "size": 12},
            cliponaxis=False,
            showlegend=False,
            hovertemplate="%{y}<br>validation loss %{x:.4f}<extra></extra>",
        )
    )
    if losses:
        span = max(max(losses) - best, 0.01)
        figure.update_xaxes(range=[best - span * 0.25, max(losses) + span * 0.6])
    figure.update_xaxes(title_text="validation loss at the end of the trial")
    figure.update_yaxes(ticklabelstandoff=10, type="category")
    return style_figure(figure, title, height=52 * len(done) + 170, width=1000)


def plot_approach_leaderboard(leaderboard: list[dict[str, Any]], selected: dict[str, Any]) -> Any:
    """Forest plot: share of fully correct answers per approach, with 95 % intervals.

    Args:
        leaderboard: One row per approach and answer guard, with ``approach``,
            ``answer_guard``, ``verifiable_correct_rate`` (0 to 1), ``verifiable_correct_ci95``
            (low, high), ``verifiable_correct``, ``examples`` and ``invented_value_rate``
            (None counts as 0).
        selected: The retained row, matched on ``approach`` and ``answer_guard`` and
            highlighted; an empty dict highlights nothing.

    Returns:
        A Plotly figure with one row per approach, the best at the top.
    """
    import plotly.graph_objects as go

    rows = sorted(
        leaderboard, key=lambda row: (row["verifiable_correct_rate"], -(row["invented_value_rate"] or 0.0))
    )
    labels = [f"{row['approach']}  ·  {row['answer_guard']}" for row in rows]
    figure = go.Figure()
    for label, row in zip(labels, rows):
        is_selected = (row["approach"], row["answer_guard"]) == (selected.get("approach"), selected.get("answer_guard"))
        colour = SERIES_BLUE if is_selected else NOT_SELECTED_GREY
        rate = row["verifiable_correct_rate"]
        low, high = row["verifiable_correct_ci95"]
        note = f"{row['verifiable_correct']}/{row['examples']}  ·  invented {row['invented_value_rate'] or 0:.0%}"
        figure.add_trace(
            go.Scatter(
                x=[rate],
                y=[label],
                mode="markers",
                marker={"color": colour, "size": 12, "line": {"color": "white", "width": 2}},
                error_x={"type": "data", "symmetric": False, "array": [high - rate], "arrayminus": [rate - low],
                         "color": colour, "thickness": 2.5, "width": 0},
                showlegend=False,
                hovertemplate=f"{label}<br>{rate:.0%} [{low:.0%}, {high:.0%}]<br>{note}<extra></extra>",
            )
        )  # fmt: skip
        figure.add_annotation(
            x=1.02, xref="paper", y=label, xanchor="left", showarrow=False,
            text=f"<b>{note}   ← selected</b>" if is_selected else note,
            font={"size": 12, "color": SERIES_BLUE if is_selected else INK_MUTED},
        )  # fmt: skip
    figure.update_xaxes(
        range=[0, 1.0],
        tickformat=".0%",
        title_text="share of test chunks with all verifiable fields correct (95 % Wilson interval)",
    )
    figure.update_yaxes(categoryorder="array", categoryarray=labels, showgrid=False, ticklabelstandoff=8)
    style_figure(figure, "Approaches compared on the test paper", height=38 * len(rows) + 150, width=1150)
    figure.update_layout(margin={"r": 250})
    return figure


def plot_experiment_dashboard(
    loss_history: list[dict[str, Any]],
    evaluations: dict[str, dict[str, Any]],
    training_metrics: dict[str, Any],
    title: str,
) -> Any:
    """Loss curves, metrics per evaluation, answer validity, run facts.

    Args:
        loss_history: Trainer log entries; an entry with ``loss`` is a training step,
            one with ``eval_loss`` an evaluation. Both carry ``step``.
        evaluations: Evaluation label to its metrics: the keys of ``DASHBOARD_METRICS`` (rates
            between 0 and 1, None shown as n/a) and ``answer_status_counts``, the number of
            answers per status in ``ANSWER_STATUS_COLOURS``.
        training_metrics: Facts about the run, listed as a two-column table of names and
            values.
        title: Chart title.

    Returns:
        A Plotly figure with four panels.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "table"}]],
        subplot_titles=("Loss", "Quality on the test paper", "Answer validity", "Run facts"),
        vertical_spacing=0.2,
        horizontal_spacing=0.1,
        row_heights=[0.52, 0.48],
    )
    _add_loss_traces(figure, loss_history, row=1, col=1)

    labels = list(evaluations)
    for offset, label in enumerate(labels):
        values = [evaluations[label].get(metric) for metric in DASHBOARD_METRICS]
        figure.add_trace(
            go.Bar(x=list(DASHBOARD_METRICS.values()), y=[value or 0.0 for value in values], name=label,
                   marker={"color": [SERIES_ORANGE, SERIES_BLUE][offset % 2], "cornerradius": 4},
                   offsetgroup=label, legend="legend2",
                   text=["n/a" if value is None else f"{value:.2f}" for value in values], textposition="outside",
                   textfont={"size": 11}, cliponaxis=False),
            row=1, col=2,
        )  # fmt: skip

    # Same offset group: the statuses of one evaluation are stacked.
    for status, colour in ANSWER_STATUS_COLOURS.items():
        counts = [evaluations[label].get("answer_status_counts", {}).get(status, 0) for label in labels]
        if any(counts):
            figure.add_trace(
                go.Bar(x=counts, y=labels, orientation="h", name=status.replace("_", " "), marker_color=colour,
                       marker_line={"color": "white", "width": 2}, offsetgroup="status", legend="legend3", width=0.5,
                       text=[count or "" for count in counts], textposition="inside", insidetextanchor="middle",
                       textfont={"color": "white", "size": 12}),
                row=2, col=1,
            )  # fmt: skip

    figure.add_trace(
        go.Table(
            columnwidth=[1.4, 1],
            header={"values": ["", ""], "height": 0, "fill_color": "white", "line_color": "white"},
            cells={
                "values": [
                    [name.replace("_", " ") for name in training_metrics],
                    [str(value) for value in training_metrics.values()],
                ],
                "align": "left",
                "fill_color": "white",
                "line_color": "#ebe9e4",
                "height": 26,
                "font": {"size": 12, "color": [INK_MUTED, INK]},
            },
        ),
        row=2, col=2,
    )  # fmt: skip

    figure.update_yaxes(range=[0, 1.18], tickvals=[0, 0.25, 0.5, 0.75, 1.0], row=1, col=2)
    figure.update_xaxes(title_text="evaluation chunks", dtick=2, row=2, col=1)
    figure.update_yaxes(showgrid=False, autorange="reversed", row=2, col=1)
    figure.update_layout(
        barmode="stack",
        bargroupgap=0.08,
        legend=legend_inside(figure, row=1, col=1),
        legend2=legend_inside(figure, row=1, col=2, above=True),
        legend3=legend_inside(figure, row=2, col=1, above=True),
    )
    return style_figure(figure, title, height=980, width=1300)
