"""One-figure summary of a fine-tuning experiment."""

from __future__ import annotations

from typing import Any

from metadata_extraction.dataset_exploration import SERIES_AQUA, SERIES_BLUE, SERIES_ORANGE

DASHBOARD_METRICS = {
    "valid_json_rate": "valid JSON",
    "title_accuracy": "title",
    "author_f1": "authors (F1)",
    "key_metric_recall": "metrics\n(recall)",
    "verifiable_fields_correct_rate": "verifiable\nfields correct",
    "fully_correct_rate": "fully\ncorrect",
}
ANSWER_STATUS_COLOURS = {"ok": SERIES_AQUA, "schema_error": "#eda100", "invalid_json": SERIES_ORANGE, "no_json": "#e34948"}


def plot_experiment_dashboard(
    loss_history: list[dict[str, Any]],
    evaluations: dict[str, dict[str, Any]],
    training_metrics: dict[str, Any],
    title: str,
) -> Any:
    """Loss curves, metrics per evaluation, answer validity, run facts."""
    import matplotlib.pyplot as plt

    figure, ((loss_axis, metric_axis), (status_axis, facts_axis)) = plt.subplots(2, 2, figsize=(14, 8))
    figure.suptitle(title, fontsize=14, fontweight="bold")

    train = [(entry["step"], entry["loss"]) for entry in loss_history if "loss" in entry]
    validation = [(entry["step"], entry["eval_loss"]) for entry in loss_history if "eval_loss" in entry]
    if train:
        loss_axis.plot(*zip(*train), color=SERIES_BLUE, linewidth=1.6, label="train (per step)")
    if validation:
        loss_axis.plot(*zip(*validation), "o-", color=SERIES_ORANGE, linewidth=2, label="validation (held-out paper)")
        loss_axis.annotate(f"{validation[-1][1]:.3f}", validation[-1], textcoords="offset points", xytext=(6, 6))
    loss_axis.set(title="Loss", xlabel="step", ylabel="cross-entropy")
    loss_axis.legend(frameon=False)

    labels = list(evaluations)
    width = 0.8 / max(len(labels), 1)
    for offset, label in enumerate(labels):
        values = [evaluations[label].get(metric) or 0.0 for metric in DASHBOARD_METRICS]
        positions = [index + offset * width for index in range(len(DASHBOARD_METRICS))]
        bars = metric_axis.bar(positions, values, width=width, label=label, color=[SERIES_ORANGE, SERIES_BLUE][offset % 2])
        for bar, metric in zip(bars, DASHBOARD_METRICS):
            value = evaluations[label].get(metric)
            metric_axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                "n/a" if value is None else f"{value:.2f}",
                ha="center",
                fontsize=8,
            )
    metric_axis.set_xticks([index + width * (len(labels) - 1) / 2 for index in range(len(DASHBOARD_METRICS))])
    metric_axis.set_xticklabels(DASHBOARD_METRICS.values(), fontsize=9)
    metric_axis.set(title="Extraction quality on the held-out paper", ylim=(0, 1.32))
    metric_axis.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    metric_axis.legend(frameon=False, loc="upper left", ncol=2)

    bottoms = [0] * len(labels)
    for status, colour in ANSWER_STATUS_COLOURS.items():
        counts = [evaluations[label].get("answer_status_counts", {}).get(status, 0) for label in labels]
        if any(counts):
            status_axis.barh(labels, counts, left=bottoms, height=0.45, color=colour, label=status, edgecolor="white")
            bottoms = [bottom + count for bottom, count in zip(bottoms, counts)]
    status_axis.set(title="Answer validity", xlabel="evaluation chunks")
    status_axis.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.45))

    facts_axis.axis("off")
    facts = [f"{name.replace('_', ' ')}: {value}" for name, value in training_metrics.items()]
    facts_axis.text(0, 0.95, "Run facts", fontsize=12, fontweight="bold", va="top")
    facts_axis.text(0, 0.82, "\n".join(facts), fontsize=10.5, va="top", linespacing=1.7)

    for axis in (loss_axis, metric_axis, status_axis):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure


def plot_approach_leaderboard(leaderboard: list[dict[str, Any]], selected: dict[str, Any]) -> Any:
    """Forest plot: share of fully correct answers per approach, with 95 % intervals."""
    import matplotlib.pyplot as plt

    rows = sorted(
        leaderboard, key=lambda row: (row["verifiable_correct_rate"], -(row["invented_value_rate"] or 0.0))
    )
    labels = [f"{row['approach']}  ·  {row['answer_guard']}" for row in rows]
    figure, axis = plt.subplots(figsize=(11, 0.42 * len(rows) + 1.6))
    for position, row in enumerate(rows):
        is_selected = (row["approach"], row["answer_guard"]) == (selected.get("approach"), selected.get("answer_guard"))
        colour = SERIES_BLUE if is_selected else "#898781"
        low, high = row["verifiable_correct_ci95"]
        axis.plot([low, high], [position, position], color=colour, linewidth=2.2, solid_capstyle="round")
        axis.plot(row["verifiable_correct_rate"], position, "o", color=colour, markersize=8)
        note = f"{row['verifiable_correct']}/{row['examples']} · invented {row['invented_value_rate'] or 0:.0%}"
        axis.text(1.03, position, note + ("   ← selected" if is_selected else ""), va="center", fontsize=9,
                  fontweight="bold" if is_selected else "normal")  # fmt: skip
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels(labels, fontsize=9)
    axis.set(xlim=(0, 1.0), xlabel="share of test chunks with all verifiable fields correct (95 % Wilson interval)")
    axis.set_title("Approaches compared on the test paper", loc="left", fontweight="bold")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    return figure
