"""Self-contained HTML dashboard of a run, plus SVG charts for the README.

Usage:
    python -m pdf_chunking.performance_dashboard --report results.run_report.json --assets assets

One file, no external dependency. Every figure is read from the run report
or a benchmark result file. A system keeps its colour across charts.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any, Sequence

# ---------------------------------------------------------------- appearance

# Categorical slots (light, dark). The order is fixed and colour-vision-safe
# for adjacent series; a system keeps its slot in every chart.
SERIES_COLOURS = {
    "s1": ("#2a78d6", "#3987e5"),  # blue   - structure-aware chunking / this pipeline
    "s2": ("#eb6834", "#d95926"),  # orange - fixed-size baseline
    "s3": ("#1baf7a", "#199e70"),  # aqua   - structure + context header
    "s4": ("#eda100", "#c98500"),  # yellow - ... without references
    "mute": ("#c3c2b7", "#52514e"),  # de-emphasised marks (other parsers)
}
SYSTEM_SLOTS = {
    "structure": "s1",
    "fixed": "s2",
    "structure + context": "s3",
    "structure + context, no references": "s4",
}

_THEME_LIGHT = {
    "plane": "#f9f9f7", "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
    "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7", "border": "rgba(11,11,11,0.10)",
    "good": "#006300", "bad": "#d03b3b",
}  # fmt: skip
_THEME_DARK = {
    "plane": "#0d0d0d", "surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7",
    "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835", "border": "rgba(255,255,255,0.10)",
    "good": "#0ca30c", "bad": "#e66767",
}  # fmt: skip

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _variables(dark: bool) -> str:
    """CSS custom properties for one theme."""
    theme = _THEME_DARK if dark else _THEME_LIGHT
    lines = [f"--{name}: {value};" for name, value in theme.items()]
    lines += [f"--{slot}: {pair[1 if dark else 0]};" for slot, pair in SERIES_COLOURS.items()]
    return " ".join(lines)


def theme_css(root: str) -> str:
    """Theme rules: light by default, dark from the OS setting or a toggle."""
    return (
        f"{root} {{ color-scheme: light; {_variables(False)} }}\n"
        f"@media (prefers-color-scheme: dark) {{ {root}:not([data-theme='light']) "
        f"{{ color-scheme: dark; {_variables(True)} }} }}\n"
        f"{root}[data-theme='dark'] {{ color-scheme: dark; {_variables(True)} }}\n"
    )


CHART_CSS = f"""
.chart text {{ font-family: {FONT}; font-size: 12px; fill: var(--ink2); }}
.chart .value {{ fill: var(--ink); font-weight: 600; font-variant-numeric: tabular-nums; }}
.chart .tick {{ fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }}
.chart .grid {{ stroke: var(--grid); stroke-width: 1; }}
.chart .axis {{ stroke: var(--axis); stroke-width: 1; }}
.chart .note {{ fill: var(--muted); font-size: 11px; }}
.chart .mark {{ transition: opacity .12s; }}
.chart .mark:hover {{ opacity: .78; }}
"""


def _esc(value: object) -> str:
    """HTML-escape a value."""
    return html.escape(str(value), quote=True)


def _text_width(text: str, size: float = 12.0) -> float:
    """Rough rendered width of ``text`` (no font metrics available here)."""
    return len(text) * size * 0.56


def _format(value: float, unit: str = "") -> str:
    """Compact number: 97564 -> 97.6K, 0.884 with unit '%' -> 88.4%."""
    if unit == "%":
        return f"{value * 100:.1f}%"
    if unit == "s":
        return f"{value:.1f} s"
    if abs(value) >= 10_000:
        return f"{value / 1000:.1f}K"
    if isinstance(value, float) and not value.is_integer():
        return f"{value:.3f}" if abs(value) < 1 else f"{value:.1f}"
    return f"{int(value):,}"


# -------------------------------------------------------------------- charts


def _bar_path(x: float, y: float, width: float, height: float, radius: float = 4.0) -> str:
    """Horizontal bar: square at the baseline (left), rounded at the data end."""
    radius = min(radius, width, height / 2)
    return (
        f"M{x:.1f},{y:.1f} h{width - radius:.1f} a{radius},{radius} 0 0 1 {radius},{radius} "
        f"v{height - 2 * radius:.1f} a{radius},{radius} 0 0 1 -{radius},{radius} h-{width - radius:.1f} z"
    )


def _column_path(x: float, y: float, width: float, height: float, radius: float = 4.0) -> str:
    """Vertical column: square at the baseline (bottom), rounded on top."""
    radius = min(radius, height, width / 2)
    return (
        f"M{x:.1f},{y + height:.1f} v-{height - radius:.1f} a{radius},{radius} 0 0 1 {radius},-{radius} "
        f"h{width - 2 * radius:.1f} a{radius},{radius} 0 0 1 {radius},{radius} v{height - radius:.1f} z"
    )


def _legend(series: Sequence[tuple[str, str]], x: float, y: float) -> tuple[str, float]:
    """Legend row of swatch + name pairs. Returns (svg, total width)."""
    parts: list[str] = []
    cursor = x
    for name, slot in series:
        parts.append(f'<rect x="{cursor:.1f}" y="{y - 9:.1f}" width="10" height="10" rx="2" fill="var(--{slot})"/>')
        parts.append(f'<text x="{cursor + 15:.1f}" y="{y:.1f}">{_esc(name)}</text>')
        cursor += 15 + _text_width(name) + 18
    return "".join(parts), cursor - x


def draw_grouped_bar_chart(
    rows: Sequence[tuple[str, Sequence[float | None]]],
    series: Sequence[tuple[str, str]],
    unit: str = "",
    maximum: float | None = None,
    width: int = 720,
    title: str = "",
) -> str:
    """Horizontal grouped bars: one group per row, one bar per series.

    Args:
        rows: ``(label, [value per series])``; ``None`` skips a bar.
        series: ``(name, colour slot)`` in display order.
        unit: ``"%"`` (values are fractions), ``"s"`` or ``""``.
        maximum: Value at the right edge (default: largest value).
    """
    values = [v for _, row in rows for v in row if v is not None]
    maximum = maximum or (max(values) if values else 1.0) or 1.0
    label_width = max(_text_width(label) for label, _ in rows) + 30
    value_width = max(_text_width(_format(v, unit)) for v in values) + 28 if values else 40
    plot_x = label_width
    plot_width = width - plot_x - value_width
    bar_height = 14 if len(series) > 1 else 18
    bar_gap = 2  # surface gap between adjacent bars
    group_gap = 16
    group_height = len(series) * bar_height + (len(series) - 1) * bar_gap
    show_legend = len(series) > 1
    top = 30 if show_legend else 8
    height = top + len(rows) * (group_height + group_gap) + 6

    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height:.0f}" width="{width}" role="img" '
        f'aria-label="{_esc(title)}" xmlns="http://www.w3.org/2000/svg">'
    ]
    if show_legend:
        parts.append(_legend(series, plot_x, 14)[0])
    parts.append(f'<line class="axis" x1="{plot_x:.1f}" y1="{top - 4}" x2="{plot_x:.1f}" y2="{height - 8:.0f}"/>')
    for row_index, (label, row_values) in enumerate(rows):
        group_y = top + row_index * (group_height + group_gap)
        parts.append(
            f'<text x="{plot_x - 10:.1f}" y="{group_y + group_height / 2 + 4:.1f}" text-anchor="end">{_esc(label)}</text>'
        )
        for series_index, value in enumerate(row_values):
            if value is None:
                continue
            name, slot = series[series_index]
            y = group_y + series_index * (bar_height + bar_gap)
            bar_width = max(2.0, plot_width * value / maximum)
            tip = f"{label} · {name}: {_format(value, unit)}"
            parts.append(
                f'<path class="mark" d="{_bar_path(plot_x, y, bar_width, bar_height)}" fill="var(--{slot})" '
                f'data-tip="{_esc(tip)}"><title>{_esc(tip)}</title></path>'
            )
            parts.append(
                f'<text class="value" x="{plot_x + bar_width + 6:.1f}" y="{y + bar_height / 2 + 4:.1f}">'
                f"{_esc(_format(value, unit))}</text>"
            )
    parts.append("</svg>")
    return "".join(parts)


def draw_column_chart(
    bins: Sequence[tuple[str, float]],
    slot: str = "s1",
    width: int = 720,
    height: int = 240,
    title: str = "",
    x_title: str = "",
) -> str:
    """Single-series column chart (used for the token-length histogram)."""
    maximum = max((value for _, value in bins), default=1.0) or 1.0
    # Round the axis maximum up to a clean tick.
    step = max(1, round(maximum / 4 / 10) * 10) if maximum > 40 else max(1, round(maximum / 4))
    axis_max = step * (int(maximum / step) + 1)
    left, right, top, bottom = 44, 12, 14, 44
    plot_width, plot_height = width - left - right, height - top - bottom
    slot_width = plot_width / max(len(bins), 1)
    column_width = min(24.0, slot_width - 2)  # capped thickness, >= 2px gap

    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" width="{width}" role="img" '
        f'aria-label="{_esc(title)}" xmlns="http://www.w3.org/2000/svg">'
    ]
    for tick in range(0, axis_max + 1, step):
        y = top + plot_height * (1 - tick / axis_max)
        css = "axis" if tick == 0 else "grid"
        parts.append(f'<line class="{css}" x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{tick}</text>')
    peak = max(range(len(bins)), key=lambda i: bins[i][1]) if bins else -1
    for index, (label, value) in enumerate(bins):
        column_height = plot_height * value / axis_max
        x = left + index * slot_width + (slot_width - column_width) / 2
        y = top + plot_height - column_height
        tip = f"{label} tokens: {int(value)} chunks"
        if value > 0:
            parts.append(
                f'<path class="mark" d="{_column_path(x, y, column_width, column_height)}" fill="var(--{slot})" '
                f'data-tip="{_esc(tip)}"><title>{_esc(tip)}</title></path>'
            )
        if index == peak:  # label only the extreme; the axis carries the rest
            parts.append(
                f'<text class="value" x="{x + column_width / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle">{int(value)}</text>'
            )
        parts.append(
            f'<text class="tick" x="{left + (index + 0.5) * slot_width:.1f}" y="{top + plot_height + 16}" '
            f'text-anchor="middle">{_esc(label)}</text>'
        )
    if x_title:
        parts.append(
            f'<text class="note" x="{left + plot_width / 2:.1f}" y="{height - 6}" text-anchor="middle">{_esc(x_title)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def wrap_chart_as_standalone_svg(chart: str) -> str:
    """Wrap a chart so it renders correctly as an image file (README).

    The theme variables and a background card are embedded in the SVG itself,
    because an ``<img>`` cannot inherit the page's CSS.
    """
    head, body = chart.split(">", 1)
    view_box = head.split('viewBox="')[1].split('"')[0].split()
    width, height = float(view_box[2]), float(view_box[3])
    pad = 16
    style = theme_css("svg") + CHART_CSS
    head = head.replace(
        f'viewBox="0 0 {view_box[2]} {view_box[3]}"',
        f'viewBox="{-pad} {-pad} {width + 2 * pad:.0f} {height + 2 * pad:.0f}"',
    ).replace(f'width="{view_box[2]}"', f'width="{width + 2 * pad:.0f}"')
    background = (
        f'<rect x="{-pad}" y="{-pad}" width="{width + 2 * pad:.0f}" height="{height + 2 * pad:.0f}" '
        f'rx="10" fill="var(--surface)" stroke="var(--border)"/>'
    )
    return f"{head}><style>{style}</style>{background}{body}"


# ------------------------------------------------------- report -> chart data


def collect_quality_chart_rows(report: dict[str, Any]) -> list[tuple[str, list[float | None]]]:
    """Rows of the structure-vs-baseline quality chart (values are fractions)."""
    quality, baseline = report["quality"], report.get("baseline_quality")

    def pick(source: dict[str, Any] | None, *path: str) -> float | None:
        """Follow ``path`` through nested dictionaries; ``None`` if a key is missing."""
        for key in path:
            if not isinstance(source, dict):
                return None
            source = source.get(key)
        return source if isinstance(source, (int, float)) else None

    specs = [
        ("Chunks starting and ending on a sentence", ("boundary_integrity", "clean_both_rate")),
        ("Chunks starting on a sentence", ("boundary_integrity", "clean_start_rate")),
        ("Chunks ending on a sentence", ("boundary_integrity", "clean_end_rate")),
        ("Tables kept in a single chunk", ("table_integrity", "intact_rate")),
    ]
    return [(label, [pick(quality, *path), pick(baseline, *path)]) for label, path in specs]


def collect_retrieval_chart_rows(
    retrieval: dict[str, Any], retriever: str
) -> tuple[list[tuple[str, list[float | None]]], list[tuple[str, str]]]:
    """Rows and series of the retrieval chart for one retriever."""
    per_system = retrieval["results"][retriever]
    systems = [name for name in SYSTEM_SLOTS if name in per_system]
    # Keep the baseline first so the eye reads "from" then "to".
    systems.sort(key=lambda name: name != "fixed")
    metrics = [("Hit@1", "hit@1"), ("Hit@3", "hit@3"), ("Hit@5", "hit@5"), ("MRR@10", "mrr@10")]
    rows = [(label, [per_system[name][key] for name in systems]) for label, key in metrics]
    return rows, [(name, SYSTEM_SLOTS[name]) for name in systems]


# ---------------------------------------------------------------------- page

PAGE_CSS = f"""
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--plane); color: var(--ink); font-family: {FONT}; font-size: 14px; line-height: 1.5; }}
main {{ max-width: 1120px; margin: 0 auto; padding: 32px 20px 64px; }}
h1 {{ font-size: 26px; margin: 0 0 4px; letter-spacing: -0.01em; }}
h2 {{ font-size: 17px; margin: 0 0 2px; }}
.sub {{ color: var(--ink2); margin: 0 0 14px; }}
.meta {{ color: var(--muted); font-size: 12px; }}
.grid {{ display: grid; gap: 16px; margin-top: 16px; }}
.cols-2 {{ grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); }}
.tiles {{ grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }}
.card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 18px 20px; overflow: hidden; }}
.card svg {{ max-width: 100%; height: auto; display: block; margin-top: 8px; }}
.hero {{ display: flex; flex-wrap: wrap; gap: 28px; align-items: center; }}
.hero .figure {{ font-size: 68px; font-weight: 650; line-height: 1; letter-spacing: -0.03em; }}
.hero .versus {{ color: var(--ink2); font-size: 15px; max-width: 520px; }}
.hero .versus b {{ color: var(--ink); }}
.tile .label {{ color: var(--ink2); font-size: 12px; }}
.tile .value {{ font-size: 26px; font-weight: 620; letter-spacing: -0.02em; }}
.tile .hint {{ color: var(--muted); font-size: 12px; }}
.status {{ display: inline-flex; align-items: center; gap: 6px; font-weight: 600; }}
.status.good {{ color: var(--good); }} .status.bad {{ color: var(--bad); }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 8px; }}
th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--grid); }}
th {{ color: var(--ink2); font-weight: 600; }}
td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
tr:last-child td {{ border-bottom: 0; }}
.bar {{ display: inline-block; height: 8px; border-radius: 4px; background: var(--s1); vertical-align: middle; margin-right: 8px; }}
details {{ margin-top: 10px; }} summary {{ cursor: pointer; color: var(--ink2); font-size: 12px; }}
.note {{ color: var(--ink2); font-size: 13px; margin: 10px 0 0; }}
.swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; }}
#tip {{ position: fixed; pointer-events: none; background: var(--ink); color: var(--surface); padding: 5px 9px; border-radius: 6px; font-size: 12px; opacity: 0; transition: opacity .1s; z-index: 10; white-space: nowrap; }}
.toggle {{ float: right; background: var(--surface); color: var(--ink2); border: 1px solid var(--border); border-radius: 8px; padding: 6px 10px; cursor: pointer; font: inherit; font-size: 12px; }}
@media (max-width: 520px) {{ .cols-2 {{ grid-template-columns: 1fr; }} .hero .figure {{ font-size: 52px; }} }}
"""

PAGE_JS = """
const tip = document.getElementById('tip');
document.querySelectorAll('[data-tip]').forEach(mark => {
  mark.addEventListener('mousemove', event => {
    tip.textContent = mark.dataset.tip; tip.style.opacity = 1;
    const x = Math.min(event.clientX + 12, window.innerWidth - tip.offsetWidth - 8);
    tip.style.left = x + 'px'; tip.style.top = (event.clientY + 14) + 'px';
  });
  mark.addEventListener('mouseleave', () => { tip.style.opacity = 0; });
});
document.getElementById('theme').addEventListener('click', () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === 'dark'
                                  : matchMedia('(prefers-color-scheme: dark)').matches;
  root.dataset.theme = dark ? 'light' : 'dark';
});
"""


def _tile(label: str, value: str, hint: str = "") -> str:
    """One headline figure: label, value and optional hint."""
    hint_html = f'<div class="hint">{_esc(hint)}</div>' if hint else ""
    return (
        f'<div class="card tile"><div class="label">{_esc(label)}</div>'
        f'<div class="value">{_esc(value)}</div>{hint_html}</div>'
    )


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]], numeric_from: int = 1) -> str:
    """Plain HTML table; columns from ``numeric_from`` on are right-aligned."""
    head = "".join(
        f'<th class="{"num" if index >= numeric_from else ""}">{_esc(header)}</th>'
        for index, header in enumerate(headers)
    )
    body = "".join(
        "<tr>"
        + "".join(
            f'<td class="{"num" if index >= numeric_from else ""}">{cell}</td>' for index, cell in enumerate(row)
        )
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _details(summary: str, content: str) -> str:
    """A collapsible section."""
    return f"<details><summary>{_esc(summary)}</summary>{content}</details>"


def build_all_charts(
    report: dict[str, Any],
    retrieval: dict[str, Any] | None = None,
    parsers: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Build every chart available from the given data, keyed by name."""
    charts: dict[str, str] = {}
    has_baseline = "baseline_quality" in report
    series = [("Structure-aware (this pipeline)", "s1")] + ([("Fixed-size baseline", "s2")] if has_baseline else [])
    rows = [(label, values[: len(series)]) for label, values in collect_quality_chart_rows(report)]
    rows = [(label, values) for label, values in rows if any(v is not None for v in values)]
    if rows:
        charts["quality"] = draw_grouped_bar_chart(rows, series, unit="%", maximum=1.0, title="Chunk quality by strategy")

    histogram = report["quality"].get("token_histogram") or {}
    if histogram:
        charts["tokens"] = draw_column_chart(
            [(label.split("-")[0], count) for label, count in histogram.items()],
            title="Distribution of chunk length in tokens",
            x_title="chunk length (tokens, lower bound of each 50-token bin)",
        )
    if retrieval:
        for retriever in retrieval["results"]:
            key = "retrieval_dense" if retriever.startswith("dense") else "retrieval_tfidf"
            chart_rows, chart_series = collect_retrieval_chart_rows(retrieval, retriever)
            charts[key] = draw_grouped_bar_chart(
                chart_rows, chart_series, maximum=1.0, title=f"Retrieval quality, {retriever}"
            )
    if parsers:
        rows = [(row["parser"], [row["seconds"]]) for row in parsers["parsers"] if row.get("seconds") is not None]
        charts["parsers"] = draw_grouped_bar_chart(
            rows, [("seconds", "s1")], unit="s", title="Text extraction time per parser, 102 pages"
        )
    return charts


def render_dashboard_html(
    report: dict[str, Any],
    retrieval: dict[str, Any] | None = None,
    parsers: dict[str, Any] | None = None,
    parameter_sweep: dict[str, Any] | None = None,
) -> str:
    """Render the run report (and optional benchmarks) as an HTML page."""
    run, summary, quality = report["run"], report["summary"], report["quality"]
    baseline = report.get("baseline_quality")
    validation = report["validation"]
    charts = build_all_charts(report, retrieval, parsers)
    sections: list[str] = []

    # --- hero: the single headline number -------------------------------
    clean = quality.get("boundary_integrity", {}).get("clean_both_rate")
    if clean is not None:
        versus = ""
        if baseline and baseline.get("boundary_integrity", {}).get("clean_both_rate") is not None:
            base = baseline["boundary_integrity"]["clean_both_rate"]
            versus = (
                f" A fixed-size splitter on the same documents manages <b>{base * 100:.1f}%</b>"
                + (f": a {clean / base:.0f}&times; difference." if base > 0 else ".")
            )
        sections.append(
            '<div class="card hero"><div class="figure">'
            f"{clean * 100:.1f}%</div><div class=\"versus\"><b>of prose chunks start and end on a "
            f"sentence boundary.</b>{versus}</div></div>"
        )

    # --- stat tiles ------------------------------------------------------
    tables = quality.get("table_integrity", {})
    tiles = [
        _tile("Documents processed", f"{summary['documents_processed']} / {summary['documents_found']}",
              f"error rate {summary['error_rate'] * 100:.0f}%" if summary.get("error_rate") is not None else ""),
        _tile("Pages", _format(summary["pages"])),
        _tile("Chunks", _format(summary["chunks"]), f"{_format(summary['tokens'])} tokens"),
        _tile("Median chunk", f"{quality['token_length'].get('median', 0):.0f} tok",
              f"max {quality['token_length'].get('max', 0)} / budget {report['config']['chunking']['max_tokens']}"),
        _tile("Tables located", _format(tables.get("tables", 0)),
              f"{tables.get('intact', 0)} whole, rest split by rows"),
        _tile("Processing time", f"{summary.get('processing_wall_seconds', run['duration_seconds']):.0f} s",
              f"{summary.get('pages_per_second', 0)} pages/s"),
    ]  # fmt: skip
    sections.append(f'<div class="grid tiles">{"".join(tiles)}</div>')

    # --- quality vs baseline + token histogram ---------------------------
    row: list[str] = []
    if "quality" in charts:
        table = _table(
            ["Metric", "Structure-aware", "Fixed-size"],
            [
                [_esc(label)] + [("&ndash;" if v is None else f"{v * 100:.1f}%") for v in (values + [None])[:2]]
                for label, values in collect_quality_chart_rows(report)
            ],
        )
        note = (
            "Same documents, same token budget, same overlap. Only the splitting rule differs."
            if baseline
            else "Run with --benchmark-against-fixed-windows to add the fixed-size reference."
        )
        row.append(
            f'<div class="card"><h2>Chunk quality</h2><p class="sub">{note}</p>{charts["quality"]}'
            f'{_details("Table view", table)}</div>'
        )
    if "tokens" in charts:
        length = quality["token_length"]
        table = _table(["Bin (tokens)", "Chunks"], [[_esc(k), v] for k, v in quality["token_histogram"].items()])
        row.append(
            '<div class="card"><h2>Chunk length</h2>'
            f'<p class="sub">Mean {length.get("mean")} tokens, 95th percentile {length.get("p95")}; '
            f'nothing exceeds the {report["config"]["chunking"]["max_tokens"]}-token budget.</p>'
            f'{charts["tokens"]}{_details("Table view", table)}</div>'
        )
    if row:
        sections.append(f'<div class="grid cols-2">{"".join(row)}</div>')

    # --- retrieval benchmark --------------------------------------------
    if retrieval:
        row = []
        for retriever, per_system in retrieval["results"].items():
            key = "retrieval_dense" if retriever.startswith("dense") else "retrieval_tfidf"
            table_rows = []
            for name, metrics in per_system.items():
                gain = (
                    f"{metrics['mrr_gain_vs_fixed']:+.3f} [{metrics['mrr_gain_ci95'][0]:+.3f}, "
                    f"{metrics['mrr_gain_ci95'][1]:+.3f}]"
                    if "mrr_gain_vs_fixed" in metrics
                    else "reference"
                )
                swatch = f'<span class="swatch" style="background:var(--{SYSTEM_SLOTS.get(name, "mute")})"></span>'
                table_rows.append(
                    [swatch + _esc(name), metrics["passages"], f"{metrics['hit@1']:.3f}", f"{metrics['hit@5']:.3f}",
                     f"{metrics['mrr@10']:.3f}", gain, f"{metrics['top5_tokens']:.0f}"]
                )  # fmt: skip
            table = _table(
                ["System", "Passages", "Hit@1", "Hit@5", "MRR@10", "MRR gain vs fixed (95% CI)", "Top-5 tokens"],
                table_rows,
            )
            row.append(
                f'<div class="card"><h2>Retrieval &mdash; {_esc(retriever)}</h2>'
                f'<p class="sub">{retrieval["n_questions"]} questions; a hit is a retrieved chunk that contains '
                f"the answer.</p>{charts[key]}{_details('Table view, with confidence intervals', table)}</div>"
            )
        sections.append(f'<div class="grid cols-2">{"".join(row)}</div>')

    # --- chunk size / overlap sweep ---------------------------------------
    if parameter_sweep:
        table = _table(
            ["Strategy", "Indexed text", "Max tokens", "Overlap", "Passages", "Hit@1", "Hit@5", "MRR@10", "Top-5 tokens"],
            [
                [_esc(c["strategy"]), _esc(c["indexing"]), c["max_tokens"], c["overlap_tokens"], c["passages"],
                 f"{c['hit@1']:.3f}", f"{c['hit@5']:.3f}", f"{c['mrr@10']:.3f}", c["top5_tokens"]]
                for c in parameter_sweep["configurations"]
            ],
            numeric_from=2,
        )  # fmt: skip
        sections.append(
            '<div class="grid"><div class="card"><h2>Chunk size and overlap sweep</h2>'
            f'<p class="sub">Dense retrieval on {parameter_sweep["n_questions"]} questions, best MRR@10 first. '
            "Top-5 tokens is what a generator would have to read.</p>" + table + "</div></div>"
        )

    # --- parser benchmark -------------------------------------------------
    if parsers:
        table = _table(
            ["Parser", "Seconds", "Leftover ligatures", "Words cut by a hyphen", "Column flow"],
            [
                [_esc(r["parser"]), f"{r['seconds']:.1f}", r["ligatures"], r["hyphen_breaks"], _esc(r["column_flow"])]
                for r in parsers["parsers"] if r.get("seconds") is not None
            ],
        )  # fmt: skip
        sections.append(
            '<div class="grid"><div class="card"><h2>Parser benchmark</h2>'
            f'<p class="sub">Default text extraction of each library on the same {parsers.get("n_pdfs", "")} PDFs. '
            "Column flow: sentences spanning a line break inside one column that come out intact.</p>"
            f'{charts["parsers"]}{table}</div></div>'
        )

    # --- per-document table ----------------------------------------------
    ok = [entry for entry in report["documents"] if entry["status"] == "ok"]
    most = max((entry["chunks"] for entry in ok), default=1)
    doc_rows = []
    for entry in report["documents"]:
        if entry["status"] != "ok":
            doc_rows.append(
                [_esc(entry["source_document"]), f'<span class="status bad">&#10007; {_esc(entry["status"])}</span>',
                 "", "", "", "", "", _esc((entry.get("error") or "")[:80])]
            )  # fmt: skip
            continue
        stats = entry["extraction"]
        bar = f'<span class="bar" style="width:{60 * entry["chunks"] / most:.0f}px"></span>'
        doc_rows.append(
            [_esc(entry["source_document"]), '<span class="status good">&#10003; ok</span>', entry["pages"],
             bar + str(entry["chunks"]), f"{entry['token_length'].get('mean', 0):.0f}",
             stats.get("blocks_table", 0),
             stats.get("figure_text_blocks_dropped", 0) + stats.get("header_footer_blocks_dropped", 0),
             f"{entry['extraction_seconds']:.1f} s"]
        )  # fmt: skip
    sections.append(
        '<div class="grid"><div class="card"><h2>Per document</h2>'
        + _table(
            ["Document", "Status", "Pages", "Chunks", "Mean tokens", "Tables", "Noise blocks removed", "Extraction"],
            doc_rows,
            numeric_from=2,
        )
        + "</div></div>"
    )

    # --- what the extractor did + validation ------------------------------
    totals: dict[str, int] = {}
    for entry in ok:
        for key, value in entry["extraction"].items():
            if isinstance(value, int) and not isinstance(value, bool):
                totals[key] = totals.get(key, 0) + value
    cleaning = [
        _tile("Section headings found", _format(totals.get("blocks_heading", 0))),
        _tile("Figure-text fragments removed", _format(totals.get("figure_text_blocks_dropped", 0)),
              "axis ticks, legends, diagram labels"),
        _tile("Headers / footers removed", _format(totals.get("header_footer_blocks_dropped", 0)),
              "page numbers, running titles"),
        _tile("Footnotes isolated", _format(totals.get("blocks_footnote", 0))),
        _tile("Two-column pages reordered", _format(totals.get("two_column_pages", 0))),
    ]  # fmt: skip
    valid = validation.get("is_valid")
    status = (
        '<span class="status good">&#10003; valid</span>' if valid else '<span class="status bad">&#10007; invalid</span>'
    )
    cleaning.append(
        f'<div class="card tile"><div class="label">Dataset validation</div><div class="value">{status}</div>'
        f'<div class="hint">{validation.get("records_checked", 0)} records, {validation.get("schema_errors", 0)} schema '
        f'errors, {validation.get("duplicate_ids", 0)} duplicate ids</div></div>'
    )
    sections.append(f'<div class="grid tiles">{"".join(cleaning)}</div>')

    environment = ", ".join(f"{name} {version}" for name, version in report.get("environment", {}).items())
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Pipeline performance report</title>"
        f"<style>{theme_css(':root')}{PAGE_CSS}{CHART_CSS}</style></head><body><main>"
        "<button class='toggle' id='theme' type='button'>Light / dark</button>"
        "<h1>PDF &rarr; RAG chunks: performance report</h1>"
        f"<p class='sub'>Run {_esc(run['run_id'])} &middot; {_esc(run['started_at'])} &middot; status "
        f"<b>{_esc(run['status'])}</b> &middot; strategy <b>{_esc(report['config']['chunking']['strategy'])}</b>, "
        f"table backend <b>{_esc(report['config']['extraction'].get('table_detection', ''))}</b></p>"
        + "".join(sections)
        + f"<p class='meta' style='margin-top:20px'>Every number on this page is read from the run report and "
        f"benchmark files of a measured run. Environment: {_esc(environment)}.</p>"
        f"<div id='tip'></div></main><script>{PAGE_JS}</script></body></html>"
    )


def _load_optional(path: Path | None) -> dict[str, Any] | None:
    """Read a JSON file if it exists; ``None`` otherwise."""
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_performance_dashboard(
    report: dict[str, Any],
    output: Path,
    benchmarks_dir: Path | None = None,
    assets_dir: Path | None = None,
) -> Path:
    """Write the dashboard, and one standalone SVG per chart if ``assets_dir`` is set.

    Benchmark files (``retrieval_results.json``, ``parser_benchmark.json``,
    ``chunking_parameter_sweep.json``) are read from ``benchmarks_dir`` when present.
    """
    retrieval = _load_optional(benchmarks_dir / "retrieval_results.json") if benchmarks_dir else None
    parsers = _load_optional(benchmarks_dir / "parser_benchmark.json") if benchmarks_dir else None
    sweep = _load_optional(benchmarks_dir / "chunking_parameter_sweep.json") if benchmarks_dir else None
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_dashboard_html(report, retrieval, parsers, sweep), encoding="utf-8")
    if assets_dir is not None:
        assets_dir.mkdir(parents=True, exist_ok=True)
        for name, chart in build_all_charts(report, retrieval, parsers).items():
            (assets_dir / f"{name}.svg").write_text(wrap_chart_as_standalone_svg(chart), encoding="utf-8")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Render a run report as an HTML dashboard.")
    parser.add_argument("--report", type=Path, default=Path("results.run_report.json"))
    parser.add_argument("--output", type=Path, default=None, help="Default: <dataset stem>.dashboard.html")
    parser.add_argument("--benchmarks", type=Path, default=Path("benchmarks"))
    parser.add_argument("--assets", type=Path, default=None, help="Also write one SVG per chart here.")
    args = parser.parse_args(argv)
    report = _load_optional(args.report)
    if report is None:
        print(f"Cannot read run report: {args.report}", file=sys.stderr)
        return 1
    output = args.output or args.report.with_name(args.report.name.replace(".run_report.json", ".dashboard.html"))
    print(f"Wrote {write_performance_dashboard(report, output, args.benchmarks, args.assets)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
