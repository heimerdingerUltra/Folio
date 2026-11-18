"""Layout-aware PDF text extraction on top of PyMuPDF.

PyMuPDF returns positioned spans (box, font, size); this module turns them
into an ordered list of typed :class:`TextBlock` (title, heading, paragraph,
equation, caption, footnote, table):

1. spans -> lines -> blocks, dropping footnote markers;
2. tables: located by ``Page.find_tables``, rebuilt from word positions;
3. reading order for one- and two-column pages;
4. block classification from typography;
5. clean-up: running headers, text inside figures, hyphenation, sections.
"""

from __future__ import annotations

import importlib.util
import logging
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pymupdf

from pdf_chunking.text_normalization import count_document_words, join_wrapped_lines, normalize_pdf_text

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Public data structures
# --------------------------------------------------------------------------

#: TextBlock types produced by the extractor.
TEXT_BLOCK_TYPES = (
    "title",
    "heading",
    "paragraph",
    "equation",
    "caption",
    "footnote",
    "table",
)


class PdfExtractionError(Exception):
    """Raised when a PDF cannot be opened or parsed at all."""


class NoTextLayerError(PdfExtractionError):
    """Raised when a PDF has no extractable text (e.g. a scanned document)."""


TableDetectionMode = Literal["auto", "layout", "heuristic", "off"]


@dataclass
class PdfExtractionConfig:
    """PDF extraction options.

    Attributes:
        table_detection: ``layout`` (pymupdf-layout ONNX model: accurate, about
            1 s/page), ``heuristic`` (numeric-density rule: fast, lower recall),
            ``auto`` (layout if installed) or ``off``.
        skip_pages_without_table_signals: In ``layout`` mode, run the model only
            on pages with a "Table N" caption, ruling lines, or a table on the
            previous page.
        drop_text_inside_figures: Remove axis labels and diagram text.
        min_chars_per_page: Below this average, the PDF has no text layer.
    """

    table_detection: TableDetectionMode = "auto"
    skip_pages_without_table_signals: bool = True
    drop_text_inside_figures: bool = True
    min_chars_per_page: int = 25


def is_layout_model_installed() -> bool:
    """Return True if the optional ``pymupdf-layout`` package is installed."""
    try:
        return importlib.util.find_spec("pymupdf.layout") is not None
    except (ImportError, ValueError):
        return False


def resolve_table_detection_mode(mode: TableDetectionMode) -> Literal["layout", "heuristic", "off"]:
    """Resolve ``auto`` to a concrete backend and check availability.

    Raises:
        PdfExtractionError: If ``layout`` is requested but ``pymupdf-layout``
            is not installed.
    """
    if mode == "auto":
        return "layout" if is_layout_model_installed() else "heuristic"
    if mode == "layout" and not is_layout_model_installed():
        raise PdfExtractionError(
            "table_detection='layout' requires the 'pymupdf-layout' package "
            "(pip install pymupdf-layout), or use table_detection='heuristic'."
        )
    return mode


@dataclass
class TextBlock:
    """A typed unit of text with its position.

    Attributes:
        text: Cleaned text (set during document post-processing).
        block_type: One of :data:`TEXT_BLOCK_TYPES`.
        page_number: 1-based page index.
        bbox: ``(x0, y0, x1, y1)`` in points, origin top-left.
        lines: Visual lines, before de-hyphenation.
        font_size: Dominant font size (character-weighted).
        max_font_size: Largest font size in the block.
        bold_ratio: Share of characters in a bold font.
        math_ratio: Share of characters in a math font.
        starts_with_marker: Opens with a superscript marker (footnote signature).
        heading_level: Heading depth (1 = top level), 0 otherwise.
        section_path: Enclosing headings, outermost first.
    """

    text: str
    block_type: str
    page_number: int
    bbox: tuple[float, float, float, float]
    lines: list[str] = field(default_factory=list)
    font_size: float = 0.0
    max_font_size: float = 0.0
    bold_ratio: float = 0.0
    math_ratio: float = 0.0
    starts_with_marker: bool = False
    heading_level: int = 0
    section_path: tuple[str, ...] = ()

    @property
    def char_count(self) -> int:
        """Number of characters over all visual lines."""
        return sum(len(line) for line in self.lines)


@dataclass
class ParsedDocument:
    """Result of parsing one PDF.

    Attributes:
        source_document: File name of the PDF.
        title: Best-effort document title.
        page_count: Number of pages in the PDF.
        blocks: Typed blocks in reading order.
        stats: Counters describing what the extractor did (tables found,
            blocks dropped, pages by layout...). Surfaced in the run report.
        warnings: Human-readable, non-fatal problems.
    """

    source_document: str
    title: str
    page_count: int
    blocks: list[TextBlock]
    stats: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Constants and regular expressions
# --------------------------------------------------------------------------

# Extraction flags: expand ligatures ("fi" instead of the single glyph) and
# skip images. Whitespace is preserved so word gaps survive.
_TEXT_FLAGS = (
    pymupdf.TEXTFLAGS_DICT
    & ~pymupdf.TEXT_PRESERVE_LIGATURES
    & ~pymupdf.TEXT_PRESERVE_IMAGES
)

_SUPERSCRIPT_FLAG = 1  # bit 0 of a span's "flags"
_BOLD_FLAG = 16  # bit 4 of a span's "flags"

# Fonts used by LaTeX for mathematics. Text set in these fonts is treated
# as part of a formula.
_MATH_FONT_RE = re.compile(
    r"^(CM(MI|SY|EX|R|BX|TI|SS|MIB|BSY)\d|MS[AB]M|EU[FRS]M|rsfs|LASY|Symbol|"
    r"STIX.*Math|.*Math)",
    re.IGNORECASE,
)
_BOLD_FONT_RE = re.compile(r"(Bold|Medi|Black|Heavy|Demi|CMBX)", re.IGNORECASE)
# Footnote / affiliation markers: digits, asterisks, daggers, section signs.
_FOOTNOTE_MARKER_RE = re.compile(r"^[\d\*∗†‡§¶,]+$")
_EXPONENT_RE = re.compile(r"^[−\-+]?\d+$")

_NUMBERED_HEADING_RE = re.compile(
    r"^(?P<num>(?:\d{1,2}(?:\.\d{1,2}){0,3}|[A-Z](?:\.\d{1,2}){0,3}))\.?\s+(?P<title>\S.{1,140})$"
)
_CAPTION_RE = re.compile(r"^(Table|Figure|Fig\.|Algorithm|Listing)\s+\d+\s*[:.]", re.IGNORECASE)
_TABLE_CAPTION_RE = re.compile(r"^Table\s+\d+\s*[:.]", re.IGNORECASE)
_PAGE_NUMBER_RE = re.compile(r"^(page\s+)?\d{1,4}(\s*(of|/)\s*\d{1,4})?$", re.IGNORECASE)
_SENTENCE_END_RE = re.compile(r"[.!?:][\"'”\)\]]?$")
_NUMERIC_TOKEN_RE = re.compile(r"^[\d.,%±+\-−/x×]+$")

_KNOWN_HEADINGS = frozenset(
    {
        "abstract", "introduction", "related work", "related works", "background",
        "conclusion", "conclusions", "references", "bibliography", "acknowledgements",
        "acknowledgments", "acknowledgement", "acknowledgment", "appendix", "limitations",
        "discussion", "methods", "method", "experiments", "results", "broader impact",
        "ethics statement", "future work", "supplementary material",
    }
)  # fmt: skip
_REFERENCE_HEADINGS = frozenset({"references", "bibliography"})

_MARGIN_FRACTION = 0.08  # top/bottom band where headers and footers live


# --------------------------------------------------------------------------
# Step 1 - spans -> lines -> block
# --------------------------------------------------------------------------


def _is_math_font(font: str) -> bool:
    """Return True if ``font`` is a (LaTeX) mathematics font."""
    return bool(_MATH_FONT_RE.match(font))


def _join_spans_into_line_text(spans: list[dict[str, Any]]) -> str:
    """Rebuild one visual line from its spans.

    * insert a space wherever there is a visible gap between spans;
    * drop superscript digits set in a text font (footnote markers: "PyMuPDF5");
    * keep numeric superscripts set in a math font as ``^n`` ("10^20").
    """
    parts: list[str] = []
    previous_x1: float | None = None
    previous_size = 0.0
    for index, span in enumerate(spans):
        text = span["text"]
        if not text:
            continue
        stripped = text.strip()
        is_super = bool(span["flags"] & _SUPERSCRIPT_FLAG)
        if (
            index == 0
            and len(spans) > 1
            and _FOOTNOTE_MARKER_RE.match(stripped)
            and (is_super or span["size"] <= 0.8 * spans[1]["size"])
        ):
            # A marker opening the line introduces the footnote itself:
            # keep it, detached from the first word ("1 https://...").
            text = stripped + " "
        elif is_super and stripped:
            if _is_math_font(span["font"]):
                if (
                    parts
                    and _EXPONENT_RE.match(stripped)
                    and span["size"] <= 0.85 * previous_size
                    and not parts[-1].endswith(" ")
                ):
                    text = "^" + stripped
            elif _FOOTNOTE_MARKER_RE.match(stripped) and index > 0:
                # Mid-line footnote / affiliation marker: drop it.
                previous_x1 = span["bbox"][2]
                continue
        gap = span["bbox"][0] - previous_x1 if previous_x1 is not None else 0.0
        needs_space = (
            parts
            and gap > 0.2 * span["size"]
            and not parts[-1].endswith(" ")
            and not text.startswith(" ")
        )
        parts.append((" " if needs_space else "") + text)
        previous_x1 = span["bbox"][2]
        if stripped:
            previous_size = span["size"]
    return "".join(parts)


def _build_block_from_pymupdf_dict(raw: dict[str, Any], page_number: int) -> TextBlock | None:
    """Convert a PyMuPDF ``dict`` text block into a :class:`TextBlock`.

    Returns ``None`` for blocks with no usable text and for rotated text
    (e.g. the vertical arXiv identifier printed in the left margin).
    """
    lines = raw.get("lines") or []
    if not lines:
        return None
    # Text direction (cos, sin): anything not left-to-right horizontal is a
    # margin watermark or a rotated figure label, never body text.
    direction = lines[0].get("dir", (1.0, 0.0))
    if abs(direction[0] - 1.0) > 0.01 or abs(direction[1]) > 0.01:
        return None

    # PyMuPDF sometimes splits one visual line into several "lines" (wide
    # gaps, sub/superscripts). Group them back by vertical position.
    # Two pieces share a visual line when their baselines (nearly) coincide.
    # Bounding boxes cannot be used for this: a line containing a tall glyph
    # such as a square-root sign overlaps the line below it. Sub/superscripts
    # shift the baseline by less than half the font size; consecutive lines
    # are a full line height apart.
    def baseline_of(line: dict[str, Any]) -> tuple[float, float]:
        """Baseline ordinate and font size of the largest span of a line."""
        largest = max(line["spans"], key=lambda span: span["size"])
        return largest["origin"][1], largest["size"]

    pieces = []
    for line in lines:
        spans = [span for span in line["spans"] if span["text"]]
        if spans:
            baseline, size = baseline_of(line)
            pieces.append((baseline, size, line["bbox"][0], spans))
    pieces.sort(key=lambda piece: (piece[0], piece[2]))

    # Each row: ([reference baseline, reference size, reference chars], spans)
    rows: list[tuple[list[float], list[dict[str, Any]]]] = []
    for baseline, size, _, spans in pieces:
        n_chars = sum(len(span["text"]) for span in spans)
        if rows:
            ref_baseline, ref_size, ref_chars = rows[-1][0]
            if abs(baseline - ref_baseline) < 0.45 * min(size, ref_size):
                rows[-1][1].extend(spans)
                # The piece with the most text defines the line's baseline.
                if n_chars > ref_chars:
                    rows[-1][0][:] = [baseline, size, n_chars]
                continue
        rows.append(([baseline, size, float(n_chars)], list(spans)))

    text_lines: list[str] = []
    size_weights: Counter[float] = Counter()
    total = bold = math = 0
    max_size = 0.0
    for _, spans in rows:
        spans.sort(key=lambda span: span["bbox"][0])
        text = normalize_pdf_text(_join_spans_into_line_text(spans))
        if text:
            text_lines.append(text)
        for span in spans:
            n_chars = len(span["text"].strip())
            if not n_chars:
                continue
            total += n_chars
            # Round to the nearest half point: LaTeX micro-typography
            # produces 9.9 / 10.0 / 10.1 for the same nominal size.
            size_weights[round(span["size"] * 2) / 2] += n_chars
            max_size = max(max_size, span["size"])
            if span["flags"] & _BOLD_FLAG or _BOLD_FONT_RE.search(span["font"]):
                bold += n_chars
            if _is_math_font(span["font"]):
                math += n_chars
    if not text_lines or not total:
        return None

    # A footnote opens with a marker that is superscript or clearly smaller
    # than the text that follows it.
    first_span = rows[0][1][0]
    starts_with_marker = bool(_FOOTNOTE_MARKER_RE.match(first_span["text"].strip() or "x")) and (
        bool(first_span["flags"] & _SUPERSCRIPT_FLAG) or first_span["size"] <= 0.8 * max_size
    )
    return TextBlock(
        text="",
        block_type="paragraph",
        page_number=page_number,
        bbox=tuple(raw["bbox"]),  # type: ignore[arg-type]
        lines=text_lines,
        font_size=size_weights.most_common(1)[0][0],
        max_font_size=max_size,
        bold_ratio=bold / total,
        math_ratio=math / total,
        starts_with_marker=starts_with_marker,
    )


# --------------------------------------------------------------------------
# Step 2 - tables
# --------------------------------------------------------------------------


def _cluster_words_into_rows(words: list[tuple[Any, ...]], tolerance: float) -> list[list[tuple[Any, ...]]]:
    """Group positioned words into visual rows by vertical centre."""
    rows: list[list[tuple[Any, ...]]] = []
    row_center = 0.0
    for word in sorted(words, key=lambda item: ((item[1] + item[3]) / 2, item[0])):
        y_center = (word[1] + word[3]) / 2
        if rows and abs(y_center - row_center) <= tolerance:
            rows[-1].append(word)
            # Running mean keeps the row stable when sub/superscripts shift.
            row_center += (y_center - row_center) / len(rows[-1])
        else:
            rows.append([word])
            row_center = y_center
    for row in rows:
        row.sort(key=lambda item: item[0])
    return rows


def _find_column_gutters(
    rows: list[list[tuple[Any, ...]]], x_min: float, x_max: float, min_gutter: float
) -> list[float]:
    """Find column separators as vertical whitespace gutters.

    This is a projection profile: for every x position we count the rows in
    which some word covers it. A run of positions covered by (almost) no row
    and at least ``min_gutter`` wide is a gutter between two columns. A few
    rows may cross a gutter, which tolerates headers spanning several
    columns.

    Returns:
        The x coordinate of the centre of each gutter, left to right.
    """
    width = int(x_max - x_min) + 1
    coverage = [0] * (width + 1)
    for row in rows:
        covered: set[int] = set()
        for word in row:
            covered.update(range(int(word[0] - x_min), int(word[2] - x_min) + 1))
        for position in covered:
            if 0 <= position <= width:
                coverage[position] += 1
    allowed_crossings = len(rows) // 7  # ~15 % of rows may span a gutter

    boundaries: list[float] = []
    run_start: int | None = None
    for position in range(width + 1):
        is_free = coverage[position] <= allowed_crossings
        if is_free and run_start is None:
            run_start = position
        elif not is_free and run_start is not None:
            # Ignore the margins: a gutter needs text on both sides.
            if run_start > 0 and position - run_start >= min_gutter:
                boundaries.append(x_min + (run_start + position) / 2)
            run_start = None
    return boundaries


def _rebuild_table_rows_from_words(words: list[tuple[Any, ...]]) -> list[str]:
    """Rebuild table rows from word positions.

    ``Table.extract()`` loses spaces and decimal points on these PDFs, so only
    the table's bounding box is trusted: words are clustered into rows,
    columns are inferred from whitespace gutters, and empty cells are kept.

    Returns:
        One string per row, cells separated by `` | ``.
    """
    if not words:
        return []
    median_height = statistics.median(word[3] - word[1] for word in words)
    rows = _cluster_words_into_rows(words, tolerance=0.5 * median_height)
    x_min = min(word[0] for word in words)
    x_max = max(word[2] for word in words)
    boundaries = _find_column_gutters(rows, x_min, x_max, min_gutter=0.35 * median_height)

    lines: list[str] = []
    for row in rows:
        if boundaries:
            cells: list[list[str]] = [[] for _ in range(len(boundaries) + 1)]
            for word in row:
                center = (word[0] + word[2]) / 2
                column = sum(1 for boundary in boundaries if center > boundary)
                cells[column].append(word[4])
            text = " | ".join(" ".join(cell) for cell in cells)
        else:
            # No stable gutter (e.g. a framed text box): fall back to marking
            # unusually wide gaps between consecutive words.
            pieces = [row[0][4]]
            for previous, current in zip(row, row[1:]):
                wide = current[0] - previous[2] > 0.75 * median_height
                pieces.append((" | " if wide else " ") + current[4])
            text = "".join(pieces)
        text = normalize_pdf_text(text)
        if text.replace("|", "").strip():
            lines.append(text)
    return lines


# A ruled table draws at least three lines (top, header and bottom rules).
_MIN_RULES_FOR_TABLE = 3


def _page_may_contain_table(page: pymupdf.Page, blocks: list[TextBlock], previous_page_had_table: bool) -> bool:
    """Cheap test: is this page worth sending to the layout model?

    True for a "Table N" caption, at least three vector drawings (table
    rules), or a table on the previous page (continuation).
    """
    if previous_page_had_table:
        return True
    if any(_TABLE_CAPTION_RE.match(block.lines[0]) for block in blocks if block.lines):
        return True
    return len(page.get_cdrawings()) >= _MIN_RULES_FOR_TABLE


def _detect_table_blocks(page: pymupdf.Page, page_number: int) -> list[TextBlock]:
    """Detect tables on ``page`` and return them as ``table`` blocks.

    Any failure of the table finder is logged and treated as "no tables":
    the text is then still extracted as ordinary blocks, so nothing is lost.
    """
    try:
        tables = page.find_tables().tables
    except Exception as exc:  # noqa: BLE001 - third-party heuristic code
        logger.warning("Table detection failed on page %d: %s", page_number, exc)
        return []

    blocks: list[TextBlock] = []
    for table in tables:
        rect = pymupdf.Rect(table.bbox) & page.rect
        if rect.is_empty or rect.width < 20 or rect.height < 8:
            continue
        rows = _rebuild_table_rows_from_words(page.get_text("words", clip=rect))
        # Reject false positives and leave their text to the normal flow:
        # fewer than two rows (a framed box, a rule under a heading), or
        # mostly single-cell rows (a column of figure labels).
        multi_cell_rows = sum(1 for row in rows if " | " in row)
        if len(rows) < 2 or multi_cell_rows < len(rows) / 4:
            continue
        blocks.append(
            TextBlock(
                text="",
                block_type="table",
                page_number=page_number,
                bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                lines=rows,
            )
        )
    return blocks


def _is_mostly_inside(block: TextBlock, region: TextBlock, min_overlap: float = 0.6) -> bool:
    """Return True if most of ``block``'s area lies inside ``region``."""
    x0, y0 = max(block.bbox[0], region.bbox[0]), max(block.bbox[1], region.bbox[1])
    x1, y1 = min(block.bbox[2], region.bbox[2]), min(block.bbox[3], region.bbox[3])
    if x1 <= x0 or y1 <= y0:
        return False
    area = (block.bbox[2] - block.bbox[0]) * (block.bbox[3] - block.bbox[1])
    return area > 0 and (x1 - x0) * (y1 - y0) / area >= min_overlap


def _looks_like_numeric_table(block: TextBlock) -> bool:
    """Fallback for tables the finder missed: dense in numbers, several rows."""
    if len(block.lines) < 3:
        return False
    tokens = " ".join(block.lines).split()
    # At least three values per row on average: a single column of numbers
    # (or of "Doc 1", "Doc 2" labels) is a chart axis, not a table.
    if len(tokens) < 9 or len(tokens) < 3 * len(block.lines):
        return False
    numeric = sum(1 for token in tokens if _NUMERIC_TOKEN_RE.match(token))
    return numeric / len(tokens) >= 0.4


# --------------------------------------------------------------------------
# Step 3 - reading order
# --------------------------------------------------------------------------


def is_two_column_page(blocks: list[TextBlock], page_width: float) -> bool:
    """Decide whether a page is laid out in two text columns.

    A page is two-column when both halves hold a substantial amount of text
    in blocks about one column wide. Tables are ignored: two small tables
    side by side do not make a page two-column.
    """
    middle = page_width / 2
    tolerance = 0.03 * page_width
    left = right = 0
    for block in blocks:
        if block.block_type == "table":
            continue
        width = block.bbox[2] - block.bbox[0]
        if not 0.25 * page_width <= width <= 0.55 * page_width:
            continue
        if block.bbox[2] <= middle + tolerance:
            left += block.char_count
        elif block.bbox[0] >= middle - tolerance:
            right += block.char_count
    return left >= 300 and right >= 300


def sort_blocks_in_reading_order(blocks: list[TextBlock], page_width: float) -> list[TextBlock]:
    """Sort the blocks of one page into reading order.

    Single-column pages: top to bottom. Two-column pages: cut into bands by
    the blocks spanning both columns, then left column before right column
    inside each band.
    """
    by_position = sorted(blocks, key=lambda block: (round(block.bbox[1]), block.bbox[0]))
    if not is_two_column_page(blocks, page_width):
        return by_position

    middle = page_width / 2
    tolerance = 0.03 * page_width
    ordered: list[TextBlock] = []
    left: list[TextBlock] = []
    right: list[TextBlock] = []

    def flush_band() -> None:
        """Emit the pending left column, then the right one."""
        ordered.extend(left)
        ordered.extend(right)
        left.clear()
        right.clear()

    for block in by_position:
        spans_both = block.bbox[0] < middle - tolerance and block.bbox[2] > middle + tolerance
        if spans_both:
            flush_band()
            ordered.append(block)
        elif (block.bbox[0] + block.bbox[2]) / 2 < middle:
            left.append(block)
        else:
            right.append(block)
    flush_band()
    return ordered


# --------------------------------------------------------------------------
# Step 4 - classification
# --------------------------------------------------------------------------


def _heading_comparison_key(text: str) -> str:
    """Canonical form used to compare a block with a table-of-contents entry."""
    match = _NUMBERED_HEADING_RE.match(text)
    if match:
        text = match.group("title")
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _infer_heading_level(text: str, toc_level: int | None) -> int:
    """Infer heading depth from its numbering ("3.2.1" -> 3) or the TOC."""
    match = _NUMBERED_HEADING_RE.match(text)
    if match:
        return match.group("num").count(".") + 1
    return toc_level or 1


def _classify_block_type(
    block: TextBlock,
    body_size: float,
    page_height: float,
    toc: dict[str, int],
) -> None:
    """Assign ``block.block_type`` (and ``heading_level``) in place.

    The rules are ordered from most to least specific. A block that matches
    none of them stays a ``paragraph``.
    """
    if block.block_type == "table":
        return
    text = " ".join(block.lines)

    if _CAPTION_RE.match(text):
        block.block_type = "caption"
        return

    # Footnotes open with a superscript marker and are set smaller than the
    # body text, in the lower part of the page.
    if (
        block.starts_with_marker
        and block.font_size < body_size
        and block.bbox[1] > 0.4 * page_height
    ):
        block.block_type = "footnote"
        return

    is_short = len(text) <= 150 and len(block.lines) <= 3
    if is_short:
        emphasized = block.bold_ratio >= 0.9 or block.max_font_size >= 1.15 * body_size
        numbered = _NUMBERED_HEADING_RE.match(text)
        normalized = _heading_comparison_key(text)
        toc_level = toc.get(normalized)
        letters = [char for char in text if char.isalpha()]
        all_caps = len(letters) >= 4 and all(char.isupper() for char in letters)
        known = text.lower().strip(" .:") in _KNOWN_HEADINGS
        if (
            (toc_level is not None and (emphasized or numbered or all_caps))
            or (emphasized and (numbered or known))
            # Small-caps headings (ICLR style) are neither bold nor larger.
            or (numbered and all_caps and len(block.lines) == 1)
        ):
            block.block_type = "heading"
            block.heading_level = _infer_heading_level(text, toc_level)
            return

    # Displayed equations: mostly math-font glyphs, and short.
    if block.math_ratio >= 0.5 and len(text) <= 400:
        block.block_type = "equation"


# --------------------------------------------------------------------------
# Step 5 - document-level clean-up
# --------------------------------------------------------------------------


def _drop_running_headers_and_page_numbers(
    blocks: list[TextBlock], page_heights: dict[int, float], stats: dict[str, Any]
) -> list[TextBlock]:
    """Remove page numbers and headers/footers repeated across pages.

    A margin block is "running" when its text, with digits masked, appears on
    at least half of the pages (minimum three), e.g. "Published as a
    conference paper at ICLR 2022" or "Page 3 of 12".
    """

    def in_margin(block: TextBlock) -> bool:
        """Is the block in the top or bottom margin of its page?"""
        height = page_heights[block.page_number]
        return block.bbox[3] < _MARGIN_FRACTION * height or block.bbox[1] > (1 - _MARGIN_FRACTION) * height

    def signature(block: TextBlock) -> str:
        """Block text with digits masked, to match repeated headers and footers."""
        return re.sub(r"\d+", "#", " ".join(block.lines).lower())

    n_pages = len(page_heights)
    pages_by_signature: dict[str, set[int]] = defaultdict(set)
    for block in blocks:
        if in_margin(block) and len(block.lines) <= 2:
            pages_by_signature[signature(block)].add(block.page_number)
    repeated = {
        sig for sig, pages in pages_by_signature.items() if len(pages) >= max(3, n_pages // 2)
    }

    kept: list[TextBlock] = []
    for block in blocks:
        if in_margin(block) and len(block.lines) <= 2:
            text = " ".join(block.lines)
            if _PAGE_NUMBER_RE.match(text) or signature(block) in repeated:
                stats["header_footer_blocks_dropped"] += 1
                continue
        kept.append(block)
    return kept


def _drop_text_inside_figures(blocks: list[TextBlock], stats: dict[str, Any]) -> list[TextBlock]:
    """Remove axis ticks, legends and diagram labels.

    For each "Figure N" caption, the figure region is the band above it up
    to the nearest heading, caption, table or real paragraph. Short blocks in
    that band are dropped, unless they read as a one-line paragraph (starts
    at the text margin and is a complete sentence).
    """
    paragraphs = [b for b in blocks if b.block_type == "paragraph" and len(b.lines) >= 3]
    if not paragraphs:
        return blocks
    column_width = statistics.median(b.bbox[2] - b.bbox[0] for b in paragraphs)

    def is_wide(block: TextBlock) -> bool:
        """Does the block span more than a column?"""
        return block.bbox[2] - block.bbox[0] >= 0.6 * column_width

    drop: set[int] = set()
    by_page: dict[int, list[int]] = defaultdict(list)
    for index, block in enumerate(blocks):
        by_page[block.page_number].append(index)

    for indices in by_page.values():
        # Left edges of the page's body paragraphs = its text margins.
        margins = {round(blocks[i].bbox[0]) for i in indices if is_wide(blocks[i])}
        for caption_index in indices:
            caption = blocks[caption_index]
            if caption.block_type != "caption" or _TABLE_CAPTION_RE.match(" ".join(caption.lines)):
                continue

            def above_caption(block: TextBlock) -> bool:
                """Is the block directly above the caption, horizontally overlapping it?"""
                overlaps = block.bbox[0] < caption.bbox[2] and block.bbox[2] > caption.bbox[0]
                return overlaps and block.bbox[3] <= caption.bbox[1] + 2

            candidates: list[int] = []
            region_top = 0.0
            for i in indices:
                block = blocks[i]
                if i == caption_index or not above_caption(block):
                    continue
                text = " ".join(block.lines)
                at_margin = any(abs(block.bbox[0] - margin) <= 3 for margin in margins)
                is_sentence = (
                    at_margin
                    and len(text.split()) >= 4
                    and text[0].isupper()
                    and bool(_SENTENCE_END_RE.search(text))
                )
                is_fragment = (
                    block.block_type in ("paragraph", "equation")
                    and len(text.split()) < 15
                    and not is_sentence
                )
                if is_fragment:
                    candidates.append(i)
                else:
                    region_top = max(region_top, block.bbox[3])  # an anchor
            drop.update(i for i in candidates if blocks[i].bbox[1] >= region_top - 2)

    stats["figure_text_blocks_dropped"] += len(drop)
    return [block for index, block in enumerate(blocks) if index not in drop]


def _merge_footnote_continuations(blocks: list[TextBlock]) -> list[TextBlock]:
    """Attach stray pieces of a footnote to the footnote they belong to.

    Footnotes sit at the bottom of a column, so any body-type block that
    lies *below* the start of a footnote, in the same column, is the rest of
    that footnote (typically a line broken off by inline math). Without this
    step such a fragment would be stitched into the running text.
    """
    result: list[TextBlock] = []
    open_footnote: TextBlock | None = None
    for block in blocks:
        if open_footnote is not None and block.page_number != open_footnote.page_number:
            open_footnote = None
        if block.block_type == "footnote":
            open_footnote = block
            result.append(block)
            continue
        if (
            open_footnote is not None
            and block.block_type in ("paragraph", "equation")
            and block.bbox[1] >= open_footnote.bbox[1] - 2
            # Same column: horizontal ranges overlap.
            and block.bbox[0] < open_footnote.bbox[2]
            and block.bbox[2] > open_footnote.bbox[0]
        ):
            open_footnote.lines.extend(block.lines)
            continue
        result.append(block)
    return result


def _merge_equation_fragments(blocks: list[TextBlock]) -> list[TextBlock]:
    """Glue the pieces of a displayed equation back together.

    A fraction, a root sign or an equation number is often emitted as its
    own block ("Attention(Q,K,V) = softmax(QKT", then "sqrt(dk)", then
    ")V (1)"). Consecutive fragments on the same vertical band are merged
    into the preceding equation so the formula stays one indivisible unit.
    """
    merged: list[TextBlock] = []
    for block in blocks:
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and previous.block_type == "equation"
            and previous.page_number == block.page_number
            and (
                block.block_type == "equation"
                or (block.block_type == "paragraph" and block.char_count <= 25)
            )
            # Same band: vertical ranges touch (3 pt of slack).
            and block.bbox[1] <= previous.bbox[3] + 3
            and block.bbox[3] >= previous.bbox[1] - 3
        ):
            previous.lines.extend(block.lines)
            previous.bbox = (
                min(previous.bbox[0], block.bbox[0]),
                min(previous.bbox[1], block.bbox[1]),
                max(previous.bbox[2], block.bbox[2]),
                max(previous.bbox[3], block.bbox[3]),
            )
            continue
        merged.append(block)
    return merged


def _resolve_block_text(blocks: list[TextBlock]) -> None:
    """Fill ``TextBlock.text`` from the visual lines, resolving hyphenation.

    Tables keep one row per line. Everything else is flowed into a single
    line, using a document-wide vocabulary to decide which line-end hyphens
    are real (see :func:`pdf_chunking.text_normalization.join_wrapped_lines`).
    """
    vocabulary = count_document_words(
        line for block in blocks if block.block_type != "table" for line in block.lines
    )
    for block in blocks:
        if block.block_type == "table":
            block.text = "\n".join(block.lines)
        elif block.block_type == "equation":
            block.text = " ".join(block.lines)
        else:
            block.text = normalize_pdf_text(join_wrapped_lines(block.lines, vocabulary))


def assign_section_paths(blocks: list[TextBlock]) -> None:
    """Give every block the path of headings that encloses it."""
    stack: list[tuple[int, str]] = []
    for block in blocks:
        if block.block_type == "heading":
            while stack and stack[-1][0] >= block.heading_level:
                stack.pop()
            stack.append((block.heading_level, block.text))
        block.section_path = tuple(title for _, title in stack)


def is_bibliography_section(section_path: tuple[str, ...]) -> bool:
    """Return True if ``section_path`` points inside the bibliography."""
    for title in section_path:
        match = _NUMBERED_HEADING_RE.match(title)
        name = (match.group("title") if match else title).lower().strip(" .:")
        if name in _REFERENCE_HEADINGS:
            return True
    return False


def _detect_document_title(blocks: list[TextBlock], body_size: float, metadata_title: str, fallback: str) -> str:
    """Pick the document title: largest text on page 1, else PDF metadata."""
    first_page = [block for block in blocks if block.page_number == 1 and block.block_type != "table"]
    if first_page:
        largest = max(block.max_font_size for block in first_page)
        if largest >= 1.2 * body_size:
            title_blocks = [block for block in first_page if block.max_font_size >= largest - 0.3]
            for block in title_blocks:
                block.block_type = "title"
            title = normalize_pdf_text(join_wrapped_lines([line for block in title_blocks for line in block.lines]))
            if 5 <= len(title) <= 300:
                return title
    return metadata_title.strip() or fallback


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def _open_pdf_safely(path: Path) -> pymupdf.Document:
    """Open ``path`` defensively and reject unusable files early."""
    if not path.is_file():
        raise PdfExtractionError(f"File not found: {path}")
    if path.stat().st_size == 0:
        raise PdfExtractionError(f"File is empty (0 bytes): {path.name}")
    try:
        document = pymupdf.open(path)
    except Exception as exc:  # noqa: BLE001 - pymupdf raises several types
        raise PdfExtractionError(f"Cannot open {path.name}: {exc}") from exc
    if not document.is_pdf:
        document.close()
        raise PdfExtractionError(f"Not a PDF file: {path.name}")
    if document.needs_pass:
        document.close()
        raise PdfExtractionError(f"Password-protected PDF: {path.name}")
    if document.page_count == 0:
        document.close()
        raise PdfExtractionError(f"PDF has no pages: {path.name}")
    return document


def extract_pdf_document(path: Path | str, config: PdfExtractionConfig | None = None) -> ParsedDocument:
    """Parse one PDF into typed text blocks in reading order.

    Raises:
        NoTextLayerError: Almost no extractable text (scanned PDF, needs OCR).
        PdfExtractionError: Missing, empty, corrupted, encrypted or non-PDF file.
    """
    path = Path(path)
    config = config or PdfExtractionConfig()
    table_detection = resolve_table_detection_mode(config.table_detection)
    document = _open_pdf_safely(path)
    stats: dict[str, Any] = Counter()
    warnings: list[str] = []
    blocks: list[TextBlock] = []
    page_heights: dict[int, float] = {}
    page_widths: dict[int, float] = {}

    try:
        try:
            # get_toc() -> [[level, title, page], ...]; empty if no outline.
            toc = {_heading_comparison_key(title): level for level, title, _ in document.get_toc()}
        except Exception:  # noqa: BLE001 - a broken outline must not be fatal
            toc = {}

        previous_page_had_table = False
        for page_index in range(document.page_count):
            page_number = page_index + 1
            try:
                page = document[page_index]
                raw_blocks = page.get_text("dict", flags=_TEXT_FLAGS)["blocks"]
                page_blocks = [
                    block
                    for raw in raw_blocks
                    if raw.get("type", 0) == 0
                    and (block := _build_block_from_pymupdf_dict(raw, page_number)) is not None
                ]
                if not page_blocks:
                    stats["pages_without_text"] += 1
                    if page.get_images():
                        warnings.append(
                            f"Page {page_number} has images but no text layer (scanned page? OCR needed)."
                        )
                    continue
                run_finder = table_detection == "layout" and (
                    not config.skip_pages_without_table_signals
                    or _page_may_contain_table(page, page_blocks, previous_page_had_table)
                )
                previous_page_had_table = False
                if table_detection == "layout":
                    stats["table_finder_pages" if run_finder else "table_finder_pages_skipped"] += 1
                if run_finder:
                    tables = _detect_table_blocks(page, page_number)
                    previous_page_had_table = bool(tables)
                    if tables:
                        stats["tables_detected"] += len(tables)
                        # Text inside a table region is replaced by the table.
                        page_blocks = [
                            block
                            for block in page_blocks
                            if not any(_is_mostly_inside(block, table) for table in tables)
                        ] + tables
            except Exception as exc:  # noqa: BLE001 - keep going on a bad page
                stats["pages_failed"] += 1
                warnings.append(f"Page {page_number} could not be parsed: {exc}")
                logger.warning("%s: page %d failed: %s", path.name, page_number, exc)
                continue

            page_heights[page_number] = page.rect.height
            page_widths[page_number] = page.rect.width
            if is_two_column_page(page_blocks, page.rect.width):
                stats["two_column_pages"] += 1
            blocks.extend(sort_blocks_in_reading_order(page_blocks, page.rect.width))

        metadata_title = (document.metadata or {}).get("title") or ""
        page_count = document.page_count
    finally:
        document.close()

    total_chars = sum(block.char_count for block in blocks)
    if total_chars < config.min_chars_per_page * page_count:
        raise NoTextLayerError(
            f"{path.name}: only {total_chars} characters over {page_count} page(s); "
            "the PDF appears to be scanned or image-only and needs OCR."
        )

    # Body font size = the size carrying the most characters in the document.
    size_weights: Counter[float] = Counter()
    for block in blocks:
        if block.block_type != "table":
            size_weights[block.font_size] += block.char_count
    body_size = size_weights.most_common(1)[0][0] if size_weights else 10.0

    blocks = _drop_running_headers_and_page_numbers(blocks, page_heights, stats)
    for block in blocks:
        _classify_block_type(block, body_size, page_heights[block.page_number], toc)
    title = _detect_document_title(blocks, body_size, metadata_title, fallback=path.stem.replace("_", " "))
    if config.drop_text_inside_figures:
        blocks = _drop_text_inside_figures(blocks, stats)
    if table_detection != "off":
        # Numeric-density fallback for tables the finder missed (or for
        # 'heuristic' mode). Runs after figure clean-up so that chart axes
        # are not mistaken for tables.
        for block in blocks:
            if block.block_type == "paragraph" and _looks_like_numeric_table(block):
                block.block_type = "table"
    blocks = _merge_footnote_continuations(blocks)
    blocks = _merge_equation_fragments(blocks)
    _resolve_block_text(blocks)
    blocks = [block for block in blocks if block.text]
    assign_section_paths(blocks)

    for block_type, count in Counter(block.block_type for block in blocks).items():
        stats[f"blocks_{block_type}"] = count
    stats["table_backend"] = table_detection
    stats["body_font_size"] = body_size
    stats["characters"] = sum(len(block.text) for block in blocks)
    stats.setdefault("tables_detected", 0)
    stats.setdefault("two_column_pages", 0)
    stats.setdefault("pages_without_text", 0)
    stats.setdefault("pages_failed", 0)

    return ParsedDocument(
        source_document=path.name,
        title=title,
        page_count=page_count,
        blocks=blocks,
        stats=dict(stats),
        warnings=warnings,
    )
