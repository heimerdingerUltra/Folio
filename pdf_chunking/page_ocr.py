"""OCR for pages without a text layer (scanned documents).

A page is rendered to an image, read by RapidOCR (ONNX models, CPU, no
system dependency), and its lines are grouped into the same
:class:`TextBlock` objects the text-layer path produces. Reading order,
header removal, heading detection and chunking then run unchanged.

What OCR cannot give: fonts. The font size is estimated from the line
height and bold is unknown, so heading detection on scanned pages relies on
size and numbering only.
"""

from __future__ import annotations

import importlib.util
import statistics
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import pymupdf

from pdf_chunking.text_normalization import normalize_pdf_text

OcrMode = Literal["auto", "off", "force"]

# 200 dpi: 10-point text is about 28 pixels high, enough for the recogniser.
OCR_RENDER_DPI = 200
_POINTS_PER_INCH = 72.0
# A line box spans ascender to descender plus padding; the nominal font size is smaller.
_FONT_SIZE_PER_LINE_HEIGHT = 0.8


@dataclass(frozen=True)
class OcrLine:
    """One recognised line.

    Attributes:
        text: Recognised text.
        bbox: ``(x0, y0, x1, y1)`` in page points, origin top-left.
        confidence: Recogniser score in [0, 1].
    """

    text: str
    bbox: tuple[float, float, float, float]
    confidence: float

    @property
    def height(self) -> float:
        """Height of the line box in points."""
        return self.bbox[3] - self.bbox[1]


def is_ocr_installed() -> bool:
    """Return True if the optional ``rapidocr`` package is installed."""
    try:
        return importlib.util.find_spec("rapidocr") is not None
    except (ImportError, ValueError):
        return False


@lru_cache(maxsize=1)
def _load_rapidocr_engine() -> Any:
    """Create the RapidOCR engine once per process (loading the models takes seconds)."""
    import logging

    from rapidocr import RapidOCR

    logging.getLogger("RapidOCR").setLevel(logging.WARNING)
    return RapidOCR()


def recognize_page_lines(page: pymupdf.Page, min_confidence: float = 0.5, dpi: int = OCR_RENDER_DPI) -> list[OcrLine]:
    """Render ``page`` and return its recognised lines, in page coordinates.

    Lines scored below ``min_confidence`` are dropped: they are mostly
    noise read from figures and stamps.
    """
    import numpy

    pixmap = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
    image = numpy.frombuffer(pixmap.samples, dtype=numpy.uint8).reshape(pixmap.height, pixmap.width, 3)
    result = _load_rapidocr_engine()(image)
    if result.boxes is None or result.txts is None:
        return []
    scale = _POINTS_PER_INCH / dpi
    lines = []
    for quad, text, score in zip(result.boxes, result.txts, result.scores):
        text = normalize_pdf_text(text)
        if not text or score < min_confidence:
            continue
        xs = [float(point[0]) * scale for point in quad]
        ys = [float(point[1]) * scale for point in quad]
        bbox = (min(xs), min(ys), max(xs), max(ys))
        # Taller than wide with several characters: rotated margin text, as on arXiv first pages.
        if len(text) > 2 and bbox[3] - bbox[1] > bbox[2] - bbox[0]:
            continue
        lines.append(OcrLine(text=text, bbox=bbox, confidence=float(score)))
    return lines


def group_lines_into_paragraphs(lines: list[OcrLine]) -> list[list[OcrLine]]:
    """Group recognised lines into paragraphs.

    A line continues the paragraph above it when the two overlap
    horizontally, have a similar height and are less than one line apart.
    Lines of two columns at the same height do not overlap horizontally, so
    columns stay separate.
    """
    paragraphs: list[list[OcrLine]] = []
    for line in sorted(lines, key=lambda line: (line.bbox[1], line.bbox[0])):
        for paragraph in reversed(paragraphs):
            last = paragraph[-1]
            overlap = min(line.bbox[2], last.bbox[2]) - max(line.bbox[0], last.bbox[0])
            narrower = min(line.bbox[2] - line.bbox[0], last.bbox[2] - last.bbox[0])
            gap = line.bbox[1] - last.bbox[3]
            similar_height = max(line.height, last.height) <= 1.35 * min(line.height, last.height)
            if overlap >= 0.5 * narrower and similar_height and -0.5 * last.height <= gap <= 0.7 * last.height:
                paragraph.append(line)
                break
        else:
            paragraphs.append([line])
    return paragraphs


def estimate_font_size(lines: list[OcrLine]) -> float:
    """Font size estimated from the median line height, rounded to half a point."""
    return round(statistics.median(line.height for line in lines) * _FONT_SIZE_PER_LINE_HEIGHT * 2) / 2
