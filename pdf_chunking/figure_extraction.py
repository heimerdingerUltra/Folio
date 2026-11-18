"""Locate figures, link them to their caption and export them as images.

A PDF has no "figure" object. A figure is either an embedded raster image
or a group of vector drawings, and what identifies it is the "Figure N"
caption printed below it. The steps:

1. per page, collect the graphic regions (images and clusters of drawings);
2. for each figure caption, take the regions stacked directly above it;
3. render that area of the page to a PNG file.

Figures whose caption is printed above them are not matched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

BoundingBox = tuple[float, float, float, float]

_FIGURE_LABEL_RE = re.compile(r"^(Figure|Fig\.)\s+(\d+)", re.IGNORECASE)
_MIN_REGION_SIDE = 30.0  # points; smaller graphics are bullets, rules and icons
_MAX_PAGE_SHARE = 0.9  # a region covering the page is a scan or a background
_MAX_GAP_TO_CAPTION = 45.0  # points between a figure and its caption
_MAX_GAP_BETWEEN_PARTS = 25.0  # points between sub-figures of one figure
# Axis labels and sub-figure titles are text, printed just outside the
# graphic: the area is widened to keep them (left, top, right, in points).
_LABEL_MARGINS = (24.0, 14.0, 8.0)


@dataclass
class Figure:
    """A figure of the document.

    Attributes:
        label: ``Figure N`` as printed in the caption.
        page_number: 1-based page index.
        bbox: Area of the graphic, in points, origin top-left.
        caption: Caption text.
        image_path: PNG file written by :func:`export_figure_images`, if any.
    """

    label: str
    page_number: int
    bbox: BoundingBox
    caption: str
    image_path: str | None = None


def find_graphic_regions(page: pymupdf.Page) -> list[BoundingBox]:
    """Bounding boxes of the raster images and drawing clusters of a page."""
    rectangles = [pymupdf.Rect(info["bbox"]) for info in page.get_image_info()]
    rectangles += list(page.cluster_drawings(x_tolerance=6, y_tolerance=6))
    page_area = page.rect.width * page.rect.height
    regions = []
    for rectangle in rectangles:
        rectangle = rectangle & page.rect
        if rectangle.is_empty or min(rectangle.width, rectangle.height) < _MIN_REGION_SIDE:
            continue
        if rectangle.width * rectangle.height > _MAX_PAGE_SHARE * page_area:
            continue
        regions.append((rectangle.x0, rectangle.y0, rectangle.x1, rectangle.y1))
    return regions


def _union(first: BoundingBox, second: BoundingBox) -> BoundingBox:
    """Smallest box containing both."""
    return (min(first[0], second[0]), min(first[1], second[1]), max(first[2], second[2]), max(first[3], second[3]))


def match_figures_to_captions(
    captions: list[tuple[int, BoundingBox, str]], regions_by_page: dict[int, list[BoundingBox]]
) -> list[Figure]:
    """Build one :class:`Figure` per "Figure N" caption that has a graphic above it.

    Args:
        captions: ``(page_number, bbox, text)`` of every caption block.
        regions_by_page: Output of :func:`find_graphic_regions`, per page.
    """
    figures = []
    for page_number, caption_box, text in captions:
        match = _FIGURE_LABEL_RE.match(text)
        if not match:
            continue
        # Regions above the caption that overlap it horizontally, nearest first.
        above = sorted(
            (
                region
                for region in regions_by_page.get(page_number, [])
                if region[3] <= caption_box[1] + 2 and region[0] < caption_box[2] and region[2] > caption_box[0]
            ),
            key=lambda region: -region[3],
        )
        if not above or caption_box[1] - above[0][3] > _MAX_GAP_TO_CAPTION:
            continue
        area = above[0]
        for region in above[1:]:
            if area[1] - region[3] <= _MAX_GAP_BETWEEN_PARTS:
                area = _union(area, region)
        # Side-by-side sub-figures: any region level with the area belongs to it.
        for region in regions_by_page[page_number]:
            if region[1] < area[3] and region[3] > area[1] and region[3] <= caption_box[1] + 2:
                area = _union(area, region)
        left, top, right = _LABEL_MARGINS
        # Downwards, everything up to the caption belongs to the figure (x-axis labels).
        area = (max(area[0] - left, 0.0), max(area[1] - top, 0.0), area[2] + right, max(area[3], caption_box[1] - 2))
        figures.append(Figure(label=f"Figure {match.group(2)}", page_number=page_number, bbox=area, caption=text))
    return figures


def export_figure_images(pdf_path: Path | str, figures: list[Figure], output_dir: Path, dpi: int = 150) -> int:
    """Render each figure to ``output_dir/figure_<N>_p<page>.png`` and set its ``image_path``.

    Returns:
        The number of images written.
    """
    if not figures:
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with pymupdf.open(pdf_path) as document:
        for figure in figures:
            page = document[figure.page_number - 1]
            clip = pymupdf.Rect(figure.bbox) & page.rect
            number = figure.label.split()[-1]
            path = output_dir / f"figure_{int(number):02d}_p{figure.page_number}.png"
            page.get_pixmap(clip=clip, dpi=dpi, alpha=False).save(path)
            figure.image_path = path.as_posix()
            written += 1
    return written
