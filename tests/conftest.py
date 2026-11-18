"""Shared fixtures: small synthetic PDFs built on the fly with PyMuPDF.

The tests never touch the network or the ``data/`` folder, so they run
anywhere in a couple of seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf
import pytest

# Make the project root importable when pytest is run from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LOREM = (
    "Retrieval systems split documents into chunks before embedding them. "
    "A chunk that stops in the middle of a sentence loses part of its meaning. "
    "Good chunking therefore respects the structure of the source document. "
)


def _write_paragraph(page: pymupdf.Page, rect: pymupdf.Rect, text: str, **kwargs: object) -> None:
    """Insert wrapped text and fail loudly if it does not fit the box."""
    leftover = page.insert_textbox(rect, text, fontsize=10, fontname="helv", **kwargs)
    assert leftover >= 0, "test text does not fit in its box"


@pytest.fixture
def simple_pdf(tmp_path: Path) -> Path:
    """Two-page, single-column PDF with numbered bold headings."""
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 80), "A Study of Chunking", fontsize=18, fontname="hebo")
    page.insert_text((72, 120), "1 Introduction", fontsize=12, fontname="hebo")
    _write_paragraph(page, pymupdf.Rect(72, 130, 520, 260), LOREM * 3)
    page.insert_text((72, 290), "2 Method", fontsize=12, fontname="hebo")
    _write_paragraph(page, pymupdf.Rect(72, 300, 520, 430), LOREM * 3)
    page.insert_text((300, 800), "1", fontsize=10, fontname="helv")  # page number

    page = document.new_page()
    page.insert_text((72, 80), "2.1 Details", fontsize=11, fontname="hebo")
    _write_paragraph(page, pymupdf.Rect(72, 90, 520, 220), LOREM * 3)
    page.insert_text((300, 800), "2", fontsize=10, fontname="helv")

    path = tmp_path / "simple.pdf"
    document.save(path)
    document.close()
    return path


@pytest.fixture
def two_column_pdf(tmp_path: Path) -> Path:
    """One page: full-width title, then two columns of marked paragraphs."""
    document = pymupdf.open()
    page = document.new_page()  # A4: 595 x 842 pt
    page.insert_text((72, 60), "Two Column Layout", fontsize=18, fontname="hebo")
    filler = "This sentence only pads the paragraph so that the block is wide and tall enough. " * 4
    # Written in a scrambled order on purpose: the extractor must recover the
    # reading order from geometry, not from the order of the content stream.
    _write_paragraph(page, pymupdf.Rect(310, 270, 535, 430), "RIGHT-BOTTOM starts here. " + filler)
    _write_paragraph(page, pymupdf.Rect(60, 270, 285, 430), "LEFT-BOTTOM starts here. " + filler)
    _write_paragraph(page, pymupdf.Rect(310, 90, 535, 250), "RIGHT-TOP starts here. " + filler)
    _write_paragraph(page, pymupdf.Rect(60, 90, 285, 250), "LEFT-TOP starts here. " + filler)
    path = tmp_path / "two_column.pdf"
    document.save(path)
    document.close()
    return path


@pytest.fixture
def image_only_pdf(tmp_path: Path) -> Path:
    """A "scanned" PDF: one page holding an image and no text layer."""
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 200, 200))
    pixmap.clear_with(200)
    document = pymupdf.open()
    page = document.new_page()
    page.insert_image(pymupdf.Rect(72, 72, 400, 400), pixmap=pixmap)
    path = tmp_path / "scanned.pdf"
    document.save(path)
    document.close()
    return path


@pytest.fixture
def corrupted_pdf(tmp_path: Path) -> Path:
    """A file with a PDF header followed by garbage."""
    path = tmp_path / "corrupted.pdf"
    path.write_bytes(b"%PDF-1.7\n" + b"\x00\xff garbage that is not a pdf body " * 50)
    return path
