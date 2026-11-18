"""Tests for PDF extraction: layout handling and defensive error paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from pdf_chunking.pdf_layout_extraction import (
    TextBlock,
    PdfExtractionConfig,
    NoTextLayerError,
    PdfExtractionError,
    _rebuild_table_rows_from_words,
    extract_pdf_document,
    is_bibliography_section,
    sort_blocks_in_reading_order,
)

# The synthetic PDFs have no tables: skip the (slow, optional) layout model.
FAST = PdfExtractionConfig(table_detection="heuristic")


def _block(x0: float, y0: float, x1: float, y1: float, text: str) -> TextBlock:
    """A paragraph block with enough characters to count as a column."""
    return TextBlock(text=text, block_type="paragraph", page_number=1, bbox=(x0, y0, x1, y1), lines=[text * 80])


class TestExtractDocument:
    def test_title_headings_and_sections(self, simple_pdf: Path) -> None:
        document = extract_pdf_document(simple_pdf, FAST)
        assert document.title == "A Study of Chunking"
        assert document.page_count == 2
        headings = [(b.text, b.heading_level) for b in document.blocks if b.block_type == "heading"]
        assert headings == [("1 Introduction", 1), ("2 Method", 1), ("2.1 Details", 2)]
        last = document.blocks[-1]
        assert last.block_type == "paragraph"
        assert last.section_path == ("2 Method", "2.1 Details")
        assert last.page_number == 2

    def test_page_numbers_are_dropped(self, simple_pdf: Path) -> None:
        document = extract_pdf_document(simple_pdf, FAST)
        assert all(block.text not in {"1", "2"} for block in document.blocks)
        assert document.stats["header_footer_blocks_dropped"] == 2

    def test_paragraph_lines_are_flowed(self, simple_pdf: Path) -> None:
        document = extract_pdf_document(simple_pdf, FAST)
        paragraph = next(b for b in document.blocks if b.block_type == "paragraph")
        assert "\n" not in paragraph.text
        assert paragraph.text.startswith("Retrieval systems split documents")

    def test_two_column_reading_order(self, two_column_pdf: Path) -> None:
        document = extract_pdf_document(two_column_pdf, FAST)
        assert document.stats["two_column_pages"] == 1
        text = " ".join(block.text for block in document.blocks)
        positions = [text.index(m) for m in ("LEFT-TOP", "LEFT-BOTTOM", "RIGHT-TOP", "RIGHT-BOTTOM")]
        # Left column top-to-bottom first, then the right column.
        assert positions == sorted(positions)


class TestErrorHandling:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(PdfExtractionError, match="not found"):
            extract_pdf_document(tmp_path / "nope.pdf", FAST)

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.pdf"
        path.write_bytes(b"")
        with pytest.raises(PdfExtractionError, match="empty"):
            extract_pdf_document(path, FAST)

    def test_corrupted_file(self, corrupted_pdf: Path) -> None:
        with pytest.raises(PdfExtractionError):
            extract_pdf_document(corrupted_pdf, FAST)

    def test_not_a_pdf(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.pdf"
        path.write_text("just some text pretending to be a pdf", encoding="utf-8")
        with pytest.raises(PdfExtractionError):
            extract_pdf_document(path, FAST)

    def test_scanned_pdf_has_no_text_layer(self, image_only_pdf: Path) -> None:
        with pytest.raises(NoTextLayerError, match="OCR"):
            extract_pdf_document(image_only_pdf, FAST)


class TestOrderBlocks:
    def test_single_column_is_top_to_bottom(self) -> None:
        blocks = [_block(72, 300, 520, 400, "b"), _block(72, 100, 520, 200, "a")]
        assert [b.text for b in sort_blocks_in_reading_order(blocks, page_width=595)] == ["a", "b"]

    def test_full_width_block_splits_columns_into_bands(self) -> None:
        blocks = [
            _block(310, 100, 535, 200, "right-1"),
            _block(60, 100, 285, 200, "left-1"),
            _block(60, 220, 535, 260, "wide"),
            _block(310, 280, 535, 380, "right-2"),
            _block(60, 280, 285, 380, "left-2"),
        ]
        ordered = [b.text for b in sort_blocks_in_reading_order(blocks, page_width=595)]
        assert ordered == ["left-1", "right-1", "wide", "left-2", "right-2"]


class TestTableRows:
    def test_columns_and_empty_cells_are_preserved(self) -> None:
        # (x0, y0, x1, y1, text): three columns, middle cell missing in row 2.
        words = [
            (10, 10, 40, 20, "Model"), (100, 10, 130, 20, "BLEU"), (200, 10, 230, 20, "Cost"),
            (10, 30, 40, 40, "Base"), (100, 30, 130, 40, "27.3"), (200, 30, 230, 40, "3.3"),
            (10, 50, 40, 60, "Big"), (200, 50, 230, 60, "2.3"),
        ]  # fmt: skip
        assert _rebuild_table_rows_from_words(words) == ["Model | BLEU | Cost", "Base | 27.3 | 3.3", "Big | | 2.3"]

    def test_empty_input(self) -> None:
        assert _rebuild_table_rows_from_words([]) == []


def test_is_reference_section() -> None:
    assert is_bibliography_section(("References",))
    assert is_bibliography_section(("8 Conclusion", "REFERENCES"))
    assert is_bibliography_section(("7 Bibliography",))
    assert not is_bibliography_section(("2 Related Work",))
    assert not is_bibliography_section(())
