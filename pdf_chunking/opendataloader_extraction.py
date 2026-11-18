"""Alternative extraction backend: OpenDataLoader PDF.

OpenDataLoader PDF (https://github.com/opendataloader-project/opendataloader-pdf)
is a Java engine with a Python wrapper. It returns a typed JSON tree per PDF
(headings, paragraphs, lists, tables, captions). This module maps that tree
onto the pipeline's own :class:`ParsedDocument`, so both backends feed the
same chunker and are compared on equal terms.

The mapping is deliberately thin: element types, heading levels and reading
order are taken as delivered. Only Unicode clean-up is applied to the text,
otherwise the comparison would measure this adapter instead of the parser.

Requires ``pip install opendataloader-pdf`` and Java 11+.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence

from pdf_chunking.pdf_layout_extraction import (
    ParsedDocument,
    PdfExtractionError,
    TextBlock,
    assign_section_paths,
)
from pdf_chunking.text_normalization import normalize_pdf_text

logger = logging.getLogger(__name__)

#: ``default`` finds tables from ruling lines; ``cluster`` also groups aligned text.
TableMethod = Literal["default", "cluster"]

TABLE_CELL_SEPARATOR = " | "
_DOCUMENT_TITLE_LEVEL = "Doctitle"
_BLOCK_TYPE_BY_ELEMENT_TYPE = {"paragraph": "paragraph", "caption": "caption", "formula": "equation"}


def _collect_text(element: dict[str, Any]) -> str:
    """Text of an element and of everything nested in it, in document order."""
    parts = [element.get("content", "")]
    for key in ("kids", "list items"):
        parts.extend(_collect_text(child) for child in element.get(key, []))
    return " ".join(part for part in parts if part)


def _format_table_rows(table: dict[str, Any]) -> list[str]:
    """One line per table row, cells separated by :data:`TABLE_CELL_SEPARATOR`."""
    rows = []
    for row in table.get("rows", []):
        cells = [normalize_pdf_text(_collect_text(cell)) for cell in row.get("cells", [])]
        if any(cells):
            rows.append(TABLE_CELL_SEPARATOR.join(cells))
    return rows


def _build_block(element: dict[str, Any], block_type: str, lines: list[str], heading_level: int = 0) -> TextBlock:
    """Create a ``TextBlock`` from an OpenDataLoader element and its text lines."""
    return TextBlock(
        text="\n".join(lines),
        block_type=block_type,
        page_number=int(element.get("page number", 1)),
        # As delivered: (left, bottom, right, top) in PDF points, origin bottom-left.
        bbox=tuple(element.get("bounding box", (0.0, 0.0, 0.0, 0.0))),
        lines=lines,
        font_size=float(element.get("font size", 0.0)),
        max_font_size=float(element.get("font size", 0.0)),
        heading_level=heading_level,
    )


def _iterate_text_blocks(elements: Sequence[dict[str, Any]]) -> Iterator[TextBlock]:
    """Flatten the element tree into typed blocks, keeping the delivered order."""
    for element in elements:
        element_type = element.get("type")
        if element_type == "heading":
            is_title = element.get("level") == _DOCUMENT_TITLE_LEVEL
            text = normalize_pdf_text(element.get("content", ""))
            if text:
                level = 0 if is_title else int(element.get("heading level", 1))
                yield _build_block(element, "title" if is_title else "heading", [text], level)
        elif element_type == "table":
            if rows := _format_table_rows(element):
                yield _build_block(element, "table", rows)
        elif element_type == "list":
            for item in element.get("list items", []):
                if text := normalize_pdf_text(_collect_text(item)):
                    yield _build_block(item, "paragraph", [text])
        elif element_type in _BLOCK_TYPE_BY_ELEMENT_TYPE:
            if text := normalize_pdf_text(element.get("content", "")):
                yield _build_block(element, _BLOCK_TYPE_BY_ELEMENT_TYPE[element_type], [text])
        elif element.get("kids"):  # "text block" and other containers
            yield from _iterate_text_blocks(element["kids"])


def build_parsed_document_from_opendataloader_json(data: dict[str, Any], fallback_name: str) -> ParsedDocument:
    """Map one OpenDataLoader JSON tree onto a :class:`ParsedDocument`.

    Args:
        data: Root object of the JSON file written by OpenDataLoader for one PDF.
        fallback_name: Source file name to use when ``data`` has no ``file name`` entry.

    Returns:
        The document, with section paths assigned. The title is the ``Doctitle`` heading, else
        the ``title`` entry of ``data``, else the file stem with underscores turned into spaces.

    Raises:
        PdfExtractionError: If the tree holds no text at all.
    """
    source_document = data.get("file name") or fallback_name
    blocks = list(_iterate_text_blocks(data.get("kids", [])))
    if not blocks:
        raise PdfExtractionError(f"{source_document}: OpenDataLoader returned no text.")
    assign_section_paths(blocks)

    title_block = next((block for block in blocks if block.block_type == "title"), None)
    title = (title_block.text if title_block else data.get("title")) or Path(source_document).stem.replace("_", " ")

    stats: dict[str, Any] = {f"blocks_{kind}": count for kind, count in Counter(b.block_type for b in blocks).items()}
    stats.update(
        parser="opendataloader-pdf",
        tables_detected=stats.get("blocks_table", 0),
        characters=sum(len(block.text) for block in blocks),
    )
    return ParsedDocument(
        source_document=source_document,
        title=title,
        page_count=int(data.get("number of pages", max(block.page_number for block in blocks))),
        blocks=blocks,
        stats=stats,
    )


def convert_pdfs_to_opendataloader_json(
    paths: Sequence[Path], output_directory: Path, table_method: TableMethod = "default"
) -> dict[str, Path]:
    """Run OpenDataLoader once over ``paths`` (one JVM start for the batch).

    Args:
        paths: PDFs to convert. Their stems must be distinct: each JSON is named after its PDF.
        output_directory: Folder that receives one ``<stem>.json`` per PDF.
        table_method: ``default`` finds tables from ruling lines; ``cluster`` also groups
            aligned text.

    Returns:
        ``{pdf file name: path of its JSON}`` for every PDF that was converted.

    Raises:
        PdfExtractionError: Package or Java missing, or the conversion failed.
    """
    try:
        import opendataloader_pdf
    except ImportError as error:
        raise PdfExtractionError("The 'opendataloader-pdf' package is not installed.") from error
    if shutil.which("java") is None:
        raise PdfExtractionError("OpenDataLoader PDF needs Java 11 or later on the PATH.")
    try:
        opendataloader_pdf.convert(
            input_path=[str(path) for path in paths],
            output_dir=str(output_directory),
            format="json",
            image_output="off",
            table_method=table_method,
            quiet=True,
        )
    except Exception as error:  # noqa: BLE001 - the wrapper raises several types
        raise PdfExtractionError(f"OpenDataLoader conversion failed: {error}") from error
    return {
        path.name: json_path for path in paths if (json_path := output_directory / f"{path.stem}.json").is_file()
    }


def extract_pdf_documents_with_opendataloader(
    paths: Sequence[Path], table_method: TableMethod = "default"
) -> dict[str, ParsedDocument | PdfExtractionError]:
    """Parse ``paths`` with OpenDataLoader.

    Args:
        paths: PDFs to parse, converted together in a temporary folder that is then deleted.
        table_method: ``default`` finds tables from ruling lines; ``cluster`` also groups
            aligned text.

    Returns:
        ``{pdf file name: ParsedDocument}``; a PDF that could not be parsed maps
        to the error instead, so one bad file does not stop the batch.

    Raises:
        PdfExtractionError: The backend itself is unavailable.
    """
    results: dict[str, ParsedDocument | PdfExtractionError] = {}
    with tempfile.TemporaryDirectory(prefix="opendataloader_") as directory:
        json_paths = convert_pdfs_to_opendataloader_json(paths, Path(directory), table_method)
        for path in paths:
            if path.name not in json_paths:
                results[path.name] = PdfExtractionError(f"{path.name}: OpenDataLoader produced no output.")
                continue
            try:
                data = json.loads(json_paths[path.name].read_text(encoding="utf-8"))
                results[path.name] = build_parsed_document_from_opendataloader_json(data, path.name)
            except PdfExtractionError as error:
                results[path.name] = error
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                results[path.name] = PdfExtractionError(f"{path.name}: unreadable OpenDataLoader output ({error}).")
    return results
