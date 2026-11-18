"""Structured view of a parsed document: section tree, Markdown, figures.

The chunk dataset is flat, which is what a vector store wants. This module
keeps the hierarchy instead: every block under its section, in reading
order, with its page and position, and every figure with its image file.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from pdf_chunking.chunk_dataset_export import write_json_file
from pdf_chunking.figure_extraction import export_figure_images
from pdf_chunking.pdf_layout_extraction import ParsedDocument, TextBlock


def _new_section(title: str, level: int, page_number: int | None) -> dict[str, Any]:
    """An empty node of the section tree."""
    return {"title": title, "level": level, "page_number": page_number, "blocks": [], "sections": []}


def _block_record(block: TextBlock, reading_order: int) -> dict[str, Any]:
    """The exported fields of one block."""
    return {
        "reading_order": reading_order,
        "type": block.block_type,
        "page_number": block.page_number,
        "bbox": [round(value, 1) for value in block.bbox],
        "from_ocr": block.from_ocr,
        "text": block.text,
    }


def build_document_structure(document: ParsedDocument) -> dict[str, Any]:
    """Nest the blocks of ``document`` under their headings.

    Returns:
        ``{"source_document", "title", "page_count", "sections", "figures"}``.
        ``sections`` is a tree: each node has a ``title``, a ``level``, its
        own ``blocks`` (in reading order) and its child ``sections``. Blocks
        placed before the first heading sit in a level-0 node without title.
    """
    root = _new_section(document.title, 0, 1)
    open_sections = [root]  # path from the root to the current section
    for reading_order, block in enumerate(document.blocks):
        if block.block_type == "heading":
            level = max(block.heading_level, 1)
            while len(open_sections) > 1 and open_sections[-1]["level"] >= level:
                open_sections.pop()
            section = _new_section(block.text, level, block.page_number)
            open_sections[-1]["sections"].append(section)
            open_sections.append(section)
        else:
            open_sections[-1]["blocks"].append(_block_record(block, reading_order))
    front_matter = {**root, "title": "", "sections": []}
    return {
        "source_document": document.source_document,
        "title": document.title,
        "page_count": document.page_count,
        "sections": ([front_matter] if root["blocks"] else []) + root["sections"],
        "figures": [asdict(figure) for figure in document.figures],
    }


def render_document_markdown(document: ParsedDocument) -> str:
    """Render ``document`` as Markdown, in reading order.

    Headings keep their depth, tables and equations are fenced so their
    alignment survives, footnotes become quotes, and each exported figure is
    inserted after its caption.
    """
    figure_by_caption = {(figure.page_number, figure.caption): figure for figure in document.figures}
    parts = [f"# {document.title}"]
    for block in document.blocks:
        kind, text = block.block_type, block.text
        if kind == "title":
            continue
        if kind == "heading":
            parts.append(f"{'#' * min(max(block.heading_level, 1) + 1, 6)} {text}")
        elif kind in ("table", "equation"):
            parts.append(f"```\n{text}\n```")
        elif kind == "footnote":
            parts.append(f"> {text}")
        elif kind == "caption":
            parts.append(f"*{text}*")
            figure = figure_by_caption.get((block.page_number, text))
            if figure is not None and figure.image_path:
                parts.append(f"![{figure.label}]({figure.image_path})")
        else:
            parts.append(text)
    return "\n\n".join(parts) + "\n"


def export_structured_document(pdf_path: Path, document: ParsedDocument, output_dir: Path) -> dict[str, Any]:
    """Write ``<stem>.json``, ``<stem>.md`` and the figure images of one document.

    Image paths are stored relative to ``output_dir`` so the folder can be moved.

    Returns:
        The paths written and the number of figure images.
    """
    stem = Path(document.source_document).stem
    output_dir.mkdir(parents=True, exist_ok=True)
    images = export_figure_images(pdf_path, document.figures, output_dir / stem)
    for figure in document.figures:
        if figure.image_path:
            figure.image_path = f"{stem}/{Path(figure.image_path).name}"
    structure_path = write_json_file(build_document_structure(document), output_dir / f"{stem}.json")
    markdown_path = output_dir / f"{stem}.md"
    markdown_path.write_text(render_document_markdown(document), encoding="utf-8")
    return {"structure": str(structure_path), "markdown": str(markdown_path), "figure_images": images}
