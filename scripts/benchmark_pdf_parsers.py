"""Compare open-source PDF parsers on the dataset (evidence for the README).

Usage:
    python scripts/benchmark_pdf_parsers.py [--input ./data]

For each library we run its *default* plain-text extraction and measure:

* ``seconds``        - wall time to extract every page of every PDF;
* ``chars``          - characters extracted (a coverage sanity check);
* ``ligatures``      - leftover ligature glyphs (U+FB00..U+FB06), which break
                       tokenisation ("e<ffi>cient" is not "efficient");
* ``hyphen_breaks``  - lines ending in a hyphen, i.e. words left cut in two;
* ``column_flow``    - on two-column pages, whether a column is read top to
                       bottom without mixing in the other column (probes
                       below).

pdfplumber and pypdf are optional: a missing library is reported, not fatal.
The last row is this repository's pipeline, which adds reading-order
reconstruction, de-hyphenation and clean-up on top of PyMuPDF.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pdf_chunking.parser_comparison import COLUMN_PROBES  # noqa: E402

_LIGATURE_RE = re.compile("[ﬀ-ﬆ]")
_HYPHEN_BREAK_RE = re.compile(r"[A-Za-z]-\n")
_SPACES_RE = re.compile(r"\s+")


def extract_pymupdf(path: Path) -> list[str]:
    """PyMuPDF default: ``page.get_text()``."""
    import pymupdf

    with pymupdf.open(path) as document:
        return [page.get_text() for page in document]


def extract_pdfplumber(path: Path) -> list[str]:
    """pdfplumber default: ``page.extract_text()``."""
    import pdfplumber

    with pdfplumber.open(path) as document:
        return [page.extract_text() or "" for page in document.pages]


def extract_pypdf(path: Path) -> list[str]:
    """pypdf default: ``page.extract_text()``."""
    from pypdf import PdfReader

    return [page.extract_text() or "" for page in PdfReader(path).pages]


def extract_pipeline(path: Path) -> list[str]:
    """This repository's extractor (text mode, table finder disabled)."""
    from pdf_chunking.pdf_layout_extraction import PdfExtractionConfig, extract_pdf_document

    document = extract_pdf_document(path, PdfExtractionConfig(table_detection="off"))
    pages = [""] * document.page_count
    for block in document.blocks:
        pages[block.page_number - 1] += block.text + "\n"
    return pages


EXTRACTORS: dict[str, Callable[[Path], list[str]]] = {
    "PyMuPDF (get_text)": extract_pymupdf,
    "pdfplumber": extract_pdfplumber,
    "pypdf": extract_pypdf,
    "this pipeline (PyMuPDF + layout logic)": extract_pipeline,
}


def column_flow_score(pages_by_file: dict[str, list[str]]) -> str:
    """Return ``"k/n"``: probes found verbatim (column read without mixing)."""
    found = total = 0
    for filename, page_index, fragment in COLUMN_PROBES:
        pages = pages_by_file.get(filename)
        if pages is None or page_index >= len(pages):
            continue
        # Compare on whitespace-free text so line wrapping and lost spaces
        # are not held against a parser; only the reading order is tested.
        total += 1
        found += _SPACES_RE.sub("", fragment) in _SPACES_RE.sub("", pages[page_index])
    return f"{found}/{total}"


def main() -> int:
    """Run the comparison and print a Markdown table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=Path("./data"))
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("benchmarks/parser_benchmark.json"),
        help="Where to save the results (read by the dashboard).",
    )
    args = parser.parse_args()
    paths = sorted(args.input.glob("*.pdf"))
    if not paths:
        print(f"No PDF found in {args.input}", file=sys.stderr)
        return 1

    rows: list[dict[str, object]] = []
    n_pages = 0
    print(f"{len(paths)} PDFs\n")
    print("| Parser | seconds | chars | ligatures | hyphen_breaks | column_flow |")
    print("|---|---:|---:|---:|---:|:---:|")
    for name, extract in EXTRACTORS.items():
        pages_by_file: dict[str, list[str]] = {}
        started = time.perf_counter()
        try:
            for path in paths:
                pages_by_file[path.name] = extract(path)
        except ImportError as exc:
            print(f"| {name} | not installed ({exc.name}) | | | | |")
            continue
        seconds = time.perf_counter() - started
        text = "\n".join(page for pages in pages_by_file.values() for page in pages)
        print(
            f"| {name} | {seconds:.1f} | {len(text):,} | {len(_LIGATURE_RE.findall(text))} "
            f"| {len(_HYPHEN_BREAK_RE.findall(text))} | {column_flow_score(pages_by_file)} |"
        )
        n_pages = sum(len(pages) for pages in pages_by_file.values())
        rows.append(
            {
                "parser": name,
                "seconds": round(seconds, 2),
                "chars": len(text),
                "ligatures": len(_LIGATURE_RE.findall(text)),
                "hyphen_breaks": len(_HYPHEN_BREAK_RE.findall(text)),
                "column_flow": column_flow_score(pages_by_file),
            }
        )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps({"n_pdfs": len(paths), "n_pages": n_pages, "parsers": rows}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
