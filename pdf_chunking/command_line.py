#!/usr/bin/env python
"""Parse PDFs into RAG-ready chunks.

Example:
    process-pdfs --input ./data --output ./results.json

Exit codes: 0 = every document processed, 1 = some failed (the others are
exported), 2 = nothing exported.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from pdf_chunking.chunk_dataset_export import SUPPORTED_FORMATS
from pdf_chunking.chunking_pipeline import ChunkingPipeline, ChunkingPipelineConfig
from pdf_chunking.chunking_strategies import ChunkingConfig
from pdf_chunking.pdf_layout_extraction import PdfExtractionConfig
from pdf_chunking.run_logging import configure_logging

logger = logging.getLogger(__name__)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Parse PDFs into clean, RAG-ready text chunks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=Path("./data"),
                        help="Folder of PDFs, or a single PDF.")
    parser.add_argument("--output", type=Path, default=Path("./results.json"),
                        help=f"Chunk dataset; format from extension ({', '.join(SUPPORTED_FORMATS)}).")

    chunking = parser.add_argument_group("chunking")
    chunking.add_argument("--strategy", choices=("structure", "fixed"), default=ChunkingConfig.strategy,
                          help="'structure' = section- and sentence-aware; 'fixed' = naive baseline.")
    chunking.add_argument("--max-tokens", type=int, default=ChunkingConfig.max_tokens,
                          help="Maximum chunk size.")
    chunking.add_argument("--overlap-tokens", type=int, default=ChunkingConfig.overlap_tokens,
                          help="Sentence-aligned overlap between consecutive chunks.")
    chunking.add_argument("--min-tokens", type=int, default=ChunkingConfig.min_tokens,
                          help="Smaller chunks are merged into a neighbour.")
    chunking.add_argument("--benchmark-against-fixed-windows", action="store_true",
                          help="Also score and export the fixed-window baseline.")

    extraction = parser.add_argument_group("extraction")
    extraction.add_argument("--pdf-parser", choices=("pymupdf", "opendataloader"), default="pymupdf",
                            help="pymupdf: this repository's layout logic. opendataloader: OpenDataLoader PDF (needs Java 11+).")
    extraction.add_argument("--table-detection", choices=("auto", "layout", "heuristic", "off"), default="auto",
                            help="'layout' = pymupdf-layout model (accurate, slow); 'heuristic' = "
                                 "numeric-density rule (fast); 'auto' = layout if installed.")
    extraction.add_argument("--scan-every-page-for-tables", action="store_true",
                            help="Run the layout model on all pages, not only likely table pages.")
    extraction.add_argument("--keep-text-inside-figures", action="store_true",
                            help="Keep axis labels and diagram text.")
    extraction.add_argument("--ocr", choices=("auto", "off", "force"), default="auto",
                            help="'auto' = OCR pages without a text layer (needs rapidocr); "
                                 "'force' = OCR every page; 'off' = never.")
    extraction.add_argument("--workers", type=int, default=1,
                            help="Processes used to parse PDFs (pymupdf parser).")
    extraction.add_argument("--export-structure", action="store_true",
                            help="Also write the section tree (JSON), Markdown and figure images of each document.")

    log = parser.add_argument_group("logging")
    log.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    log.add_argument("--log-file", type=Path, default=None, help="Also write a DEBUG log here.")
    return parser  # fmt: skip


def build_pipeline_config(args: argparse.Namespace) -> ChunkingPipelineConfig:
    """Translate parsed arguments into a pipeline configuration.

    Raises:
        ValueError: If the output format or a chunking parameter is invalid.
    """
    if args.output.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported output extension {args.output.suffix!r}; use {', '.join(SUPPORTED_FORMATS)}.")
    return ChunkingPipelineConfig(
        input_path=args.input,
        output_path=args.output,
        pdf_parser=args.pdf_parser,
        extraction=PdfExtractionConfig(
            table_detection=args.table_detection,
            skip_pages_without_table_signals=not args.scan_every_page_for_tables,
            drop_text_inside_figures=not args.keep_text_inside_figures,
            ocr=args.ocr,
        ),
        chunking=ChunkingConfig(
            strategy=args.strategy,
            max_tokens=args.max_tokens,
            overlap_tokens=args.overlap_tokens,
            min_tokens=args.min_tokens,
        ),
        benchmark_against_fixed_windows=args.benchmark_against_fixed_windows,
        export_structure=args.export_structure,
        workers=max(args.workers, 1),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pipeline; returns the process exit code."""
    args = build_argument_parser().parse_args(argv)
    configure_logging(args.log_level, args.log_file)
    try:
        return (
            ChunkingPipeline(build_pipeline_config(args))
            .discover_pdfs()
            .extract_documents()
            .chunk_documents()
            .validate_chunks()
            .measure_chunk_quality()
            .benchmark_against_fixed_windows()
            .export_chunk_dataset()
            .export_structured_documents()
            .write_run_report()
            .render_performance_dashboard()
            .exit_code
        )
    except (ValueError, FileNotFoundError, OSError) as error:
        logger.error("%s", error)
        return 2


if __name__ == "__main__":
    sys.exit(main())
