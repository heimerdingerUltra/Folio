"""Fluent pipeline: PDFs -> validated chunk dataset -> run report -> dashboard.

Each stage returns the pipeline itself, so a run reads as the list of its
steps (see ``command_line``). Stages only fill attributes; nothing is hidden in
module state, which keeps every stage testable on its own.
"""

from __future__ import annotations

import logging
import platform
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

import pymupdf

from pdf_chunking import __version__
from pdf_chunking.chunk_dataset_export import write_chunk_dataset, write_json_file
from pdf_chunking.chunk_quality_metrics import compute_chunk_quality_report, summarize_token_lengths
from pdf_chunking.chunk_record_schema import ChunkRecord, validate_chunk_dataset
from pdf_chunking.chunking_strategies import ChunkingConfig, chunk_parsed_document
from pdf_chunking.opendataloader_extraction import extract_pdf_documents_with_opendataloader
from pdf_chunking.pdf_layout_extraction import (
    NoTextLayerError,
    ParsedDocument,
    PdfExtractionConfig,
    PdfExtractionError,
    extract_pdf_document,
)
from pdf_chunking.performance_dashboard import write_performance_dashboard

logger = logging.getLogger(__name__)

EXIT_CODE_BY_STATUS = {"success": 0, "partial": 1, "failed": 2}
_REPORTED_PACKAGES = ("pymupdf", "pymupdf-layout", "opendataloader-pdf", "pydantic", "tiktoken", "scikit-learn")

PdfParser = Literal["pymupdf", "opendataloader"]


@dataclass(frozen=True)
class ChunkingPipelineConfig:
    """Everything needed to reproduce a run.

    Attributes:
        input_path: Folder of PDFs, or a single PDF.
        output_path: Chunk dataset (``.json``, ``.jsonl`` or ``.csv``).
        pdf_parser: ``pymupdf`` (this repository's layout logic) or
            ``opendataloader`` (OpenDataLoader PDF, needs Java).
        extraction: PDF extraction options (``pymupdf`` parser only).
        chunking: Chunking options.
        benchmark_against_fixed_windows: Also chunk with the naive baseline,
            score it, and export it for the retrieval benchmark.
    """

    input_path: Path
    output_path: Path
    pdf_parser: PdfParser = "pymupdf"
    extraction: PdfExtractionConfig = field(default_factory=PdfExtractionConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    benchmark_against_fixed_windows: bool = False

    def _sibling(self, suffix: str) -> Path:
        """A path next to the output file, with ``suffix`` appended to its stem."""
        return self.output_path.with_name(self.output_path.stem + suffix)

    @property
    def run_report_path(self) -> Path:
        """Where the JSON run report is written."""
        return self._sibling(".run_report.json")

    @property
    def dashboard_path(self) -> Path:
        """Where the HTML dashboard is written."""
        return self._sibling(".dashboard.html")

    @property
    def fixed_window_baseline_path(self) -> Path:
        """Where the fixed-window baseline dataset is written."""
        return self._sibling(".fixed_window_baseline" + self.output_path.suffix)


def find_pdf_files(input_path: Path) -> list[Path]:
    """List the PDFs under ``input_path`` (a folder or one file), sorted.

    Raises:
        FileNotFoundError: If ``input_path`` does not exist.
    """
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    return sorted(
        (path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"),
        key=lambda path: path.name.lower(),
    )


def _installed_version(package: str) -> str:
    """Installed version of ``package``, or ``"not installed"``."""
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


def _drain_mupdf_warnings() -> list[str]:
    """MuPDF buffers its repair messages; collect them for the report."""
    try:
        text = pymupdf.TOOLS.mupdf_warnings()
    except Exception:  # noqa: BLE001 - diagnostic only
        return []
    return sorted({line.strip() for line in text.splitlines() if line.strip()})[:10]


class ChunkingPipeline:
    """PDF folder -> chunk dataset, one chainable stage at a time.

    Example:
        >>> ChunkingPipeline(config).discover_pdfs().extract_documents().chunk_documents()
    """

    def __init__(self, config: ChunkingPipelineConfig) -> None:
        """Create a pipeline with empty state; stages fill it in order."""
        self.config = config
        self.started_at = datetime.now(timezone.utc)
        self._clock = time.perf_counter()
        self.pdf_paths: list[Path] = []
        self.document_reports: list[dict[str, Any]] = []
        self.parsed_documents: list[ParsedDocument] = []
        self.chunks: list[ChunkRecord] = []
        self.validation: dict[str, Any] = {}
        self.quality: dict[str, Any] = {}
        self.fixed_window_quality: dict[str, Any] | None = None
        self.run_report: dict[str, Any] = {}

    # ------------------------------------------------------------- stages

    def discover_pdfs(self) -> ChunkingPipeline:
        """Find the PDFs to process."""
        self.pdf_paths = find_pdf_files(self.config.input_path)
        logger.info("Found %d PDF file(s) in %s", len(self.pdf_paths), self.config.input_path)
        return self

    def extract_documents(self) -> ChunkingPipeline:
        """Parse every PDF; a bad file is recorded and skipped, never fatal."""
        # MuPDF would otherwise print repair messages straight to stderr.
        pymupdf.TOOLS.mupdf_display_errors(False)
        pymupdf.TOOLS.mupdf_display_warnings(False)
        opendataloader_results = self._extract_batch_with_opendataloader()
        for path in self.pdf_paths:
            entry: dict[str, Any] = {"source_document": path.name, "status": "ok", "error": None}
            started = time.perf_counter()
            try:
                if opendataloader_results is None:
                    document = extract_pdf_document(path, self.config.extraction)
                elif isinstance(result := opendataloader_results[path.name], PdfExtractionError):
                    raise result
                else:
                    document = result
            except NoTextLayerError as error:
                entry.update(status="no_text_layer", error=str(error))
            except PdfExtractionError as error:
                entry.update(status="failed", error=str(error))
            except Exception as error:  # noqa: BLE001 - one bad file must not stop the batch
                entry.update(status="failed", error=f"Unexpected {type(error).__name__}: {error}")
                logger.debug("Traceback for %s", path.name, exc_info=True)
            else:
                self.parsed_documents.append(document)
                entry.update(
                    title=document.title,
                    pages=document.page_count,
                    extraction=document.stats,
                    warnings=document.warnings,
                )
            entry["extraction_seconds"] = round(time.perf_counter() - started + self._batch_seconds_per_pdf, 3)
            if parser_warnings := _drain_mupdf_warnings():
                entry["parser_warnings"] = parser_warnings
            if entry["status"] != "ok":
                logger.error("%s: %s", path.name, entry["error"])
            self.document_reports.append(entry)
        return self

    def _extract_batch_with_opendataloader(self) -> dict[str, ParsedDocument | PdfExtractionError] | None:
        """Parse all PDFs in one JVM run; ``None`` when the PyMuPDF parser is selected."""
        self._batch_seconds_per_pdf = 0.0
        if self.config.pdf_parser != "opendataloader" or not self.pdf_paths:
            return None
        started = time.perf_counter()
        try:
            # 'cluster' recovered 9 tables with text against 3 for the default method
            # (benchmarks/opendataloader_comparison.json).
            results = extract_pdf_documents_with_opendataloader(self.pdf_paths, table_method="cluster")
        except PdfExtractionError as error:  # backend unavailable: every file fails the same way
            results = {path.name: error for path in self.pdf_paths}
        # The JVM start is shared by the batch; spread it so per-file timings add up.
        self._batch_seconds_per_pdf = (time.perf_counter() - started) / len(self.pdf_paths)
        return results

    def chunk_documents(self) -> ChunkingPipeline:
        """Chunk every parsed document with the configured strategy."""
        entries = {entry["source_document"]: entry for entry in self.document_reports}
        for document in self.parsed_documents:
            entry = entries[document.source_document]
            started = time.perf_counter()
            try:
                chunks, rejected = chunk_parsed_document(document, self.config.chunking)
            except Exception as error:  # noqa: BLE001 - keep the other documents
                entry.update(status="failed", error=f"Chunking failed: {type(error).__name__}: {error}")
                logger.error("%s: %s", document.source_document, entry["error"])
                continue
            self.chunks.extend(chunks)
            entry.update(
                chunking_seconds=round(time.perf_counter() - started, 3),
                chunks=len(chunks),
                chunks_rejected_by_validation=rejected,
                token_length=summarize_token_lengths([chunk.token_count for chunk in chunks]),
            )
            logger.info(
                "%-32s %3d pages -> %4d chunks  (extract %.1fs, chunk %.2fs)",
                document.source_document,
                document.page_count,
                len(chunks),
                entry["extraction_seconds"],
                entry["chunking_seconds"],
            )
            for warning in document.warnings:
                logger.warning("%s: %s", document.source_document, warning)
        return self

    def validate_chunks(self) -> ChunkingPipeline:
        """Check ids, indices and the token budget across the whole dataset."""
        self.validation = validate_chunk_dataset(self.chunks, max_tokens=self.config.chunking.max_tokens)
        if not self.validation["is_valid"]:
            logger.warning("Dataset validation found problems: %s", self.validation["errors"][:3])
        return self

    def measure_chunk_quality(self) -> ChunkingPipeline:
        """Compute size, boundary, table and cohesion metrics."""
        chunking = self.config.chunking
        self.quality = compute_chunk_quality_report(
            self.parsed_documents, self.chunks, chunking.min_tokens, chunking.max_tokens
        )
        return self

    def benchmark_against_fixed_windows(self) -> ChunkingPipeline:
        """Score and export the naive fixed-window baseline (if enabled)."""
        if not self.config.benchmark_against_fixed_windows or not self.parsed_documents:
            return self
        baseline = replace(self.config.chunking, strategy="fixed")
        baseline_chunks = [
            chunk
            for document in self.parsed_documents
            for chunk in chunk_parsed_document(document, baseline)[0]
        ]
        self.fixed_window_quality = compute_chunk_quality_report(
            self.parsed_documents, baseline_chunks, baseline.min_tokens, baseline.max_tokens
        )
        write_chunk_dataset(baseline_chunks, self.config.fixed_window_baseline_path)
        logger.info("Wrote %d baseline chunks to %s", len(baseline_chunks), self.config.fixed_window_baseline_path)
        return self

    def export_chunk_dataset(self) -> ChunkingPipeline:
        """Write the chunks in the format given by the output extension."""
        if self.chunks:
            path = write_chunk_dataset(self.chunks, self.config.output_path)
            logger.info("Wrote %d chunks to %s", len(self.chunks), path)
        else:
            logger.error("No chunks produced; nothing written to %s", self.config.output_path)
        return self

    def write_run_report(self) -> ChunkingPipeline:
        """Write the JSON run report: config, timings, quality, errors."""
        duration = time.perf_counter() - self._clock
        failed = [entry for entry in self.document_reports if entry["status"] != "ok"]
        pages = sum(entry.get("pages", 0) for entry in self.document_reports)
        extraction_seconds = sum(entry.get("extraction_seconds", 0) for entry in self.document_reports)
        chunking_seconds = sum(entry.get("chunking_seconds", 0) for entry in self.document_reports)
        processing_seconds = extraction_seconds + chunking_seconds
        self.run_report = {
            "run": {
                "run_id": uuid.uuid4().hex[:12],
                "status": self.status,
                "started_at": self.started_at.isoformat(timespec="seconds"),
                "duration_seconds": round(duration, 3),
                "output_file": str(self.config.output_path) if self.chunks else None,
            },
            "config": {
                "input_path": str(self.config.input_path),
                "output_path": str(self.config.output_path),
                "pdf_parser": self.config.pdf_parser,
                "extraction": asdict(self.config.extraction),
                "chunking": asdict(self.config.chunking),
            },
            "environment": {
                "pipeline": __version__,
                "python": platform.python_version(),
                "platform": platform.platform(),
                **{package: _installed_version(package) for package in _REPORTED_PACKAGES},
            },
            "summary": {
                "documents_found": len(self.pdf_paths),
                "documents_processed": len(self.parsed_documents),
                "documents_failed": len(failed),
                "error_rate": round(len(failed) / len(self.pdf_paths), 3) if self.pdf_paths else None,
                "pages": pages,
                "chunks": len(self.chunks),
                "tokens": sum(chunk.token_count for chunk in self.chunks),
                "chunks_per_document": summarize_token_lengths(
                    [entry["chunks"] for entry in self.document_reports if "chunks" in entry]
                ),
                "extraction_seconds": round(extraction_seconds, 3),
                "chunking_seconds": round(chunking_seconds, 3),
                "pages_per_second": round(pages / processing_seconds, 2) if processing_seconds else None,
            },
            "quality": self.quality,
            "validation": self.validation,
            "documents": self.document_reports,
        }
        if self.fixed_window_quality is not None:
            self.run_report["baseline_quality"] = self.fixed_window_quality
        path = write_json_file(self.run_report, self.config.run_report_path)
        logger.info("Wrote run report to %s", path)
        logger.info(
            "Run %s: %d/%d documents, %d chunks, %.1fs",
            self.status,
            len(self.parsed_documents),
            len(self.pdf_paths),
            len(self.chunks),
            duration,
        )
        return self

    def render_performance_dashboard(self) -> ChunkingPipeline:
        """Render the run report (plus any benchmark results) as HTML."""
        try:
            path = write_performance_dashboard(self.run_report, self.config.dashboard_path, Path("benchmarks"))
            logger.info("Wrote dashboard to %s", path)
        except Exception as error:  # noqa: BLE001 - a rendering bug must not fail the run
            logger.warning("Could not render the dashboard: %s", error)
        return self

    # ------------------------------------------------------------ outcome

    @property
    def status(self) -> str:
        """``success``, ``partial`` (some documents failed) or ``failed``."""
        if not self.chunks:
            return "failed"
        if len(self.parsed_documents) < len(self.pdf_paths):
            return "partial"
        return "success"

    @property
    def exit_code(self) -> int:
        """0 = all documents processed, 1 = some failed, 2 = nothing exported."""
        return EXIT_CODE_BY_STATUS[self.status]
