"""Tests of the OpenDataLoader backend: JSON mapping, error paths, pipeline wiring."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from pdf_chunking import command_line
from pdf_chunking import chunking_pipeline, opendataloader_extraction
from pdf_chunking.chunking_strategies import ChunkingConfig, chunk_parsed_document
from pdf_chunking.opendataloader_extraction import (
    build_parsed_document_from_opendataloader_json,
    extract_pdf_documents_with_opendataloader,
)
from pdf_chunking.pdf_layout_extraction import PdfExtractionError


def element(element_type: str, content: str = "", page: int = 1, **extra: object) -> dict:
    return {"type": element_type, "content": content, "page number": page, "bounding box": [0, 0, 10, 10], **extra}


def table(*rows: list[str], page: int = 1) -> dict:
    return element(
        "table",
        page=page,
        rows=[{"type": "table row", "cells": [{"type": "table cell", "kids": [element("paragraph", cell)] if cell else []} for cell in row]} for row in rows],
    )


SAMPLE_TREE = {
    "file name": "paper.pdf",
    "number of pages": 2,
    "title": "",
    "kids": [
        element("heading", "A Study of Efﬁcient Parsing", level="Doctitle", **{"heading level": 1}),
        element("heading", "1 Introduction", level="Subtitle", **{"heading level": 2}),
        element("paragraph", "Parsing PDFs is hard."),
        element("list", **{"list items": [element("list item", "First point."), element("list item", "Second point.")]}),
        element("heading", "1.1 Scope", level="Subtitle", **{"heading level": 3}),
        {"type": "text block", "page number": 1, "kids": [element("paragraph", "Nested paragraph.")]},
        element("heading", "2 Results", page=2, level="Subtitle", **{"heading level": 2}),
        element("caption", "Table 1: Scores.", page=2),
        table(["Model", "Score"], ["Ours", "0.9"], ["", ""], page=2),
        table(["", ""]),
        element("image", page=2),
    ],
}


class TestJsonMapping:
    def test_elements_become_typed_blocks_in_order(self) -> None:
        document = build_parsed_document_from_opendataloader_json(SAMPLE_TREE, "fallback.pdf")
        assert (document.source_document, document.page_count) == ("paper.pdf", 2)
        assert [(block.block_type, block.text) for block in document.blocks] == [
            ("title", "A Study of Efficient Parsing"),  # ligature expanded
            ("heading", "1 Introduction"),
            ("paragraph", "Parsing PDFs is hard."),
            ("paragraph", "First point."),
            ("paragraph", "Second point."),
            ("heading", "1.1 Scope"),
            ("paragraph", "Nested paragraph."),
            ("heading", "2 Results"),
            ("caption", "Table 1: Scores."),
            ("table", "Model | Score\nOurs | 0.9"),  # empty rows and empty tables are dropped
        ]

    def test_title_sections_and_statistics(self) -> None:
        document = build_parsed_document_from_opendataloader_json(SAMPLE_TREE, "fallback.pdf")
        assert document.title == "A Study of Efficient Parsing"
        by_text = {block.text: block for block in document.blocks}
        assert by_text["Nested paragraph."].section_path == ("1 Introduction", "1.1 Scope")
        assert by_text["Table 1: Scores."].section_path == ("2 Results",)
        assert document.stats["parser"] == "opendataloader-pdf"
        assert document.stats["tables_detected"] == 1 and document.stats["blocks_heading"] == 3

    def test_title_falls_back_to_metadata_then_file_name(self) -> None:
        body = [element("paragraph", "Only text.")]
        assert build_parsed_document_from_opendataloader_json({"title": "From Metadata", "kids": body}, "a_b.pdf").title == "From Metadata"
        assert build_parsed_document_from_opendataloader_json({"kids": body}, "a_b.pdf").title == "a b"

    def test_a_tree_without_text_is_an_error(self) -> None:
        with pytest.raises(PdfExtractionError, match="no text"):
            build_parsed_document_from_opendataloader_json({"file name": "empty.pdf", "kids": [element("image")]}, "empty.pdf")

    def test_mapped_document_goes_through_the_chunker(self) -> None:
        document = build_parsed_document_from_opendataloader_json(SAMPLE_TREE, "fallback.pdf")
        chunks, rejected = chunk_parsed_document(document, ChunkingConfig(min_tokens=1))
        assert chunks and rejected == 0
        assert any("Model | Score" in chunk.chunk_text and "Table 1: Scores." in chunk.chunk_text for chunk in chunks)


class TestBackendErrors:
    def test_missing_java_is_reported(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        pytest.importorskip("opendataloader_pdf")
        monkeypatch.setattr(opendataloader_extraction.shutil, "which", lambda name: None)
        with pytest.raises(PdfExtractionError, match="Java"):
            extract_pdf_documents_with_opendataloader([tmp_path / "a.pdf"])

    def test_one_unreadable_output_does_not_stop_the_batch(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        def fake_convert(paths, output_directory, table_method="default"):
            (output_directory / "good.json").write_text(json.dumps({**SAMPLE_TREE, "file name": "good.pdf"}), encoding="utf-8")
            (output_directory / "broken.json").write_text("{not json", encoding="utf-8")
            return {"good.pdf": output_directory / "good.json", "broken.pdf": output_directory / "broken.json"}

        monkeypatch.setattr(opendataloader_extraction, "convert_pdfs_to_opendataloader_json", fake_convert)
        results = extract_pdf_documents_with_opendataloader([tmp_path / name for name in ("good.pdf", "broken.pdf", "missing.pdf")])
        assert results["good.pdf"].title == "A Study of Efficient Parsing"
        assert isinstance(results["broken.pdf"], PdfExtractionError)
        assert isinstance(results["missing.pdf"], PdfExtractionError)


class TestPipelineWiring:
    def run_cli(self, simple_pdf: Path, tmp_path: Path) -> tuple[int, Path]:
        folder = tmp_path / "pdfs"
        folder.mkdir()
        shutil.copy(simple_pdf, folder / "paper.pdf")
        output = tmp_path / "results.json"
        code = command_line.main(
            ["--input", str(folder), "--output", str(output), "--pdf-parser", "opendataloader", "--min-tokens", "1", "--log-level", "ERROR"]
        )
        return code, output

    def test_cli_selects_the_opendataloader_parser(self, monkeypatch: pytest.MonkeyPatch, simple_pdf: Path, tmp_path: Path) -> None:
        calls: list[str] = []

        def fake_extract(paths, table_method="default"):
            calls.append(table_method)
            return {path.name: build_parsed_document_from_opendataloader_json(SAMPLE_TREE, path.name) for path in paths}

        monkeypatch.setattr(chunking_pipeline, "extract_pdf_documents_with_opendataloader", fake_extract)
        code, output = self.run_cli(simple_pdf, tmp_path)
        assert code == 0 and calls == ["cluster"]
        report = json.loads(output.with_name("results.run_report.json").read_text(encoding="utf-8"))
        assert report["config"]["pdf_parser"] == "opendataloader"
        assert report["documents"][0]["extraction"]["parser"] == "opendataloader-pdf"

    def test_unavailable_backend_fails_every_file_cleanly(self, monkeypatch: pytest.MonkeyPatch, simple_pdf: Path, tmp_path: Path) -> None:
        def unavailable(paths, table_method="default"):
            raise PdfExtractionError("OpenDataLoader PDF needs Java 11 or later on the PATH.")

        monkeypatch.setattr(chunking_pipeline, "extract_pdf_documents_with_opendataloader", unavailable)
        code, output = self.run_cli(simple_pdf, tmp_path)
        assert code == 2 and not output.exists()
        report = json.loads(output.with_name("results.run_report.json").read_text(encoding="utf-8"))
        assert "Java" in report["documents"][0]["error"]

    def test_real_backend_on_a_synthetic_pdf(self, simple_pdf: Path, tmp_path: Path) -> None:
        pytest.importorskip("opendataloader_pdf")
        if shutil.which("java") is None:
            pytest.skip("Java is not installed")
        code, output = self.run_cli(simple_pdf, tmp_path)
        assert code == 0
        assert json.loads(output.read_text(encoding="utf-8"))


class TestParserComparison:
    def test_same_chunker_and_questions_for_every_parser(self, simple_pdf: Path) -> None:
        import numpy as np

        from pdf_chunking.parser_comparison import compare_pdf_parsers, tabulate_parser_comparison
        from pdf_chunking.pdf_layout_extraction import PdfExtractionConfig, extract_pdf_document

        def reference_parser(paths):
            return [extract_pdf_document(path, PdfExtractionConfig(table_detection="off")) for path in paths]

        def json_tree_parser(paths):
            return [build_parsed_document_from_opendataloader_json({**SAMPLE_TREE, "file name": p.name}, p.name) for p in paths]

        def constant_scorer(queries, texts):
            return np.ones((len(queries), len(texts)))

        comparison = compare_pdf_parsers(
            paths=[simple_pdf],
            questions=[{"question": "Is parsing hard?", "document": simple_pdf.name, "evidence": [["Parsing PDFs is hard."]]}],
            scorers={"constant": constant_scorer},
            gold_titles={simple_pdf.name: "A Study of Efficient Parsing"},
            parsers={"reference": reference_parser, "json tree": json_tree_parser},
            chunking=ChunkingConfig(min_tokens=1),
        )
        by_parser = {metrics["parser"]: metrics for metrics in comparison["parsers"]}
        assert by_parser["json tree"]["titles_correct"] == "1/1"
        assert by_parser["json tree"]["blocks_by_type"]["table"] == 1
        retrieval = comparison["retrieval"]["constant"]
        assert retrieval["json tree"]["answerable_questions"] == "1/1"
        assert "mrr_gain_vs_reference" in retrieval["json tree"] and "mrr_gain_vs_reference" not in retrieval["reference"]

        rows = tabulate_parser_comparison(comparison)
        assert set(rows[0]) == {"measure", "reference", "json tree"}
        assert rows[-1]["reference"] == "reference"
