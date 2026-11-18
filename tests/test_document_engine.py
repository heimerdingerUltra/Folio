"""OCR, figures, document structure, parallel parsing and arXiv-labelled training data."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pymupdf
import pytest

from metadata_extraction.arxiv_training_examples import (
    build_arxiv_training_examples,
    label_chunk_with_arxiv_metadata,
    opening_sentences_found_in,
    quotes_a_result_figure,
)
from metadata_extraction.document_metadata_extraction import (
    extract_document_metadata,
    merge_chunk_metadata,
    parse_author_names,
    select_chunks_worth_reading,
)
from metadata_extraction.grounded_training_examples import comparison_key
from metadata_extraction.lora_finetuning_experiment import LoraFinetuningConfig, MetadataExtractionExperiment
from metadata_extraction.paper_metadata_schema import PaperMetadata
from pdf_chunking import command_line
from pdf_chunking.arxiv_papers import TRAINING_CORPUS_ARXIV_IDS, parse_arxiv_api_response
from pdf_chunking.document_structure import build_document_structure, render_document_markdown
from pdf_chunking.figure_extraction import match_figures_to_captions
from pdf_chunking.page_ocr import OcrLine, estimate_font_size, group_lines_into_paragraphs, is_ocr_installed
from pdf_chunking.pdf_layout_extraction import (
    NoTextLayerError,
    ParsedDocument,
    PdfExtractionConfig,
    TextBlock,
    extract_pdf_document,
)

FAST = PdfExtractionConfig(table_detection="heuristic")

ARXIV_ANSWER = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1810.04805v2</id>
    <title>BERT: Pre-training of Deep
  Bidirectional Transformers</title>
    <summary>  We introduce a new language representation model.
It obtains new results. </summary>
    <author><name>Jacob Devlin</name></author>
    <author><name>Ming-Wei Chang</name></author>
  </entry>
</feed>"""


def block(text: str, kind: str = "paragraph", page: int = 1, level: int = 0, y: float = 100.0) -> TextBlock:
    return TextBlock(
        text=text, block_type=kind, page_number=page, bbox=(72.0, y, 500.0, y + 12), lines=text.split("\n"),
        heading_level=level,
    )  # fmt: skip


def chunk(text: str, page: int = 1, section: str = "", kind: str = "text", index: int = 0) -> dict:
    return {
        "chunk_id": f"paper-{index:04d}", "source_document": "paper.pdf", "chunk_text": text,
        "page_number": page, "section": section, "content_type": kind,
    }  # fmt: skip


@pytest.fixture
def paper_document() -> ParsedDocument:
    blocks = [
        block("A Study of Things", "title", y=60),
        block("Ada Lovelace∗, Alan M. Turing and Grace Hopper\nUniversity of Somewhere\nada@somewhere.org", y=90),
        block("Abstract", "heading", level=1, y=130),
        block("Existing systems are slow. We ask whether they can be faster. Our method reaches 91.5% accuracy.", y=150),
        block("1 Introduction", "heading", level=1, y=200),
        block("Some context.", y=220),
        block("1.1 Scope", "heading", level=2, y=250),
        block("Figure 1: The system.", "caption", y=400),
        block("2 Method", "heading", level=1, page=2),
        block("a b\n1 2", "table", page=2, y=200),
    ]
    return ParsedDocument(source_document="paper.pdf", title="A Study of Things", page_count=2, blocks=blocks)


class TestArxivCorpus:
    def test_api_answer_is_parsed_and_whitespace_collapsed(self) -> None:
        [paper] = parse_arxiv_api_response(ARXIV_ANSWER)
        assert paper["arxiv_id"] == "1810.04805"
        assert paper["title"] == "BERT: Pre-training of Deep Bidirectional Transformers"
        assert paper["authors"] == ["Jacob Devlin", "Ming-Wei Chang"]
        assert paper["abstract"].startswith("We introduce") and "\n" not in paper["abstract"]

    def test_training_corpus_excludes_the_evaluation_papers(self) -> None:
        evaluation_ids = {"1706.03762", "2302.13971", "2106.09685", "2005.11401", "2602.00010"}
        assert not evaluation_ids & set(TRAINING_CORPUS_ARXIV_IDS)
        assert len(set(TRAINING_CORPUS_ARXIV_IDS)) == len(TRAINING_CORPUS_ARXIV_IDS)


class TestArxivLabels:
    paper = {
        "title": "A Study of Things",
        "authors": ["Ada Lovelace", "Alan Turing"],
        "abstract": "Existing systems are slow. We ask whether they can be faster. Results follow.",
    }

    def test_first_page_chunk_gets_title_and_the_authors_it_contains(self) -> None:
        target = label_chunk_with_arxiv_metadata(chunk("A Study of Things\nAda Lovelace, Bob"), self.paper)
        assert target.title == "A Study of Things" and target.main_authors == ["Ada Lovelace"]

    def test_title_is_not_labelled_after_the_first_page(self) -> None:
        target = label_chunk_with_arxiv_metadata(chunk("A Study of Things again", page=3), self.paper)
        assert target.is_empty

    def test_problem_statement_is_the_abstract_opening_found_verbatim(self) -> None:
        text = comparison_key("Abstract existing systems are slow. We ask whether they can be faster. More.")
        assert opening_sentences_found_in(self.paper["abstract"], text) == (
            "Existing systems are slow. We ask whether they can be faster."
        )
        assert opening_sentences_found_in(self.paper["abstract"], "unrelated text") is None

    def test_chunks_with_unknown_metric_labels_are_left_out(self) -> None:
        assert quotes_a_result_figure("accuracy of 91.5") and quotes_a_result_figure("up 3 %")
        assert not quotes_a_result_figure("in 2019, across 12 layers")
        assert label_chunk_with_arxiv_metadata(chunk("We reach 91.5 points."), self.paper) is None
        assert label_chunk_with_arxiv_metadata(chunk("a b", kind="table"), self.paper) is None
        assert label_chunk_with_arxiv_metadata(chunk("[1] A. Author", kind="references"), self.paper) is None

    def test_only_papers_with_a_record_are_labelled(self) -> None:
        other = {**chunk("text"), "source_document": "unknown.pdf"}
        examples = build_arxiv_training_examples([chunk("A Study of Things"), other], {"paper.pdf": self.paper})
        assert [example.document for example in examples] == ["paper.pdf"]

    def test_extra_papers_go_to_the_training_split_only(self, tmp_path: Path) -> None:
        chunks_path, metadata_path = tmp_path / "chunks.json", tmp_path / "arxiv_metadata.json"
        chunks_path.write_text(json.dumps([chunk("A Study of Things\nAda Lovelace")]), encoding="utf-8")
        metadata_path.write_text(json.dumps({"paper.pdf": self.paper}), encoding="utf-8")
        config = LoraFinetuningConfig(arxiv_chunk_dataset_path=str(chunks_path), arxiv_metadata_path=str(metadata_path))
        experiment = (
            MetadataExtractionExperiment(config)
            .load_chunk_dataset()
            .build_grounded_examples()
            .add_arxiv_training_examples()
            .split_by_document()
        )
        assert "paper.pdf" in {example.document for example in experiment.train_examples}
        held_out = experiment.validation_examples + experiment.test_examples
        assert "paper.pdf" not in {example.document for example in held_out}
        assert experiment.arxiv_label_coverage["papers_with_title"] == 1

    def test_an_evaluation_paper_cannot_be_added_as_training_data(self, tmp_path: Path) -> None:
        chunks_path, metadata_path = tmp_path / "chunks.json", tmp_path / "arxiv_metadata.json"
        chunks_path.write_text("[]", encoding="utf-8")
        metadata_path.write_text(json.dumps({"RAG.pdf": self.paper}), encoding="utf-8")
        config = LoraFinetuningConfig(arxiv_chunk_dataset_path=str(chunks_path), arxiv_metadata_path=str(metadata_path))
        experiment = MetadataExtractionExperiment(config).load_chunk_dataset().build_grounded_examples()
        with pytest.raises(ValueError, match="overlap"):
            experiment.add_arxiv_training_examples()


class TestOcrLayout:
    @staticmethod
    def line(text: str, x0: float, y0: float, width: float = 200.0, height: float = 12.0) -> OcrLine:
        return OcrLine(text=text, bbox=(x0, y0, x0 + width, y0 + height), confidence=0.99)

    def test_consecutive_lines_form_one_paragraph(self) -> None:
        lines = [self.line("first", 72, 100), self.line("second", 72, 114), self.line("far below", 72, 200)]
        assert [[l.text for l in p] for p in group_lines_into_paragraphs(lines)] == [["first", "second"], ["far below"]]

    def test_columns_are_not_mixed(self) -> None:
        lines = [self.line("left 1", 72, 100), self.line("right 1", 320, 100), self.line("left 2", 72, 114)]
        paragraphs = [[l.text for l in p] for p in group_lines_into_paragraphs(lines)]
        assert ["left 1", "left 2"] in paragraphs and ["right 1"] in paragraphs

    def test_a_larger_line_starts_a_new_paragraph(self) -> None:
        lines = [self.line("Heading", 72, 100, height=20), self.line("body", 72, 124)]
        assert len(group_lines_into_paragraphs(lines)) == 2

    def test_font_size_is_estimated_from_line_height(self) -> None:
        assert estimate_font_size([self.line("a", 0, 0, height=12.5)]) == 10.0

    def test_ocr_off_reports_a_scanned_pdf(self, image_only_pdf: Path) -> None:
        with pytest.raises(NoTextLayerError):
            extract_pdf_document(image_only_pdf, PdfExtractionConfig(table_detection="heuristic", ocr="off"))

    @pytest.mark.skipif(not is_ocr_installed(), reason="rapidocr is not installed")
    def test_a_scanned_page_is_read(self, simple_pdf: Path, tmp_path: Path) -> None:
        scanned_path = tmp_path / "scanned.pdf"
        with pymupdf.open(simple_pdf) as source, pymupdf.open() as scanned:
            for page in source:
                target = scanned.new_page(width=page.rect.width, height=page.rect.height)
                target.insert_image(target.rect, pixmap=page.get_pixmap(dpi=200))
            scanned.save(scanned_path)
        native = extract_pdf_document(simple_pdf, FAST)
        read = extract_pdf_document(scanned_path, FAST)
        assert read.stats["pages_read_by_ocr"] == native.page_count
        assert all(block.from_ocr for block in read.blocks)
        expected = set(comparison_key(" ".join(block.text for block in native.blocks)).split())
        found = set(comparison_key(" ".join(block.text for block in read.blocks)).split())
        assert len(expected & found) / len(expected) >= 0.9


class TestFigures:
    def test_a_graphic_above_a_figure_caption_is_matched(self) -> None:
        captions = [(2, (100.0, 300.0, 400.0, 320.0), "Figure 3: Training curves."), (2, (0, 500, 9, 9), "Table 1: x")]
        regions = {2: [(120.0, 150.0, 380.0, 290.0), (120.0, 600.0, 380.0, 700.0)]}
        [figure] = match_figures_to_captions(captions, regions)
        assert figure.label == "Figure 3" and figure.page_number == 2
        assert figure.bbox[1] <= 150.0 and 290.0 <= figure.bbox[3] <= 300.0

    def test_a_distant_graphic_is_not_a_figure(self) -> None:
        captions = [(1, (100.0, 500.0, 400.0, 520.0), "Figure 1: Far away.")]
        assert match_figures_to_captions(captions, {1: [(120.0, 50.0, 380.0, 150.0)]}) == []

    def test_side_by_side_parts_form_one_figure(self) -> None:
        captions = [(1, (100.0, 300.0, 500.0, 320.0), "Figure 2: Two panels.")]
        regions = {1: [(100.0, 150.0, 280.0, 290.0), (320.0, 160.0, 500.0, 285.0)]}
        [figure] = match_figures_to_captions(captions, regions)
        assert figure.bbox[0] <= 100.0 and figure.bbox[2] >= 500.0


class TestDocumentStructure:
    def test_blocks_are_nested_under_their_headings(self, paper_document: ParsedDocument) -> None:
        structure = build_document_structure(paper_document)
        titles = [section["title"] for section in structure["sections"]]
        assert titles == ["", "Abstract", "1 Introduction", "2 Method"]
        introduction = structure["sections"][2]
        assert [child["title"] for child in introduction["sections"]] == ["1.1 Scope"]
        assert introduction["sections"][0]["blocks"][0]["type"] == "caption"

    def test_reading_order_is_kept(self, paper_document: ParsedDocument) -> None:
        def orders(section: dict) -> list[int]:
            own = [record["reading_order"] for record in section["blocks"]]
            return own + [order for child in section["sections"] for order in orders(child)]

        flat = [order for section in build_document_structure(paper_document)["sections"] for order in orders(section)]
        assert flat == sorted(flat)

    def test_markdown_keeps_heading_depth_and_fences_tables(self, paper_document: ParsedDocument) -> None:
        markdown = render_document_markdown(paper_document)
        assert markdown.startswith("# A Study of Things\n")
        assert "## 1 Introduction" in markdown and "### 1.1 Scope" in markdown
        assert "```\na b\n1 2\n```" in markdown and "*Figure 1: The system.*" in markdown


class TestPipelineOptions:
    @pytest.fixture
    def input_dir(self, tmp_path: Path, simple_pdf: Path, two_column_pdf: Path) -> Path:
        folder = tmp_path / "pdfs"
        folder.mkdir()
        shutil.copy(simple_pdf, folder / "simple.pdf")
        shutil.copy(two_column_pdf, folder / "two_column.pdf")
        return folder

    @staticmethod
    def run(*args: object) -> int:
        return command_line.main([str(a) for a in args] + ["--table-detection", "heuristic", "--log-level", "ERROR"])

    def test_parallel_parsing_gives_the_same_dataset(self, input_dir: Path, tmp_path: Path) -> None:
        sequential, parallel = tmp_path / "sequential.json", tmp_path / "parallel.json"
        assert self.run("--input", input_dir, "--output", sequential) == 0
        assert self.run("--input", input_dir, "--output", parallel, "--workers", 2) == 0
        assert sequential.read_bytes() == parallel.read_bytes()

    def test_structure_export_writes_json_and_markdown(self, input_dir: Path, tmp_path: Path) -> None:
        output = tmp_path / "results.json"
        assert self.run("--input", input_dir, "--output", output, "--export-structure") == 0
        folder = tmp_path / "results.documents"
        structure = json.loads((folder / "simple.json").read_text(encoding="utf-8"))
        assert structure["source_document"] == "simple.pdf" and structure["sections"]
        assert (folder / "two_column.md").read_text(encoding="utf-8").startswith("# ")


class TestDocumentMetadata:
    @pytest.mark.parametrize(
        ("lines", "expected"),
        [
            (["Ada Lovelace∗, Alan M. Turing and Grace Hopper"], ["Ada Lovelace", "Alan M. Turing", "Grace Hopper"]),
            (["Jacob Devlin Ming-Wei Chang Kenton Lee Kristina Toutanova"],
             ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"]),
            (["Colin Raffel∗ craffel@gmail.com"], ["Colin Raffel"]),
            (["Zhenzhong Lan^1 Mingda Chen^2∗"], ["Zhenzhong Lan", "Mingda Chen"]),
            (["Google Research", "{ada, alan}@somewhere.org", "University of Somewhere"], []),
        ],
    )  # fmt: skip
    def test_author_names(self, lines: list[str], expected: list[str]) -> None:
        assert parse_author_names(lines) == expected

    def test_fields_and_their_evidence(self, paper_document: ParsedDocument) -> None:
        result = extract_document_metadata(paper_document)
        assert result.metadata.title == "A Study of Things"
        assert result.metadata.main_authors == ["Ada Lovelace", "Alan M. Turing", "Grace Hopper"]
        assert result.metadata.core_problem_statement == "Existing systems are slow. We ask whether they can be faster."
        assert result.metadata.key_metrics == ["Our method reaches 91.5% accuracy."]
        metric = next(item for item in result.evidence if item.field == "key_metrics")
        assert paper_document.blocks[metric.reading_order].text.endswith(metric.value) and metric.page_number == 1

    def test_routing_drops_the_bibliography_and_plain_body_chunks(self) -> None:
        chunks = [
            chunk("front matter", page=1, index=0),
            chunk("plain method text", page=4, section="3 Method", index=1),
            chunk("we obtain 28.4 BLEU", page=7, section="3 Method", index=2),
            chunk("closing remarks", page=9, section="7 Conclusion", index=3),
            chunk("[1] Someone. 28.4", page=10, kind="references", index=4),
        ]
        assert [c["chunk_id"] for c in select_chunks_worth_reading(chunks)] == ["paper-0000", "paper-0002", "paper-0003"]

    def test_chunk_answers_are_merged_in_reading_order(self) -> None:
        merged = merge_chunk_metadata(
            [
                PaperMetadata(title="A Study of Things", main_authors=["Ada Lovelace"]),
                PaperMetadata(title="Wrong", core_problem_statement="Systems are slow.", key_metrics=["91.5%"]),
                PaperMetadata(title="A Study of Things", main_authors=["Ada Lovelace", "Alan Turing"], key_metrics=["91.5%"]),
            ]
        )
        assert merged.title == "A Study of Things" and merged.main_authors == ["Ada Lovelace", "Alan Turing"]
        assert merged.core_problem_statement == "Systems are slow." and merged.key_metrics == ["91.5%"]
