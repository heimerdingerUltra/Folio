"""Tests for unit building and the chunking strategies."""

from __future__ import annotations

import pytest

from pdf_chunking.chunking_strategies import ChunkingConfig, build_logical_units, chunk_parsed_document, slugify_document_name
from pdf_chunking.text_normalization import split_sentences
from pdf_chunking.pdf_layout_extraction import TextBlock, ParsedDocument
from pdf_chunking.chunk_quality_metrics import measure_sentence_boundary_integrity, measure_table_integrity
from pdf_chunking.token_counting import load_token_counter

SENTENCE = "Chunking quality drives retrieval quality in every pipeline we have measured so far."


def block(text: str, kind: str = "paragraph", page: int = 1, section: tuple[str, ...] = ()) -> TextBlock:
    return TextBlock(text=text, block_type=kind, page_number=page, bbox=(0, 0, 1, 1), section_path=section)


def document(blocks: list[TextBlock]) -> ParsedDocument:
    return ParsedDocument("Test_Doc.pdf", "Test Doc", page_count=3, blocks=blocks)


def long_paragraph(n_sentences: int, tag: str) -> str:
    return " ".join(f"{tag} sentence number {i}. {SENTENCE}" for i in range(n_sentences))


class TestBuildUnits:
    def test_stitches_paragraph_split_by_page_break(self) -> None:
        units = build_logical_units(
            [
                block("The model is trained on a large", page=1),
                block("Figure 1: An unrelated figure.", kind="caption", page=1),
                block("corpus of text. It works well.", page=2),
            ]
        )
        assert [u.text for u in units] == [
            "The model is trained on a large corpus of text. It works well.",
            "Figure 1: An unrelated figure.",
        ]
        assert (units[0].page_start, units[0].page_end) == (1, 2)

    def test_repairs_hyphenation_across_the_break(self) -> None:
        units = build_logical_units([block("This is an approx-"), block("imation of the truth.")])
        assert units[0].text == "This is an approximation of the truth."

    def test_does_not_stitch_complete_paragraphs(self) -> None:
        units = build_logical_units([block("First paragraph."), block("second starts lowercase.")])
        assert len(units) == 2

    def test_does_not_stitch_across_headings(self) -> None:
        units = build_logical_units(
            [
                block("An unfinished thought", section=("1 A",)),
                block("2 B", kind="heading", section=("2 B",)),
                block("continues here.", section=("2 B",)),
            ]
        )
        assert [u.kind for u in units] == ["text", "heading", "text"]

    def test_attaches_equation_and_its_where_clause(self) -> None:
        units = build_logical_units(
            [
                block("We compute the output as:"),
                block("y = Wx + b (1)", kind="equation"),
                block("where W is the weight matrix."),
            ]
        )
        assert len(units) == 1
        assert units[0].text == "We compute the output as:\ny = Wx + b (1)\nwhere W is the weight matrix."

    def test_pairs_table_with_caption(self) -> None:
        units = build_logical_units(
            [
                block("Table 1: Results on the dev set.", kind="caption"),
                block("Model | Acc\nA | 0.9", kind="table"),
            ]
        )
        assert len(units) == 1 and units[0].kind == "table"
        assert units[0].text.startswith("Table 1: Results on the dev set.\nModel | Acc")

    def test_reference_entries_absorb_their_hanging_indent(self) -> None:
        section = ("References",)
        units = build_logical_units(
            [
                block("[1] A. Author. A very long title that", section=section),
                block("Wraps onto a second line. In Proc., 2020.", section=section),
                block("[2] B. Author. Another title. 2021.", section=section),
            ]
        )
        assert len(units) == 2


class TestStructureChunking:
    def chunks(self, blocks: list[TextBlock], **config: int):
        records, rejected = chunk_parsed_document(document(blocks), ChunkingConfig(**config))
        assert rejected == 0
        return records

    def test_respects_token_budget(self) -> None:
        records = self.chunks([block(long_paragraph(60, "Alpha"))], max_tokens=120, overlap_tokens=20)
        assert len(records) > 3
        assert all(record.token_count <= 120 for record in records)

    def test_never_cuts_a_sentence(self) -> None:
        text = long_paragraph(60, "Alpha")
        sentences = set(split_sentences(text))
        records = self.chunks([block(text)], max_tokens=120, overlap_tokens=20)
        for record in records:
            assert set(split_sentences(record.chunk_text)) <= sentences
        assert measure_sentence_boundary_integrity(records)["clean_both_rate"] == 1.0

    def test_no_text_is_lost(self) -> None:
        text = long_paragraph(60, "Alpha")
        records = self.chunks([block(text)], max_tokens=120, overlap_tokens=0)
        assert " ".join(record.chunk_text for record in records) == text

    def test_overlap_repeats_whole_trailing_sentences(self) -> None:
        records = self.chunks([block(long_paragraph(40, "Alpha"))], max_tokens=120, overlap_tokens=40)
        first, second = records[0], records[1]
        carried = split_sentences(second.chunk_text)[0]
        assert first.chunk_text.endswith(carried)

    def test_heading_starts_a_new_chunk_and_sets_the_section(self) -> None:
        records = self.chunks(
            [
                block("1 Intro", kind="heading", section=("1 Intro",)),
                block(long_paragraph(6, "Alpha"), section=("1 Intro",)),
                block("2 Method", kind="heading", page=2, section=("2 Method",)),
                block("2.1 Setup", kind="heading", page=2, section=("2 Method", "2.1 Setup")),
                block(long_paragraph(6, "Beta"), page=2, section=("2 Method", "2.1 Setup")),
            ],
            max_tokens=400,
        )
        assert len(records) == 2
        assert records[0].section == "1 Intro"
        assert records[0].chunk_text.startswith("1 Intro\n\nAlpha")
        # Consecutive headings stay together, on top of the body they introduce.
        assert records[1].chunk_text.startswith("2 Method\n\n2.1 Setup\n\nBeta")
        assert records[1].section == "2 Method > 2.1 Setup"
        assert records[1].page_number == 2
        # No overlap is carried across a section boundary.
        assert "Alpha" not in records[1].chunk_text

    def test_heading_without_body_is_not_glued_to_a_sibling_section(self) -> None:
        records = self.chunks(
            [
                block("A Data", kind="heading", section=("A Data",)),
                block("Table 1: Sizes.\nSplit | N\nTrain | 10", kind="table", section=("A Data",)),
                block("B Setup", kind="heading", section=("B Setup",)),
                block(long_paragraph(6, "Beta"), section=("B Setup",)),
            ]
        )
        text_chunk = next(r for r in records if r.content_type == "text")
        assert text_chunk.chunk_text.startswith("B Setup\n\nBeta")
        assert "A Data" not in text_chunk.chunk_text

    def test_table_is_a_single_chunk_even_mid_paragraph(self) -> None:
        table = block("Model | Acc\nA | 0.9\nB | 0.8", kind="table")
        parsed = document([block("The model is trained on a"), table, block("large corpus. Done.")])
        records, _ = chunk_parsed_document(parsed, ChunkingConfig())
        assert [r.content_type for r in records] == ["text", "table"]
        assert records[0].chunk_text == "The model is trained on a large corpus. Done."
        assert measure_table_integrity(parsed, records) == {"tables": 1, "intact": 1, "intact_rate": 1.0}

    def test_oversized_table_is_split_by_rows_with_header_repeated(self) -> None:
        rows = [f"model-{i} | {i}.5 | {i * 3}.25 | some longer description" for i in range(80)]
        text = "Table 2: A long table.\nModel | A | B | Notes\n" + "\n".join(rows)
        records = self.chunks([block(text, kind="table")], max_tokens=150)
        assert len(records) > 1
        for record in records:
            assert record.token_count <= 150
            assert record.chunk_text.startswith("Table 2: A long table.\nModel | A | B | Notes\n")
        body = "\n".join(r.chunk_text.split("\n", 2)[2] for r in records)
        assert body == "\n".join(rows)  # every row exactly once, in order

    def test_small_chunk_is_merged_into_neighbour(self) -> None:
        records = self.chunks(
            [
                block("1 A", kind="heading", section=("1 A",)),
                block(long_paragraph(5, "Alpha"), section=("1 A",)),
                block("1.1 Tiny", kind="heading", section=("1 A", "1.1 Tiny")),
                block("Just a few words.", section=("1 A", "1.1 Tiny")),
            ],
            max_tokens=400,
            min_tokens=40,
        )
        assert len(records) == 1
        assert records[0].chunk_text.endswith("1.1 Tiny\n\nJust a few words.")

    def test_references_are_tagged(self) -> None:
        section = ("References",)
        records = self.chunks(
            [
                block("References", kind="heading", section=section),
                block("[1] A. Author. " + SENTENCE + " 2020.", section=section),
            ],
            min_tokens=0,
        )
        assert [r.content_type for r in records] == ["references"]

    def test_ids_and_indices_are_stable_and_contiguous(self) -> None:
        blocks = [block(long_paragraph(60, "Alpha"))]
        first = self.chunks(blocks, max_tokens=120)
        second = self.chunks(blocks, max_tokens=120)
        assert [r.chunk_id for r in first] == [f"test_doc-{i:04d}" for i in range(len(first))]
        assert first == second  # deterministic

    def test_unsplittable_text_is_hard_split(self) -> None:
        text = "word " * 600  # 600 tokens, no sentence boundary at all
        records = self.chunks([block(text.strip())], max_tokens=100, overlap_tokens=0)
        assert len(records) >= 6
        assert all(record.token_count <= 100 for record in records)


class TestFixedBaseline:
    def test_windows_fit_budget_and_overlap(self) -> None:
        config = ChunkingConfig(strategy="fixed", max_tokens=100, overlap_tokens=20)
        records, _ = chunk_parsed_document(document([block(long_paragraph(40, "Alpha"))]), config)
        assert len(records) > 3
        assert all(record.token_count <= 100 for record in records)
        # The baseline cuts mid-sentence, which is what the metric must reveal.
        assert measure_sentence_boundary_integrity(records)["clean_both_rate"] < 0.5


class TestConfig:
    @pytest.mark.parametrize(
        "kwargs",
        [{"max_tokens": 10}, {"overlap_tokens": 300}, {"overlap_tokens": -1}, {"min_tokens": 400}],
    )
    def test_rejects_inconsistent_parameters(self, kwargs: dict[str, int]) -> None:
        with pytest.raises(ValueError):
            ChunkingConfig(**kwargs)


def test_document_slug() -> None:
    assert slugify_document_name("Attention_Is_All_You_Need.pdf") == "attention_is_all_you_need"
    assert slugify_document_name("My Report (v2).PDF") == "my_report_v2"
    assert slugify_document_name("---.pdf") == "document"


def test_token_counter_handles_special_token_strings() -> None:
    count = load_token_counter()
    assert count("hello world") >= 2
    assert count("literal <|endoftext|> marker") > 0  # must not raise
