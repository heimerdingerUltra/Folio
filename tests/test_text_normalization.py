"""Tests for text normalisation, de-hyphenation and sentence splitting."""

from __future__ import annotations

import pytest

from pdf_chunking.text_normalization import count_document_words, join_wrapped_lines, normalize_pdf_text, split_sentences


class TestNormalizeText:
    def test_expands_ligatures(self) -> None:
        assert normalize_pdf_text("eﬃcient ﬁne-tuning") == "efficient fine-tuning"

    def test_removes_invisible_characters_and_collapses_spaces(self) -> None:
        assert normalize_pdf_text("a  b­​c   d\x00") == "a bc d"

    def test_removes_space_before_punctuation(self) -> None:
        assert normalize_pdf_text("accessed , 2026 .") == "accessed, 2026."

    def test_keeps_decimals_and_newlines(self) -> None:
        assert normalize_pdf_text("lr = .5 \n next") == "lr = .5\nnext"

    def test_does_not_rewrite_math_symbols(self) -> None:
        # NFKC would turn the double-struck R into a plain "R".
        assert normalize_pdf_text("W ∈ ℝ") == "W ∈ ℝ"


class TestJoinLines:
    def test_removes_line_break_hyphen(self) -> None:
        assert join_wrapped_lines(["In RAG applica-", "tions, retrieval matters."]) == (
            "In RAG applications, retrieval matters."
        )

    def test_keeps_hyphen_of_known_compound(self) -> None:
        vocabulary = count_document_words(["we use byte-pair encoding", "byte-pair merges"])
        assert join_wrapped_lines(["the byte-", "pair encoding"], vocabulary) == "the byte-pair encoding"

    def test_joins_when_only_the_joined_form_is_known(self) -> None:
        vocabulary = count_document_words(["ChunkNorris is fast", "ChunkNorris wins"])
        assert join_wrapped_lines(["we call it Chun-", "kNorris."], vocabulary) == "we call it ChunkNorris."

    def test_keeps_hyphen_before_digit_or_capital(self) -> None:
        assert join_wrapped_lines(["we fine-tune GPT-", "3 on it"]) == "we fine-tune GPT-3 on it"
        assert join_wrapped_lines(["the Kneser-", "Ney estimator"]) == "the Kneser-Ney estimator"

    def test_joins_all_caps_title(self) -> None:
        assert join_wrapped_lines(["ADAPTATION OF LARGE LAN-", "GUAGE MODELS"]) == (
            "ADAPTATION OF LARGE LANGUAGE MODELS"
        )

    def test_rejoins_wrapped_url(self) -> None:
        assert join_wrapped_lines(["see https://github.com/org/bench-chunk", "norris-acl2025"]) == (
            "see https://github.com/org/bench-chunknorris-acl2025"
        )
        assert join_wrapped_lines(["URL https:", "//example.org/paper."]) == "URL https://example.org/paper."

    def test_does_not_glue_next_word_onto_url(self) -> None:
        assert join_wrapped_lines(["see https://example.org/a", "for details"]) == (
            "see https://example.org/a for details"
        )

    def test_skips_empty_lines(self) -> None:
        assert join_wrapped_lines(["", "one", "  ", "two"]) == "one two"


class TestSplitSentences:
    def test_basic_split(self) -> None:
        assert split_sentences("First one. Second one! Third one?") == [
            "First one.",
            "Second one!",
            "Third one?",
        ]

    @pytest.mark.parametrize(
        "text",
        [
            "We follow Vaswani et al. (2017) in this work.",
            "Results are shown in Fig. 3 and Sec. 4 of the paper.",
            "Some optimizers, e.g. Adam, converge faster.",
            "Training took 3.5 days on 8 GPUs.",
            "The work of J. Smith is related.",
            "This was shown by Lewis et al. and later confirmed.",
        ],
    )
    def test_does_not_split_inside_a_sentence(self, text: str) -> None:
        assert split_sentences(text) == [text]

    def test_keeps_citation_with_its_sentence(self) -> None:
        assert split_sentences("Dropout helps [33]. We use a rate of 0.1.") == [
            "Dropout helps [33].",
            "We use a rate of 0.1.",
        ]

    def test_newline_separated_lines_are_never_merged(self) -> None:
        text = "We compute the output as:\nAttention(Q, K, V) = softmax(QK / d)V (1)\nwhere d is a scale."
        assert split_sentences(text) == [
            "We compute the output as:",
            "Attention(Q, K, V) = softmax(QK / d)V (1)",
            "where d is a scale.",
        ]

    def test_empty_input(self) -> None:
        assert split_sentences("") == []
        assert split_sentences("  \n ") == []
