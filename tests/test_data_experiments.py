"""Experiments on the training data and one-factor ablation of the LoRA configuration."""

from __future__ import annotations

import pytest

from metadata_extraction.data_experiments import (
    LABELLING_RULES,
    run_chunk_size_experiment,
    run_class_ratio_experiment,
    run_labelling_rule_experiment,
    run_training_corpus_experiment,
)
from metadata_extraction.grounded_training_examples import TrainingExample
from metadata_extraction.hyperparameter_search import ATTENTION_PROJECTIONS, build_config_ablations, run_config_ablation
from metadata_extraction.paper_metadata_schema import PaperMetadata
from pdf_chunking.pdf_layout_extraction import ParsedDocument, TextBlock

TITLE = "Attention Is All You Need"


def example(document: str, index: int, **fields: object) -> TrainingExample:
    return TrainingExample(f"{document}-{index:04d}", document, "some text", PaperMetadata(**fields))


def chunk(text: str, page: int, index: int, kind: str = "text") -> dict:
    return {
        "chunk_id": f"attention-{index:04d}", "source_document": "Attention_Is_All_You_Need.pdf", "chunk_text": text,
        "page_number": page, "section": "", "content_type": kind,
    }  # fmt: skip


@pytest.fixture
def paper() -> ParsedDocument:
    sentences = " ".join(f"Sentence number {n} of the body text explains one more detail." for n in range(120))
    blocks = [
        TextBlock(text=TITLE, block_type="title", page_number=1, bbox=(72, 60, 500, 80)),
        TextBlock(text="Ashish Vaswani and Noam Shazeer", block_type="paragraph", page_number=1, bbox=(72, 90, 500, 100)),
        TextBlock(text="1 Introduction", block_type="heading", page_number=1, bbox=(72, 120, 500, 130), heading_level=1,
                  section_path=("1 Introduction",)),
        TextBlock(text=sentences, block_type="paragraph", page_number=2, bbox=(72, 140, 500, 700),
                  section_path=("1 Introduction",)),
    ]  # fmt: skip
    return ParsedDocument(source_document="Attention_Is_All_You_Need.pdf", title=TITLE, page_count=2, blocks=blocks)


class TestChunkSize:
    def test_smaller_chunks_mean_more_chunks_and_a_lower_informative_share(self, paper: ParsedDocument) -> None:
        results = run_chunk_size_experiment([paper], max_tokens_grid=(100, 400)).set_index("max_tokens")
        assert results.loc[100, "chunks"] > results.loc[400, "chunks"]
        assert results.loc[100, "informative_share"] < results.loc[400, "informative_share"]

    def test_the_title_is_found_at_every_size(self, paper: ParsedDocument) -> None:
        results = run_chunk_size_experiment([paper], max_tokens_grid=(100, 400))
        assert (results["title"] >= 1).all() and (results["informative"] >= 1).all()


class TestLabellingRule:
    def test_the_grounded_rule_has_no_unsupported_label_and_the_naive_one_does(self) -> None:
        chunks = [chunk(f"{TITLE}\nAshish Vaswani", 1, 0), chunk("A paragraph about training.", 5, 1)]
        results = run_labelling_rule_experiment(chunks).set_index("labelling_rule")
        grounded, first_page, everywhere = (results.loc[name] for name in LABELLING_RULES)
        assert grounded["title_labels"] == 1 and grounded["title_labels_unsupported"] == 0.0
        assert first_page["title_labels"] == 1 and first_page["author_labels_unsupported"] > 0.5
        assert everywhere["informative"] == 2 and everywhere["title_labels_unsupported"] == 0.5

    def test_bibliography_chunks_are_never_labelled(self) -> None:
        results = run_labelling_rule_experiment([chunk("[1] A reference.", 9, 0, kind="references")])
        assert (results["informative"] == 0).all()


class TestTrainingCorpus:
    def test_each_batch_of_papers_adds_examples(self) -> None:
        hand_labelled = [example("gold.pdf", 0, title="T"), example("gold.pdf", 1)]
        extra = [example(f"arxiv_{n}.pdf", 0, title=f"Paper {n}", main_authors=["A B"]) for n in range(4)]
        results = run_training_corpus_experiment(hand_labelled, extra, extra_papers_grid=(0, 2, 4, 99))
        assert results["extra_papers"].tolist() == [0, 2, 4]  # 99 papers are not available
        assert results["papers"].tolist() == [1, 3, 5]
        assert results["informative"].tolist() == [1, 3, 5] and results["authors"].tolist() == [0, 2, 4]


class TestClassRatio:
    def test_the_ratio_sets_the_balance_and_the_number_of_passes(self) -> None:
        train = [example("gold.pdf", n, title="T") for n in range(4)] + [example("gold.pdf", 10 + n) for n in range(40)]
        results = run_class_ratio_experiment(train, seed=0, ratios=(1.0, 4.0)).set_index("empty_per_informative")
        assert results.loc[1.0, ["examples", "informative", "empty"]].tolist() == [8, 4, 4]
        assert results.loc[4.0, "informative_share"] == 0.2
        assert results.loc[1.0, "epochs_in_20_steps"] > results.loc[4.0, "epochs_in_20_steps"]


class TestConfigAblation:
    reference = {"lora_rank": 8, "lora_alpha": 16, "learning_rate": 4e-4, "empty_per_informative": 1.0}

    def test_each_variant_changes_one_setting(self) -> None:
        variants = build_config_ablations(self.reference)
        assert next(iter(variants.items())) == ("reference", {})
        assert variants["rank 16"] == {"lora_rank": 16, "lora_alpha": 32}  # alpha keeps its ratio to the rank
        assert "rank 8" not in variants and variants["alpha = rank"] == {"lora_alpha": 8}
        assert variants["attention projections only"] == {"lora_target_modules": ATTENTION_PROJECTIONS}
        assert variants["learning rate / 4"]["learning_rate"] == pytest.approx(1e-4)

    def test_alpha_variant_when_the_reference_already_has_alpha_equal_to_rank(self) -> None:
        variants = build_config_ablations({**self.reference, "lora_alpha": 8})
        assert variants["alpha = 2 x rank"] == {"lora_alpha": 16}

    def test_losses_are_reported_against_the_reference_and_failures_are_kept(self) -> None:
        def objective(overrides: dict) -> float:
            if overrides.get("lora_rank") == 32:
                raise MemoryError("out of memory")
            return 0.5 - 0.01 * overrides.get("lora_rank", 0)

        rows = run_config_ablation(objective, {"reference": {}, "rank 4": {"lora_rank": 4}, "rank 32": {"lora_rank": 32}})
        assert [row["variant"] for row in rows] == ["reference", "rank 4", "rank 32"]
        assert rows[0]["difference_to_reference"] == 0.0 and rows[1]["difference_to_reference"] == pytest.approx(-0.04)
        assert rows[2]["validation_loss"] is None and rows[2]["state"].startswith("failed: MemoryError")
        assert rows[2]["difference_to_reference"] is None and rows[1]["changed"] == "lora_rank=4"
