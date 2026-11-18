"""CPU-only tests of the metadata-extraction package (no GPU, no model)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from metadata_extraction.dataset_exploration import (
    plot_dataset_overview,
    summarize_chunks_per_document,
    summarize_label_distribution,
)
from metadata_extraction.experiment_dashboard import plot_experiment_dashboard
from metadata_extraction.extraction_evaluation import build_error_analysis, score_extraction_answers
from metadata_extraction.extraction_prompt import build_chat_messages
from metadata_extraction.gold_paper_metadata import GOLD_PAPER_METADATA
from metadata_extraction.grounded_training_examples import (
    TrainingExample,
    comparison_key,
    downsample_empty_examples,
    label_chunk_with_supported_metadata,
    split_by_document,
)
from metadata_extraction.lora_finetuning_experiment import LoraFinetuningConfig, MetadataExtractionExperiment
from metadata_extraction.paper_metadata_schema import PaperMetadata, parse_model_answer
from metadata_extraction.training_data_validation import (
    TrainingDataValidationError,
    raise_if_any_check_failed,
    validate_training_data,
)

ROOT = Path(__file__).resolve().parents[1]
GOLD = {
    "title": "A Great Paper",
    "authors": ["Ada Lovelace", "Alan Turing"],
    "problem": "Things are slow.",
    "problem_anchors": ["slow", "sequential"],
    "metrics": [{"text": "28.4 BLEU", "evidence": ["28.4", "BLEU"]}],
}


def chunk(text: str, page: int = 1, section: str = "", content_type: str = "text") -> dict:
    return {"chunk_text": text, "page_number": page, "section": section, "content_type": content_type}


def example(document: str, target: PaperMetadata, chunk_id: str = "c") -> TrainingExample:
    return TrainingExample(chunk_id=chunk_id, document=document, chunk_text="text", target=target)


class TestPaperMetadata:
    def test_normalises_blanks_and_duplicates(self) -> None:
        metadata = PaperMetadata(title="  ", main_authors=["Ada", " Ada ", ""], core_problem_statement=" Slow. ")
        assert metadata.title is None
        assert metadata.main_authors == ["Ada"]
        assert metadata.core_problem_statement == "Slow."

    def test_is_empty(self) -> None:
        assert PaperMetadata().is_empty
        assert not PaperMetadata(key_metrics=["28.4 BLEU"]).is_empty

    def test_rejects_wrong_types_and_unknown_keys(self) -> None:
        with pytest.raises(ValidationError):
            PaperMetadata.model_validate({"main_authors": "Ada Lovelace"})
        with pytest.raises(ValidationError):
            PaperMetadata.model_validate({"year": 2017})

    @pytest.mark.parametrize(
        ("raw", "status"),
        [
            ('Sure! {"title": "T", "main_authors": [], "core_problem_statement": null, "key_metrics": []}', "ok"),
            ("no json here", "no_json"),
            ('{"title": "T" "x"}', "invalid_json"),
            ('{"title": "T", "year": 2017}', "schema_error"),
        ],
    )
    def test_parse_model_answer(self, raw: str, status: str) -> None:
        metadata, parsed_status = parse_model_answer(raw)
        assert parsed_status == status
        assert (metadata is not None) == (status == "ok")


class TestGroundedLabels:
    def test_first_page_chunk_gets_title_and_present_authors_only(self) -> None:
        target = label_chunk_with_supported_metadata(chunk("A GREAT  PAPER\nAda Lovelace, Google"), GOLD)
        assert target.title == "A Great Paper"
        assert target.main_authors == ["Ada Lovelace"]
        assert target.core_problem_statement is None and target.key_metrics == []

    def test_title_is_not_labelled_outside_the_first_page(self) -> None:
        target = label_chunk_with_supported_metadata(chunk("As shown in A Great Paper by Ada Lovelace", page=7), GOLD)
        assert target.is_empty

    def test_problem_statement_needs_opening_section_and_all_anchors(self) -> None:
        text = "Sequential models are slow to train."
        assert label_chunk_with_supported_metadata(chunk(text, 2, "1 Introduction"), GOLD).core_problem_statement
        assert not label_chunk_with_supported_metadata(chunk(text, 6, "5 Results"), GOLD).core_problem_statement
        assert not label_chunk_with_supported_metadata(chunk("Slow.", 2, "1 Introduction"), GOLD).core_problem_statement

    def test_metric_needs_all_evidence_and_may_come_from_a_table(self) -> None:
        table = chunk("Model | BLEU\nOurs | 28.4", page=8, content_type="table")
        assert label_chunk_with_supported_metadata(table, GOLD).key_metrics == ["28.4 BLEU"]
        assert label_chunk_with_supported_metadata(chunk("We reach 28.4.", page=8), GOLD).key_metrics == []

    def test_downsampling_keeps_all_informative_examples_and_is_seeded(self) -> None:
        informative = [example("a.pdf", PaperMetadata(title="T"), f"i{i}") for i in range(3)]
        empty = [example("a.pdf", PaperMetadata(), f"e{i}") for i in range(20)]
        first = downsample_empty_examples(informative + empty, empty_per_informative=1.0, seed=7)
        second = downsample_empty_examples(informative + empty, empty_per_informative=1.0, seed=7)
        assert first == second
        assert sum(not e.target.is_empty for e in first) == 3 and len(first) == 6

    def test_split_holds_out_one_whole_document(self) -> None:
        examples = [example("a.pdf", PaperMetadata()), example("b.pdf", PaperMetadata())]
        train, validation = split_by_document(examples, "b.pdf")
        assert {e.document for e in train} == {"a.pdf"} and {e.document for e in validation} == {"b.pdf"}
        with pytest.raises(ValueError):
            split_by_document(examples, "missing.pdf")


def test_prompt_has_user_then_assistant_json() -> None:
    messages = build_chat_messages("Some chunk.", PaperMetadata(title="T"))
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[0]["content"].endswith("### Chunk\nSome chunk.")
    assert PaperMetadata.model_validate_json(messages[1]["content"]).title == "T"
    assert len(build_chat_messages("Some chunk.")) == 1


def test_scoring_of_model_answers() -> None:
    chunk = "A Great Paper by Ada Lovelace and Alan Turing"
    examples = [
        TrainingExample("c1", "a.pdf", chunk, PaperMetadata(title="A Great Paper", main_authors=["Ada Lovelace", "Alan Turing"])),
        TrainingExample("c2", "a.pdf", "Body text.", PaperMetadata()),
        TrainingExample("c3", "a.pdf", "More body text.", PaperMetadata()),
    ]
    answers = [
        '{"title": "a great paper", "main_authors": ["Ada Lovelace", "Bob"], "core_problem_statement": null, "key_metrics": []}',
        '{"title": null, "main_authors": [], "core_problem_statement": null, "key_metrics": []}',
        "I think the title is unknown.",
    ]
    assert score_extraction_answers(examples, answers) == {
        "examples": 3,
        "valid_json_rate": 0.667,
        "title_accuracy": 1.0,
        "author_f1": 0.5,
        "key_metric_recall": None,
        "problem_statement_agreement": 1.0,
        "verifiable_fields_correct_rate": 0.333,
        "fully_correct_rate": 0.333,
        "invented_value_rate": 0.333,  # "Bob" is not in the chunk
        "over_extraction_rate": 0.0,
        "answer_status_counts": {"ok": 2, "no_json": 1},
    }
    assert [row["verdict"] for row in build_error_analysis(examples, answers)] == [
        "1 author(s) missed; 1 wrong author(s); 1 invented value(s)",
        "correct",
        "answer no_json",
    ]


def test_invention_and_over_extraction_are_measured_separately() -> None:
    empty = TrainingExample("c", "a.pdf", "This paragraph reviews prior retrieval work.", PaperMetadata())
    invented = '{"title": "Invented", "main_authors": [], "core_problem_statement": null, "key_metrics": []}'
    summarised = '{"title": null, "main_authors": [], "core_problem_statement": "Reviews prior work.", "key_metrics": []}'

    metrics = score_extraction_answers([empty], [invented])
    assert metrics["invented_value_rate"] == 1.0 and metrics["over_extraction_rate"] == 1.0
    assert build_error_analysis([empty], [invented])[0]["verdict"] == "wrong title; 1 invented value(s)"

    # A problem statement on an unlabelled chunk over-extracts but invents nothing verifiable.
    metrics = score_extraction_answers([empty], [summarised])
    assert metrics["invented_value_rate"] == 0.0 and metrics["over_extraction_rate"] == 1.0
    assert metrics["verifiable_fields_correct_rate"] == 1.0 and metrics["fully_correct_rate"] == 0.0


def test_metrics_are_matched_on_figures_and_grounded_extras_are_not_errors() -> None:
    chunk = "RAG-Seq reaches 44.5 EM on NQ. The model has 626M trainable parameters."
    gold = PaperMetadata(key_metrics=["44.5 Exact Match on Natural Questions (RAG-Sequence)", "State of the art on three open-domain QA tasks"])
    item = TrainingExample("c", "a.pdf", chunk, gold)

    def answer(*metrics: str) -> str:
        return PaperMetadata(key_metrics=list(metrics)).model_dump_json()

    reworded = answer("EM of 44.5 on NQ", "State-of-the-art on three open domain QA tasks", "626M trainable parameters")
    assert score_extraction_answers([item], [reworded])["key_metric_recall"] == 1.0
    assert build_error_analysis([item], [reworded])[0]["verdict"] == "correct"

    wrong_figure = answer("EM of 51.2 on NQ")
    metrics = score_extraction_answers([item], [wrong_figure])
    assert metrics["key_metric_recall"] == 0.0 and metrics["invented_value_rate"] == 1.0


class TestTrainingDataValidation:
    gold = {"a.pdf": GOLD, "b.pdf": GOLD}

    def make(self, document: str, chunk_id: str, informative: bool) -> TrainingExample:
        target = PaperMetadata(title="A Great Paper", main_authors=["Ada Lovelace"]) if informative else PaperMetadata()
        return TrainingExample(chunk_id, document, "A Great Paper by Ada Lovelace", target)

    def failed(self, train: list[TrainingExample], validation: list[TrainingExample]) -> set[str]:
        checks = validate_training_data([*train, *validation], train, validation, self.gold)
        return {check.name for check in checks if not check.passed}

    def test_detects_document_and_chunk_leakage(self) -> None:
        shared = self.make("a.pdf", "a-1", True)
        failed = self.failed([shared, self.make("a.pdf", "a-2", False)], [shared, self.make("a.pdf", "a-3", False)])
        assert "no document shared by train and validation" in failed
        assert "no chunk shared by train and validation" in failed

    def test_detects_ungrounded_label_and_degenerate_balance(self) -> None:
        ungrounded = TrainingExample("a-1", "a.pdf", "nothing relevant here", PaperMetadata(title="A Great Paper"))
        failed = self.failed([ungrounded], [self.make("b.pdf", "b-1", False)])
        assert "titles and authors are grounded in their chunk" in failed
        assert "both classes present in each split" in failed

    def test_raises_only_when_a_check_failed(self) -> None:
        clean = validate_training_data([], [], [], {})
        with pytest.raises(TrainingDataValidationError, match="both classes"):
            raise_if_any_check_failed(clean)
        raise_if_any_check_failed([check for check in clean if check.passed])


@pytest.fixture(scope="module")
def experiment() -> MetadataExtractionExperiment:
    """Data stages of the experiment on the sample dataset."""
    dataset = ROOT / "results.json"
    if not dataset.is_file():
        pytest.skip("results.json not found; run main.py first")
    config = LoraFinetuningConfig(chunk_dataset_path=str(dataset))
    return (
        MetadataExtractionExperiment(config)
        .load_chunk_dataset()
        .build_grounded_examples()
        .split_by_document()
    )


def test_every_gold_field_is_supported_by_some_chunk(experiment: MetadataExtractionExperiment) -> None:
    # A parser or chunker change that breaks an evidence string would
    # otherwise silently remove a label from the training set.
    for document, gold in GOLD_PAPER_METADATA.items():
        targets = [e.target for e in experiment.examples if e.document == document]
        assert sum(bool(t.title) for t in targets) == 1, document
        assert {a for t in targets for a in t.main_authors} == set(gold["authors"]), document
        assert any(t.core_problem_statement for t in targets), document
        assert {m for t in targets for m in t.key_metrics} == {m["text"] for m in gold["metrics"]}, document


def test_labels_are_grounded_and_split_does_not_leak(experiment: MetadataExtractionExperiment) -> None:
    for item in experiment.examples:
        text = comparison_key(item.chunk_text)
        assert all(comparison_key(author) in text for author in item.target.main_authors)
    train = {e.document for e in experiment.train_examples}
    validation = {e.document for e in experiment.validation_examples}
    test = {e.document for e in experiment.test_examples}
    assert validation == {experiment.config.validation_document} and test == {experiment.config.test_document}
    assert not train & (validation | test) and not validation & test
    summary = experiment.describe_dataset()
    assert summary["train"]["informative"] >= summary["train"]["empty"] > 0
    assert 0 < len(experiment.evaluation_examples) <= experiment.config.max_evaluation_examples


def test_data_checks_pass_on_the_sample_dataset(experiment: MetadataExtractionExperiment) -> None:
    experiment.validate_training_data()
    assert len(experiment.validation_checks) == 10 and all(check.passed for check in experiment.validation_checks)


def test_exploration_tables_and_figures(experiment: MetadataExtractionExperiment) -> None:
    per_document = summarize_chunks_per_document(experiment.chunks)
    assert len(per_document) == 5 and per_document["chunks"].sum() == len(experiment.chunks)
    splits = experiment.splits
    distribution = summarize_label_distribution(splits).set_index("split")
    assert distribution.loc["train", "examples"] == len(experiment.train_examples)
    assert distribution.loc["validation", "documents"] == 1
    figure = plot_dataset_overview(experiment.chunks, splits)
    assert [title.text for title in figure.layout.annotations] == [
        "Chunk length by content type", "Chunks per document", "Examples supporting each field",
    ]  # fmt: skip
    assert {trace.name for trace in figure.data if trace.legendgroup == "split"} == set(splits)


def test_experiment_dashboard_renders_from_recorded_results() -> None:
    loss_history = [{"step": 1, "loss": 0.98}, {"step": 10, "loss": 0.4}, {"step": 10, "eval_loss": 0.53}]
    evaluations = {
        "before fine-tuning": {"valid_json_rate": 0.25, "author_f1": 0.0, "answer_status_counts": {"schema_error": 6, "ok": 2}},
        "after fine-tuning": {"valid_json_rate": 1.0, "author_f1": 0.96, "answer_status_counts": {"ok": 8}},
    }
    figure = plot_experiment_dashboard(loss_history, evaluations, {"train_runtime_seconds": 208}, "Dashboard")
    assert {trace.type for trace in figure.data} == {"scatter", "bar", "table"}
    assert [trace.name for trace in figure.data if getattr(trace, "legendgroup", None) == "evaluation"] == list(evaluations)
    assert "train runtime seconds" in figure.data[-1].cells.values[0]


def test_overfitting_is_flagged_when_validation_loss_rebounds() -> None:
    run = MetadataExtractionExperiment()
    assert run.diagnose_overfitting() == {}
    run.loss_history = [{"step": 10, "eval_loss": 0.47}, {"step": 20, "eval_loss": 0.48}, {"step": 40, "eval_loss": 0.65}]
    assert run.diagnose_overfitting() == {"best_validation_loss": 0.47, "best_validation_step": 10, "overfitting": True}
    run.loss_history = [{"step": 10, "eval_loss": 0.53}, {"step": 20, "eval_loss": 0.45}]
    assert run.diagnose_overfitting()["overfitting"] is False
