"""Tests of the statistics, guards, baselines, search, tracking and selection logic."""

from __future__ import annotations

import json
import os
import sys
import types

import pytest

from metadata_extraction.approach_comparison import build_approach_leaderboard, select_best_approach
from metadata_extraction.behavioural_tests import (
    DISTRACTOR_SENTENCE,
    perturb_chunk_without_changing_its_meaning,
    score_abstention_on_bibliography,
    score_invariance,
    select_bibliography_chunks,
    select_invariance_cases,
)
from metadata_extraction.experiment_dashboard import plot_approach_leaderboard
from metadata_extraction.experiment_tracking import ExperimentTracker, create_experiment_tracker
from metadata_extraction.extraction_evaluation import score_extraction_answers
from metadata_extraction.extraction_prompt import build_few_shot_messages, select_few_shot_demonstrations
from metadata_extraction.grounded_training_examples import (
    TrainingExample,
    split_train_validation_test_by_document,
)
from metadata_extraction.hallucination_guards import (
    ANSWER_GUARDS,
    remove_ungrounded_fields,
    repair_json_and_remove_ungrounded_fields,
    repair_json_answer,
)
from metadata_extraction.hyperparameter_search import search_hyperparameters
from metadata_extraction.paper_metadata_schema import PaperMetadata, parse_model_answer
from metadata_extraction.rule_based_extractor import extract_metadata_with_rules
from metadata_extraction.statistical_tests import (
    bootstrap_interval,
    exact_mcnemar_test,
    kolmogorov_smirnov_statistic,
    permutation_test_of_distribution_shift,
    sample_size_for_margin,
    wilson_interval,
)

EMPTY_JSON = PaperMetadata().model_dump_json()
FRONT_PAGE = "A Great Paper\nAda Lovelace ada@lab.org\nAlan Turing University of Cambridge"
TITLE_TARGET = PaperMetadata(title="A Great Paper", main_authors=["Ada Lovelace", "Alan Turing"])


def example(chunk_id: str, text: str, target: PaperMetadata, document: str = "a.pdf") -> TrainingExample:
    return TrainingExample(chunk_id, document, text, target)


class TestStatisticalTests:
    def test_wilson_interval_stays_in_bounds_at_the_extremes(self) -> None:
        assert wilson_interval(0, 12) == (0.0, 0.243)
        assert wilson_interval(12, 12) == (0.757, 1.0)
        assert wilson_interval(0, 0) == (0.0, 1.0)
        low, high = wilson_interval(6, 12)
        assert low < 0.5 < high

    def test_mcnemar_uses_only_discordant_pairs(self) -> None:
        strong = exact_mcnemar_test([False] * 12, [True] * 11 + [False])
        assert strong == {"candidate_only": 11, "reference_only": 0, "p_value": 0.001}
        assert exact_mcnemar_test([True, False], [False, True])["p_value"] == 1.0
        assert exact_mcnemar_test([True, True], [True, True])["p_value"] == 1.0
        # 2 improvements out of 2 discordant pairs cannot be significant.
        assert exact_mcnemar_test([False, False, True], [True, True, True])["p_value"] == 0.5

    def test_distribution_shift_is_detected_only_when_present(self) -> None:
        same = permutation_test_of_distribution_shift([1, 2, 3, 4, 5] * 6, [1, 2, 3, 4, 5] * 3, resamples=300)
        shifted = permutation_test_of_distribution_shift(list(range(30)), list(range(50, 70)), resamples=300)
        assert same["p_value"] > 0.5 and shifted["p_value"] < 0.01
        assert kolmogorov_smirnov_statistic([1, 2], [3, 4]) == 1.0

    def test_bootstrap_and_sample_size(self) -> None:
        low, high = bootstrap_interval([0, 1, 1, 1, 0, 1], resamples=500)
        assert 0 <= low < 0.667 < high <= 1
        assert bootstrap_interval([0.5] * 10, resamples=100) == (0.5, 0.5)
        assert sample_size_for_margin(0.10) == 97 and sample_size_for_margin(0.05) == 385


class TestHallucinationGuards:
    def test_repair_recovers_json_from_prose_and_drops_unknown_keys(self) -> None:
        raw = 'Sure, here it is:\n```json\n{"title": "T", "main_authors": "Ada", "year": 2017, "key_metrics": null}\n```'
        assert parse_model_answer(raw)[1] == "schema_error"
        metadata, status = parse_model_answer(repair_json_answer(raw))
        assert status == "ok" and metadata == PaperMetadata(title="T", main_authors=["Ada"])

    def test_repair_leaves_unsalvageable_answers_alone(self) -> None:
        for raw in ("no json at all", '{"title": "T" "broken"}', "[1, 2]"):
            assert repair_json_answer(raw) == raw

    def test_grounding_filter_removes_only_what_the_chunk_does_not_contain(self) -> None:
        predicted = PaperMetadata(
            title="A Great Paper",
            main_authors=["Ada Lovelace", "Grace Hopper"],
            core_problem_statement="Abstractive, cannot be string-checked.",
            key_metrics=["28.4 BLEU", "99.9 accuracy", "state of the art"],
        )
        chunk = "a great   paper by ADA LOVELACE reaches 28.4 BLEU"
        assert remove_ungrounded_fields(predicted, chunk) == PaperMetadata(
            title="A Great Paper",
            main_authors=["Ada Lovelace"],
            core_problem_statement="Abstractive, cannot be string-checked.",
            key_metrics=["28.4 BLEU", "state of the art"],
        )

    def test_guard_turns_a_hallucination_into_a_correct_abstention(self) -> None:
        item = example("c", "Nothing about people here.", PaperMetadata())
        raw = '{"title": "Invented", "main_authors": ["Nobody Real"], "core_problem_statement": null, "key_metrics": []}'
        assert score_extraction_answers([item], [raw])["invented_value_rate"] == 1.0
        guarded = repair_json_and_remove_ungrounded_fields(raw, item.chunk_text)
        after = score_extraction_answers([item], [guarded])
        assert after["invented_value_rate"] == 0.0 and after["over_extraction_rate"] == 0.0
        assert set(ANSWER_GUARDS) == {"no guard", "JSON repair", "JSON repair + grounding filter"}


class TestBaselinesAndPrompts:
    def test_rules_find_title_and_authors_on_front_matter_only(self) -> None:
        front = PaperMetadata.model_validate_json(extract_metadata_with_rules(FRONT_PAGE))
        assert front.title == "A Great Paper" and {"Ada Lovelace", "Alan Turing"} <= set(front.main_authors)
        body = PaperMetadata.model_validate_json(extract_metadata_with_rules("We trained it. It reached 28.4 BLEU."))
        assert body.title is None and body.main_authors == [] and body.key_metrics == ["It reached 28.4 BLEU."]
        assert PaperMetadata.model_validate_json(extract_metadata_with_rules("Plain prose, nothing else.")).is_empty

    def test_few_shot_prompt_shows_one_informative_and_one_empty_example(self) -> None:
        train = [
            example("long", FRONT_PAGE + " padding" * 200, TITLE_TARGET),
            example("short", FRONT_PAGE, TITLE_TARGET),
            example("empty", "Body text.", PaperMetadata()),
        ]
        demonstrations = select_few_shot_demonstrations(train, max_chunk_characters=20)
        assert [d.chunk_id for d in demonstrations] == ["short", "empty"]
        assert all(len(d.chunk_text) <= 20 for d in demonstrations)
        messages = build_few_shot_messages("Query chunk.", demonstrations)
        assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant", "user"]
        assert messages[-1]["content"].endswith("Query chunk.") and messages[3]["content"] == EMPTY_JSON


class TestHyperparameterSearch:
    start = {"lora_rank": 16, "lora_alpha": 16, "learning_rate": 2e-4, "empty_per_informative": 1.0}

    def test_starting_point_is_evaluated_first_and_a_crash_is_survived(self) -> None:
        seen: list[dict] = []

        def objective(overrides: dict) -> float:
            seen.append(overrides)
            if len(seen) == 2:
                raise MemoryError("simulated out-of-memory")
            return 1.0 / overrides["lora_rank"]

        result = search_hyperparameters(objective, self.start, n_trials=5, seed=0)
        assert seen[0] == self.start and len(result.trials) == 5
        assert result.trials[1]["state"].startswith("failed: MemoryError")
        completed = [t for t in result.trials if t["validation_loss"] is not None]
        assert result.best_validation_loss == min(t["validation_loss"] for t in completed)
        assert result.best_overrides["lora_alpha"] % result.best_overrides["lora_rank"] == 0

    def test_search_is_reproducible_and_keeps_nothing_when_everything_fails(self) -> None:
        def objective(overrides: dict) -> float:
            return overrides["learning_rate"]

        first = search_hyperparameters(objective, self.start, n_trials=4, seed=3)
        second = search_hyperparameters(objective, self.start, n_trials=4, seed=3)
        assert [t["learning_rate"] for t in first.trials] == [t["learning_rate"] for t in second.trials]

        def always_fails(overrides: dict) -> float:
            raise RuntimeError("boom")

        failed = search_hyperparameters(always_fails, self.start, n_trials=3)
        assert failed.best_overrides == {} and failed.best_validation_loss is None


class TestExperimentTracking:
    def test_a_failing_backend_is_dropped_without_stopping_the_run(self) -> None:
        class Recorder:
            name = "recorder"

            def __init__(self) -> None:
                self.calls: list[tuple] = []

            def start_run(self, run_name, parameters):
                self.calls.append(("start", run_name))

            def log_metrics(self, metrics, step=None):
                self.calls.append(("metrics", metrics, step))

            def log_artifact(self, path):
                self.calls.append(("artifact", path.name))

            def end_run(self):
                self.calls.append(("end",))

        class Broken(Recorder):
            name = "broken"

            def log_metrics(self, metrics, step=None):
                raise ConnectionError("offline")

        recorder, broken = Recorder(), Broken()
        tracker = ExperimentTracker([broken, recorder])
        tracker.start_run("run", {"rank": 16})
        tracker.log_loss_history([{"step": 1, "loss": 0.9, "learning_rate": 1e-4}, {"step": 2, "eval_loss": 0.5}, {"epoch": 1}])
        tracker.log_metrics({"accuracy": 1, "note": "text is skipped"})
        tracker.end_run()
        assert tracker.active_backends == ["recorder"]
        assert ("metrics", {"train_loss": 0.9, "learning_rate": 1e-4}, 1) in recorder.calls
        assert ("metrics", {"validation_loss": 0.5}, 2) in recorder.calls
        assert ("metrics", {"accuracy": 1.0}, None) in recorder.calls and recorder.calls[-1] == ("end",)

    def test_unavailable_backends_are_skipped(self) -> None:
        assert create_experiment_tracker(()).active_backends == []
        assert create_experiment_tracker(("no-such-backend",)).active_backends == []
        assert ExperimentTracker([]).list_mlflow_runs() is None

    def test_mlflow_records_runs_locally(self, tmp_path, monkeypatch) -> None:
        pytest.importorskip("mlflow")
        monkeypatch.chdir(tmp_path)
        tracker = create_experiment_tracker(("mlflow",), project="unit-test")
        tracker.start_run("trial", {"lora_rank": 8})
        tracker.log_metrics({"final_validation_loss": 0.42})
        artifact = tmp_path / "results.json"
        artifact.write_text(json.dumps({"ok": True}))
        tracker.log_artifact(artifact)
        tracker.end_run()
        runs = tracker.list_mlflow_runs()
        assert len(runs) == 1 and runs["metrics.final_validation_loss"].iloc[0] == 0.42
        assert runs["params.lora_rank"].iloc[0] == "8"
        assert tracker.backend_destinations == {"mlflow": "local file store (./mlruns)"}

    @pytest.mark.parametrize(
        ("api_key", "mode", "destination"),
        [
            ("test-key", "online", "https://wandb.ai/team/unit-test"),
            (None, "offline", "offline (./wandb, `wandb sync` to upload)"),
        ],
    )
    def test_wandb_logs_online_only_when_an_api_key_exists(self, monkeypatch, api_key, mode, destination) -> None:
        init_calls: list[dict] = []

        class FakeRun:
            def get_project_url(self) -> str:
                return "https://wandb.ai/team/unit-test"

            def finish(self) -> None:
                pass

        def init(**arguments):
            init_calls.append(arguments)
            return FakeRun()

        fake_wandb = types.SimpleNamespace(init=init, util=types.SimpleNamespace(generate_id=lambda: "group-1"))
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
        monkeypatch.setitem(sys.modules, "google.colab", None)  # no Colab secrets in the test
        monkeypatch.delenv("WANDB_MODE", raising=False)
        monkeypatch.delenv("WANDB_SILENT", raising=False)
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        if api_key:
            monkeypatch.setenv("WANDB_API_KEY", api_key)

        tracker = create_experiment_tracker(("wandb",), project="unit-test")
        tracker.start_run("search-trial-1", {"stage": "search", "lora_rank": 8})
        tracker.end_run()

        assert os.environ["WANDB_MODE"] == mode
        assert ("WANDB_SILENT" in os.environ) == (mode == "offline")
        assert tracker.backend_destinations == {"wandb": destination}
        assert init_calls[0]["group"] == "group-1" and init_calls[0]["job_type"] == "search"


class TestBehaviouralTests:
    def test_perturbation_keeps_the_words_and_adds_the_distractor(self) -> None:
        perturbed = perturb_chunk_without_changing_its_meaning("Title\n\nAda   Lovelace\n")
        assert perturbed == f"Title Ada Lovelace {DISTRACTOR_SENTENCE}"

    def test_case_selection(self) -> None:
        items = [example("e", "x", PaperMetadata()), example("i", FRONT_PAGE, TITLE_TARGET)]
        assert [c.chunk_id for c in select_invariance_cases(items, limit=1)] == ["i"]
        chunks = [
            {"source_document": "a.pdf", "content_type": "references", "chunk_text": "[1] ..."},
            {"source_document": "a.pdf", "content_type": "text", "chunk_text": "body"},
            {"source_document": "b.pdf", "content_type": "references", "chunk_text": "[2] ..."},
        ]
        assert select_bibliography_chunks(chunks, "a.pdf") == [chunks[0]]

    def test_scoring(self) -> None:
        answer = TITLE_TARGET.model_dump_json()
        reordered = '{"main_authors": ["Ada Lovelace", "Alan Turing"], "title": "A Great Paper", "core_problem_statement": null, "key_metrics": []}'
        invariance = score_invariance([answer, answer], [reordered, EMPTY_JSON])
        assert (invariance["passed"], invariance["cases"], invariance["pass_rate"]) == (1, 2, 0.5)
        abstention = score_abstention_on_bibliography([EMPTY_JSON, answer, "not json"], "bibliography")
        assert (abstention["passed"], abstention["cases"]) == (1, 3)


class TestApproachComparison:
    examples = [
        example("front", FRONT_PAGE, TITLE_TARGET),
        example("empty-1", "Body text without metadata.", PaperMetadata()),
        example("empty-2", "More body text.", PaperMetadata()),
    ]
    answers = {
        "zero-shot": ["Here you go: " + TITLE_TARGET.model_dump_json() + " hope it helps", "I cannot tell.", EMPTY_JSON],
        "hallucinating": [
            TITLE_TARGET.model_dump_json(),
            '{"title": "Invented", "main_authors": [], "core_problem_statement": null, "key_metrics": []}',
            EMPTY_JSON,
        ],
        "fine-tuned": [TITLE_TARGET.model_dump_json(), EMPTY_JSON, EMPTY_JSON],
    }

    def leaderboard(self) -> list[dict]:
        return build_approach_leaderboard(self.examples, self.answers, reference_approach="zero-shot")

    def test_every_approach_is_scored_under_every_guard(self) -> None:
        board = self.leaderboard()
        assert len(board) == len(self.answers) * len(ANSWER_GUARDS)
        by_key = {(row["approach"], row["answer_guard"]): row for row in board}
        assert by_key[("zero-shot", "no guard")]["p_value_vs_reference"] is None
        assert by_key[("zero-shot", "no guard")]["verifiable_correct"] == 2
        assert by_key[("hallucinating", "no guard")]["invented_value_rate"] == 0.333
        assert by_key[("hallucinating", "no guard")]["over_extraction_rate"] == 0.5
        # The grounding filter removes the invented title; the answer becomes a correct abstention.
        guarded = by_key[("hallucinating", "JSON repair + grounding filter")]
        assert guarded["invented_value_rate"] == 0.0 and guarded["verifiable_correct"] == 3
        assert by_key[("fine-tuned", "no guard")]["verifiable_correct_ci95"] == (0.438, 1.0)

    def test_selection_prefers_no_invention_then_correctness_then_simplicity(self) -> None:
        board = self.leaderboard()
        selected = select_best_approach(board)
        assert (selected["approach"], selected["answer_guard"]) == ("fine-tuned", "no guard")
        without_fine_tuned = [row for row in board if row["approach"] != "fine-tuned"]
        runner_up = select_best_approach(without_fine_tuned)
        assert runner_up["invented_value_rate"] == 0.0 and runner_up["verifiable_correct"] == 3
        assert runner_up["answer_guard"] == "JSON repair + grounding filter"

    def test_leaderboard_plot(self) -> None:
        pytest.importorskip("plotly")
        board = self.leaderboard()
        figure = plot_approach_leaderboard(board, select_best_approach(board))
        assert len(figure.data) == len(board)
        assert sum("selected" in annotation.text for annotation in figure.layout.annotations) == 1


def test_three_way_split_keeps_papers_apart() -> None:
    items = [example(f"{d}-{i}", "text", PaperMetadata(), f"{d}.pdf") for d in "abcd" for i in range(2)]
    train, validation, test = split_train_validation_test_by_document(items, "c.pdf", "d.pdf")
    assert {e.document for e in train} == {"a.pdf", "b.pdf"}
    assert {e.document for e in validation} == {"c.pdf"} and {e.document for e in test} == {"d.pdf"}
    with pytest.raises(ValueError):
        split_train_validation_test_by_document(items, "c.pdf", "c.pdf")
    with pytest.raises(ValueError):
        split_train_validation_test_by_document(items, "c.pdf", "missing.pdf")
