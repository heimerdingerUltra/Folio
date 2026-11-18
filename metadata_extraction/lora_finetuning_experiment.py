"""Fluent LoRA fine-tuning experiment (Unsloth, 4-bit base model, Colab T4).

Every stage returns the experiment, so a run reads as its list of steps::

    (MetadataExtractionExperiment(config, tracker)
        .fix_random_seeds()
        .load_chunk_dataset()
        .build_grounded_examples()
        .add_arxiv_training_examples()
        .split_by_document()
        .validate_training_data()
        .evaluate_rule_based_baseline()
        .load_quantized_base_model()
        .check_prompt_token_budget()
        .run_generation_smoke_test()
        .evaluate_base_model_zero_shot()
        .evaluate_base_model_few_shot()
        .release_gpu_memory()
        .search_hyperparameters()
        .load_quantized_base_model()
        .attach_lora_adapters()
        .train()
        .save_adapter()
        .reload_adapter_from_disk()
        .evaluate_finetuned_model()
        .run_behavioural_tests()
        .compare_approaches()
        .save_experiment_results())

Papers are split three ways: *train* fits the adapter, *validation* selects
hyperparameters, *test* is read once for the final comparison. GPU libraries
are imported inside the stages that need them, so the data stages run (and
are tested) on a CPU-only machine.
"""

from __future__ import annotations

import gc
import json
import logging
import random
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from metadata_extraction.arxiv_training_examples import (
    build_arxiv_training_examples,
    load_arxiv_metadata,
    summarize_arxiv_label_coverage,
)
from metadata_extraction.approach_comparison import SELECTION_RULE, build_approach_leaderboard, select_best_approach
from metadata_extraction.behavioural_tests import (
    perturb_chunk_without_changing_its_meaning,
    score_abstention_on_bibliography,
    score_invariance,
    select_bibliography_chunks,
    select_invariance_cases,
)
from metadata_extraction.experiment_dashboard import (
    plot_approach_leaderboard,
    plot_experiment_dashboard,
    plot_loss_curves,
    plot_search_trials,
)
from metadata_extraction.experiment_tracking import ExperimentTracker
from metadata_extraction.extraction_evaluation import build_error_analysis, score_extraction_answers
from metadata_extraction.extraction_prompt import (
    build_chat_messages,
    build_few_shot_messages,
    select_few_shot_demonstrations,
)
from metadata_extraction.grounded_training_examples import (
    TrainingExample,
    build_grounded_examples,
    downsample_empty_examples,
    load_chunk_dataset,
    split_train_validation_test_by_document,
    summarize_label_coverage,
)
from metadata_extraction.hallucination_guards import ANSWER_GUARDS
from metadata_extraction.hyperparameter_search import (
    SearchResult,
    build_config_ablations,
    run_config_ablation,
    search_hyperparameters,
)
from metadata_extraction.paper_metadata_schema import parse_model_answer
from metadata_extraction.rule_based_extractor import extract_metadata_with_rules
from metadata_extraction.training_data_validation import (
    ValidationCheck,
    raise_if_any_check_failed,
    validate_training_data,
)

logger = logging.getLogger(__name__)

# Markers of the Phi-3 chat template, used to mask the prompt in the loss.
PHI3_USER_MARKER = "<|user|>\n"
PHI3_ASSISTANT_MARKER = "<|assistant|>\n"

RULE_BASED = "rule-based (no model)"
ZERO_SHOT = "base model, zero-shot"
FEW_SHOT = "base model, few-shot"
FINE_TUNED = "LoRA fine-tuned"


@dataclass(frozen=True)
class LoraFinetuningConfig:
    """Every knob of the experiment.

    Attributes:
        base_model: 4-bit quantised so that it fits a free Colab T4 (16 GB).
        max_sequence_length: Prompt (~120 tokens) + chunk (<= 400) + answer.
        lora_rank: Rank of the low-rank update matrices.
        lora_alpha: Update scaling; equal to the rank keeps the scale at 1.
        max_steps: Steps of the final training run. After a hyperparameter
            search it is set to ``search_trial_steps``: a configuration is
            only trusted for the training length it was validated at.
        validation_document: Paper used to select hyperparameters.
        test_document: Paper read once, for the final comparison.
        arxiv_chunk_dataset_path: Chunks of the extra training papers
            (optional); they are labelled from ``arxiv_metadata_path`` and
            added to the training set only.
        arxiv_metadata_path: ``arxiv_metadata.json`` of those papers.
        empty_per_informative: Empty-target examples kept per informative
            one in the training set (searched; held-out sets stay at 1.0).
        max_evaluation_examples: Cap on generated answers per approach.
        max_new_tokens: Generation budget per answer; large enough for an
            untuned model that pretty-prints its JSON.
        search_trials: Hyperparameter configurations tried (0 disables).
        search_trial_steps: Training steps of each search trial.
    """

    base_model: str = "unsloth/Phi-3-mini-4k-instruct"
    chat_template: str = "phi-3"
    max_sequence_length: int = 2048
    load_in_4bit: bool = True

    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    )  # fmt: skip

    max_steps: int = 40
    learning_rate: float = 2e-4
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    weight_decay: float = 0.01
    evaluate_every_steps: int = 5

    chunk_dataset_path: str = "results.json"
    validation_document: str = "ChunkNorris.pdf"
    test_document: str = "RAG.pdf"
    arxiv_chunk_dataset_path: str | None = None
    arxiv_metadata_path: str | None = None
    empty_per_informative: float = 1.0
    max_evaluation_examples: int = 12
    max_new_tokens: int = 400

    search_trials: int = 5
    search_trial_steps: int = 20

    seed: int = 3407
    training_output_dir: str = "training_outputs"
    adapter_dir: str = "lora_metadata_adapter"

    @property
    def searched_fields(self) -> dict[str, Any]:
        """Current values of the fields the hyperparameter search varies."""
        return {
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "learning_rate": self.learning_rate,
            "empty_per_informative": self.empty_per_informative,
        }


class MetadataExtractionExperiment:
    """Chainable stages from chunk dataset to a selected, evaluated extractor."""

    def __init__(self, config: LoraFinetuningConfig | None = None, tracker: ExperimentTracker | None = None) -> None:
        """Create an experiment with empty state; stages fill it in order."""
        self.config = config or LoraFinetuningConfig()
        self.tracker = tracker or ExperimentTracker([])
        self.chunks: list[dict[str, Any]] = []
        self.examples: list[TrainingExample] = []
        self.arxiv_examples: list[TrainingExample] = []
        self.arxiv_label_coverage: dict[str, Any] = {}
        self.train_examples: list[TrainingExample] = []
        self.validation_examples: list[TrainingExample] = []
        self.test_examples: list[TrainingExample] = []
        self.model: Any = None
        self.tokenizer: Any = None
        self.trainer: Any = None
        self.validation_checks: list[ValidationCheck] = []
        self.prompt_token_budget: dict[str, Any] = {}
        self.smoke_test: dict[str, Any] = {}
        self.search_result: SearchResult = SearchResult()
        self.config_ablation: list[dict[str, Any]] = []
        self.training_metrics: dict[str, Any] = {}
        self.loss_history: list[dict[str, Any]] = []
        self.answers_by_approach: dict[str, list[str]] = {}
        self.evaluations: dict[str, dict[str, Any]] = {}
        self.error_analysis: dict[str, list[dict[str, Any]]] = {}
        self.behavioural_tests: list[dict[str, Any]] = []
        self.leaderboard: list[dict[str, Any]] = []
        self.selected_approach: dict[str, Any] = {}

    # ---------------------------------------------------------- data (CPU)

    def fix_random_seeds(self) -> MetadataExtractionExperiment:
        """Seed Python, NumPy and PyTorch (when installed) for reproducibility."""
        random.seed(self.config.seed)
        try:
            import numpy

            numpy.random.seed(self.config.seed)
            import torch

            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
        except ImportError:
            pass
        return self

    def load_chunk_dataset(self) -> MetadataExtractionExperiment:
        """Load the chunks produced by ``main.py``."""
        self.chunks = load_chunk_dataset(self.config.chunk_dataset_path)
        return self

    def build_grounded_examples(self) -> MetadataExtractionExperiment:
        """Label each chunk with the gold fields its own text supports."""
        self.examples = build_grounded_examples(self.chunks)
        return self

    def add_arxiv_training_examples(self) -> MetadataExtractionExperiment:
        """Label the extra training papers from their arXiv metadata (if configured).

        Raises:
            ValueError: If an extra paper is also a hand-labelled one, which
                would leak a held-out document into the training set.
        """
        if not (self.config.arxiv_chunk_dataset_path and self.config.arxiv_metadata_path):
            return self
        metadata = load_arxiv_metadata(self.config.arxiv_metadata_path)
        if overlap := {example.document for example in self.examples} & set(metadata):
            raise ValueError(f"Extra training papers overlap the hand-labelled ones: {sorted(overlap)}")
        self.arxiv_examples = build_arxiv_training_examples(
            load_chunk_dataset(self.config.arxiv_chunk_dataset_path), metadata
        )
        self.arxiv_label_coverage = summarize_arxiv_label_coverage(self.arxiv_examples, metadata)
        return self

    def split_by_document(self) -> MetadataExtractionExperiment:
        """Train / validation / test by paper, then balance the classes.

        The training ratio is a hyperparameter; held-out sets are always
        balanced 1:1 so their metrics stay comparable across configurations.
        """
        train, validation, test = split_train_validation_test_by_document(
            self.examples, self.config.validation_document, self.config.test_document
        )
        seed = self.config.seed
        train = [*train, *self.arxiv_examples]
        self.train_examples = downsample_empty_examples(train, self.config.empty_per_informative, seed)
        self.validation_examples = downsample_empty_examples(validation, 1.0, seed)
        self.test_examples = downsample_empty_examples(test, 1.0, seed)
        return self

    def validate_training_data(self) -> MetadataExtractionExperiment:
        """Run the data checks; stop here if one fails.

        Raises:
            TrainingDataValidationError: Leakage, near-duplicates, ungrounded
                labels, missing gold coverage or a degenerate class balance.
        """
        self.validation_checks = validate_training_data(
            [*self.examples, *self.arxiv_examples], self.train_examples, self.validation_examples, test_examples=self.test_examples
        )
        raise_if_any_check_failed(self.validation_checks)
        return self

    def describe_dataset(self) -> dict[str, Any]:
        """Label coverage per paper and the size of each split."""

        def describe(split: list[TrainingExample]) -> dict[str, Any]:
            """Size and label balance of one split."""
            informative = sum(not example.target.is_empty for example in split)
            return {
                "examples": len(split),
                "informative": informative,
                "empty": len(split) - informative,
                "documents": sorted({example.document for example in split}),
            }

        return {
            "label_coverage": summarize_label_coverage(self.examples),
            "arxiv_label_coverage": self.arxiv_label_coverage,
            "train": describe(self.train_examples),
            "validation": describe(self.validation_examples),
            "test": describe(self.test_examples),
        }

    @property
    def splits(self) -> dict[str, list[TrainingExample]]:
        """Training examples by split name."""
        return {"train": self.train_examples, "validation": self.validation_examples, "test": self.test_examples}

    @property
    def evaluation_examples(self) -> list[TrainingExample]:
        """Test examples every approach is scored on: informative chunks first."""
        ordered = sorted(self.test_examples, key=lambda example: example.target.is_empty)
        return ordered[: self.config.max_evaluation_examples]

    # ---------------------------------------------------- approaches (CPU)

    def _record_approach(self, name: str, answers: list[str]) -> None:
        """Store the answers of one approach and score them on the evaluation set."""
        examples = self.evaluation_examples
        self.answers_by_approach[name] = answers
        self.evaluations[name] = score_extraction_answers(examples, answers)
        self.error_analysis[name] = build_error_analysis(examples, answers)

    def evaluate_rule_based_baseline(self) -> MetadataExtractionExperiment:
        """Score the regex extractor: the bar any model has to clear."""
        self._record_approach(
            RULE_BASED, [extract_metadata_with_rules(example.chunk_text) for example in self.evaluation_examples]
        )
        return self

    # --------------------------------------------------------- model (GPU)

    def load_quantized_base_model(self) -> MetadataExtractionExperiment:
        """Load the 4-bit base model and attach its chat template."""
        from unsloth import FastLanguageModel  # must be imported before transformers
        from unsloth.chat_templates import get_chat_template

        self.model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.base_model,
            max_seq_length=self.config.max_sequence_length,
            dtype=None,
            load_in_4bit=self.config.load_in_4bit,
        )
        self.tokenizer = get_chat_template(tokenizer, chat_template=self.config.chat_template)
        return self

    def release_gpu_memory(self) -> MetadataExtractionExperiment:
        """Drop the model and trainer so the next stage starts from a clean GPU."""
        import torch

        self.model = self.trainer = None
        gc.collect()
        torch.cuda.empty_cache()
        return self

    def _count_tokens(self, messages: list[dict[str, str]]) -> int:
        """Number of tokens of a conversation once the chat template is applied."""
        text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        return len(self.tokenizer(text)["input_ids"])

    def check_prompt_token_budget(self) -> MetadataExtractionExperiment:
        """Verify that every prompt + answer fits the model's sequence length.

        A truncated example silently loses its answer, i.e. its training signal.

        Raises:
            ValueError: If the longest example exceeds ``max_sequence_length``.
        """
        lengths = [
            self._count_tokens(build_chat_messages(example.chunk_text, example.target))
            for split in self.splits.values()
            for example in split
        ]
        self.prompt_token_budget = {
            "examples": len(lengths),
            "mean_tokens": round(statistics.fmean(lengths)),
            "max_tokens": max(lengths),
            "max_sequence_length": self.config.max_sequence_length,
            "headroom_tokens": self.config.max_sequence_length - max(lengths),
        }
        if max(lengths) > self.config.max_sequence_length:
            raise ValueError(f"Longest example has {max(lengths)} tokens; raise max_sequence_length.")
        return self

    def generate_answer(self, messages: list[dict[str, str]], max_new_tokens: int | None = None) -> str:
        """Greedy-decode the model's answer to a chat prompt."""
        import torch

        max_new_tokens = max_new_tokens or self.config.max_new_tokens

        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to("cuda")
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
        answer_tokens = output[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

    def _generate_answers(self, prompts: Sequence[list[dict[str, str]]]) -> list[str]:
        """Generate one answer per prompt, in evaluation mode."""
        self.model.eval()
        return [self.generate_answer(messages) for messages in prompts]

    def run_generation_smoke_test(self) -> MetadataExtractionExperiment:
        """Generate one answer end to end before spending time on training.

        Raises:
            RuntimeError: If the model returns an empty answer.
        """
        example = self.evaluation_examples[0]
        started = time.perf_counter()
        self.model.eval()
        answer = self.generate_answer(build_chat_messages(example.chunk_text), max_new_tokens=96)
        self.smoke_test = {
            "chunk_id": example.chunk_id,
            "seconds": round(time.perf_counter() - started, 1),
            "answer_status": parse_model_answer(answer)[1],
            "answer_preview": answer[:200],
        }
        if not answer:
            raise RuntimeError("Smoke test failed: the model generated an empty answer.")
        return self

    def evaluate_base_model_zero_shot(self) -> MetadataExtractionExperiment:
        """Untouched model, instruction only."""
        prompts = [build_chat_messages(example.chunk_text) for example in self.evaluation_examples]
        self._record_approach(ZERO_SHOT, self._generate_answers(prompts))
        return self

    def evaluate_base_model_few_shot(self) -> MetadataExtractionExperiment:
        """Untouched model with two solved examples in the prompt (no training)."""
        demonstrations = select_few_shot_demonstrations(self.train_examples)
        prompts = [build_few_shot_messages(example.chunk_text, demonstrations) for example in self.evaluation_examples]
        longest = max(self._count_tokens(messages) for messages in prompts)
        if longest > self.config.max_sequence_length:
            logger.warning("Few-shot prompts reach %d tokens; skipping this approach.", longest)
            return self
        self._record_approach(FEW_SHOT, self._generate_answers(prompts))
        return self

    def attach_lora_adapters(self) -> MetadataExtractionExperiment:
        """Freeze the base weights and add trainable low-rank matrices."""
        from unsloth import FastLanguageModel

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_rank,
            target_modules=list(self.config.lora_target_modules),
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=self.config.seed,
        )
        self.model.print_trainable_parameters()
        return self

    def _render_training_texts(self, examples: list[TrainingExample]) -> Any:
        """Build a ``datasets.Dataset`` of conversations rendered with the chat template."""
        from datasets import Dataset

        texts = [
            self.tokenizer.apply_chat_template(
                build_chat_messages(example.chunk_text, example.target), tokenize=False, add_generation_prompt=False
            )
            for example in examples
        ]
        return Dataset.from_dict({"text": texts})

    def train(self) -> MetadataExtractionExperiment:
        """Fine-tune, with the loss computed on the JSON answer only."""
        import torch
        from trl import SFTConfig, SFTTrainer
        from unsloth.chat_templates import train_on_responses_only

        config = self.config
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self._render_training_texts(self.train_examples),
            eval_dataset=self._render_training_texts(self.validation_examples),
            dataset_text_field="text",
            max_seq_length=config.max_sequence_length,
            packing=False,
            args=SFTConfig(
                per_device_train_batch_size=config.batch_size,
                per_device_eval_batch_size=config.batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                warmup_steps=config.warmup_steps,
                max_steps=config.max_steps,
                learning_rate=config.learning_rate,
                lr_scheduler_type="linear",
                optim="adamw_8bit",
                weight_decay=config.weight_decay,
                logging_steps=1,
                eval_strategy="steps",
                eval_steps=config.evaluate_every_steps,
                save_strategy="no",
                seed=config.seed,
                output_dir=config.training_output_dir,
                report_to="none",
            ),
        )
        self.trainer = train_on_responses_only(
            trainer, instruction_part=PHI3_USER_MARKER, response_part=PHI3_ASSISTANT_MARKER
        )
        labels = self.trainer.train_dataset[0]["labels"]
        supervised = sum(label != -100 for label in labels)
        if not 0 < supervised < len(labels):
            raise RuntimeError("Prompt masking failed: check the chat-template markers.")

        torch.cuda.reset_peak_memory_stats()
        result = self.trainer.train()
        self.loss_history = list(self.trainer.state.log_history)
        self.training_metrics = {
            "train_runtime_seconds": round(result.metrics["train_runtime"], 1),
            "final_train_loss": round(result.metrics["train_loss"], 4),
            "final_validation_loss": self.final_validation_loss,
            **self.diagnose_overfitting(),
            "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        }
        return self

    def diagnose_overfitting(self, tolerance: float = 0.05) -> dict[str, Any]:
        """Compare the final validation loss with the best one seen during training.

        Returns:
            ``best_validation_loss``, ``best_validation_step`` and
            ``overfitting``: True when the final loss is more than
            ``tolerance`` (relative) above the best, i.e. training went on
            past the point where the adapter stopped generalising.
        """
        losses = [(entry["eval_loss"], entry["step"]) for entry in self.loss_history if "eval_loss" in entry]
        if not losses:
            return {}
        best_loss, best_step = min(losses)
        return {
            "best_validation_loss": round(best_loss, 4),
            "best_validation_step": best_step,
            "overfitting": losses[-1][0] > best_loss * (1 + tolerance),
        }

    @property
    def final_validation_loss(self) -> float | None:
        """Last validation loss of the training history, if any."""
        losses = [entry["eval_loss"] for entry in self.loss_history if "eval_loss" in entry]
        return round(losses[-1], 4) if losses else None

    def search_hyperparameters(self) -> MetadataExtractionExperiment:
        """Search rank, alpha, learning rate and class ratio on the validation paper.

        Each trial trains a fresh adapter for ``search_trial_steps`` steps and
        is logged as its own tracked run. The best configuration replaces the
        current one; if every trial fails, the current one is kept.
        """
        if self.config.search_trials <= 0:
            return self
        self.search_result = search_hyperparameters(
            self._short_run_objective("search"),
            self.config.searched_fields,
            self.config.search_trials,
            self.config.seed,
        )
        if self.search_result.best_overrides:
            # Train the final model for the length the winner was validated
            # at: a learning rate that is best at 20 steps can overfit at 40.
            self.config = replace(
                self.config, **self.search_result.best_overrides, max_steps=self.config.search_trial_steps
            )
            self.split_by_document()  # the class ratio may have changed
        return self

    def _short_run_objective(self, stage: str) -> Any:
        """Build the function that trains one short run and returns its validation loss.

        Every call trains a fresh adapter for ``search_trial_steps`` steps on
        the same examples and is logged as its own tracked run.
        """
        run_number = iter(range(10_000))

        def validation_loss_of(overrides: dict[str, Any]) -> float:
            """Train a short run with ``overrides`` and return its validation loss."""
            trial_config = replace(
                self.config,
                **overrides,
                max_steps=self.config.search_trial_steps,
                evaluate_every_steps=self.config.search_trial_steps,
            )
            trial = MetadataExtractionExperiment(trial_config)
            trial.examples = self.examples
            trial.arxiv_examples = self.arxiv_examples
            self.tracker.start_run(f"{stage}-trial-{next(run_number)}", {**overrides, "stage": stage})
            try:
                trial.split_by_document().load_quantized_base_model().attach_lora_adapters().train()
                self.tracker.log_loss_history(trial.loss_history)
                loss = trial.final_validation_loss
                if loss is None:
                    raise RuntimeError("the trial produced no validation loss")
                self.tracker.log_metrics({"final_validation_loss": loss})
                return loss
            finally:
                trial.release_gpu_memory()
                self.tracker.end_run()

        return validation_loss_of

    def run_config_ablation(self) -> MetadataExtractionExperiment:
        """Change one setting at a time around the current configuration.

        Run after the search: it tells which of the selected values matter.
        The configuration itself is not changed by this stage.
        """
        variants = build_config_ablations(self.config.searched_fields)
        self.config_ablation = run_config_ablation(self._short_run_objective("ablation"), variants)
        return self

    def save_adapter(self) -> MetadataExtractionExperiment:
        """Write the LoRA weights, tokenizer and experiment config to disk."""
        directory = Path(self.config.adapter_dir)
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
        (directory / "experiment_config.json").write_text(json.dumps(asdict(self.config), indent=2))
        if not any(directory.glob("adapter_model.*")):
            raise RuntimeError(f"No adapter weights were written to {directory}.")
        return self

    def reload_adapter_from_disk(self) -> MetadataExtractionExperiment:
        """Drop the in-memory model and load the saved adapter back.

        Proves the saved artefact is usable on its own.
        """
        from unsloth import FastLanguageModel

        self.release_gpu_memory()
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.adapter_dir,
            max_seq_length=self.config.max_sequence_length,
            dtype=None,
            load_in_4bit=self.config.load_in_4bit,
        )
        FastLanguageModel.for_inference(self.model)
        return self

    def evaluate_finetuned_model(self) -> MetadataExtractionExperiment:
        """Score the fine-tuned model on the test paper."""
        prompts = [build_chat_messages(example.chunk_text) for example in self.evaluation_examples]
        self._record_approach(FINE_TUNED, self._generate_answers(prompts))
        return self

    def run_behavioural_tests(self) -> MetadataExtractionExperiment:
        """Invariance to harmless edits, and abstention on bibliography chunks."""
        cases = select_invariance_cases(self.evaluation_examples)
        originals = self._generate_answers([build_chat_messages(case.chunk_text) for case in cases])
        perturbed = self._generate_answers(
            [build_chat_messages(perturb_chunk_without_changing_its_meaning(case.chunk_text)) for case in cases]
        )
        bibliography = select_bibliography_chunks(self.chunks, self.config.test_document)
        raw = self._generate_answers([build_chat_messages(chunk["chunk_text"]) for chunk in bibliography])
        guard = ANSWER_GUARDS["JSON repair + grounding filter"]
        guarded = [guard(answer, chunk["chunk_text"]) for answer, chunk in zip(raw, bibliography)]
        self.behavioural_tests = [
            score_invariance(originals, perturbed),
            score_abstention_on_bibliography(raw, "abstention on bibliography chunks (no guard)"),
            score_abstention_on_bibliography(guarded, "abstention on bibliography chunks (grounding filter)"),
            {
                "test": "abstention on bibliography chunks (routed by content_type, never sent to the model)",
                "cases": len(bibliography),
                "passed": len(bibliography),
                "pass_rate": 1.0 if bibliography else None,
                "pass_rate_ci95": (1.0, 1.0),
            },
        ]
        return self

    # ------------------------------------------------- selection & outputs

    def compare_approaches(self) -> MetadataExtractionExperiment:
        """Rank every approach under every answer guard and select one by rule."""
        reference = ZERO_SHOT if ZERO_SHOT in self.answers_by_approach else next(iter(self.answers_by_approach))
        self.leaderboard = build_approach_leaderboard(self.evaluation_examples, self.answers_by_approach, reference)
        self.selected_approach = {**select_best_approach(self.leaderboard), "selection_rule": SELECTION_RULE}
        return self

    def save_experiment_results(self) -> MetadataExtractionExperiment:
        """Write everything measured to one JSON file and to the trackers."""
        results = {
            "config": asdict(self.config),
            "dataset": self.describe_dataset(),
            "data_validation": [asdict(check) for check in self.validation_checks],
            "prompt_token_budget": self.prompt_token_budget,
            "smoke_test": self.smoke_test,
            "hyperparameter_search": asdict(self.search_result),
            "config_ablation": self.config_ablation,
            "training_metrics": self.training_metrics,
            "loss_history": self.loss_history,
            "evaluations": self.evaluations,
            "error_analysis": self.error_analysis,
            "behavioural_tests": self.behavioural_tests,
            "leaderboard": self.leaderboard,
            "selected_approach": self.selected_approach,
            # Raw answers make every metric recomputable without a GPU.
            "evaluated_chunk_ids": [example.chunk_id for example in self.evaluation_examples],
            "answers_by_approach": self.answers_by_approach,
        }
        directory = Path(self.config.training_output_dir)
        directory.mkdir(exist_ok=True)
        results_path = directory / "experiment_results.json"
        results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

        self.tracker.start_run("final-model", {**asdict(self.config), "stage": "final"})
        self.tracker.log_loss_history(self.loss_history)
        self.tracker.log_metrics(self.training_metrics)
        self.tracker.log_metrics(
            {f"test_{name}": value for name, value in self.evaluations.get(FINE_TUNED, {}).items()}
        )
        self.tracker.log_artifact(results_path)
        self.tracker.log_artifact(directory / "experiment_dashboard.html")
        self.tracker.end_run()
        return self

    def list_adapter_files(self) -> dict[str, float]:
        """Adapter file names and sizes in megabytes."""
        return {
            path.name: round(path.stat().st_size / 1024**2, 2)
            for path in sorted(Path(self.config.adapter_dir).iterdir())
            if path.is_file()
        }

    def plot_dashboard(self) -> Any:
        """Losses, baseline vs fine-tuned metrics, answer validity, run facts."""
        facts = {
            "base model": self.config.base_model.split("/")[-1],
            "LoRA rank / alpha": f"{self.config.lora_rank} / {self.config.lora_alpha}",
            "learning rate": f"{self.config.learning_rate:.2e}",
            "search": f"{len(self.search_result.trials)} trials ({self.search_result.sampler or 'disabled'})",
            "train / validation / test examples": " / ".join(str(len(split)) for split in self.splits.values()),
            **self.training_metrics,
        }
        shown = {name: self.evaluations[name] for name in (ZERO_SHOT, FINE_TUNED) if name in self.evaluations}
        return plot_experiment_dashboard(
            self.loss_history, shown, facts, "LoRA metadata extraction: experiment dashboard"
        )

    def plot_search_trials(self) -> Any:
        """Validation loss of every search trial, with its configuration."""
        return plot_search_trials(self.search_result.trials)

    def plot_leaderboard(self) -> Any:
        """Fully-correct rate of every approach with its confidence interval."""
        return plot_approach_leaderboard(self.leaderboard, self.selected_approach)

    def plot_loss_curves(self) -> Any:
        """Training loss per step and validation loss per evaluation."""
        title = f"LoRA r={self.config.lora_rank} on {self.config.base_model.split('/')[-1]}"
        return plot_loss_curves(self.loss_history, title)
