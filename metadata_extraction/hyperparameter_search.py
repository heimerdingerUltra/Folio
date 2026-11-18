"""Hyperparameter search for the LoRA adapter (Optuna TPE, random-search fallback).

The objective is the validation loss on a paper that is used for model
selection only; the test paper is never seen here. The current
configuration is always evaluated first, so the search can only match or
improve on it. A trial that crashes is recorded and the search continues.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

#: Trains with the given config overrides and returns the validation loss.
Objective = Callable[[dict[str, Any]], float]

LORA_RANK_CHOICES = (8, 16, 32)
LORA_ALPHA_MULTIPLIER_CHOICES = (1, 2)
LEARNING_RATE_RANGE = (5e-5, 5e-4)
EMPTY_PER_INFORMATIVE_CHOICES = (0.5, 1.0, 2.0)


@dataclass
class SearchResult:
    """Every trial, and the overrides of the best one."""

    trials: list[dict[str, Any]] = field(default_factory=list)
    best_overrides: dict[str, Any] = field(default_factory=dict)
    best_validation_loss: float | None = None
    sampler: str = ""


def _to_overrides(lora_rank: int, alpha_multiplier: int, learning_rate: float, empty_ratio: float) -> dict[str, Any]:
    """Translate search parameters into configuration overrides."""
    return {
        "lora_rank": lora_rank,
        "lora_alpha": lora_rank * alpha_multiplier,
        "learning_rate": learning_rate,
        "empty_per_informative": empty_ratio,
    }


def _run_trial(objective: Objective, overrides: dict[str, Any], number: int) -> dict[str, Any]:
    """Evaluate one configuration, never raising."""
    started = time.perf_counter()
    row: dict[str, Any] = {"trial": number, **overrides, "validation_loss": None, "state": "complete"}
    try:
        row["validation_loss"] = round(float(objective(overrides)), 4)
    except Exception as error:  # noqa: BLE001 - an out-of-memory trial must not end the search
        row["state"] = f"failed: {type(error).__name__}: {str(error)[:120]}"
        logger.warning("Trial %d failed: %s", number, error)
    row["seconds"] = round(time.perf_counter() - started, 1)
    return row


def search_hyperparameters(
    objective: Objective,
    starting_point: dict[str, Any],
    n_trials: int = 5,
    seed: int = 0,
) -> SearchResult:
    """Minimise the validation loss over rank, alpha, learning rate and class ratio.

    Args:
        objective: Trains one configuration and returns its validation loss.
        starting_point: Current values of the searched fields; evaluated first.
        n_trials: Total budget, starting point included.
        seed: Seed of the sampler.
    """
    result = SearchResult()
    result.trials.append(_run_trial(objective, starting_point, number=0))

    try:
        import optuna
    except ImportError:
        optuna = None

    if optuna is not None:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        result.sampler = "Optuna TPE"
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
        first = result.trials[0]
        if first["validation_loss"] is not None:
            # Tell the sampler what the starting point achieved.
            study.add_trial(
                optuna.trial.create_trial(
                    params={
                        "lora_rank": first["lora_rank"],
                        "alpha_multiplier": first["lora_alpha"] // first["lora_rank"],
                        "learning_rate": first["learning_rate"],
                        "empty_per_informative": first["empty_per_informative"],
                    },
                    distributions={
                        "lora_rank": optuna.distributions.CategoricalDistribution(LORA_RANK_CHOICES),
                        "alpha_multiplier": optuna.distributions.CategoricalDistribution(LORA_ALPHA_MULTIPLIER_CHOICES),
                        "learning_rate": optuna.distributions.FloatDistribution(*LEARNING_RATE_RANGE, log=True),
                        "empty_per_informative": optuna.distributions.CategoricalDistribution(
                            EMPTY_PER_INFORMATIVE_CHOICES
                        ),
                    },
                    value=first["validation_loss"],
                )
            )
        for number in range(1, n_trials):
            trial = study.ask()
            overrides = _to_overrides(
                trial.suggest_categorical("lora_rank", LORA_RANK_CHOICES),
                trial.suggest_categorical("alpha_multiplier", LORA_ALPHA_MULTIPLIER_CHOICES),
                trial.suggest_float("learning_rate", *LEARNING_RATE_RANGE, log=True),
                trial.suggest_categorical("empty_per_informative", EMPTY_PER_INFORMATIVE_CHOICES),
            )
            row = _run_trial(objective, overrides, number)
            if row["validation_loss"] is None:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
            else:
                study.tell(trial, row["validation_loss"])
            result.trials.append(row)
    else:
        result.sampler = "random search (Optuna not installed)"
        rng = random.Random(seed)
        low, high = LEARNING_RATE_RANGE
        for number in range(1, n_trials):
            overrides = _to_overrides(
                rng.choice(LORA_RANK_CHOICES),
                rng.choice(LORA_ALPHA_MULTIPLIER_CHOICES),
                low * (high / low) ** rng.random(),  # log-uniform
                rng.choice(EMPTY_PER_INFORMATIVE_CHOICES),
            )
            result.trials.append(_run_trial(objective, overrides, number))

    completed = [row for row in result.trials if row["validation_loss"] is not None]
    if completed:
        best = min(completed, key=lambda row: row["validation_loss"])
        result.best_validation_loss = best["validation_loss"]
        result.best_overrides = {
            key: best[key] for key in ("lora_rank", "lora_alpha", "learning_rate", "empty_per_informative")
        }
    return result


ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")


def build_config_ablations(reference: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """One-factor-at-a-time variants of a configuration.

    The search optimises four values together; an ablation changes a single
    one and keeps the rest, which shows what each choice is worth. The
    alpha / rank ratio of the reference is kept when the rank changes.

    Args:
        reference: ``lora_rank``, ``lora_alpha`` and ``learning_rate`` of the
            configuration to vary.

    Returns:
        Variant name -> configuration overrides. The first one is the
        reference itself (no override).
    """
    rank, alpha, learning_rate = reference["lora_rank"], reference["lora_alpha"], reference["learning_rate"]
    ratio = alpha / rank
    variants: dict[str, dict[str, Any]] = {"reference": {}}
    for other_rank in (4, 8, 16, 32):
        if other_rank != rank:
            variants[f"rank {other_rank}"] = {"lora_rank": other_rank, "lora_alpha": int(other_rank * ratio)}
    variants["alpha = rank" if ratio != 1 else "alpha = 2 x rank"] = {"lora_alpha": rank if ratio != 1 else 2 * rank}
    variants["learning rate / 4"] = {"learning_rate": learning_rate / 4}
    variants["learning rate x 2"] = {"learning_rate": learning_rate * 2}
    variants["attention projections only"] = {"lora_target_modules": ATTENTION_PROJECTIONS}
    variants["LoRA dropout 0.05"] = {"lora_dropout": 0.05}
    return variants


def run_config_ablation(objective: Objective, variants: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Evaluate each variant; report its validation loss and the gap to the reference.

    A variant that fails (out of memory, for instance) is recorded and skipped.

    Args:
        objective: Trains with the given configuration overrides and returns the validation loss.
        variants: Variant name -> configuration overrides. The first entry is taken as the
            reference.

    Returns:
        One row per variant, in input order, with ``variant``, ``changed``, ``validation_loss``
        (``None`` for a failed variant), ``state``, ``seconds`` and ``difference_to_reference``
        (``None`` when either loss is missing).
    """
    rows = []
    for number, (name, overrides) in enumerate(variants.items()):
        row = _run_trial(objective, overrides, number)
        rows.append(
            {
                "variant": name,
                "changed": ", ".join(f"{key}={value}" for key, value in overrides.items()) or "nothing",
                "validation_loss": row["validation_loss"],
                "state": row["state"],
                "seconds": row["seconds"],
            }
        )
    reference_loss = rows[0]["validation_loss"] if rows else None
    for row in rows:
        both = reference_loss is not None and row["validation_loss"] is not None
        row["difference_to_reference"] = round(row["validation_loss"] - reference_loss, 4) if both else None
    return rows

