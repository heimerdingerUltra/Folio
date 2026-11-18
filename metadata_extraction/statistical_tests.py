"""Small-sample statistics for the experiment's evaluation sets.

Pure standard library + NumPy: exact or resampling-based, because the
evaluation sets here have a dozen examples and normal approximations fail.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Sequence

import numpy as np


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """95 % confidence interval of a proportion (Wilson score).

    Unlike the normal approximation it stays inside [0, 1] and is usable
    for small ``trials`` and rates of 0 or 1.

    Args:
        successes: Number of successes observed.
        trials: Number of trials. With 0 trials the interval is ``(0.0, 1.0)``.
        z: Normal quantile of the confidence level; 1.96 gives 95 %.

    Returns:
        ``(low, high)``, rounded to 3 decimals.
    """
    if trials == 0:
        return 0.0, 1.0
    rate = successes / trials
    denominator = 1 + z**2 / trials
    centre = (rate + z**2 / (2 * trials)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / trials + z**2 / (4 * trials**2)) / denominator
    return round(max(0.0, centre - margin), 3), round(min(1.0, centre + margin), 3)


def exact_mcnemar_test(reference_correct: Sequence[bool], candidate_correct: Sequence[bool]) -> dict[str, float]:
    """Paired test: does the candidate get different examples right than the reference?

    Only discordant pairs carry information. Under the null hypothesis they
    split 50/50, so the two-sided p-value is an exact binomial tail.

    Args:
        reference_correct: Per example, whether the reference system is correct.
        candidate_correct: The same for the candidate, on the same examples in the same order.

    Returns:
        ``candidate_only`` / ``reference_only`` (discordant counts) and ``p_value``.
    """
    candidate_only = sum(c and not r for r, c in zip(reference_correct, candidate_correct))
    reference_only = sum(r and not c for r, c in zip(reference_correct, candidate_correct))
    discordant = candidate_only + reference_only
    if discordant == 0:
        return {"candidate_only": 0, "reference_only": 0, "p_value": 1.0}
    tail = sum(math.comb(discordant, k) for k in range(min(candidate_only, reference_only) + 1)) / 2**discordant
    return {
        "candidate_only": candidate_only,
        "reference_only": reference_only,
        "p_value": round(min(1.0, 2 * tail), 4),
    }


def bootstrap_interval(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    resamples: int = 5000,
    seed: int = 0,
) -> tuple[float, float]:
    """95 % percentile bootstrap interval of ``statistic`` over ``values``.

    Args:
        values: Observed sample.
        statistic: Function reducing a resampled array to one number; the mean by default.
        resamples: Number of resamples drawn with replacement, each of the size of ``values``.
        seed: Seed of the random generator.

    Returns:
        ``(low, high)``: the 2.5th and 97.5th percentiles of the resampled statistic, rounded to
        3 decimals; ``(nan, nan)`` for an empty sample.
    """
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = [statistic(rng.choice(data, size=data.size, replace=True)) for _ in range(resamples)]
    low, high = np.percentile(estimates, [2.5, 97.5])
    return round(float(low), 3), round(float(high), 3)


def kolmogorov_smirnov_statistic(first: Sequence[float], second: Sequence[float]) -> float:
    """Largest gap between the two empirical distribution functions.

    Args:
        first: First sample; must not be empty.
        second: Second sample; must not be empty.

    Returns:
        The statistic, between 0 (identical distributions) and 1 (disjoint ranges).
    """
    a, b = np.sort(np.asarray(first, dtype=float)), np.sort(np.asarray(second, dtype=float))
    grid = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, grid, side="right") / a.size
                               - np.searchsorted(b, grid, side="right") / b.size)))  # fmt: skip


def permutation_test_of_distribution_shift(
    first: Sequence[float], second: Sequence[float], resamples: int = 5000, seed: int = 0
) -> dict[str, float]:
    """Are two samples drawn from the same distribution? (KS statistic, permutation p-value)

    Used as a covariate-shift check between splits: a small p-value means
    the model would be evaluated on inputs unlike those it was trained on.

    Args:
        first: First sample, for instance a feature of the training chunks.
        second: Second sample, the same feature on another split.
        resamples: Number of random reassignments of the pooled values to the two samples.
        seed: Seed of the shuffling.

    Returns:
        ``ks_statistic`` (rounded to 3 decimals) and ``p_value``: the share of reassignments
        whose statistic is at least the observed one, with an add-one correction.
    """
    observed = kolmogorov_smirnov_statistic(first, second)
    pooled = list(first) + list(second)
    rng = random.Random(seed)
    at_least_as_extreme = 0
    for _ in range(resamples):
        rng.shuffle(pooled)
        if kolmogorov_smirnov_statistic(pooled[: len(first)], pooled[len(first) :]) >= observed:
            at_least_as_extreme += 1
    return {"ks_statistic": round(observed, 3), "p_value": round((at_least_as_extreme + 1) / (resamples + 1), 4)}


def sample_size_for_margin(margin: float, expected_rate: float = 0.5, z: float = 1.96) -> int:
    """Examples needed to estimate a rate within ``± margin`` at 95 % confidence.

    Args:
        margin: Half-width of the interval, as a proportion (0.05 for five points).
        expected_rate: Anticipated rate; 0.5 is the most demanding case.
        z: Normal quantile of the confidence level; 1.96 gives 95 %.

    Returns:
        The number of examples, rounded up.
    """
    return math.ceil(z**2 * expected_rate * (1 - expected_rate) / margin**2)
