"""Retrieval benchmark: is a chunk holding the answer ranked near the top?

Relevance is defined on text (a chunk is relevant when it contains the
question's evidence strings), so any chunking strategy can be scored against
the same questions. Used by ``scripts/benchmark_retrieval.py`` and
``scripts/tune_chunking_parameters.py``.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
_WHITESPACE_RE = re.compile(r"\s+")

Passage = dict[str, Any]
Question = dict[str, Any]
#: Maps (queries, passage texts) to a score matrix of shape (queries, passages).
SimilarityScorer = Callable[[list[str], list[str]], np.ndarray]


#: The 40 benchmark questions, shipped with the package.
DEFAULT_QUESTIONS_PATH = Path(__file__).with_name("retrieval_questions.json")


def load_questions(path: Path = DEFAULT_QUESTIONS_PATH) -> list[Question]:
    """Read the benchmark questions.

    Args:
        path: UTF-8 JSON file holding a ``questions`` array. Defaults to the file shipped with
            the package.

    Returns:
        The questions, in file order.
    """
    return json.loads(path.read_text(encoding="utf-8"))["questions"]


def _comparison_key(text: str) -> str:
    """Lower-case text with whitespace collapsed."""
    return _WHITESPACE_RE.sub(" ", text).lower()


def find_relevant_passages(passages: Sequence[Passage], question: Question) -> np.ndarray:
    """Boolean mask of the passages that contain the answer to ``question``.

    Args:
        passages: Chunk dictionaries with at least ``chunk_text`` and ``source_document``.
        question: Benchmark question. ``document`` names the source file holding the answer and
            ``evidence`` is a list of groups of strings.

    Returns:
        One boolean per passage: True when the passage comes from the question's document and
        contains every string of at least one evidence group, ignoring case and whitespace
        differences.
    """
    evidence_groups = [[_comparison_key(item) for item in group] for group in question["evidence"]]
    mask = []
    for passage in passages:
        text = _comparison_key(passage["chunk_text"])
        mask.append(
            passage["source_document"] == question["document"]
            and any(all(item in text for item in group) for group in evidence_groups)
        )
    return np.array(mask)


def create_dense_scorer(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SimilarityScorer:
    """Cosine similarity between bi-encoder embeddings (GPU if available).

    Args:
        model_name: Name or path of a sentence-transformers model. When it contains ``bge``,
            the BGE retrieval instruction is prepended to each query.

    Returns:
        A scorer mapping ``(queries, passage texts)`` to a matrix of cosine similarities of
        shape ``(queries, passages)``.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    instruction = _BGE_QUERY_INSTRUCTION if "bge" in model_name.lower() else ""

    def score(queries: list[str], texts: list[str]) -> np.ndarray:
        """Cosine similarity between query and passage embeddings."""
        passage_vectors = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        query_vectors = model.encode(
            [instruction + query for query in queries], normalize_embeddings=True, show_progress_bar=False
        )
        return query_vectors @ passage_vectors.T

    return score


def create_tfidf_scorer() -> SimilarityScorer:
    """Lexical reference: TF-IDF over unigrams and bigrams."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    def score(queries: list[str], texts: list[str]) -> np.ndarray:
        """TF-IDF similarity, the vocabulary being fitted on the passages."""
        vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2))
        passage_matrix = vectorizer.fit_transform(texts)
        return (vectorizer.transform(queries) @ passage_matrix.T).toarray()

    return score


def rank_of_first_relevant_passage(scores: np.ndarray, relevant: Sequence[np.ndarray]) -> list[int | None]:
    """1-based rank of the best-placed relevant passage per query (None if absent).

    Args:
        scores: Similarity matrix of shape ``(queries, passages)``; higher means more similar.
        relevant: One boolean mask over the passages per query, in the order of the rows of
            ``scores``.

    Returns:
        One rank per query, passages being sorted by decreasing score and ties keeping passage
        order. None when no passage is relevant to the query.
    """
    ranks: list[int | None] = []
    for query_scores, mask in zip(scores, relevant):
        order = np.argsort(-query_scores, kind="stable")
        positions = np.flatnonzero(mask[order])
        ranks.append(int(positions[0]) + 1 if positions.size else None)
    return ranks


def reciprocal_ranks(ranks: Sequence[int | None], cutoff: int = 10) -> np.ndarray:
    """Reciprocal rank per query; 0 when the answer is beyond ``cutoff``.

    Args:
        ranks: 1-based rank of the first relevant passage per query; None when there is none.
        cutoff: Largest rank that still scores (10 for MRR@10).

    Returns:
        An array with ``1 / rank`` per query, or 0.0 for a rank that is None or above
        ``cutoff``.
    """
    return np.array([1.0 / rank if rank is not None and rank <= cutoff else 0.0 for rank in ranks])


def paired_bootstrap_interval(
    per_question_gain: np.ndarray, resamples: int = 5000, seed: int = 0
) -> tuple[float, float]:
    """95 % interval of the mean gain, resampling questions with replacement.

    Args:
        per_question_gain: Difference between two systems on each question (same question
            order for both); must not be empty.
        resamples: Number of bootstrap samples, each of the size of ``per_question_gain``.
        seed: Seed of the random generator, so the interval is reproducible.

    Returns:
        ``(lower, upper)``: the 2.5th and 97.5th percentiles of the resampled means, rounded to
        3 decimals.
    """
    rng = random.Random(seed)
    n = len(per_question_gain)
    means = sorted(
        float(np.mean([per_question_gain[rng.randrange(n)] for _ in range(n)])) for _ in range(resamples)
    )
    return round(means[int(0.025 * resamples)], 3), round(means[int(0.975 * resamples)], 3)


def score_retrieval(
    passages: Sequence[Passage],
    indexed_texts: list[str],
    questions: Sequence[Question],
    scorer: SimilarityScorer,
) -> dict[str, Any]:
    """Hit@k, MRR@10 and context cost of one indexed set of passages.

    Args:
        passages: Chunk dictionaries with ``chunk_text``, ``source_document`` and
            ``token_count``; relevance is judged on ``chunk_text``.
        indexed_texts: The text given to the scorer for each passage, in the same order as
            ``passages`` (for instance ``embedding_text`` or ``chunk_text``).
        questions: Benchmark questions, each with ``question``, ``document`` and ``evidence``;
            must not be empty.
        scorer: Maps ``(queries, passage texts)`` to a score matrix of shape
            ``(queries, passages)``.

    Returns:
        Metrics plus ``ranks`` (rank of the first relevant passage per
        question), which paired comparisons need.
    """
    scores = scorer([question["question"] for question in questions], indexed_texts)
    ranks = rank_of_first_relevant_passage(
        scores, [find_relevant_passages(passages, question) for question in questions]
    )
    token_counts = np.array([passage["token_count"] for passage in passages])
    top_five = np.argsort(-scores, axis=1, kind="stable")[:, :5]

    def hit_rate(k: int) -> float:
        """Share of questions whose first relevant passage is in the top ``k``."""
        return round(sum(rank is not None and rank <= k for rank in ranks) / len(ranks), 3)

    return {
        "passages": len(passages),
        "hit@1": hit_rate(1),
        "hit@3": hit_rate(3),
        "hit@5": hit_rate(5),
        "mrr@10": round(float(reciprocal_ranks(ranks).mean()), 3),
        "top5_tokens": round(float(token_counts[top_five].sum(axis=1).mean())),
        "ranks": ranks,
    }


def add_gain_over_reference(metrics: dict[str, Any], reference: dict[str, Any]) -> None:
    """Add the MRR gain over ``reference`` and its bootstrap interval, in place.

    Args:
        metrics: Output of :func:`score_retrieval` for the system under test. Receives
            ``mrr_gain_vs_fixed`` (mean reciprocal-rank difference, rounded to 3 decimals) and
            ``mrr_gain_ci95`` (``[lower, upper]``).
        reference: Output of :func:`score_retrieval` for the reference system, on the same
            questions in the same order.
    """
    gain = reciprocal_ranks(metrics["ranks"]) - reciprocal_ranks(reference["ranks"])
    metrics["mrr_gain_vs_fixed"] = round(float(gain.mean()), 3)
    metrics["mrr_gain_ci95"] = list(paired_bootstrap_interval(gain))
