"""Compare chunking strategies on retrieval quality.

Usage:
    python main.py --input ./data --output ./results.json --benchmark-against-fixed-windows
    python scripts/benchmark_retrieval.py

Systems compared (same 40 questions, same retrievers):
    fixed                  naive fixed-size token windows
    structure              this pipeline's chunks, text only
    structure + context    same chunks, indexed on embedding_text (title + section + text)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pdf_chunking.retrieval_benchmark import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_QUESTIONS_PATH,
    Passage,
    SimilarityScorer,
    add_gain_over_reference,
    create_dense_scorer,
    create_tfidf_scorer,
    find_relevant_passages,
    load_questions,
    score_retrieval,
)


def load_chunk_dataset(path: Path) -> list[Passage]:
    """Read an exported chunk dataset.

    Args:
        path: JSON file exported by ``main.py``.

    Returns:
        The passages, one dict per chunk.

    Raises:
        FileNotFoundError: If ``path`` is not an existing file.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run: python main.py --input ./data --output ./results.json "
            "--benchmark-against-fixed-windows"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def print_markdown_tables(results: dict[str, Any]) -> None:
    """Print one Markdown table per retriever.

    Args:
        results: Retriever name -> system name -> metrics: ``passages``, ``hit@1``, ``hit@3``,
            ``hit@5``, ``mrr@10``, ``top5_tokens`` and, for every system but the reference,
            ``mrr_gain_vs_fixed`` and ``mrr_gain_ci95`` (low, high).
    """
    for retriever, systems in results.items():
        print(f"\n### {retriever}\n")
        print("| System | passages | hit@1 | hit@3 | hit@5 | MRR@10 | MRR gain vs fixed (95 % CI) | top-5 tokens |")
        print("|---|---:|---:|---:|---:|---:|:---:|---:|")
        for name, m in systems.items():
            gain = (
                f"{m['mrr_gain_vs_fixed']:+.3f} [{m['mrr_gain_ci95'][0]:+.3f}, {m['mrr_gain_ci95'][1]:+.3f}]"
                if "mrr_gain_vs_fixed" in m
                else "reference"
            )
            print(
                f"| {name} | {m['passages']} | {m['hit@1']:.3f} | {m['hit@3']:.3f} | {m['hit@5']:.3f} "
                f"| {m['mrr@10']:.3f} | {gain} | {m['top5_tokens']} |"
            )


def main() -> int:
    """Score both chunk datasets and write the results."""
    parser = argparse.ArgumentParser(description="Retrieval benchmark for chunking strategies.")
    parser.add_argument("--structure", type=Path, default=ROOT / "results.json")
    parser.add_argument("--fixed", type=Path, default=ROOT / "results.fixed_window_baseline.json")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--tfidf-only", action="store_true", help="Skip the dense retriever (no model download).")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks" / "retrieval_results.json")
    args = parser.parse_args()

    try:
        structure, fixed = load_chunk_dataset(args.structure), load_chunk_dataset(args.fixed)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    questions = load_questions(args.questions)
    systems: dict[str, tuple[list[Passage], list[str]]] = {
        "fixed": (fixed, [chunk["chunk_text"] for chunk in fixed]),
        "structure": (structure, [chunk["chunk_text"] for chunk in structure]),
        "structure + context": (structure, [chunk["embedding_text"] for chunk in structure]),
    }
    # A question nobody can answer would measure the benchmark, not the system.
    for name, (passages, _) in systems.items():
        unanswerable = [q["question"] for q in questions if not find_relevant_passages(passages, q).any()]
        if unanswerable:
            print(f"{name}: no relevant passage for {unanswerable}", file=sys.stderr)
            return 1

    scorers: dict[str, SimilarityScorer] = {}
    if not args.tfidf_only:
        try:
            scorers[f"dense ({args.embedding_model})"] = create_dense_scorer(args.embedding_model)
        except Exception as error:  # noqa: BLE001 - package missing, no network...
            print(f"Dense retriever unavailable ({error}); using TF-IDF only.", file=sys.stderr)
    scorers["tf-idf"] = create_tfidf_scorer()

    results: dict[str, Any] = {}
    for retriever, scorer in scorers.items():
        results[retriever] = {
            name: score_retrieval(passages, texts, questions, scorer) for name, (passages, texts) in systems.items()
        }
        for name, metrics in results[retriever].items():
            if name != "fixed":
                add_gain_over_reference(metrics, results[retriever]["fixed"])

    print(f"{len(questions)} questions")
    print_markdown_tables(results)
    args.output.write_text(json.dumps({"n_questions": len(questions), "results": results}, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
