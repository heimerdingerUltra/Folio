"""Grid-search chunk size and overlap against the retrieval benchmark.

Usage:
    python scripts/tune_chunking_parameters.py

PDFs are parsed once (cached in ``.cache/``); each configuration is then
re-chunked and scored with the dense retriever. Results are written to
``benchmarks/chunking_parameter_sweep.json``, best MRR@10 first.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pdf_chunking.chunking_strategies import ChunkingConfig, chunk_parsed_document  # noqa: E402
from pdf_chunking.pdf_layout_extraction import ParsedDocument, extract_pdf_document  # noqa: E402
from pdf_chunking.retrieval_benchmark import (  # noqa: E402
    DEFAULT_QUESTIONS_PATH,
    create_dense_scorer,
    find_relevant_passages,
    load_questions,
    score_retrieval,
)

MAX_TOKENS_GRID = (256, 400, 512)
OVERLAP_TOKENS_GRID = (0, 50)


def load_parsed_documents(data_dir: Path, cache_file: Path) -> list[ParsedDocument]:
    """Parse the PDFs, reusing the cache when it is newer than every PDF.

    Args:
        data_dir: Directory whose ``*.pdf`` files are parsed (sub-directories are not searched).
        cache_file: Pickle of the parsed documents; written when missing or older than a PDF.

    Returns:
        The parsed documents, in the order of the sorted PDF paths.
    """
    pdf_paths = sorted(data_dir.glob("*.pdf"))
    if cache_file.is_file() and all(cache_file.stat().st_mtime > path.stat().st_mtime for path in pdf_paths):
        return pickle.loads(cache_file.read_bytes())
    documents = [extract_pdf_document(path) for path in pdf_paths]
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_bytes(pickle.dumps(documents))
    return documents


def main() -> int:
    """Sweep chunk size and overlap, print the grid and save it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks" / "chunking_parameter_sweep.json")
    args = parser.parse_args()

    documents = load_parsed_documents(args.data, ROOT / ".cache" / "parsed_documents.pickle")
    questions = load_questions(args.questions)
    scorer = create_dense_scorer()

    rows = []
    for strategy, max_tokens, overlap_tokens in itertools.product(
        ("structure", "fixed"), MAX_TOKENS_GRID, OVERLAP_TOKENS_GRID
    ):
        if overlap_tokens >= max_tokens // 2:
            continue
        config = ChunkingConfig(strategy=strategy, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        chunks = [chunk.model_dump() for document in documents for chunk in chunk_parsed_document(document, config)[0]]
        if not all(find_relevant_passages(chunks, question).any() for question in questions):
            continue  # a window boundary split an evidence string: not comparable
        indexings = {"text": [chunk["chunk_text"] for chunk in chunks]}
        if strategy == "structure":
            indexings["text + section context"] = [chunk["embedding_text"] for chunk in chunks]
        for indexing, texts in indexings.items():
            metrics = score_retrieval(chunks, texts, questions, scorer)
            del metrics["ranks"]
            rows.append(
                {"strategy": strategy, "indexing": indexing, "max_tokens": max_tokens,
                 "overlap_tokens": overlap_tokens, **metrics}
            )  # fmt: skip

    rows.sort(key=lambda row: (-row["mrr@10"], -row["hit@5"], row["top5_tokens"]))
    print("| strategy | indexing | max | overlap | passages | hit@1 | hit@5 | MRR@10 | top-5 tokens |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['strategy']} | {row['indexing']} | {row['max_tokens']} | {row['overlap_tokens']} "
            f"| {row['passages']} | {row['hit@1']:.3f} | {row['hit@5']:.3f} | {row['mrr@10']:.3f} | {row['top5_tokens']} |"
        )
    args.output.write_text(json.dumps({"n_questions": len(questions), "configurations": rows}, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
