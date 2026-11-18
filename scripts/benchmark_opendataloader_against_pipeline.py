"""Compare OpenDataLoader PDF with this repository's PyMuPDF extraction.

Usage:
    python scripts/benchmark_opendataloader_against_pipeline.py [--input ./data] [--tfidf-only]

Both parsers feed the same chunker with the same settings; the measures are
described in :mod:`pdf_chunking.parser_comparison`. Rows marked ``*`` favour
the PyMuPDF pipeline by construction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metadata_extraction.gold_paper_metadata import GOLD_PAPER_METADATA  # noqa: E402
from pdf_chunking.parser_comparison import compare_pdf_parsers, tabulate_parser_comparison  # noqa: E402
from pdf_chunking.retrieval_benchmark import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_QUESTIONS_PATH,
    SimilarityScorer,
    create_dense_scorer,
    create_tfidf_scorer,
    load_questions,
)


def main() -> int:
    """Run the comparison, print it as Markdown and save it as JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=ROOT / "data")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--tfidf-only", action="store_true", help="Skip the dense retriever (no model download).")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks" / "opendataloader_comparison.json")
    args = parser.parse_args()
    paths = sorted(args.input.glob("*.pdf"))
    if not paths:
        print(f"No PDF found in {args.input}", file=sys.stderr)
        return 1

    scorers: dict[str, SimilarityScorer] = {"tf-idf": create_tfidf_scorer()}
    if not args.tfidf_only:
        try:
            scorers[f"dense ({DEFAULT_EMBEDDING_MODEL})"] = create_dense_scorer(DEFAULT_EMBEDDING_MODEL)
        except Exception as error:  # noqa: BLE001 - package missing, no network...
            print(f"Dense retriever unavailable ({error}); using TF-IDF only.", file=sys.stderr)

    gold_titles = {name: metadata["title"] for name, metadata in GOLD_PAPER_METADATA.items()}
    comparison = compare_pdf_parsers(paths, load_questions(args.questions), scorers, gold_titles)

    rows = tabulate_parser_comparison(comparison)
    columns = list(rows[0])
    print("| " + " | ".join(columns) + " |")
    print("|---|" + "---:|" * (len(columns) - 1))
    for row in rows:
        print("| " + " | ".join(str(row[column]) for column in columns) + " |")
    print("\n* favours the PyMuPDF pipeline by construction.")
    args.output.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
