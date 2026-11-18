"""Score the structure-driven metadata extractor against arXiv records.

Usage:
    python scripts/evaluate_document_metadata.py [--corpus ./data/training_corpus] [--papers ./data]

Titles and authors of the extra papers come from the arXiv API; those of the
five assessment papers from the hand-written gold. The script also measures
what chunk routing saves on the hand-labelled chunk dataset. Results are
printed and written to ``benchmarks/document_metadata_extraction.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metadata_extraction.arxiv_training_examples import load_arxiv_metadata  # noqa: E402
from metadata_extraction.document_metadata_extraction import (  # noqa: E402
    extract_document_metadata,
    select_chunks_worth_reading,
)
from metadata_extraction.gold_paper_metadata import GOLD_PAPER_METADATA  # noqa: E402
from metadata_extraction.grounded_training_examples import (  # noqa: E402
    build_grounded_examples,
    comparison_key,
    load_chunk_dataset,
)
from pdf_chunking.arxiv_papers import ARXIV_METADATA_FILENAME  # noqa: E402
from pdf_chunking.pdf_layout_extraction import PdfExtractionConfig, PdfExtractionError, extract_pdf_document  # noqa: E402


def score_documents(reference_by_path: dict[Path, dict]) -> dict:
    """Title accuracy, author precision / recall and abstract localisation."""
    rows = []
    for path, reference in sorted(reference_by_path.items()):
        try:
            document = extract_pdf_document(path, PdfExtractionConfig(table_detection="heuristic"))
        except PdfExtractionError as error:
            print(f"skipped {path.name}: {error}", file=sys.stderr)
            continue
        found = extract_document_metadata(document).metadata
        expected = {comparison_key(name) for name in reference["authors"]}
        predicted = {comparison_key(name) for name in found.main_authors}
        row = {
            "document": path.name,
            "title_correct": comparison_key(found.title or "") == comparison_key(reference["title"]),
            "authors_expected": len(expected),
            "authors_predicted": len(predicted),
            "authors_correct": len(expected & predicted),
        }
        if abstract := reference.get("abstract"):
            # The statement is the opening of the abstract: is it the right paragraph?
            opening = comparison_key(abstract)[:60]
            row["abstract_located"] = comparison_key(found.core_problem_statement or "").startswith(opening)
        rows.append(row)

    correct = sum(row["authors_correct"] for row in rows)
    with_abstract = [row for row in rows if "abstract_located" in row]
    return {
        "documents": len(rows),
        "title_accuracy": round(sum(row["title_correct"] for row in rows) / len(rows), 3),
        "author_precision": round(correct / max(sum(row["authors_predicted"] for row in rows), 1), 3),
        "author_recall": round(correct / max(sum(row["authors_expected"] for row in rows), 1), 3),
        "documents_with_all_authors": sum(
            row["authors_correct"] == row["authors_expected"] == row["authors_predicted"] for row in rows
        ),
        "abstract_located": f"{sum(row['abstract_located'] for row in with_abstract)}/{len(with_abstract)}",
        "per_document": rows,
    }


def score_chunk_routing(chunk_dataset_path: Path) -> dict:
    """Share of chunks read after routing, and informative chunks kept."""
    chunks = load_chunk_dataset(chunk_dataset_path)
    informative = {example.chunk_id for example in build_grounded_examples(chunks) if not example.target.is_empty}
    selected = {chunk["chunk_id"] for chunk in select_chunks_worth_reading(chunks)}
    return {
        "chunks": len(chunks),
        "chunks_read": len(selected),
        "share_read": round(len(selected) / len(chunks), 3),
        "informative_chunks": len(informative),
        "informative_chunks_kept": len(informative & selected),
    }


def main() -> int:
    """Run both evaluations, print them and save them."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=Path("./data/training_corpus"))
    parser.add_argument("--papers", type=Path, default=Path("./data"))
    parser.add_argument("--chunks", type=Path, default=Path("./results.json"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/document_metadata_extraction.json"))
    args = parser.parse_args()

    references = {args.papers / name: gold for name, gold in GOLD_PAPER_METADATA.items()}
    metadata_path = args.corpus / ARXIV_METADATA_FILENAME
    if metadata_path.exists():
        references.update({args.corpus / name: paper for name, paper in load_arxiv_metadata(metadata_path).items()})
    results = {"extraction": score_documents(references), "chunk_routing": score_chunk_routing(args.chunks)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {key: value for key, value in results["extraction"].items() if key != "per_document"}
    print(json.dumps({"extraction": summary, "chunk_routing": results["chunk_routing"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
