"""End-to-end tests of the CLI: outputs, run report and exit codes."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest

from pdf_chunking import command_line
from pdf_chunking import dataset_validation_cli as validation_cli
from pdf_chunking.chunk_record_schema import ChunkRecord


def run_cli(*args: object) -> int:
    """Invoke the CLI in-process; the heuristic table mode keeps it fast."""
    return command_line.main([str(arg) for arg in args] + ["--table-detection", "heuristic", "--log-level", "ERROR"])


@pytest.fixture
def input_dir(tmp_path: Path, simple_pdf: Path, two_column_pdf: Path) -> Path:
    folder = tmp_path / "pdfs"
    folder.mkdir()
    shutil.copy(simple_pdf, folder / "simple.pdf")
    shutil.copy(two_column_pdf, folder / "two_column.pdf")
    return folder


def test_success_writes_dataset_and_report(input_dir: Path, tmp_path: Path) -> None:
    output = tmp_path / "out" / "results.json"
    assert run_cli("--input", input_dir, "--output", output, "--benchmark-against-fixed-windows") == 0

    records = json.loads(output.read_text(encoding="utf-8"))
    assert records, "no chunks exported"
    for record in records:
        ChunkRecord.model_validate(record)  # exported rows satisfy the schema
        assert {"chunk_id", "source_document", "page_number", "chunk_text"} <= record.keys()
    assert {r["source_document"] for r in records} == {"simple.pdf", "two_column.pdf"}

    report = json.loads((tmp_path / "out" / "results.run_report.json").read_text(encoding="utf-8"))
    assert report["run"]["status"] == "success"
    assert report["summary"]["documents_processed"] == 2
    assert report["summary"]["error_rate"] == 0.0
    assert report["summary"]["chunks"] == len(records)
    assert report["validation"]["is_valid"] is True
    assert "token_length" in report["quality"] and "baseline_quality" in report
    assert all("extraction_seconds" in entry for entry in report["documents"])


def test_bad_files_are_reported_but_do_not_stop_the_batch(
    input_dir: Path, corrupted_pdf: Path, image_only_pdf: Path, tmp_path: Path
) -> None:
    shutil.copy(corrupted_pdf, input_dir / "corrupted.pdf")
    shutil.copy(image_only_pdf, input_dir / "scanned.pdf")
    (input_dir / "empty.pdf").write_bytes(b"")
    output = tmp_path / "results.json"

    assert run_cli("--input", input_dir, "--output", output) == 1  # partial success

    report = json.loads((tmp_path / "results.run_report.json").read_text(encoding="utf-8"))
    status = {entry["source_document"]: entry["status"] for entry in report["documents"]}
    assert status == {
        "corrupted.pdf": "failed",
        "empty.pdf": "failed",
        "scanned.pdf": "no_text_layer",
        "simple.pdf": "ok",
        "two_column.pdf": "ok",
    }
    assert report["run"]["status"] == "partial"
    assert report["summary"]["documents_failed"] == 3
    assert report["summary"]["error_rate"] == 0.6
    assert all(entry["error"] for entry in report["documents"] if entry["status"] != "ok")
    # The good documents were still exported.
    records = json.loads(output.read_text(encoding="utf-8"))
    assert {r["source_document"] for r in records} == {"simple.pdf", "two_column.pdf"}


def test_exit_code_2_when_nothing_can_be_exported(tmp_path: Path, corrupted_pdf: Path) -> None:
    output = tmp_path / "results.json"
    assert run_cli("--input", tmp_path / "missing", "--output", output) == 2
    empty = tmp_path / "empty_dir"
    empty.mkdir()
    assert run_cli("--input", empty, "--output", output) == 2
    assert run_cli("--input", corrupted_pdf, "--output", output) == 2
    assert not output.exists()


def test_rejects_bad_arguments(input_dir: Path, tmp_path: Path) -> None:
    assert run_cli("--input", input_dir, "--output", tmp_path / "results.xml") == 2
    assert run_cli("--input", input_dir, "--output", tmp_path / "r.json", "--overlap-tokens", 500) == 2


@pytest.mark.parametrize("extension", [".csv", ".jsonl"])
def test_other_formats_round_trip_through_the_validator(
    input_dir: Path, tmp_path: Path, extension: str
) -> None:
    output = tmp_path / f"results{extension}"
    assert run_cli("--input", input_dir, "--output", output) == 0
    if extension == ".csv":
        with output.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert rows and list(rows[0]) == list(ChunkRecord.model_fields)
    assert validation_cli.main([str(output), "--max-tokens", "400"]) == 0


def test_validator_flags_a_tampered_dataset(input_dir: Path, tmp_path: Path) -> None:
    output = tmp_path / "results.json"
    assert run_cli("--input", input_dir, "--output", output) == 0
    records = json.loads(output.read_text(encoding="utf-8"))
    records[0]["page_number"] = 0
    records[1]["chunk_id"] = records[2]["chunk_id"]
    output.write_text(json.dumps(records), encoding="utf-8")
    assert validation_cli.main([str(output)]) == 1


def test_single_file_input_and_fixed_strategy(simple_pdf: Path, tmp_path: Path) -> None:
    output = tmp_path / "results.json"
    assert run_cli("--input", simple_pdf, "--output", output, "--strategy", "fixed", "--max-tokens", 64,
                   "--overlap-tokens", 8, "--min-tokens", 0) == 0  # fmt: skip
    records = json.loads(output.read_text(encoding="utf-8"))
    assert len(records) > 3 and all(r["token_count"] <= 64 for r in records)


def test_command_from_the_assessment_brief_still_works(simple_pdf: Path, tmp_path: Path) -> None:
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "results.json"
    completed = subprocess.run(
        [sys.executable, str(root / "process_pdfs.py"), "--input", str(simple_pdf), "--output", str(output),
         "--table-detection", "heuristic", "--log-level", "ERROR"],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
    )  # fmt: skip
    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text(encoding="utf-8"))
