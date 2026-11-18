"""The distribution declared in ``pyproject.toml`` matches the code."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from pdf_chunking.retrieval_benchmark import DEFAULT_QUESTIONS_PATH, load_questions

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def project() -> dict:
    tomllib = pytest.importorskip("tomllib")  # Python >= 3.11
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_declared_packages_exist(project: dict) -> None:
    for package in project["tool"]["setuptools"]["packages"]:
        assert (ROOT / package / "__init__.py").is_file()


def test_console_scripts_point_to_callables(project: dict) -> None:
    for target in project["project"]["scripts"].values():
        module, function = target.split(":")
        assert callable(getattr(importlib.import_module(module), function))


def test_benchmark_questions_ship_inside_the_package(project: dict) -> None:
    assert DEFAULT_QUESTIONS_PATH.name in project["tool"]["setuptools"]["package-data"]["pdf_chunking"]
    assert len(load_questions()) == 40
