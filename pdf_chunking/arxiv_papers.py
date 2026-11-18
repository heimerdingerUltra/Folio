"""Download arXiv papers: the assessment's five PDFs, or a larger labelled corpus.

Usage:
    download-arxiv-papers [--output ./data]
    download-arxiv-papers --training-corpus [--output ./data/training_corpus]

The five URLs are copied verbatim from the assessment brief. The training
corpus is a fixed list of arXiv identifiers; for each one the arXiv API gives
the title, the authors and the abstract, which are written to
``arxiv_metadata.json`` next to the PDFs and later used as labels.
Files that already exist are skipped, so the script is safe to re-run.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import Any

# Mini-dataset of open-source AI papers (from the assessment brief).
PDF_URLS: dict[str, str] = {
    "Attention_Is_All_You_Need.pdf": "https://arxiv.org/pdf/1706.03762.pdf",
    "LLaMA.pdf": "https://arxiv.org/pdf/2302.13971.pdf",
    "LoRA.pdf": "https://arxiv.org/pdf/2106.09685.pdf",
    "RAG.pdf": "https://arxiv.org/pdf/2005.11401.pdf",
    "ChunkNorris.pdf": "https://arxiv.org/pdf/2602.00010.pdf",
}

# Extra papers used only to train the metadata extractor. None of them is one
# of the five papers above, which stay the only evaluation documents.
TRAINING_CORPUS_ARXIV_IDS: tuple[str, ...] = (
    "1301.3781", "1312.6114", "1406.2661", "1409.0473", "1409.3215", "1412.6980", "1502.03167",
    "1503.02531", "1508.04025", "1512.03385", "1607.06450", "1711.05101", "1802.05365", "1810.04805",
    "1902.00751", "1906.08237", "1907.11692", "1908.10084", "1909.11942", "1910.01108", "1910.10683",
    "1910.13461", "1911.05722", "2002.05709", "2003.10555", "2004.04906", "2004.12832", "2006.11239",
    "2007.01282", "2010.11929", "2101.00190", "2104.09864", "2111.06377", "2112.10752", "2201.11903",
    "2203.15556", "2205.14135", "2210.03629", "2212.10496", "2305.14314",
)  # fmt: skip

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ARXIV_METADATA_FILENAME = "arxiv_metadata.json"
_ATOM = {"atom": "http://www.w3.org/2005/Atom"}
_WHITESPACE_RE = re.compile(r"\s+")
_VERSION_SUFFIX_RE = re.compile(r"v\d+$")
# arXiv asks automated clients to leave three seconds between requests.
_SECONDS_BETWEEN_REQUESTS = 3.0


def _fetch(url: str, timeout: float = 60) -> bytes:
    """GET ``url``. arXiv rejects the default urllib user agent on some networks."""
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _download_pdf(url: str, filepath: Path) -> bool:
    """Download one PDF unless it is already there; returns False on failure."""
    if filepath.exists():
        print(f"Already present: {filepath.name}")
        return True
    print(f"Downloading {filepath.name}...")
    try:
        filepath.write_bytes(_fetch(url))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"  FAILED ({exc})", file=sys.stderr)
        return False
    return True


def download_arxiv_papers(output_dir: Path) -> int:
    """Download every PDF in :data:`PDF_URLS` into ``output_dir``.

    Args:
        output_dir: Destination folder; created if it does not exist.

    Returns:
        The number of files that could not be downloaded.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    return sum(not _download_pdf(url, output_dir / filename) for filename, url in PDF_URLS.items())


def parse_arxiv_api_response(xml_text: str | bytes) -> list[dict[str, Any]]:
    """Read title, authors and abstract of each entry of an arXiv API answer.

    Returns:
        One dictionary per paper: ``arxiv_id`` (without version), ``title``,
        ``authors`` (in the paper's order) and ``abstract``, whitespace collapsed.
    """

    def text_of(element: ElementTree.Element | None) -> str:
        """Element text on one line."""
        return _WHITESPACE_RE.sub(" ", (element.text or "") if element is not None else "").strip()

    papers = []
    for entry in ElementTree.fromstring(xml_text).findall("atom:entry", _ATOM):
        identifier = text_of(entry.find("atom:id", _ATOM)).rsplit("/abs/", 1)[-1]
        title = text_of(entry.find("atom:title", _ATOM))
        if not identifier or not title or title == "Error":
            continue
        papers.append(
            {
                "arxiv_id": _VERSION_SUFFIX_RE.sub("", identifier),
                "title": title,
                "authors": [text_of(author.find("atom:name", _ATOM)) for author in entry.findall("atom:author", _ATOM)],
                "abstract": text_of(entry.find("atom:summary", _ATOM)),
            }
        )
    return papers


def fetch_arxiv_metadata(arxiv_ids: tuple[str, ...] | list[str]) -> list[dict[str, Any]]:
    """Ask the arXiv API for the metadata of ``arxiv_ids`` (one request)."""
    query = urllib.parse.urlencode({"id_list": ",".join(arxiv_ids), "max_results": len(arxiv_ids)})
    return parse_arxiv_api_response(_fetch(f"{ARXIV_API_URL}?{query}"))


def download_training_corpus(
    output_dir: Path, arxiv_ids: tuple[str, ...] | list[str] = TRAINING_CORPUS_ARXIV_IDS
) -> int:
    """Download the PDFs of ``arxiv_ids`` and write their arXiv metadata.

    ``arxiv_metadata.json`` maps each PDF file name to its title, authors and
    abstract. It is rewritten at each call with the papers actually on disk.

    Returns:
        The number of papers that could not be downloaded.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_by_file: dict[str, dict[str, Any]] = {}
    failures = len(arxiv_ids)
    for paper in fetch_arxiv_metadata(arxiv_ids):
        filename = f"arxiv_{paper['arxiv_id'].replace('.', '_').replace('/', '_')}.pdf"
        already_there = (output_dir / filename).exists()
        if _download_pdf(f"https://arxiv.org/pdf/{paper['arxiv_id']}", output_dir / filename):
            metadata_by_file[filename] = paper
            failures -= 1
        if not already_there:
            time.sleep(_SECONDS_BETWEEN_REQUESTS)
    (output_dir / ARXIV_METADATA_FILENAME).write_text(
        json.dumps(metadata_by_file, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return failures


def main() -> int:
    """CLI entry point; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--training-corpus", action="store_true",
                        help="Download the extra training papers and their arXiv metadata.")
    args = parser.parse_args()  # fmt: skip
    if args.training_corpus:
        output = args.output or Path("./data/training_corpus")
        failures = download_training_corpus(output)
    else:
        output = args.output or Path("./data")
        failures = download_arxiv_papers(output)
    print("Dataset ready in the", output, "folder!" if not failures else "folder (with failures).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
