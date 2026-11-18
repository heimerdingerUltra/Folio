"""Download the assessment's mini-dataset of five arXiv papers.

Usage:
    download-arxiv-papers [--output ./data]

The URL list is copied verbatim from the assessment brief. Files that already
exist are skipped, so the script is safe to re-run.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Mini-dataset of open-source AI papers (from the assessment brief).
PDF_URLS: dict[str, str] = {
    "Attention_Is_All_You_Need.pdf": "https://arxiv.org/pdf/1706.03762.pdf",
    "LLaMA.pdf": "https://arxiv.org/pdf/2302.13971.pdf",
    "LoRA.pdf": "https://arxiv.org/pdf/2106.09685.pdf",
    "RAG.pdf": "https://arxiv.org/pdf/2005.11401.pdf",
    "ChunkNorris.pdf": "https://arxiv.org/pdf/2602.00010.pdf",
}


def download_arxiv_papers(output_dir: Path) -> int:
    """Download every PDF in :data:`PDF_URLS` into ``output_dir``.

    Args:
        output_dir: Destination folder; created if it does not exist.

    Returns:
        The number of files that could not be downloaded.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for filename, url in PDF_URLS.items():
        filepath = output_dir / filename
        if filepath.exists():
            print(f"Already present: {filename}")
            continue
        print(f"Downloading {filename}...")
        try:
            # arXiv rejects the default urllib user agent on some networks.
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                filepath.write_bytes(response.read())
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            failures += 1
            print(f"  FAILED ({exc})", file=sys.stderr)
    return failures


def main() -> int:
    """CLI entry point; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("./data"))
    args = parser.parse_args()
    failures = download_arxiv_papers(args.output)
    print("Dataset ready in the", args.output, "folder!" if not failures else "folder (with failures).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
