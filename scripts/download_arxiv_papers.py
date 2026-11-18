"""Download the assessment's five arXiv papers from a checkout.

Usage:
    python scripts/download_arxiv_papers.py [--output ./data]

The code is in ``pdf_chunking.arxiv_papers`` (``download-arxiv-papers`` once
the package is installed).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pdf_chunking.arxiv_papers import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
