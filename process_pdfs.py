#!/usr/bin/env python
"""Entry point named as in the assessment brief; same command line as ``main.py``.

Example:
    python process_pdfs.py --input ./data --output ./results.json
"""

from __future__ import annotations

import sys

from pdf_chunking.command_line import main

if __name__ == "__main__":
    sys.exit(main())
