#!/usr/bin/env python
"""Run the pipeline from a checkout; the code is in ``pdf_chunking.command_line``.

Example:
    python main.py --input ./data --output ./results.json
"""

from __future__ import annotations

import sys

from pdf_chunking.command_line import main

if __name__ == "__main__":
    sys.exit(main())
