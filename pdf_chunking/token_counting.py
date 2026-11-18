"""Token counting used to size chunks.

Chunk budgets are expressed in *tokens*, not characters, because embedding
models and LLM context windows are limited in tokens. We use ``tiktoken``'s
``cl100k_base`` encoding as a model-agnostic proxy: it is fast, deterministic
and within roughly 10-20 % of the WordPiece / SentencePiece tokenizers used by
common open-source embedding models. The default chunk budget leaves headroom
for that mismatch (see ``ChunkingConfig``).
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Callable

logger = logging.getLogger(__name__)

# Rough fallback: words and standalone punctuation marks. English prose
# averages ~1.3 BPE tokens per whitespace word, which this approximates.
_FALLBACK_TOKEN_RE = re.compile(r"\w+|[^\w\s]")

TokenCounter = Callable[[str], int]


def _estimate_token_count(text: str) -> int:
    """Count tokens with a regex heuristic (used when tiktoken is unavailable)."""
    return len(_FALLBACK_TOKEN_RE.findall(text))


@lru_cache(maxsize=4)
def load_token_counter(encoding_name: str = "cl100k_base") -> TokenCounter:
    """Return a ``str -> int`` token counter (tiktoken).

    Falls back to a regex estimate when tiktoken or its vocabulary is
    unavailable (offline machine), so the pipeline never fails on it.

    Args:
        encoding_name: Name of a tiktoken encoding, such as ``cl100k_base``.

    Returns:
        A function giving the number of tokens of a string. The result is cached per
        ``encoding_name``.
    """
    try:
        import tiktoken

        encoding = tiktoken.get_encoding(encoding_name)
    except Exception as exc:  # noqa: BLE001 - ImportError, network errors, bad name...
        logger.warning(
            "tiktoken encoding %r unavailable (%s); using approximate regex token counts.",
            encoding_name,
            exc,
        )
        return _estimate_token_count

    def count(text: str) -> int:
        """Number of tokens in ``text`` for the selected encoding."""
        # disallowed_special=() so literal strings such as "<|endoftext|>"
        # appearing in a paper are counted as plain text instead of raising.
        return len(encoding.encode(text, disallowed_special=()))

    return count
