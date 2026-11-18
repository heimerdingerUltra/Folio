"""Non-ML baseline: regular expressions and layout cues.

Any learned extractor has to beat this to justify its cost. It also shows
which fields are easy (a title is the first line of the first page) and
which are not (a problem statement cannot be found with a regex).
"""

from __future__ import annotations

import re

from metadata_extraction.paper_metadata_schema import PaperMetadata

_PERSON_NAME_RE = re.compile(
    r"\b[A-ZÀ-Þ][a-zß-ÿ]+(?:[- ][A-Z]\.)?(?:[ -][A-ZÀ-Þ][a-zß-ÿ]+){1,2}\b"
)
_METRIC_SENTENCE_RE = re.compile(
    r"[^.]*\b\d+(?:\.\d+)?\s?(?:%|BLEU|F1|EM|ms|Wh|times|days|GB|[BM] parameters)[^.]*\.", flags=re.IGNORECASE
)
_FRONT_MATTER_HINT_RE = re.compile(r"@|\bUniversity\b|\bResearch\b|\bLaboratoire\b|\bInstitute\b|\bMeta AI\b")
_NOT_A_PERSON = frozenset(
    {"Google Brain", "Google Research", "Facebook Research", "New York", "Jean Monnet", "Hubert Curien"}
)
_ORGANISATION_WORDS = frozenset({"University", "Research", "Institute", "Laboratoire", "Brain", "College", "Lab"})


def _strip_trailing_organisation_word(name: str) -> str:
    """"Alan Turing University" -> "Alan Turing" (the regex is greedy on capitalised words)."""
    words = name.split(" ")
    return " ".join(words[:-1]) if words[-1] in _ORGANISATION_WORDS else name


def extract_metadata_with_rules(chunk_text: str) -> str:
    """Return a JSON answer built from surface patterns only.

    * a chunk with e-mail addresses or affiliations is treated as front
      matter: its first line is the title and capitalised name pairs are
      authors;
    * a sentence quoting a figure with a known unit is a key metric;
    * the problem statement is never attempted.

    Args:
        chunk_text: Text of one chunk, line breaks included: the first non-blank line is the
            title candidate.

    Returns:
        JSON serialisation of a ``PaperMetadata`` with at most three key metrics and a null
        problem statement.
    """
    lines = [line.strip() for line in chunk_text.split("\n") if line.strip()]
    is_front_matter = bool(_FRONT_MATTER_HINT_RE.search(chunk_text))
    title = None
    authors: list[str] = []
    if is_front_matter and lines:
        title = lines[0]
        body = "\n".join(lines[1:])
        candidates = (_strip_trailing_organisation_word(name) for name in _PERSON_NAME_RE.findall(body))
        authors = [name for name in candidates if " " in name and name not in _NOT_A_PERSON]
    metrics = [] if is_front_matter else [m.strip() for m in _METRIC_SENTENCE_RE.findall(chunk_text)][:3]
    return PaperMetadata(title=title, main_authors=authors, key_metrics=metrics).model_dump_json()
