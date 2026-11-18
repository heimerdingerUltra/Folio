"""Text normalisation, de-hyphenation and sentence splitting.

Everything here is pure string manipulation with no PDF dependency, so it is
cheap to unit-test and reusable for other sources (HTML, DOCX...).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Sequence

# --------------------------------------------------------------------------
# Character-level normalisation
# --------------------------------------------------------------------------

# Ligatures are replaced explicitly rather than with unicodedata.normalize
# ("NFKC"), because NFKC also rewrites mathematical symbols (e.g. the
# double-struck R or superscript digits), silently changing formulas.
_LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
}
_CHAR_TRANSLATION = str.maketrans(
    {
        **_LIGATURES,
        " ": " ",  # non-breaking space
        " ": " ",  # thin space
        " ": " ",  # hair space
        " ": " ",  # narrow no-break space
        "­": "",  # soft hyphen (invisible line-break hint)
        "​": "",  # zero-width space
        "﻿": "",  # byte-order mark
        "�": "",  # replacement char produced by undecodable glyphs
    }
)
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTISPACE_RE = re.compile(r"[ \t]+")
# PDF text justification often leaves a stray space before punctuation
# ("accessed , 2026"). Only fix it after a word character and before
# whitespace/end so decimals (".5") and code are left alone.
_SPACE_BEFORE_PUNCT_RE = re.compile(r"(?<=[\w\)\]]) +([,;:.])(?=\s|$)")


def normalize_pdf_text(text: str) -> str:
    """Expand ligatures, drop invisible characters, collapse spaces.

    Newlines are kept (table rows, equations). Ligatures are mapped
    explicitly: NFKC would also rewrite math symbols.

    Args:
        text: Raw text as extracted from a PDF; may span several lines.

    Returns:
        The cleaned text: control characters removed, runs of spaces and tabs reduced to one
        space, the space before a comma, semicolon, colon or period removed, and each line
        stripped.
    """
    text = text.translate(_CHAR_TRANSLATION)
    text = _CONTROL_CHARS_RE.sub("", text)
    text = _MULTISPACE_RE.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    # Trim spaces around each line while keeping the line structure.
    return "\n".join(line.strip() for line in text.split("\n")).strip()


# --------------------------------------------------------------------------
# De-hyphenation
# --------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]*[A-Za-z]|[A-Za-z]")
_TRAILING_FRAGMENT_RE = re.compile(r"([A-Za-z]+(?:-[A-Za-z]+)*)-$")
_LEADING_FRAGMENT_RE = re.compile(r"^([A-Za-z]+)")
# A URL cut by a line break: the line ends inside the URL (or right after its scheme).
_TRAILING_URL_RE = re.compile(r"(\S*://\S*|(?<![A-Za-z])https?:)$")
_URL_CONTINUATION_RE = re.compile(r"^(?=[^\s]*[/_\-=])[\w\-./%#?=&~:]+")


def count_document_words(lines: Iterable[str]) -> Counter[str]:
    """Count lower-cased words (hyphenated compounds kept whole) in ``lines``.

    The vocabulary is used by :func:`join_wrapped_lines` to decide whether a hyphen at
    a line end is a typesetting artefact or part of a real compound word.

    Args:
        lines: Text lines of a whole document.

    Returns:
        A ``Counter`` mapping each word to its number of occurrences.
    """
    vocabulary: Counter[str] = Counter()
    for line in lines:
        # Ignore the last token of lines ending with a hyphen: that token is
        # exactly the ambiguous fragment we are trying to resolve.
        candidate = line.rstrip()
        if candidate.endswith("-"):
            candidate = candidate.rsplit(" ", 1)[0] if " " in candidate else ""
        vocabulary.update(word.lower() for word in _WORD_RE.findall(candidate))
    return vocabulary


def join_wrapped_lines(lines: Sequence[str], vocabulary: Counter[str] | None = None) -> str:
    """Join the visual lines of a paragraph, resolving line-end hyphens.

    "applica-/tions" must join but "byte-/pair" must not. The document's own
    vocabulary decides: keep the hyphen if the hyphenated form occurs
    elsewhere, join if only the joined form does. Without evidence, join when
    the next fragment is lowercase (or both are upper-case), else keep.
    A URL wrapped over two lines is re-joined.

    Args:
        lines: Visual lines of one paragraph, top to bottom. Blank lines are skipped.
        vocabulary: Word counts of the document, as built by :func:`count_document_words`.
            None means no evidence: only the letter-case rule applies.

    Returns:
        The paragraph on a single line.
    """
    vocabulary = vocabulary if vocabulary is not None else Counter()
    result = ""
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if not result:
            result = line
            continue

        trailing = _TRAILING_FRAGMENT_RE.search(result)
        leading = _LEADING_FRAGMENT_RE.match(line)
        if _TRAILING_URL_RE.search(result) and _URL_CONTINUATION_RE.match(line):
            result = result + line  # URL wrapped across two lines
        elif trailing and leading:
            head, tail = trailing.group(1), leading.group(1)
            joined = (head + tail).lower()
            hyphenated = f"{head}-{tail}".lower()
            last_part = head.rsplit("-", 1)[-1]
            is_artefact = tail.islower() or (last_part.isupper() and tail.isupper())
            if vocabulary[hyphenated] > vocabulary[joined]:
                keep_hyphen = True  # rule 1: the compound exists as such
            elif vocabulary[joined] > 0:
                keep_hyphen = False  # the joined word exists elsewhere
            else:
                keep_hyphen = not is_artefact  # rules 2 and 3
            result = result + line if keep_hyphen else result[:-1] + line
        elif result.endswith("-") and trailing:
            result = result + line  # "GPT-" + "3": keep hyphen, no space
        else:
            result = result + " " + line
    return result


# --------------------------------------------------------------------------
# Sentence splitting
# --------------------------------------------------------------------------

# Abbreviations after which a period does NOT end a sentence. Kept small and
# biased towards scientific writing; matching is case-insensitive.
_ABBREVIATIONS = frozenset(
    {
        "al", "fig", "figs", "eq", "eqs", "sec", "secs", "tab", "ref", "refs",
        "e.g", "i.e", "cf", "vs", "etc", "resp", "approx", "no", "nos", "vol",
        "pp", "ch", "dr", "mr", "mrs", "ms", "prof", "inc", "ltd", "st", "et",
        "w.r.t", "a.k.a", "viz", "appx", "app", "def", "thm", "prop", "lem",
    }
)  # fmt: skip

# Candidate boundary: terminal punctuation (optionally followed by closing
# quotes/brackets or a bracketed citation), then whitespace.
_BOUNDARY_RE = re.compile(r"([.!?][\"'”’\)\]]*)(\s+)")
_LAST_TOKEN_RE = re.compile(r"(\S+)$")
# A new sentence plausibly starts with an uppercase letter, a digit, or an
# opening quote/bracket. A lowercase start means we are mid-sentence
# (e.g. after "et al." or "i.e.").
_SENTENCE_START_RE = re.compile(r"[\"'“‘\(\[]?[A-Z0-9Α-Ω]")


def _ends_a_sentence(before: str, after: str) -> bool:
    """Decide whether the punctuation ending ``before`` closes a sentence.

    Args:
        before: Text up to and including the terminal punctuation.
        after: Text following the whitespace after the punctuation.
    """
    if not after or not _SENTENCE_START_RE.match(after):
        return False
    match = _LAST_TOKEN_RE.search(before)
    if not match:
        return True
    token = match.group(1).rstrip("\"'”’)]")
    if not token.endswith("."):
        return True  # "!" or "?" always end a sentence here
    stem = token[:-1].lstrip("([\"'").lower()
    if stem in _ABBREVIATIONS:
        return False
    # Single letters: initials ("J. Smith") or enumerations ("A. Setup").
    if len(stem) == 1 and stem.isalpha():
        return False
    # Section-style numbering ("3.", "3.2.") at the very start of a sentence.
    if re.fullmatch(r"\d+(\.\d+)*", stem) and len(before.split()) <= 1:
        return False
    return True


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, rule-based and tuned for papers.

    Does not split after "et al.", "Fig.", "e.g.", initials or decimals.
    Newline-separated lines (equations, table rows) are never merged or split.

    Args:
        text: Text to split; each newline-separated line is processed on its own.

    Returns:
        The sentences in order, stripped of surrounding whitespace. Blank lines produce
        nothing.
    """
    sentences: list[str] = []
    for segment in text.split("\n"):
        segment = segment.strip()
        if not segment:
            continue
        start = 0
        for match in _BOUNDARY_RE.finditer(segment):
            before = segment[start : match.end(1)]
            after = segment[match.end() :]
            if _ends_a_sentence(before, after):
                sentences.append(before.strip())
                start = match.end()
        tail = segment[start:].strip()
        if tail:
            sentences.append(tail)
    return sentences
