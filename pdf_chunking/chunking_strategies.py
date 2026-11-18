"""Chunking strategies.

``structure`` (default):
    1. rebuild logical units: stitch paragraphs cut by a page or column
       break, attach equations to the sentence introducing them, pair
       tables with their caption (:func:`build_logical_units`);
    2. pack units under a token budget: headings are hard boundaries,
       cuts fall on sentence boundaries, consecutive chunks of a section
       share a sentence-aligned overlap, a table is its own chunk;
    3. merge undersized chunks into a neighbour of the same section.

``fixed``: naive fixed-size token windows, the baseline to beat.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import ValidationError

from pdf_chunking.text_normalization import join_wrapped_lines, split_sentences
from pdf_chunking.pdf_layout_extraction import TextBlock, ParsedDocument, is_bibliography_section
from pdf_chunking.chunk_record_schema import ChunkRecord
from pdf_chunking.token_counting import TokenCounter, load_token_counter

logger = logging.getLogger(__name__)

Strategy = Literal["structure", "fixed"]

_TABLE_CAPTION_RE = re.compile(r"^Table\s+\d+\s*[:.]", re.IGNORECASE)
# A paragraph is "closed" when it ends with terminal punctuation, optionally
# followed by a closing quote/bracket or a citation such as "[12]".
_CLOSED_RE = re.compile(r"[.!?:;][\"'”’\)\]]*(\s*\[[\d,\s–-]+\])?$")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class ChunkingConfig:
    """Chunking parameters.

    Attributes:
        strategy: ``structure`` or the ``fixed`` baseline.
        max_tokens: Hard upper bound on chunk size.
        overlap_tokens: Budget for the sentence-aligned overlap (whole sentences).
        min_tokens: Smaller chunks are merged into a neighbour.
        encoding_name: tiktoken encoding used to count tokens.
    """

    strategy: Strategy = "structure"
    max_tokens: int = 400
    overlap_tokens: int = 50
    min_tokens: int = 40
    encoding_name: str = "cl100k_base"

    def __post_init__(self) -> None:
        """Reject inconsistent token budgets."""
        if self.max_tokens < 32:
            raise ValueError("max_tokens must be at least 32")
        if not 0 <= self.overlap_tokens < self.max_tokens // 2:
            raise ValueError("overlap_tokens must be in [0, max_tokens / 2)")
        if not 0 <= self.min_tokens < self.max_tokens:
            raise ValueError("min_tokens must be in [0, max_tokens)")


# --------------------------------------------------------------------------
# Step 1 - logical units
# --------------------------------------------------------------------------


@dataclass
class LogicalUnit:
    """A logical piece of content that the packer treats as a whole.

    Attributes:
        text: LogicalUnit text. Newlines separate indivisible lines (equations,
            table rows).
        kind: ``heading``, ``text`` or ``table``.
        page_start: First page the unit appears on.
        page_end: Last page the unit appears on.
        section_path: Enclosing headings, outermost first.
        is_float: True for captions and footnotes, which sit *between*
            paragraphs without being part of the running text.
    """

    text: str
    kind: Literal["heading", "text", "table"]
    page_start: int
    page_end: int
    section_path: tuple[str, ...]
    is_float: bool = False


def _stops_mid_sentence(text: str) -> bool:
    """Return True if ``text`` stops mid-sentence (no terminal punctuation)."""
    return not _CLOSED_RE.search(text.rstrip())


def _ends_where_no_paragraph_can(text: str) -> bool:
    """Return True if ``text`` ends where no paragraph can end.

    A trailing comma, opening bracket or dash means the paragraph was cut by
    a page or column break, whatever the next block starts with
    ("... (Smith et al.," + "2021) ...").
    """
    return text.rstrip().endswith((",", "(", "[", "\u2013", "\u2014"))


def _continues_a_sentence(text: str) -> bool:
    """Return True if ``text`` reads as the continuation of a sentence."""
    stripped = text.lstrip()
    return bool(stripped) and (stripped[0].islower() or stripped[0] in ",;)")


def build_logical_units(blocks: list[TextBlock]) -> list[LogicalUnit]:
    """Turn layout blocks into logical units.

    * Paragraph stitching: a block ending mid-sentence followed by one that
      continues it is one paragraph cut by a page/column break; captions,
      footnotes and tables floating in between are skipped over.
    * Equation attachment: a displayed equation joins the sentence that
      introduces it, and the "where ..." clause that follows.
    * Caption pairing: a table is merged with its adjacent "Table N" caption.

    Args:
        blocks: Blocks of one document in reading order, with ``section_path`` already assigned.

    Returns:
        The units in reading order. Headings and tables stay units of their own; captions and
        footnotes become ``text`` units with ``is_float`` set; stitched paragraphs span the
        pages of all the blocks they absorbed.
    """
    # Pair each table with an adjacent, not-yet-used table caption.
    caption_of: dict[int, int] = {}
    used_captions: set[int] = set()
    for index, block in enumerate(blocks):
        if block.block_type != "table":
            continue
        for neighbour in (index - 1, index + 1):
            if (
                0 <= neighbour < len(blocks)
                and neighbour not in used_captions
                and blocks[neighbour].block_type == "caption"
                and blocks[neighbour].page_number == block.page_number
                and _TABLE_CAPTION_RE.match(blocks[neighbour].text)
            ):
                caption_of[index] = neighbour
                used_captions.add(neighbour)
                break

    units: list[LogicalUnit] = []
    # Index in ``units`` of the last running-text paragraph that later
    # blocks may be stitched onto; reset at every heading.
    flow_index: int | None = None
    flow_ends_with_equation = False

    for index, block in enumerate(blocks):
        if index in used_captions:
            continue
        kind = block.block_type

        if kind == "heading":
            units.append(
                LogicalUnit(block.text, "heading", block.page_number, block.page_number, block.section_path)
            )
            flow_index = None
            continue

        if kind == "table":
            text = block.text
            if index in caption_of:
                text = blocks[caption_of[index]].text + "\n" + text
            units.append(
                LogicalUnit(text, "table", block.page_number, block.page_number, block.section_path)
            )
            continue

        if kind in ("caption", "footnote"):
            units.append(
                LogicalUnit(
                    block.text,
                    "text",
                    block.page_number,
                    block.page_number,
                    block.section_path,
                    is_float=True,
                )
            )
            continue

        flow = units[flow_index] if flow_index is not None else None
        same_flow = (
            flow is not None
            and flow.section_path == block.section_path
            and block.page_number - flow.page_end <= 1
        )

        if kind == "equation":
            if same_flow and flow is not None:
                flow.text += "\n" + block.text
                flow.page_end = block.page_number
                flow_ends_with_equation = True
                continue
            # No paragraph to attach to: the equation starts a unit itself.
            units.append(
                LogicalUnit(block.text, "text", block.page_number, block.page_number, block.section_path)
            )
            flow_index = len(units) - 1
            flow_ends_with_equation = True
            continue

        # Titles and paragraphs.
        if is_bibliography_section(block.section_path):
            # Bibliography: an entry with a hanging indent arrives as two
            # blocks ("[16] K. Grace ... When will AI" / "exceed human
            # performance? ..."). Entries end with a period, so an entry
            # that is still open simply absorbs the next block - whatever
            # its first letter, since titles are capitalised.
            can_stitch = (
                kind == "paragraph" and same_flow and flow is not None and _stops_mid_sentence(flow.text)
            )
        else:
            can_stitch = (
                kind == "paragraph"
                and same_flow
                and flow is not None
                and (
                    (_continues_a_sentence(block.text) and (flow_ends_with_equation or _stops_mid_sentence(flow.text)))
                    or (not flow_ends_with_equation and _ends_where_no_paragraph_can(flow.text))
                )
            )
        if can_stitch and flow is not None:
            # After an equation, keep the continuation on a new line so the
            # formula stays an indivisible line of its own.
            if flow_ends_with_equation:
                flow.text += "\n" + block.text
            else:
                # join_wrapped_lines also repairs a word hyphenated across the break.
                flow.text = join_wrapped_lines([flow.text, block.text])
            flow.page_end = block.page_number
        else:
            units.append(
                LogicalUnit(block.text, "text", block.page_number, block.page_number, block.section_path)
            )
            flow_index = len(units) - 1
        flow_ends_with_equation = False

    return units


# --------------------------------------------------------------------------
# Step 2 - packing
# --------------------------------------------------------------------------


@dataclass
class _ChunkDraft:
    """A chunk under construction.

    ``pieces`` holds ``(separator, text)`` pairs; the separator is the string
    placed *before* the text when the chunk is rendered ("\\n\\n" between
    units, "\\n" between lines of a unit, " " between sentences).
    """

    section_path: tuple[str, ...]
    kind: Literal["text", "table"] = "text"
    order: int = 0
    pieces: list[tuple[str, str]] = field(default_factory=list)
    tokens: int = 0
    pages: list[int] = field(default_factory=list)
    body_sentences: list[tuple[str, int]] = field(default_factory=list)
    overlap_pieces: int = 0  # number of leading pieces copied from the previous chunk
    has_body: bool = False

    def add(self, separator: str, text: str, tokens: int, pages: tuple[int, int]) -> None:
        """Append a piece of text and update the running totals.

        Args:
            separator: String placed before ``text`` when the chunk is rendered; ignored for the
                first piece of the chunk.
            text: The piece to append.
            tokens: Token count of ``text`` alone; one more token is charged for the separator.
            pages: ``(first, last)`` 1-based pages the piece appears on.
        """
        self.pieces.append((separator if self.pieces else "", text))
        self.tokens += tokens + 1  # +1 covers the separator's token
        self.pages.extend(pages)

    def render(self, skip_overlap: bool = False) -> str:
        """Return the chunk text, optionally without its overlap prefix.

        Args:
            skip_overlap: If True, leave out the leading pieces copied from the previous chunk.

        Returns:
            The pieces joined with their separators, without leading or trailing whitespace.
        """
        pieces = self.pieces[self.overlap_pieces :] if skip_overlap else self.pieces
        return "".join(separator + text for separator, text in pieces).strip()


def _split_oversized_text_on_whitespace(text: str, max_tokens: int, count: TokenCounter) -> list[str]:
    """Hard-split text that has no usable sentence boundary.

    Last-resort fallback (e.g. a 600-token "sentence" produced by a broken
    table or a long URL list): split on whitespace into windows that fit.
    """
    parts: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for word in text.split():
        word_tokens = count(" " + word)
        if current and current_tokens + word_tokens > max_tokens:
            parts.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(word)
        current_tokens += word_tokens
    if current:
        parts.append(" ".join(current))
    return parts


def _split_table_into_drafts(unit: LogicalUnit, order: int, config: ChunkingConfig, count: TokenCounter) -> list[_ChunkDraft]:
    """Turn a table unit into one chunk, or several if it exceeds the budget.

    When a table must be split, the caption and header row are repeated at
    the top of every part so each part remains interpretable on its own.
    """
    pages = (unit.page_start, unit.page_end)
    total = count(unit.text)
    if total <= config.max_tokens:
        draft = _ChunkDraft(unit.section_path, "table", order, has_body=True)
        draft.add("", unit.text, total, pages)
        return [draft]

    lines = unit.text.split("\n")
    # Prefix = caption (if any) + header row, unless that alone is too big.
    n_prefix = 2 if _TABLE_CAPTION_RE.match(lines[0]) and len(lines) > 2 else 1
    prefix = "\n".join(lines[:n_prefix])
    prefix_tokens = count(prefix) + 1
    if prefix_tokens > config.max_tokens // 2:
        prefix, prefix_tokens, n_prefix = "", 0, 0
    budget = config.max_tokens - prefix_tokens

    groups: list[list[str]] = [[]]
    used = 0
    for line in lines[n_prefix:]:
        # A single row longer than the budget is broken on whitespace.
        for part in _split_oversized_text_on_whitespace(line, budget, count) if count(line) > budget else [line]:
            part_tokens = count(part) + 1
            if groups[-1] and used + part_tokens > budget:
                groups.append([])
                used = 0
            groups[-1].append(part)
            used += part_tokens

    drafts: list[_ChunkDraft] = []
    for group in groups:
        if not group:
            continue
        text = "\n".join(([prefix] if prefix else []) + group)
        draft = _ChunkDraft(unit.section_path, "table", order, has_body=True)
        draft.add("", text, count(text), pages)
        drafts.append(draft)
    return drafts


def _select_overlap_sentences(draft: _ChunkDraft, budget: int) -> list[tuple[str, int]]:
    """Select the trailing sentences of ``draft`` that fit in ``budget``.

    Only whole sentences are carried over. If even the last sentence is
    larger than the budget, no overlap is produced: half a sentence would be
    noise, not context.
    """
    tail: list[tuple[str, int]] = []
    used = 0
    for sentence, tokens in reversed(draft.body_sentences):
        if used + tokens > budget:
            break
        tail.append((sentence, tokens))
        used += tokens
    return list(reversed(tail))


def _pack_units_into_drafts(units: list[LogicalUnit], config: ChunkingConfig, count: TokenCounter) -> list[_ChunkDraft]:
    """Greedily pack units into drafts under the token budget."""
    drafts: list[_ChunkDraft] = []
    current: _ChunkDraft | None = None

    def close(carry_overlap: bool, order: int) -> _ChunkDraft | None:
        """Finish the current draft; optionally seed the next with overlap."""
        nonlocal current
        finished = current
        current = None
        if finished is None:
            return None
        drafts.append(finished)
        if not carry_overlap or config.overlap_tokens == 0:
            return None
        tail = _select_overlap_sentences(finished, config.overlap_tokens)
        if not tail:
            return None
        seeded = _ChunkDraft(finished.section_path, order=order)
        for position, (sentence, tokens) in enumerate(tail):
            seeded.add(" " if position else "", sentence, tokens, (finished.pages[-1],))
        seeded.overlap_pieces = len(seeded.pieces)
        return seeded

    for order, unit in enumerate(units):
        pages = (unit.page_start, unit.page_end)

        if unit.kind == "table":
            # Tables are emitted on the side: the text chunk being built
            # stays open, because a table usually floats mid-paragraph.
            drafts.extend(_split_table_into_drafts(unit, order, config, count))
            continue

        if unit.kind == "heading":
            # Hard boundary - unless the current draft holds only headings
            # ("3 Model" directly followed by "3.1 Encoder"), which stay
            # together with the body that follows.
            if current is not None and current.has_body:
                close(carry_overlap=False, order=order)
            if current is None:
                current = _ChunkDraft(unit.section_path, order=order)
            else:
                # Pending headings with no body (their section held only a
                # table, say): keep the ancestors of the new heading, drop
                # stale siblings so they are not glued onto this section.
                ancestors = set(unit.section_path[:-1])
                kept = [(sep, text) for sep, text in current.pieces if text in ancestors]
                current = _ChunkDraft(unit.section_path, order=order)
                for _, text in kept:
                    current.add("\n\n", text, count(text), pages)
            current.section_path = unit.section_path
            current.add("\n\n", unit.text, count(unit.text), pages)
            continue

        if current is None:
            current = _ChunkDraft(unit.section_path, order=order)
        unit_tokens = count(unit.text)

        # Sentences eligible for the overlap. Captions and footnotes are
        # excluded: repeating a footnote URL at the top of the next chunk
        # adds noise, not context.
        def overlap_candidates(unit: LogicalUnit = unit) -> list[tuple[str, int]]:
            """Sentences of the unit with their token counts; none for a table."""
            if unit.is_float:
                return []
            return [(s, count(s)) for s in split_sentences(unit.text)]

        # Case 1 - the whole unit fits in the current chunk.
        if current.tokens + unit_tokens + 1 <= config.max_tokens:
            current.add("\n\n", unit.text, unit_tokens, pages)
            current.has_body = True
            current.body_sentences.extend(overlap_candidates())
            continue

        # Case 2 - it does not fit here but fits in a fresh chunk, and the
        # current chunk is already reasonably full: break between paragraphs.
        if (
            unit_tokens + 1 <= config.max_tokens
            and current.has_body
            and current.tokens >= config.max_tokens // 2
        ):
            section = current.section_path
            current = close(carry_overlap=True, order=order) or _ChunkDraft(section, order=order)
            if current.tokens + unit_tokens + 1 > config.max_tokens:
                current = _ChunkDraft(section, order=order)  # overlap would not fit: drop it
            current.add("\n\n", unit.text, unit_tokens, pages)
            current.has_body = True
            current.body_sentences.extend(overlap_candidates())
            continue

        # Case 3 - fill sentence by sentence, closing chunks as they fill up.
        first_in_unit = True
        for line_index, line in enumerate(unit.text.split("\n")):
            for sentence_index, sentence in enumerate(split_sentences(line)):
                sentence_tokens = count(sentence)
                # A "sentence" over the budget cannot be kept whole.
                parts = (
                    _split_oversized_text_on_whitespace(sentence, config.max_tokens - 1, count)
                    if sentence_tokens + 1 > config.max_tokens
                    else [sentence]
                )
                for part in parts:
                    part_tokens = count(part) if len(parts) > 1 else sentence_tokens
                    if current.tokens + part_tokens + 1 > config.max_tokens and current.has_body:
                        section = current.section_path
                        current = close(carry_overlap=True, order=order) or _ChunkDraft(section, order=order)
                        if current.tokens + part_tokens + 1 > config.max_tokens:
                            current = _ChunkDraft(section, order=order)
                        first_in_unit = True
                    if first_in_unit:
                        separator = "\n\n"
                    elif sentence_index == 0 and line_index > 0:
                        separator = "\n"
                    else:
                        separator = " "
                    current.add(separator, part, part_tokens, pages)
                    current.has_body = True
                    if not unit.is_float:
                        current.body_sentences.append((part, part_tokens))
                    first_in_unit = False

    # A trailing draft made only of headings carries no content: its
    # section is already recorded on the table chunks that follow it.
    if current is not None and current.has_body:
        drafts.append(current)
    # Tables were appended as encountered; restore document order.
    drafts.sort(key=lambda draft: draft.order)
    return drafts


def _merge_undersized_drafts(drafts: list[_ChunkDraft], config: ChunkingConfig, count: TokenCounter) -> list[_ChunkDraft]:
    """Merge text chunks under ``min_tokens`` into a neighbouring text chunk.

    A 15-token chunk (a lone heading, an orphan footnote) embeds poorly and
    wastes a retrieval slot. It is merged into the previous text chunk, or
    else the next one, provided both belong to the same top-level section
    and the result still fits the budget. Otherwise it is left as is.
    """

    def compatible(a: _ChunkDraft, b: _ChunkDraft) -> bool:
        """Can two drafts be merged? Both are text of the same top-level section."""
        return a.kind == b.kind == "text" and a.section_path[:1] == b.section_path[:1]

    merged: list[_ChunkDraft] = []
    pending: _ChunkDraft | None = None  # small draft waiting to be merged forward
    for draft in drafts:
        if pending is not None:
            body = pending.render(skip_overlap=True)
            if compatible(pending, draft) and count(body) + draft.tokens + 1 <= config.max_tokens:
                # Prepend the small chunk; the receiver's own overlap prefix
                # is dropped since the small chunk now provides the context.
                rest = draft.pieces[draft.overlap_pieces :]
                draft.pieces = [("", body)] + [("\n\n", rest[0][1])] + rest[1:] if rest else [("", body)]
                draft.overlap_pieces = 0
                draft.tokens = count(draft.render())
                draft.pages = pending.pages + draft.pages
                draft.section_path = pending.section_path
                draft.order = pending.order
            else:
                merged.append(pending)
            pending = None

        is_small = draft.kind == "text" and count(draft.render(skip_overlap=True)) < config.min_tokens
        if not is_small:
            merged.append(draft)
            continue
        # Look back past tables for the previous text chunk.
        previous = next((item for item in reversed(merged) if item.kind == "text"), None)
        body = draft.render(skip_overlap=True)
        if (
            previous is not None
            and compatible(previous, draft)
            and previous.tokens + count(body) + 1 <= config.max_tokens
        ):
            previous.add("\n\n", body, count(body), tuple(draft.pages))
        else:
            pending = draft
    if pending is not None:
        merged.append(pending)
    return merged


# --------------------------------------------------------------------------
# Baseline strategy
# --------------------------------------------------------------------------


def _cut_fixed_token_windows(blocks: list[TextBlock], config: ChunkingConfig, count: TokenCounter) -> list[_ChunkDraft]:
    """Naive baseline: fixed-size token windows with overlap.

    All text is flattened into one stream of words and cut every
    ``max_tokens`` tokens, stepping back ``overlap_tokens`` each time. No
    sentence, paragraph, heading or table boundary is respected. This is what
    a default "recursive character splitter" degenerates to on PDF text, and
    it is the reference against which the quality metrics are read.
    """
    words: list[tuple[str, int, int]] = []  # (word, tokens, page)
    for block in blocks:
        for word in block.text.split():
            words.append((word, count(" " + word), block.page_number))

    drafts: list[_ChunkDraft] = []
    start = 0
    while start < len(words):
        end, used = start, 0
        while end < len(words) and used + words[end][1] <= config.max_tokens:
            used += words[end][1]
            end += 1
        end = max(end, start + 1)  # always make progress
        text = " ".join(word for word, _, _ in words[start:end])
        # Per-word counts are approximate; trim until the window truly fits.
        while end > start + 1 and count(text) > config.max_tokens:
            end -= 1
            text = " ".join(word for word, _, _ in words[start:end])
        draft = _ChunkDraft((), order=len(drafts), has_body=True)
        draft.add("", text, used, (words[start][2], words[end - 1][2]))
        drafts.append(draft)
        if end >= len(words):
            break
        # Step back by roughly ``overlap_tokens`` for the next window.
        back, overlap = end, 0
        while back > start + 1 and overlap + words[back - 1][1] <= config.overlap_tokens:
            back -= 1
            overlap += words[back][1]
        start = back
    return drafts


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def slugify_document_name(source_document: str) -> str:
    """Return a lowercase identifier derived from a file name.

    Args:
        source_document: File name of the document; the text after its last dot is dropped.

    Returns:
        The lowercased stem with each run of characters other than ``a-z`` and ``0-9`` replaced
        by one underscore, or ``"document"`` if nothing is left.

    Example:
        >>> slugify_document_name("Attention_Is_All_You_Need.pdf")
        'attention_is_all_you_need'
    """
    stem = source_document.rsplit(".", 1)[0]
    return _SLUG_RE.sub("_", stem.lower()).strip("_") or "document"


def chunk_parsed_document(
    document: ParsedDocument,
    config: ChunkingConfig | None = None,
) -> tuple[list[ChunkRecord], int]:
    """Chunk a parsed document into validated records.

    Args:
        document: Output of :func:`pdf_chunking.pdf_layout_extraction.extract_pdf_document`.
        config: Chunking parameters; defaults to :class:`ChunkingConfig`.

    Returns:
        A pair ``(records, n_rejected)`` where ``n_rejected`` counts chunks
        that failed schema validation (logged and excluded, never fatal).
    """
    config = config or ChunkingConfig()
    count = load_token_counter(config.encoding_name)

    if config.strategy == "fixed":
        drafts = _cut_fixed_token_windows(document.blocks, config, count)
    else:
        units = build_logical_units(document.blocks)
        drafts = _merge_undersized_drafts(_pack_units_into_drafts(units, config, count), config, count)

    slug = slugify_document_name(document.source_document)
    records: list[ChunkRecord] = []
    rejected = 0
    for draft in drafts:
        text = draft.render()
        if not text:
            continue
        if draft.kind == "table":
            content_type = "table"
        elif is_bibliography_section(draft.section_path):
            content_type = "references"
        else:
            content_type = "text"
        index = len(records)
        section = " > ".join(draft.section_path)
        # Contextual header: a chunk saying "we set it to 0.1" is only
        # retrievable if the index also knows which paper and section it is in.
        context = " - ".join(part for part in (document.title, section) if part)
        with_context = config.strategy == "structure" and bool(context)
        try:
            records.append(
                ChunkRecord(
                    chunk_id=f"{slug}-{index:04d}",
                    source_document=document.source_document,
                    document_title=document.title,
                    page_number=min(draft.pages),
                    page_end=max(draft.pages),
                    section=section,
                    content_type=content_type,
                    chunk_index=index,
                    token_count=count(text),
                    char_count=len(text),
                    chunk_text=text,
                    embedding_text=f"{context}\n\n{text}" if with_context else text,
                )
            )
        except ValidationError as exc:
            # Defensive: a malformed chunk must not abort the document.
            rejected += 1
            logger.warning(
                "%s: dropped invalid chunk (%s)", document.source_document, exc.errors()[0]["msg"]
            )
    return records, rejected
