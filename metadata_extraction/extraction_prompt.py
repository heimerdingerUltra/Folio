"""Prompts shown to the model, as chat messages."""

from __future__ import annotations

from typing import Sequence

from metadata_extraction.grounded_training_examples import TrainingExample
from metadata_extraction.paper_metadata_schema import PaperMetadata

EXTRACTION_INSTRUCTION = (
    "You extract metadata from ONE chunk of a research paper.\n"
    "Return ONLY a JSON object with exactly these keys:\n"
    '- "title": string or null\n'
    '- "main_authors": list of strings\n'
    '- "core_problem_statement": string or null\n'
    '- "key_metrics": list of strings\n'
    "Use only information stated in the chunk. If the chunk does not state a field, "
    "use null or an empty list. Do not guess."
)


def _user_turn(chunk_text: str) -> dict[str, str]:
    """The user message: instruction followed by the chunk."""
    return {"role": "user", "content": f"{EXTRACTION_INSTRUCTION}\n\n### Chunk\n{chunk_text}"}


def build_chat_messages(chunk_text: str, target: PaperMetadata | None = None) -> list[dict[str, str]]:
    """User turn (instruction + chunk), plus the assistant answer when ``target`` is given."""
    messages = [_user_turn(chunk_text)]
    if target is not None:
        messages.append({"role": "assistant", "content": target.model_dump_json()})
    return messages


def select_few_shot_demonstrations(
    train_examples: Sequence[TrainingExample], max_chunk_characters: int = 900
) -> list[TrainingExample]:
    """Pick one informative and one empty training example as demonstrations.

    Showing both teaches the format *and* that an empty answer is legitimate.
    The shortest candidates are used and truncated: demonstrations compete
    with the query chunk for the context window.
    """
    def shortest(candidates: list[TrainingExample]) -> TrainingExample:
        """The shortest candidate, its text truncated to the character budget."""
        chosen = min(candidates, key=lambda example: len(example.chunk_text))
        return TrainingExample(
            chosen.chunk_id, chosen.document, chosen.chunk_text[:max_chunk_characters], chosen.target
        )

    with_title = [example for example in train_examples if example.target.title]
    empty = [example for example in train_examples if example.target.is_empty]
    return [shortest(group) for group in (with_title, empty) if group]


def build_few_shot_messages(chunk_text: str, demonstrations: Sequence[TrainingExample]) -> list[dict[str, str]]:
    """In-context learning: solved examples as earlier turns, then the query."""
    messages: list[dict[str, str]] = []
    for demonstration in demonstrations:
        messages.extend(build_chat_messages(demonstration.chunk_text, demonstration.target))
    messages.append(_user_turn(chunk_text))
    return messages
