"""Flashcard generation with CSV export importable into Anki (FR-9, SPEC 6.7)."""

import csv
import io

from lecturelens.errors import NoContentError
from lecturelens.schemas import FlashcardSet, RetrievedChunk
from lecturelens.tools.base import Scope, StudyResult, StudyTool, valid_ids

FLASHCARDS_TASK = "flashcards"
CSV_COLUMNS = ("front", "back", "source")
SOURCE_SEPARATOR = "; "


def source_label(item: RetrievedChunk) -> str:
    """Short human-readable reference, e.g. ``lecture3.pdf p.14``."""
    return f"{item.chunk.file_name} p.{item.chunk.page}"


def to_csv(cards: FlashcardSet, sources: list[RetrievedChunk]) -> str:
    """Render cards as CSV with ``front, back, source`` columns (Anki: Import > Text file).

    The ``source`` column lists the file and page of each supporting excerpt.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for card in cards.cards:
        refs = SOURCE_SEPARATOR.join(source_label(sources[i - 1]) for i in card.source_ids)
        writer.writerow([card.front, card.back, refs])
    return buffer.getvalue()


class FlashcardGenerator(StudyTool):
    """Generate one-fact flashcards for a topic or a document."""

    def generate(self, scope: Scope, n: int | None = None) -> StudyResult[FlashcardSet]:
        """Create about ``n`` cards (default ``flashcards.default_n``).

        Cards with empty sides or duplicate fronts are dropped; source ids are restricted to
        excerpts that exist.

        Raises:
            NoContentError: If nothing matches the scope or no card survives the checks.
            LLMError: If the LLM call fails.
        """
        cfg = self.settings.flashcards
        n = n or cfg.default_n
        output, sources, usage, label = self._retrieve_and_generate(
            FLASHCARDS_TASK, scope, cfg.context_chunks, label_var="topic", n=n
        )
        assert isinstance(output, FlashcardSet)
        seen: set[str] = set()
        cards = []
        for card in output.cards:
            front, back = card.front.strip(), card.back.strip()
            if not front or not back or front.lower() in seen:
                continue
            seen.add(front.lower())
            cards.append(
                card.model_copy(
                    update={
                        "front": front,
                        "back": back,
                        "source_ids": valid_ids(card.source_ids, len(sources)),
                    }
                )
            )
        if not cards:
            raise NoContentError("The model produced no usable flashcards; try another topic.")
        notes = (
            []
            if len(cards) == len(output.cards)
            else [f"Dropped {len(output.cards) - len(cards)} empty or duplicate card(s)."]
        )
        return StudyResult(FlashcardSet(cards=cards[:n]), sources, usage, label, notes)
