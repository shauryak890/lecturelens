"""Load and validate the evaluation dataset ``eval/qa_dataset.jsonl`` (SPEC 12.1).

One JSON object per line::

    {"id": "q01", "type": "factual", "question": "...", "answerable": true,
     "gold": [{"file": "lecture.pdf", "page": 14}], "reference_answer": "...", "history": []}

Follow-up questions carry earlier turns in ``history`` (``[{"question", "answer"}]``).
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from lecturelens.errors import ConfigError
from lecturelens.schemas import ChatTurn

QuestionType = Literal["factual", "conceptual", "comparison", "keyword", "followup", "unanswerable"]


class GoldPage(BaseModel):
    """A page that answers the question."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    file: str = Field(min_length=1)
    page: int = Field(ge=1)


class EvalItem(BaseModel):
    """One evaluation question."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    type: QuestionType
    question: str = Field(min_length=1)
    answerable: bool
    gold: list[GoldPage] = []
    reference_answer: str = ""
    history: list[ChatTurn] = []

    @model_validator(mode="after")
    def _consistent(self) -> "EvalItem":
        if self.answerable != bool(self.gold):
            raise ValueError("answerable questions need gold pages; unanswerable ones none")
        if (self.type == "unanswerable") == self.answerable:
            raise ValueError("type 'unanswerable' must match answerable=false")
        return self

    @property
    def gold_pages(self) -> set[tuple[str, int]]:
        """Gold pages as ``(file, page)`` pairs."""
        return {(g.file, g.page) for g in self.gold}


def load_dataset(path: Path) -> list[EvalItem]:
    """Read and validate the dataset.

    Raises:
        ConfigError: If the file is missing, a line is invalid, or ids repeat. The message
            names the offending line.
    """
    if not path.is_file():
        raise ConfigError(f"Evaluation dataset not found: {path}")
    items: list[EvalItem] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            items.append(EvalItem.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ConfigError(f"{path}:{number}: invalid evaluation item: {exc}") from exc
    ids = [item.id for item in items]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ConfigError(f"{path}: duplicate ids {duplicates}")
    if not items:
        raise ConfigError(f"{path} contains no questions")
    return items


def missing_gold_pages(
    items: list[EvalItem], indexed: set[tuple[str, int]]
) -> dict[str, list[GoldPage]]:
    """Gold pages that are not in the index, by question id (empty if all are present)."""
    missing: dict[str, list[GoldPage]] = {}
    for item in items:
        absent = [g for g in item.gold if (g.file, g.page) not in indexed]
        if absent:
            missing[item.id] = absent
    return missing
