"""Parse and validate ``[S#]`` citations in generated answers (SPEC 3.2 step 7, 6.7).

The model is told to cite only excerpt numbers it was given, but it can still invent one
("[S9]" with five excerpts). Invalid ids are removed from both the answer text and
``cited_sources``; an answer that claims to be answerable but is left without a single valid
citation is flagged for the one-time repair prompt.
"""

import re

from pydantic import BaseModel

from lecturelens.schemas import AnswerResponse

_CITATION = re.compile(r"\[S(\d+)\]")
# "[S1, S3]", "[S1,S3]" or "[S1, 3]" -> "[S1][S3]" so every citation has one canonical form
_CITATION_GROUP = re.compile(r"\[S\d+(?:\s*[,;]\s*S?\d+)+\]")
_GROUP_ID = re.compile(r"\d+")
_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+([.,;:!?])")
_DOUBLE_SPACE = re.compile(r"[ \t]{2,}")


class CitationCheck(BaseModel):
    """Result of validating an answer's citations."""

    response: AnswerResponse  # cleaned answer (invalid ids removed)
    removed: list[int]  # invalid ids that were stripped
    needs_repair: bool  # answerable, but no valid citation left

    @property
    def had_invalid(self) -> bool:
        """Whether any cited id did not exist in the context."""
        return bool(self.removed)


def normalize_citations(text: str) -> str:
    """Rewrite grouped citations such as ``[S1, S3]`` as ``[S1][S3]``."""
    return _CITATION_GROUP.sub(
        lambda m: "".join(f"[S{n}]" for n in _GROUP_ID.findall(m.group(0))), text
    )


def extract(text: str) -> list[int]:
    """Return cited source numbers in order of first appearance (``[S1, S3]`` included)."""
    seen: dict[int, None] = {}
    for match in _CITATION.finditer(normalize_citations(text)):
        seen.setdefault(int(match.group(1)), None)
    return list(seen)


def validate(response: AnswerResponse, n_sources: int) -> CitationCheck:
    """Remove citations outside ``1..n_sources`` and check that grounding is intact.

    ``cited_sources`` of the cleaned response is the sorted union of valid ids listed by the
    model and valid ids used in the text, so the UI can rely on it.

    Args:
        response: The model's answer.
        n_sources: Number of excerpts in the context.

    Returns:
        The cleaned response, the removed ids and whether a repair is needed.
    """
    valid = set(range(1, n_sources + 1))
    text = normalize_citations(response.answer)
    in_text = extract(text)
    removed = sorted({n for n in [*in_text, *response.cited_sources] if n not in valid})

    if removed:
        text = _CITATION.sub(lambda m: m.group(0) if int(m.group(1)) in valid else "", text)
        text = _DOUBLE_SPACE.sub(" ", _SPACE_BEFORE_PUNCT.sub(r"\1", text)).strip()
    cited = sorted(({n for n in in_text if n in valid} | set(response.cited_sources)) & valid)
    cleaned = response.model_copy(update={"answer": text, "cited_sources": cited})
    return CitationCheck(
        response=cleaned,
        removed=removed,
        needs_repair=response.answerable and not extract(text),
    )
