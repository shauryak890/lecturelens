"""Multiple-choice quiz generation (FR-7, SPEC 6.7).

One LLM call writes the questions; deterministic post-checks then make the quiz usable:

* exactly 4 distinct options, and no "all/none of the above";
* ``source_ids`` restricted to excerpts that exist, and questions with no valid source dropped;
* near-identical questions removed (word-overlap Jaccard above ``quiz.dedup_similarity``);
* options shuffled with a seeded RNG, so the correct answer is not always "A" yet the same
  quiz is reproducible.
"""

import logging
import random
import re

from lecturelens.errors import NoContentError
from lecturelens.schemas import Quiz, QuizQuestion
from lecturelens.tools.base import Scope, StudyResult, StudyTool, valid_ids

logger = logging.getLogger(__name__)

QUIZ_TASK = "quiz"
OPTIONS_PER_QUESTION = 4  # fixed by the Quiz schema and the prompt
_CATCH_ALL = re.compile(r"\b(all|none|both|neither) of the (above|options)\b", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def jaccard(a: str, b: str) -> float:
    """Word-set overlap of two texts (0 = disjoint, 1 = same words)."""
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / len(wa | wb) if wa | wb else 1.0


def shuffle_options(question: QuizQuestion, rng: random.Random) -> QuizQuestion:
    """Permute the options with ``rng`` and move ``correct_index`` along with them."""
    order = list(range(len(question.options)))
    rng.shuffle(order)
    return question.model_copy(
        update={
            "options": [question.options[i] for i in order],
            "correct_index": order.index(question.correct_index),
        }
    )


def check_question(question: QuizQuestion, n_sources: int) -> tuple[QuizQuestion | None, str]:
    """Validate one question; return it with cleaned source ids, or ``None`` and why."""
    options = [o.strip() for o in question.options]
    if len({o.lower() for o in options}) != OPTIONS_PER_QUESTION or not all(options):
        return None, "duplicate or empty options"
    if any(_CATCH_ALL.search(o) for o in options):
        return None, "'all/none of the above' option"
    sources = valid_ids(question.source_ids, n_sources)
    if not sources:
        return None, "no valid source"
    return question.model_copy(update={"options": options, "source_ids": sources}), ""


def post_check(
    quiz: Quiz, n_sources: int, n: int, seed: int, dedup_similarity: float
) -> tuple[Quiz, list[str]]:
    """Apply all post-checks; return the cleaned quiz (at most ``n`` questions) and notes."""
    kept: list[QuizQuestion] = []
    notes: list[str] = []
    for number, question in enumerate(quiz.questions, start=1):
        checked, problem = check_question(question, n_sources)
        if checked is None:
            notes.append(f"Dropped question {number}: {problem}.")
            continue
        if any(jaccard(checked.question, q.question) >= dedup_similarity for q in kept):
            notes.append(f"Dropped question {number}: near-duplicate of an earlier question.")
            continue
        kept.append(checked)
    if len(kept) > n:
        kept = kept[:n]
    rng = random.Random(seed)
    shuffled = [shuffle_options(q, rng) for q in kept]
    return quiz.model_copy(update={"questions": shuffled}), notes


class QuizGenerator(StudyTool):
    """Generate a grounded MCQ quiz for a topic or a document."""

    def generate(
        self, scope: Scope, n: int | None = None, difficulty: str | None = None
    ) -> StudyResult[Quiz]:
        """Create up to ``n`` questions (capped at ``quiz.max_n``).

        Args:
            scope: Topic or document to quiz on.
            n: Number of questions; defaults to ``quiz.default_n``.
            difficulty: ``easy``, ``medium``, ``hard`` or ``mixed``.

        Raises:
            NoContentError: If nothing matches the scope or no question survives the checks.
            LLMError: If the LLM call fails.
        """
        cfg = self.settings.quiz
        n = min(n or cfg.default_n, cfg.max_n)
        output, sources, usage, label = self._retrieve_and_generate(
            QUIZ_TASK,
            scope,
            cfg.context_chunks,
            label_var="topic",
            n=n,
            difficulty=difficulty or cfg.default_difficulty,
        )
        assert isinstance(output, Quiz)
        quiz, notes = post_check(output, len(sources), n, cfg.seed, cfg.dedup_similarity)
        if not quiz.questions:
            raise NoContentError("The model produced no usable questions; try another topic.")
        if len(quiz.questions) < n:
            notes.append(f"{len(quiz.questions)} of {n} requested questions passed the checks.")
        return StudyResult(quiz.model_copy(update={"topic": label}), sources, usage, label, notes)
