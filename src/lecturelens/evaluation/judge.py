"""LLM-as-judge for generation quality (SPEC 12.2), following the RAGAS decomposition.

* Faithfulness: the judge splits the answer into atomic claims and marks each as supported
  by the retrieved context or not; the score is supported / total claims.
* Relevancy: the judge rates on a 1-5 rubric how well the answer addresses the question; a
  correct refusal for an unanswerable question scores 5.

The judge uses ``llm.judge_model`` (a cheaper model) at temperature 0, so results are cached
and reproducible. LLM judges have biases: compare configurations, don't read scores as absolute.
"""

import logging
from dataclasses import dataclass

from lecturelens.llm.client import LLMClient
from lecturelens.schemas import FaithfulnessJudgement, RelevancyJudgement, Usage

logger = logging.getLogger(__name__)

FAITHFULNESS_TASK = "judge_faithfulness"
RELEVANCY_TASK = "judge_relevancy"


@dataclass(frozen=True)
class Faithfulness:
    """Claim-level verdicts and the resulting score (``None`` when there are no claims)."""

    claims: list[str]
    supported: list[bool]
    score: float | None
    reasoning: str


class Judge:
    """Scores answers with the prompt file's judge tasks."""

    def __init__(self, llm: LLMClient, model: str) -> None:
        """Create the judge.

        Args:
            llm: LLM client (prompts come from its registry).
            model: Judge model id (``llm.judge_model``).
        """
        self.llm = llm
        self.model = model

    def faithfulness(self, answer: str, context: str) -> tuple[Faithfulness, Usage]:
        """Share of the answer's claims supported by ``context``."""
        result, usage = self.llm.run_task(
            FAITHFULNESS_TASK, model=self.model, context=context, answer=answer
        )
        assert isinstance(result, FaithfulnessJudgement)
        # the two lists are meant to be parallel; if the judge miscounts, use the overlap
        n = min(len(result.claims), len(result.supported))
        if n != max(len(result.claims), len(result.supported)):
            logger.warning(
                "Judge returned %d claims but %d verdicts",
                len(result.claims),
                len(result.supported),
            )
        claims, supported = result.claims[:n], result.supported[:n]
        score = sum(supported) / n if n else None
        return Faithfulness(claims, supported, score, result.reasoning), usage

    def relevancy(self, question: str, answer: str, answerable: bool) -> tuple[int, Usage]:
        """1-5 rating of how well ``answer`` addresses ``question``."""
        result, usage = self.llm.run_task(
            RELEVANCY_TASK,
            model=self.model,
            question=question,
            answer=answer,
            answerable="yes" if answerable else "no",
        )
        assert isinstance(result, RelevancyJudgement)
        return result.score, usage
