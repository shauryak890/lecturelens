"""Structured summaries with cited key points and a glossary (FR-8, SPEC 6.7)."""

from lecturelens.schemas import Summary
from lecturelens.tools.base import Scope, StudyResult, StudyTool, valid_ids

SUMMARY_TASK = "summarize"


class Summarizer(StudyTool):
    """Summarise a topic or a whole document."""

    def summarize(self, scope: Scope) -> StudyResult[Summary]:
        """Return an overview, ordered key points and glossary, each item with sources.

        Items whose ``source_ids`` all point outside the context are kept but lose those ids,
        so the UI never links to a source that does not exist.

        Raises:
            NoContentError: If nothing matches the scope.
            LLMError: If the LLM call fails.
        """
        cfg = self.settings.summary
        output, sources, usage, label = self._retrieve_and_generate(
            SUMMARY_TASK,
            scope,
            cfg.context_chunks,
            label_var="scope",
            n_points=cfg.n_points,
            n_terms=cfg.n_terms,
        )
        assert isinstance(output, Summary)
        n = len(sources)
        summary = output.model_copy(
            update={
                "key_points": [
                    p.model_copy(update={"source_ids": valid_ids(p.source_ids, n)})
                    for p in output.key_points
                ],
                "glossary": [
                    g.model_copy(update={"source_ids": valid_ids(g.source_ids, n)})
                    for g in output.glossary
                ],
            }
        )
        return StudyResult(summary, sources, usage, label)
