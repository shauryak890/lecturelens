"""Build the numbered excerpt block the answer prompt cites from (SPEC 3.2 step 5, 6.7).

Each chunk becomes ``[S1] (lecture3.pdf, p.14, Unit 2 > Word2Vec)`` followed by its text. The
label format lives in prompts.yaml (``formats.excerpt``). Chunks are added in rank order until
the token budget is reached, and the chunks actually used are returned so that ``[S{i}]``
always maps to ``sources[i - 1]``.
"""

from lecturelens.llm.prompts import PromptRegistry
from lecturelens.schemas import RetrievedChunk


def format_excerpt(registry: PromptRegistry, n: int, item: RetrievedChunk) -> str:
    """Render one numbered excerpt, e.g. ``[S1] (notes.pdf, p.3, Unit 1)\\n<text>``."""
    chunk = item.chunk
    heading = (
        registry.format("excerpt_heading", heading_path=chunk.heading_path)
        if chunk.heading_path
        else ""
    )
    return registry.format(
        "excerpt", n=n, file_name=chunk.file_name, page=chunk.page, heading=heading, text=chunk.text
    )


def build_context(
    chunks: list[RetrievedChunk], max_tokens: int, registry: PromptRegistry
) -> tuple[str, list[RetrievedChunk]]:
    """Number the chunks S1..Sn and join them, stopping at the token budget.

    The first chunk is always included, even if it alone exceeds the budget, so a question
    never fails for lack of context. Token counts are the chunker's embedding-tokenizer counts.

    Args:
        chunks: Retrieved chunks, best first.
        max_tokens: Context budget (``retrieval.max_context_tokens``).
        registry: Prompt registry providing the excerpt format.

    Returns:
        The context block and the chunks it contains (``[S{i}]`` is ``used[i - 1]``).
    """
    used: list[RetrievedChunk] = []
    total = 0
    for item in chunks:
        if used and total + item.chunk.n_tokens > max_tokens:
            break
        used.append(item)
        total += item.chunk.n_tokens
    separator = registry.format("excerpt_separator")
    text = separator.join(format_excerpt(registry, i, item) for i, item in enumerate(used, 1))
    return text, used
