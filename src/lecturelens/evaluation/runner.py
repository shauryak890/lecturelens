"""Evaluation runner (SPEC 12): retrieval metrics, ablations, chunk sweep, generation metrics.

* Retrieval: Hit@k, Recall@k, nDCG@k and MRR for the default configuration and every
  ``eval.ablations`` entry. No LLM calls, except condensing follow-up questions once (cached).
* Chunk sweep: each file is parsed and cleaned once, then re-chunked at every size in
  ``eval.chunk_sweep`` into a temporary index, so OCR is not repeated per size.
* Generation: every question goes through the real pipeline; an LLM judge scores faithfulness
  and relevancy, and citation validity, citation hit rate, abstention accuracy, latency and
  tokens are measured. The same is done for the "minimal prompt" ablation (no grounding rules)
  on a smaller subset to protect the daily API quota.

Results are returned as :class:`EvalReport` and written by :mod:`lecturelens.evaluation.report`.
"""

import logging
import math
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from lecturelens.config import Settings
from lecturelens.errors import ConfigError, LLMError
from lecturelens.evaluation import retrieval_metrics as rm
from lecturelens.evaluation.dataset import EvalItem
from lecturelens.evaluation.judge import Judge
from lecturelens.indexing.bm25_index import BM25Index
from lecturelens.indexing.indexer import chunking_settings
from lecturelens.indexing.vector_store import VectorStore
from lecturelens.ingestion.chunker import chunk_pages, indexed_text
from lecturelens.ingestion.cleaner import clean_pages
from lecturelens.ingestion.loader import discover_files, doc_id_from_hash, file_hash, load_pages
from lecturelens.logging_utils import utc_timestamp
from lecturelens.rag.context import build_context
from lecturelens.rag.pipeline import RAGPipeline
from lecturelens.retrieval.retriever import HybridRetriever
from lecturelens.schemas import AskResult, Page, RetrievedChunk
from lecturelens.services import Services

logger = logging.getLogger(__name__)

DEFAULT_NAME = "default"
MINIMAL_NAME = "minimal_prompt"
MINIMAL_TASK = "answer_minimal"
MS_PER_S = 1000
P50, P95 = 50, 95

Progress = Callable[[str], None]


# ---------------------------------------------------------------- result models


class RetrievalRow(BaseModel):
    """Per-question retrieval outcome."""

    id: str
    type: str
    query: str
    gold: list[str]
    retrieved: list[str]  # "file p.N" per retrieved chunk, best first
    first_gold_rank: int | None


class RetrievalResult(BaseModel):
    """Retrieval metrics for one configuration."""

    name: str
    overrides: dict[str, Any] = {}
    metrics: dict[str, float]
    latency_ms_p50: float
    per_question: list[RetrievalRow]


class SweepResult(BaseModel):
    """Retrieval metrics at one chunk size."""

    chunk_size_tokens: int
    n_chunks: int
    avg_tokens_per_chunk: float
    metrics: dict[str, float]


class GenerationRow(BaseModel):
    """Per-question generation outcome."""

    id: str
    type: str
    question: str
    standalone_question: str = ""
    answerable: bool
    predicted_answerable: bool | None = None
    answer: str = ""
    confidence: str = ""
    cited_pages: list[str] = []
    gold: list[str] = []
    gold_cited: bool | None = None
    faithfulness: float | None = None
    unsupported_claims: list[str] = []
    relevancy: int | None = None
    removed_citations: list[int] = []
    repaired: bool = False
    llm_answered: bool = False  # False when the pipeline abstained without an LLM call
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


class GenerationResult(BaseModel):
    """Generation metrics for one prompt variant."""

    name: str
    answer_task: str
    n_questions: int
    metrics: dict[str, float | None]
    per_question: list[GenerationRow]


class EvalReport(BaseModel):
    """Everything one evaluation run produced (written to results.json)."""

    created_at: str
    prompt_version: str
    models: dict[str, str]
    dataset: str
    n_questions: int
    config: dict[str, Any]
    retrieval: list[RetrievalResult] = []
    chunk_sweep: list[SweepResult] = []
    generation: list[GenerationResult] = []
    notes: list[str] = []


# ---------------------------------------------------------------- helpers


def page_label(file_name: str, page: int) -> str:
    """``lecture.pdf p.14``."""
    return f"{file_name} p.{page}"


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (0 for an empty list)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def minimal_prompt_subset(items: list[EvalItem], n: int) -> list[EvalItem]:
    """Half unanswerable (if available), the rest answerable first-turn questions, in order."""
    unanswerable = [i for i in items if not i.answerable][: n // 2]
    answerable = [i for i in items if i.answerable and not i.history]
    return unanswerable + answerable[: n - len(unanswerable)]


def generation_metrics(rows: list[GenerationRow]) -> dict[str, float | None]:
    """Aggregate per-question generation outcomes (SPEC 12.2 table)."""
    ok = [r for r in rows if r.error is None]
    answered = [r for r in ok if r.llm_answered]
    answerable_answered = [r for r in answered if r.answerable and r.predicted_answerable]
    # abstention as the positive class: predicted "not answerable"
    tp = sum(1 for r in ok if not r.predicted_answerable and not r.answerable)
    fp = sum(1 for r in ok if not r.predicted_answerable and r.answerable)
    fn = sum(1 for r in ok if r.predicted_answerable and not r.answerable)
    latencies = [r.latency_ms for r in ok]
    return {
        "faithfulness": _mean([r.faithfulness for r in ok if r.faithfulness is not None]),
        "relevancy": _mean([float(r.relevancy) for r in ok if r.relevancy is not None]),
        "citation_validity": _mean([float(not r.removed_citations) for r in answered]),
        "repair_rate": _mean([float(r.repaired) for r in answered]),
        "citation_hit_rate": _mean([float(bool(r.gold_cited)) for r in answerable_answered]),
        "abstention_accuracy": _mean([float(r.predicted_answerable == r.answerable) for r in ok]),
        "abstention_precision": tp / (tp + fp) if tp + fp else None,
        "abstention_recall": tp / (tp + fn) if tp + fn else None,
        "no_llm_abstentions": float(sum(1 for r in ok if not r.llm_answered)),
        "latency_ms_p50": percentile(latencies, P50),
        "latency_ms_p95": percentile(latencies, P95),
        "prompt_tokens_per_query": _mean([float(r.prompt_tokens) for r in answered]),
        "errors": float(len(rows) - len(ok)),
    }


# ---------------------------------------------------------------- runner


class EvalRunner:
    """Runs the evaluation for a dataset against the configured index and models."""

    def __init__(
        self, services: Services, items: list[EvalItem], on_progress: Progress | None = None
    ) -> None:
        """Create the runner.

        Args:
            services: Shared components (index, retriever, LLM).
            items: Validated evaluation questions.
            on_progress: Optional callback receiving short status messages.
        """
        self.services = services
        self.settings: Settings = services.settings
        self.items = items
        self._progress = on_progress or (lambda message: None)
        self._queries: dict[str, str] = {}
        self.notes: list[str] = []

    # ------------------------------------------------------------ orchestration

    def run(
        self, *, generation: bool = True, ablations: bool = False, chunk_sweep: bool = False
    ) -> EvalReport:
        """Run the selected parts and return the report (nothing is written here)."""
        s = self.settings
        report = EvalReport(
            created_at=utc_timestamp(),
            prompt_version=self.services.registry.version,
            models={
                "llm": s.llm.model,
                "judge": s.llm.judge_model,
                "embeddings": s.embeddings.model,
                "reranker": s.retrieval.reranker_model,
            },
            dataset=str(s.eval.dataset),
            n_questions=len(self.items),
            config=s.snapshot(),
        )
        report.retrieval.append(self.evaluate_retrieval(DEFAULT_NAME, {}))
        if ablations:
            for ablation in s.eval.ablations:
                report.retrieval.append(self.evaluate_retrieval(ablation.name, ablation.overrides))
        if chunk_sweep:
            report.chunk_sweep = self.chunk_sweep()
        if generation:
            report.generation.append(
                self.evaluate_generation(DEFAULT_NAME, self.services.pipeline, self.items)
            )
            subset = minimal_prompt_subset(self.items, s.eval.minimal_prompt_questions)
            if subset:
                minimal = RAGPipeline(
                    s,
                    self.services.retriever,
                    self.services.llm,
                    self.services.registry,
                    answer_task=MINIMAL_TASK,
                )
                report.generation.append(self.evaluate_generation(MINIMAL_NAME, minimal, subset))
        report.notes = self.notes
        return report

    # ------------------------------------------------------------ retrieval

    def query_for(self, item: EvalItem) -> str:
        """Search query for an item: follow-ups are condensed once (LLM, cached)."""
        if item.id not in self._queries:
            query = item.question
            if item.history:
                try:
                    query, _ = self.services.pipeline.standalone_question(
                        item.question, item.history
                    )
                except (ConfigError, LLMError) as exc:
                    query = f"{item.history[-1].question} {item.question}"
                    self.notes.append(
                        f"{item.id}: follow-up not condensed ({exc}); used previous question "
                        "+ follow-up as the query."
                    )
            self._queries[item.id] = query
        return self._queries[item.id]

    def _retriever_for(self, settings: Settings, base: HybridRetriever) -> HybridRetriever:
        return HybridRetriever(
            base.embedder,
            base.store,
            base.bm25,
            base.reranker,
            settings.retrieval,
            settings.ingestion.add_context_header,
        )

    def evaluate_retrieval(self, name: str, overrides: dict[str, Any]) -> RetrievalResult:
        """Retrieval metrics for one configuration over the answerable questions."""
        self._progress(f"Retrieval: {name}")
        variant = self.settings.with_overrides(overrides)
        return self._retrieval_metrics(
            name, overrides, self._retriever_for(variant, self.services.retriever)
        )

    def _retrieval_metrics(
        self, name: str, overrides: dict[str, Any], retriever: HybridRetriever
    ) -> RetrievalResult:
        k_values = self.settings.eval.k_values
        top_k = max(k_values)
        rows, rankings, golds, latencies = [], [], [], []
        for item in (i for i in self.items if i.answerable):
            query = self.query_for(item)
            started = time.perf_counter()
            results = retriever.retrieve(query, final_k=top_k)
            latencies.append((time.perf_counter() - started) * MS_PER_S)
            ranking = [(r.chunk.file_name, r.chunk.page) for r in results]
            rankings.append(ranking)
            golds.append(item.gold_pages)
            rows.append(
                RetrievalRow(
                    id=item.id,
                    type=item.type,
                    query=query,
                    gold=[page_label(f, p) for f, p in sorted(item.gold_pages)],
                    retrieved=[page_label(f, p) for f, p in ranking],
                    first_gold_rank=rm.first_gold_rank(ranking, item.gold_pages),
                )
            )
        return RetrievalResult(
            name=name,
            overrides=overrides,
            metrics=rm.aggregate(rankings, golds, k_values),
            latency_ms_p50=percentile(latencies, P50),
            per_question=rows,
        )

    # ------------------------------------------------------------ chunk sweep

    def _parsed_pages(self) -> list[Page]:
        """Parse and clean every indexed source file once (the slow part, incl. OCR)."""
        s = self.settings
        pages: list[Page] = []
        folders = [Path(s.app.data_dir), Path(s.app.sample_dir)]
        for path in (
            f
            for folder in folders
            if folder.is_dir()
            for f in discover_files(folder, s.ingestion.extensions)
        ):
            self._progress(f"Chunk sweep: parsing {path.name}")
            doc_id = doc_id_from_hash(file_hash(path))
            raw = load_pages(path, doc_id, use_ocr=s.ingestion.use_ocr)
            pages.extend(clean_pages(raw, s.ingestion))
        return pages

    def chunk_sweep(self) -> list[SweepResult]:
        """Retrieval metrics when the corpus is re-chunked at each ``eval.chunk_sweep`` size."""
        pages = self._parsed_pages()
        counter = self.services.indexer.counter
        embedder = self.services.embedder
        base = self.services.retriever
        results = []
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            for size in self.settings.eval.chunk_sweep:
                self._progress(f"Chunk sweep: {size} tokens")
                variant = self.settings.with_overrides({"ingestion.chunk_size_tokens": size})
                chunks = chunk_pages(pages, counter, variant.ingestion)
                texts = [indexed_text(c, variant.ingestion.add_context_header) for c in chunks]
                store = VectorStore(Path(tmp) / f"chunks_{size}")
                store.upsert(chunks, embedder.embed_documents(texts))
                bm25 = BM25Index(variant.retrieval.bm25)
                bm25.build([c.chunk_id for c in chunks], texts)
                retriever = HybridRetriever(
                    embedder,
                    store,
                    bm25,
                    base.reranker,
                    variant.retrieval,
                    variant.ingestion.add_context_header,
                )
                metrics = self._retrieval_metrics(f"chunks_{size}", {}, retriever).metrics
                results.append(
                    SweepResult(
                        chunk_size_tokens=size,
                        n_chunks=len(chunks),
                        avg_tokens_per_chunk=sum(c.n_tokens for c in chunks) / max(len(chunks), 1),
                        metrics=metrics,
                    )
                )
        logger.info("Chunk sweep settings: %s", chunking_settings(self.settings))
        return results

    # ------------------------------------------------------------ generation

    def evaluate_generation(
        self, name: str, pipeline: RAGPipeline, items: list[EvalItem]
    ) -> GenerationResult:
        """Answer each item with ``pipeline`` and judge the answers."""
        judge = Judge(self.services.llm, self.settings.llm.judge_model)
        rows = []
        for number, item in enumerate(items, start=1):
            self._progress(f"Generation ({name}): {number}/{len(items)} {item.id}")
            rows.append(self._generate_one(pipeline, judge, item))
        return GenerationResult(
            name=name,
            answer_task=pipeline.answer_task,
            n_questions=len(items),
            metrics=generation_metrics(rows),
            per_question=rows,
        )

    def _generate_one(self, pipeline: RAGPipeline, judge: Judge, item: EvalItem) -> GenerationRow:
        row = GenerationRow(
            id=item.id,
            type=item.type,
            question=item.question,
            answerable=item.answerable,
            gold=[page_label(f, p) for f, p in sorted(item.gold_pages)],
        )
        try:
            result = pipeline.ask(item.question, history=item.history)
        except LLMError as exc:
            logger.warning("%s: generation failed: %s", item.id, exc)
            return row.model_copy(update={"error": str(exc)})
        row = self._describe(row, result, item)
        try:
            row = self._judge(row, result, judge, item)
        except LLMError as exc:
            logger.warning("%s: judging failed: %s", item.id, exc)
            row = row.model_copy(update={"error": f"judge: {exc}"})
        return row

    @staticmethod
    def _describe(row: GenerationRow, result: AskResult, item: EvalItem) -> GenerationRow:
        response = result.response
        cited = [result.sources[i - 1].chunk for i in response.cited_sources]
        cited_pages = {(c.file_name, c.page) for c in cited}
        return row.model_copy(
            update={
                "standalone_question": result.standalone_question,
                "predicted_answerable": response.answerable,
                "answer": response.answer,
                "confidence": response.confidence,
                "cited_pages": [page_label(f, p) for f, p in sorted(cited_pages)],
                "gold_cited": bool(cited_pages & item.gold_pages) if item.answerable else None,
                "removed_citations": result.removed_citations,
                "repaired": result.repaired,
                "llm_answered": bool(result.sources),
                "latency_ms": result.timings_ms.get("total_ms", 0.0),
                "prompt_tokens": result.usage.get("prompt_tokens", 0),
                "output_tokens": result.usage.get("output_tokens", 0),
            }
        )

    def _judge(
        self, row: GenerationRow, result: AskResult, judge: Judge, item: EvalItem
    ) -> GenerationRow:
        update: dict[str, Any] = {}
        if result.response.answerable and result.sources:
            context = self._context_of(result.sources)
            verdict, _ = judge.faithfulness(result.response.answer, context)
            update["faithfulness"] = verdict.score
            update["unsupported_claims"] = [
                claim for claim, ok in zip(verdict.claims, verdict.supported, strict=True) if not ok
            ]
        update["relevancy"], _ = judge.relevancy(
            item.question, result.response.answer, item.answerable
        )
        return row.model_copy(update=update)

    def _context_of(self, sources: list[RetrievedChunk]) -> str:
        text, _ = build_context(
            sources, self.settings.retrieval.max_context_tokens, self.services.registry
        )
        return text
