"""Tests for the evaluation package: dataset, retrieval metrics, judge, runner, report."""

import json
import math
from pathlib import Path

import pytest

from lecturelens.errors import ConfigError
from lecturelens.evaluation import retrieval_metrics as rm
from lecturelens.evaluation.dataset import EvalItem, load_dataset, missing_gold_pages
from lecturelens.evaluation.judge import Judge
from lecturelens.evaluation.report import headline_table, render_markdown, write_report
from lecturelens.evaluation.runner import (
    EvalReport,
    EvalRunner,
    GenerationRow,
    generation_metrics,
    minimal_prompt_subset,
    percentile,
)
from lecturelens.indexing.indexer import Indexer
from lecturelens.llm.prompts import PromptRegistry
from lecturelens.retrieval.retriever import HybridRetriever
from lecturelens.schemas import (
    AnswerResponse,
    CondensedQuestion,
    FaithfulnessJudgement,
    RelevancyJudgement,
)
from lecturelens.services import Services

from .conftest import ROOT
from .fakes import FakeLLMClient

A, B, C = ("a.pdf", 1), ("a.pdf", 2), ("b.pdf", 7)


# ---------------------------------------------------------------- retrieval metrics


def test_hit_recall_and_mrr_by_hand() -> None:
    ranking = [C, A, A, B]
    gold = {A, B}
    assert rm.hit_at_k(ranking, gold, 1) == 0.0
    assert rm.hit_at_k(ranking, gold, 2) == 1.0
    assert rm.recall_at_k(ranking, gold, 3) == 0.5  # A twice still counts once
    assert rm.recall_at_k(ranking, gold, 4) == 1.0
    assert rm.reciprocal_rank(ranking, gold) == 0.5
    assert rm.reciprocal_rank([C], gold) == 0.0
    assert rm.first_gold_rank(ranking, gold) == 2 and rm.first_gold_rank([C], gold) is None


def test_ndcg_by_hand() -> None:
    gold = {A, B}
    assert rm.ndcg_at_k([A, B], gold, 5) == pytest.approx(1.0)
    expected = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert rm.ndcg_at_k([C, A, B], gold, 5) == pytest.approx(expected)
    assert rm.ndcg_at_k([A, A], {A}, 5) == pytest.approx(1.0)  # duplicate page: no extra credit


def test_aggregate_means() -> None:
    m = rm.aggregate([[A], [C]], [{A}, {A}], k_values=[1, 5])
    assert m["hit@1"] == 0.5 and m["mrr"] == 0.5 and m["recall@5"] == 0.5


# ---------------------------------------------------------------- dataset


def _line(**overrides) -> str:
    item = {
        "id": "q1",
        "type": "factual",
        "question": "Q?",
        "answerable": True,
        "gold": [{"file": "a.pdf", "page": 1}],
        "reference_answer": "",
        "history": [],
    }
    return json.dumps(item | overrides)


def test_real_dataset_is_valid() -> None:
    items = load_dataset(ROOT / "eval" / "qa_dataset.jsonl")
    assert len(items) >= 25
    assert {i.type for i in items} >= {"factual", "unanswerable", "followup"}
    assert any(i.history for i in items)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"gold": []}, "need gold pages"),
        ({"type": "unanswerable"}, "must match"),
        ({"answerable": False}, "need gold pages"),
        ({"type": "trivia"}, "type"),
    ],
)
def test_invalid_items_are_rejected_with_line_number(
    tmp_path: Path, overrides: dict, message: str
) -> None:
    path = tmp_path / "qa.jsonl"
    path.write_text(_line(id="ok") + "\n" + _line(**overrides) + "\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"qa.jsonl:2"):
        load_dataset(path)


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    path = tmp_path / "qa.jsonl"
    path.write_text(_line() + "\n" + _line() + "\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate"):
        load_dataset(path)


def test_missing_gold_pages() -> None:
    item = EvalItem.model_validate(json.loads(_line()))
    assert missing_gold_pages([item], {("a.pdf", 1)}) == {}
    assert list(missing_gold_pages([item], set())) == ["q1"]


# ---------------------------------------------------------------- helpers


def test_percentile_nearest_rank() -> None:
    values = [float(v) for v in range(1, 101)]
    assert percentile(values, 50) == 50.0 and percentile(values, 95) == 95.0
    assert percentile([], 50) == 0.0


def test_minimal_prompt_subset_mixes_unanswerable_and_answerable() -> None:
    items = load_dataset(ROOT / "eval" / "qa_dataset.jsonl")
    subset = minimal_prompt_subset(items, 10)
    assert len(subset) == 10
    assert sum(not i.answerable for i in subset) == 5
    assert not any(i.history for i in subset)


def test_generation_metrics_abstention_precision_recall() -> None:
    def row(answerable: bool, predicted: bool, **kw) -> GenerationRow:
        return GenerationRow(
            id="x",
            type="factual",
            question="q",
            answerable=answerable,
            predicted_answerable=predicted,
            llm_answered=True,
            **kw,
        )

    rows = [
        row(True, True, faithfulness=1.0, relevancy=5, gold_cited=True),
        row(True, False, relevancy=2),  # wrongly abstained (false positive)
        row(False, False, relevancy=5),  # correct abstention (true positive)
        row(False, True, relevancy=1, removed_citations=[9]),  # should have abstained (FN)
    ]
    m = generation_metrics(rows)
    assert m["abstention_accuracy"] == 0.5
    assert m["abstention_precision"] == 0.5 and m["abstention_recall"] == 0.5
    assert m["faithfulness"] == 1.0 and m["relevancy"] == pytest.approx(13 / 4)
    assert m["citation_validity"] == 0.75 and m["citation_hit_rate"] == 1.0


# ---------------------------------------------------------------- judge + runner (offline)


def test_judge_scores_and_tolerates_mismatched_lists(tiny_settings, registry) -> None:
    verdict = FaithfulnessJudgement(claims=["a", "b", "c"], supported=[True, False], reasoning="r")
    llm = FakeLLMClient(
        tiny_settings.llm,
        registry,
        {
            "judge_faithfulness": [verdict],
            "judge_relevancy": [RelevancyJudgement(score=4, reasoning="ok")],
        },
    )
    judge = Judge(llm, "judge-model")
    faith, _ = judge.faithfulness("answer", "context")
    assert faith.score == 0.5 and faith.claims == ["a", "b"]
    assert judge.relevancy("q", "a", answerable=False)[0] == 4
    assert all(call.model == "judge-model" for call in llm.calls)
    assert "QUESTION IS ANSWERABLE FROM NOTES: no" in llm.calls[1].user


def _runner(
    indexer: Indexer, registry: PromptRegistry, items: list[EvalItem], responses: dict
) -> tuple[EvalRunner, FakeLLMClient]:
    s = indexer.settings
    llm = FakeLLMClient(s.llm, registry, responses)
    services = Services(s)
    retriever = HybridRetriever(
        indexer.embedder,
        indexer.store,
        indexer.load_bm25(),
        None,
        s.retrieval,
        s.ingestion.add_context_header,
    )
    services.__dict__.update(
        embedder=indexer.embedder,
        store=indexer.store,
        indexer=indexer,
        registry=registry,
        llm=llm,
        retriever=retriever,
    )
    return EvalRunner(services, items), llm


def _items() -> list[EvalItem]:
    rows = [
        {
            "id": "q1",
            "type": "factual",
            "question": "What is Kneser-Ney smoothing?",
            "answerable": True,
            "gold": [{"file": "02_ngram_language_models.md", "page": 1}],
        },
        {
            "id": "q2",
            "type": "followup",
            "question": "and what does it subtract?",
            "answerable": True,
            "gold": [{"file": "02_ngram_language_models.md", "page": 1}],
            "history": [{"question": "What is Kneser-Ney smoothing?", "answer": "A method."}],
        },
        {
            "id": "q3",
            "type": "unanswerable",
            "question": "How does RLHF work?",
            "answerable": False,
        },
    ]
    return [EvalItem.model_validate(r) for r in rows]


def test_runner_end_to_end_offline(
    tiny_index: Indexer, registry: PromptRegistry, tmp_path: Path
) -> None:
    grounded = AnswerResponse(
        answer="It uses continuation counts [S1].",
        cited_sources=[1],
        answerable=True,
        confidence="high",
    )
    abstain = AnswerResponse(
        answer="I couldn't find this in your course material.",
        cited_sources=[],
        answerable=False,
        confidence="low",
    )
    runner, llm = _runner(
        tiny_index,
        registry,
        _items(),
        {
            "condense_question": [
                CondensedQuestion(standalone_question="What does Kneser-Ney smoothing subtract?")
            ],
            "answer": [grounded, grounded, abstain],
            "answer_minimal": [grounded],
            "judge_faithfulness": [
                FaithfulnessJudgement(claims=["c"], supported=[True], reasoning="r")
            ],
            "judge_relevancy": [RelevancyJudgement(score=5, reasoning="r")],
        },
    )
    report = runner.run(generation=True, ablations=True, chunk_sweep=False)

    names = [r.name for r in report.retrieval]
    assert names == ["default", "bm25_only", "dense_only", "hybrid", "hybrid_rerank"]
    default = report.retrieval[0]
    assert default.metrics["hit@5"] == 1.0  # both answerable questions find the n-gram note
    followup = next(r for r in default.per_question if r.id == "q2")
    assert followup.query == "What does Kneser-Ney smoothing subtract?"

    gen = report.generation[0]
    assert gen.name == "default" and gen.n_questions == 3
    assert gen.metrics["abstention_accuracy"] == 1.0
    assert gen.metrics["faithfulness"] == 1.0 and gen.metrics["citation_hit_rate"] == 1.0
    minimal = report.generation[1]
    assert minimal.name == "minimal_prompt" and minimal.answer_task == "answer_minimal"
    assert minimal.metrics["abstention_accuracy"] == 0.5  # minimal prompt answered RLHF
    assert "answer_minimal" in llm.tasks_called()
    # the follow-up was condensed once for retrieval and is reused (cache) by generation
    assert llm.tasks_called().count("condense_question") >= 1

    md, js = write_report(report, tmp_path, n_examples=3)
    text = md.read_text(encoding="utf-8")
    assert "| Default config (hybrid + rerank) | 1.000 | 1.000 |" in text
    assert "Default, minimal prompt (no rules) (2 q)" in text
    assert "### Failure analysis" in text and "should have abstained" not in text
    assert EvalReport.model_validate_json(js.read_text(encoding="utf-8")) == report


def test_report_failure_analysis_explains_wrong_abstention() -> None:
    report = EvalReport(
        created_at="t",
        prompt_version="1",
        models={"llm": "m"},
        dataset="d",
        n_questions=1,
        config={
            "eval": {"k_values": [1, 5]},
            "retrieval": {"mode": "hybrid", "rerank": True, "final_k": 5},
            "ingestion": {"chunk_size_tokens": 400},
        },
    )
    from lecturelens.evaluation.runner import GenerationResult

    row = GenerationRow(
        id="q9",
        type="unanswerable",
        question="RLHF?",
        answerable=False,
        predicted_answerable=True,
        llm_answered=True,
        relevancy=1,
    )
    report.generation.append(
        GenerationResult(
            name="default",
            answer_task="answer",
            n_questions=1,
            metrics=generation_metrics([row]),
            per_question=[row],
        )
    )
    text = render_markdown(report, n_examples=3)
    assert "should have abstained" in text
    assert headline_table(report)[0].startswith("| Config |")
