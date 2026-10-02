"""Write ``report.md`` and ``results.json`` for an :class:`EvalReport` (SPEC 12.3)."""

from collections import defaultdict
from pathlib import Path

from lecturelens.evaluation.runner import (
    DEFAULT_NAME,
    MINIMAL_NAME,
    EvalReport,
    GenerationResult,
    GenerationRow,
    RetrievalResult,
)

REPORT_MD = "report.md"
RESULTS_JSON = "results.json"
DASH = "-"
SWEEP_TIE_MRR = 0.01  # MRR differences this small are noise on a ~25-question set
HEADLINE_K_LOW, HEADLINE_K_HIGH = 1, 5
ROW_LABELS = {
    "bm25_only": "BM25 only",
    "dense_only": "Dense only",
    "hybrid": "Hybrid (RRF)",
    "hybrid_rerank": "Hybrid + rerank",
    MINIMAL_NAME: "Default, minimal prompt (no rules)",
}


def _label(report: EvalReport, name: str) -> str:
    """Display name of a result row; the default row names the configuration it ran with."""
    if name == DEFAULT_NAME:
        retrieval = report.config["retrieval"]
        rerank = " + rerank" if retrieval["rerank"] else ""
        return f"Default config ({retrieval['mode']}{rerank})"
    return ROW_LABELS.get(name, name)


def _fmt(value: float | None, digits: int = 3) -> str:
    return DASH if value is None else f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    return DASH if value is None else f"{100 * value:.0f}%"


def _ms(value: float | None) -> str:
    return DASH if value is None else f"{value:,.0f} ms"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines + [""]


def _gen(report: EvalReport, name: str) -> GenerationResult | None:
    return next((g for g in report.generation if g.name == name), None)


def _retr(report: EvalReport, name: str) -> RetrievalResult | None:
    return next((r for r in report.retrieval if r.name == name), None)


# ---------------------------------------------------------------- sections


def headline_table(report: EvalReport) -> list[str]:
    """The SPEC 12.3 results table: retrieval ablations, default config and minimal prompt."""
    lo, hi = f"hit@{HEADLINE_K_LOW}", f"hit@{HEADLINE_K_HIGH}"
    default_gen = _gen(report, DEFAULT_NAME)
    rows = []
    for result in report.retrieval:
        m = result.metrics
        g = default_gen.metrics if default_gen and result.name == DEFAULT_NAME else {}
        rows.append(
            [
                _label(report, result.name),
                _fmt(m.get(lo)),
                _fmt(m.get(hi)),
                _fmt(m.get("mrr")),
                _fmt(g.get("faithfulness")),
                _fmt(g.get("relevancy"), 2),
                _pct(g.get("abstention_accuracy")),
                _ms(g.get("latency_ms_p50")),
            ]
        )
    minimal = _gen(report, MINIMAL_NAME)
    if minimal:
        g = minimal.metrics
        rows.append(
            [
                f"{ROW_LABELS[MINIMAL_NAME]} ({minimal.n_questions} q)",
                DASH,
                DASH,
                DASH,
                _fmt(g.get("faithfulness")),
                _fmt(g.get("relevancy"), 2),
                _pct(g.get("abstention_accuracy")),
                _ms(g.get("latency_ms_p50")),
            ]
        )
    header = [
        "Config",
        f"Hit@{HEADLINE_K_LOW}",
        f"Hit@{HEADLINE_K_HIGH}",
        "MRR",
        "Faithful.",
        "Relev. (1-5)",
        "Abstain acc.",
        "p50 latency",
    ]
    return _table(header, rows)


def retrieval_section(report: EvalReport) -> list[str]:
    ks = report.config["eval"]["k_values"]
    header = [
        "Config",
        *[f"Hit@{k}" for k in ks],
        *[f"Recall@{k}" for k in ks],
        f"nDCG@{max(ks)}",
        "MRR",
        "p50 retrieval",
    ]
    rows = [
        [
            _label(report, r.name),
            *[_fmt(r.metrics.get(f"hit@{k}")) for k in ks],
            *[_fmt(r.metrics.get(f"recall@{k}")) for k in ks],
            _fmt(r.metrics.get(f"ndcg@{max(ks)}")),
            _fmt(r.metrics.get("mrr")),
            _ms(r.latency_ms_p50),
        ]
        for r in report.retrieval
    ]
    n = len(report.retrieval[0].per_question) if report.retrieval else 0
    lines = [f"## Retrieval ({n} answerable questions)", ""] + _table(header, rows)
    default = _retr(report, DEFAULT_NAME)
    if default:
        by_type: dict[str, list[float]] = defaultdict(list)
        for row in default.per_question:
            rank = row.first_gold_rank
            by_type[row.type].append(1.0 if rank is not None and rank <= max(ks) else 0.0)
        type_rows = [[t, str(len(v)), _fmt(sum(v) / len(v))] for t, v in sorted(by_type.items())]
        lines += [f"Hit@{max(ks)} by question type (default config):", ""]
        lines += _table(["Type", "Questions", f"Hit@{max(ks)}"], type_rows)
    return lines


def sweep_section(report: EvalReport) -> list[str]:
    if not report.chunk_sweep:
        return []
    rows = [
        [
            str(s.chunk_size_tokens),
            str(s.n_chunks),
            f"{s.avg_tokens_per_chunk:.0f}",
            _fmt(s.metrics.get(f"hit@{HEADLINE_K_LOW}")),
            _fmt(s.metrics.get(f"hit@{HEADLINE_K_HIGH}")),
            _fmt(s.metrics.get("mrr")),
        ]
        for s in report.chunk_sweep
    ]
    verdict = sweep_verdict(report)
    header = [
        "Chunk size (tokens)",
        "Chunks",
        "Avg tokens",
        f"Hit@{HEADLINE_K_LOW}",
        f"Hit@{HEADLINE_K_HIGH}",
        "MRR",
    ]
    return (
        ["## Chunk-size sweep (retrieval only, default config)", ""]
        + _table(header, rows)
        + [verdict, ""]
    )


def sweep_verdict(report: EvalReport) -> str:
    """Best chunk size by Hit@k then MRR, calling near-equal sizes a tie (small datasets)."""
    hit = f"hit@{HEADLINE_K_HIGH}"
    best = max(report.chunk_sweep, key=lambda s: (s.metrics.get(hit, 0), s.metrics.get("mrr", 0)))
    ties = [
        s.chunk_size_tokens
        for s in report.chunk_sweep
        if s.metrics.get(hit, 0) == best.metrics.get(hit, 0)
        and best.metrics.get("mrr", 0) - s.metrics.get("mrr", 0) <= SWEEP_TIE_MRR
    ]
    configured = report.config["ingestion"]["chunk_size_tokens"]
    if len(ties) > 1 and configured in ties:
        sizes = ", ".join(str(t) for t in ties)
        return (
            f"Sizes {sizes} tie (same Hit@{HEADLINE_K_HIGH}, MRR within {SWEEP_TIE_MRR}); "
            f"the configured **{configured} tokens** is kept."
        )
    return f"Best by Hit@{HEADLINE_K_HIGH} then MRR: **{best.chunk_size_tokens} tokens**."


def generation_section(report: EvalReport) -> list[str]:
    if not report.generation:
        return []
    header = [
        "Variant",
        "Questions",
        "Faithfulness",
        "Relevancy",
        "Citation validity",
        "Repairs",
        "Citation hit",
        "Abstain acc.",
        "Abstain P / R",
        "No-LLM abstentions",
        "p50 / p95",
        "Prompt tokens/query",
        "Errors",
    ]
    rows = []
    for g in report.generation:
        m = g.metrics
        rows.append(
            [
                _label(report, g.name),
                str(g.n_questions),
                _fmt(m["faithfulness"]),
                _fmt(m["relevancy"], 2),
                _pct(m["citation_validity"]),
                _pct(m["repair_rate"]),
                _pct(m["citation_hit_rate"]),
                _pct(m["abstention_accuracy"]),
                f"{_pct(m['abstention_precision'])} / {_pct(m['abstention_recall'])}",
                f"{m['no_llm_abstentions']:.0f}",
                f"{_ms(m['latency_ms_p50'])} / {_ms(m['latency_ms_p95'])}",
                _fmt(m["prompt_tokens_per_query"], 0),
                f"{m['errors']:.0f}",
            ]
        )
    return (
        ["## Generation", ""]
        + _table(header, rows)
        + [
            "Faithfulness = supported claims / claims (LLM judge, answered questions only). "
            "Citation validity = answers whose cited ids all existed before repair. Citation hit = "
            'answered answerable questions citing a gold page. Abstention P/R treats "not '
            'answerable" as the positive class. Latency is end-to-end per question and, during '
            "evaluation, includes waiting for the client-side rate limiter that the judge calls "
            "share, so p95 is mostly queueing; p50 is representative of interactive use.",
            "",
        ]
    )


def _failure_reason(row: GenerationRow, gold_rank: int | None, k: int) -> str | None:
    """Why a question went wrong, or None if it did not."""
    if row.error:
        return f"Error: {row.error}"
    if row.predicted_answerable != row.answerable:
        if row.answerable:
            where = (
                f"the gold page was retrieved at rank {gold_rank}"
                if gold_rank
                else f"no gold page in the top {k}"
            )
            return f"Wrongly abstained ({where})."
        return "Answered a question the notes do not cover (should have abstained)."
    if row.answerable and gold_rank is None:
        return f"Retrieval miss: no gold page in the top {k}; the answer used other pages."
    if row.faithfulness is not None and row.faithfulness < 1:
        claims = "; ".join(row.unsupported_claims) or "see judge"
        return f"Partly unsupported (faithfulness {row.faithfulness:.2f}): {claims}"
    if row.answerable and row.gold_cited is False:
        return "Answered, but cited none of the gold pages."
    if row.relevancy is not None and row.relevancy <= 3:
        return f"Low relevancy ({row.relevancy}/5)."
    return None


def examples_section(report: EvalReport, n: int) -> list[str]:
    gen, retr = _gen(report, DEFAULT_NAME), _retr(report, DEFAULT_NAME)
    if not gen or n == 0:
        return []
    k = max(report.config["eval"]["k_values"])
    ranks = {r.id: r.first_gold_rank for r in retr.per_question} if retr else {}
    failures = [
        (row, why)
        for row in gen.per_question
        if (why := _failure_reason(row, ranks.get(row.id), k))
    ]
    successes = [
        row
        for row in gen.per_question
        if _failure_reason(row, ranks.get(row.id), k) is None
        and (not row.answerable or (row.faithfulness == 1.0 and row.relevancy == 5))
    ]
    # show one correct abstention (if any), then answered questions; never repeat a question
    abstentions = [r for r in successes if not r.answerable][:1]
    picked = abstentions + [r for r in successes if r.answerable][: n - len(abstentions)]
    lines = ["## Examples", "", f"### Successes ({len(picked)})", ""]
    for row in picked:
        lines += _example(row, ranks.get(row.id))
    lines += [f"### Failure analysis ({min(n, len(failures))} of {len(failures)})", ""]
    for row, why in failures[:n]:
        lines += _example(row, ranks.get(row.id), why)
    if not failures:
        lines += ["No failures in the default configuration.", ""]
    return lines


def _example(row: GenerationRow, gold_rank: int | None, why: str | None = None) -> list[str]:
    lines = [f"**{row.id} ({row.type}): {row.question}**", ""]
    if row.standalone_question and row.standalone_question != row.question:
        lines.append(f"- Condensed to: *{row.standalone_question}*")
    if row.gold:
        lines.append(
            f"- Gold: {', '.join(row.gold)} (first retrieved at rank "
            f"{gold_rank if gold_rank else 'none'})"
        )
    answer = " ".join(row.answer.split())
    lines.append(f"- Answer ({row.confidence}): {answer}")
    lines.append(
        f"- Cited: {', '.join(row.cited_pages) or 'none'} | faithfulness "
        f"{_fmt(row.faithfulness, 2)} | relevancy {row.relevancy or DASH}"
    )
    if why:
        lines.append(f"- **What went wrong:** {why}")
    return lines + [""]


def details_section(report: EvalReport) -> list[str]:
    gen, retr = _gen(report, DEFAULT_NAME), _retr(report, DEFAULT_NAME)
    if not gen:
        return []
    ranks = {r.id: r.first_gold_rank for r in retr.per_question} if retr else {}
    rows = [
        [
            r.id,
            r.type,
            str(ranks.get(r.id) or DASH) if r.answerable else DASH,
            "yes" if r.predicted_answerable == r.answerable else "**no**",
            "yes" if r.gold_cited else ("no" if r.gold_cited is False else DASH),
            _fmt(r.faithfulness, 2),
            str(r.relevancy or DASH),
            f"{r.latency_ms:,.0f}",
        ]
        for r in gen.per_question
    ]
    header = [
        "Id",
        "Type",
        "Gold rank",
        "Abstention correct",
        "Gold cited",
        "Faithful.",
        "Relev.",
        "Latency ms",
    ]
    return ["## Per-question results (default config)", ""] + _table(header, rows)


def render_markdown(report: EvalReport, n_examples: int) -> str:
    """The full Markdown report."""
    models = ", ".join(f"{k}: `{v}`" for k, v in report.models.items())
    lines = [
        "# LectureLens evaluation report",
        "",
        f"- Generated: {report.created_at} | prompt file version {report.prompt_version}",
        f"- Models: {models}",
        f"- Dataset: `{report.dataset}` ({report.n_questions} questions)",
        f"- Retrieval: mode `{report.config['retrieval']['mode']}`, rerank "
        f"`{report.config['retrieval']['rerank']}`, chunk size "
        f"{report.config['ingestion']['chunk_size_tokens']} tokens, final_k "
        f"{report.config['retrieval']['final_k']}",
        "",
        "LLM-judge scores (faithfulness, relevancy) come from a single judge model and are best "
        "read comparatively across configurations rather than as absolute quality.",
        "",
        "## Results",
        "",
        *headline_table(report),
        *retrieval_section(report),
        *sweep_section(report),
        *generation_section(report),
        *examples_section(report, n_examples),
        *details_section(report),
    ]
    if report.notes:
        lines += ["## Notes", "", *[f"- {note}" for note in report.notes], ""]
    return "\n".join(lines)


def write_report(report: EvalReport, out_dir: Path, n_examples: int) -> tuple[Path, Path]:
    """Write ``report.md`` and ``results.json`` into ``out_dir``; return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    md, js = out_dir / REPORT_MD, out_dir / RESULTS_JSON
    md.write_text(render_markdown(report, n_examples), encoding="utf-8", newline="\n")
    js.write_text(report.model_dump_json(indent=2), encoding="utf-8", newline="\n")
    return md, js
