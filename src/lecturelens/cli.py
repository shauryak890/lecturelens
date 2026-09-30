"""Typer command-line interface: ``python -m lecturelens <command>``.

This is the only module that prints to the console (via Rich). Commands are added phase by
phase: P0 ``models``; P1 ``ingest`` and ``stats``; P2 ``ask`` and ``chat``; quiz, summarize,
flashcards and eval follow. Heavy modules are imported inside commands so ``--help`` is fast.
"""

import io
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from lecturelens.config import (
    DEFAULT_CONFIG_PATH,
    AnswerMode,
    RetrievalMode,
    Settings,
    load_settings,
)
from lecturelens.errors import LectureLensError
from lecturelens.logging_utils import APP_LOG_NAME, setup_logging
from lecturelens.schemas import AskResult, ChatTurn

if TYPE_CHECKING:
    from lecturelens.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)
console = Console()
err_console = Console(stderr=True)
BYTES_PER_MB = 1024 * 1024

app = typer.Typer(
    name="lecturelens",
    help="LectureLens: a citation-grounded RAG study tutor for your course notes.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


def _tolerant_console_streams() -> None:
    """Never crash printing an answer: replace characters the console cannot encode.

    Legacy Windows consoles use cp1252, which has no glyph for symbols LLM answers often
    contain (e.g. "⇒", "≤"); by default printing one raises UnicodeEncodeError.
    """
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(errors="replace")


def _fail(exc: LectureLensError) -> typer.Exit:
    err_console.print(f"[bold red]Error:[/] {exc}")
    return typer.Exit(code=1)


@app.callback()
def main(
    ctx: typer.Context,
    config: Annotated[
        Path, typer.Option("--config", "-c", help="Path to config.yaml.")
    ] = DEFAULT_CONFIG_PATH,
) -> None:
    """Load settings and configure logging before any command runs."""
    _tolerant_console_streams()
    try:
        settings = load_settings(config)
    except LectureLensError as exc:
        raise _fail(exc) from exc
    setup_logging(settings.app.log_level, settings.app.log_dir, [settings.llm.api_key_env])
    ctx.obj = settings


@app.command()
def ingest(
    ctx: typer.Context,
    path: Annotated[
        Path | None,
        typer.Option("--path", "-p", help=r"Folder to ingest \[default: app.data_dir]."),
    ] = None,
    sample: Annotated[
        bool, typer.Option("--sample", help="Ingest the bundled sample notes (app.sample_dir).")
    ] = False,
    rebuild: Annotated[
        bool, typer.Option("--rebuild", help="Drop the index and re-index from scratch.")
    ] = False,
) -> None:
    """Parse, chunk, embed and index PDF/Markdown/text files. Unchanged files are skipped."""
    from lecturelens.indexing.indexer import build_indexer

    settings: Settings = ctx.obj
    folder = settings.app.sample_dir if sample else (path or settings.app.data_dir)
    try:
        with console.status("Loading embedding model (first run downloads it)..."):
            indexer = build_indexer(settings)
        with console.status("Indexing...") as status:
            report = indexer.ingest(
                folder, rebuild=rebuild, on_file=lambda f: status.update(f"Indexing {f.name}...")
            )
    except LectureLensError as exc:
        raise _fail(exc) from exc

    for name in report.indexed:
        console.print(f"[green]indexed[/]   {name}")
    for name in report.skipped:
        console.print(f"[dim]unchanged[/] {name}")
    for name, original in report.duplicates.items():
        console.print(f"[yellow]duplicate[/] {name} (same content as {original})")
    for name, error in report.failed.items():
        console.print(f"[red]failed[/]    {name}: {error}")
    console.print(
        f"\n{len(report.indexed)} file(s) indexed, {report.new_chunks} new chunks, "
        f"{report.total_chunks} chunks in index ({report.seconds:.1f}s)."
    )
    if report.failed:
        raise typer.Exit(code=1)


@app.command()
def stats(ctx: typer.Context) -> None:
    """Show indexed documents, chunk counts, average chunk size and index size."""
    from lecturelens.indexing.indexer import index_stats

    settings: Settings = ctx.obj
    try:
        s = index_stats(settings)
    except LectureLensError as exc:
        raise _fail(exc) from exc
    if not s.n_docs:
        console.print("The index is empty. Run [bold]python -m lecturelens ingest --sample[/].")
        return

    table = Table(title="Indexed documents")
    table.add_column("File", style="bold")
    for column in ("Pages", "Chunks", "Tokens"):
        table.add_column(column, justify="right")
    table.add_column("Indexed at (UTC)")
    for f in s.files:
        table.add_row(f.file_name, str(f.n_pages), str(f.n_chunks), str(f.n_tokens), f.indexed_at)
    console.print(table)
    console.print(
        f"Documents: {s.n_docs}   Pages: {s.n_pages}   Chunks: {s.n_chunks}   "
        f"Avg tokens/chunk: {s.avg_tokens_per_chunk:.0f}\n"
        f"Embedding model: {s.embedding_model} (dim={s.dim})   "
        f"BM25: {'ready' if s.bm25_ready else 'missing'}   "
        f"Index size: {s.index_bytes / BYTES_PER_MB:.1f} MB"
    )


ModeOpt = Annotated[
    AnswerMode | None,
    typer.Option("--mode", "-m", help=r"Answer style \[default: generation.default_mode]."),
]
RetrievalOpt = Annotated[
    RetrievalMode | None,
    typer.Option("--retrieval", "-r", help=r"Retrieval mode \[default: retrieval.mode]."),
]
NoRerankOpt = Annotated[
    bool, typer.Option("--no-rerank", help="Skip cross-encoder reranking for this run.")
]
DebugOpt = Annotated[bool, typer.Option("--debug", help="Show timings and token usage.")]
CONFIDENCE_STYLE = {"high": "green", "medium": "yellow", "low": "red"}
EXIT_COMMANDS = {"/exit", "/quit", "exit", "quit"}
CLEAR_COMMAND = "/clear"


def _render_result(result: AskResult, debug: bool) -> None:
    """Print an answer panel, its source table and (optionally) debug information."""
    response = result.response
    style = CONFIDENCE_STYLE[response.confidence]
    subtitle = f"confidence: [{style}]{response.confidence}[/]" + (
        "" if response.answerable else "  |  not answerable from your notes"
    )
    console.print(
        Panel(Markdown(response.answer), title="Answer", subtitle=subtitle, border_style=style)
    )
    if result.sources:
        table = Table(title="Sources", title_justify="left")
        for column in ("", "File", "Page", "Section", "Score", "Ranks"):
            table.add_column(column)
        cited = set(response.cited_sources)
        for i, src in enumerate(result.sources, start=1):
            chunk = src.chunk
            ranks = " ".join(f"{name}:{rank}" for name, rank in src.ranks.items())
            label = f"[bold]S{i}[/]" if i in cited else f"[dim]S{i}[/]"
            table.add_row(
                label,
                chunk.file_name,
                str(chunk.page),
                chunk.heading_path,
                f"{src.score:.3f}",
                ranks,
            )
        console.print(table)
    if debug:
        if result.standalone_question != result.question:
            console.print(f"[dim]Standalone question:[/] {result.standalone_question}")
        timings = "  ".join(
            f"{k.removesuffix('_ms')} {v:.0f} ms" for k, v in result.timings_ms.items()
        )
        usage = "  ".join(f"{k} {v}" for k, v in result.usage.items())
        console.print(f"[dim]{timings}\n{usage}[/]")


def _load_pipeline(settings: Settings) -> "RAGPipeline":
    from lecturelens.rag.pipeline import build_pipeline

    with console.status("Opening the index..."):
        return build_pipeline(settings)


@app.command()
def ask(
    ctx: typer.Context,
    question: Annotated[str, typer.Argument(help="Your question, in quotes.")],
    mode: ModeOpt = None,
    retrieval: RetrievalOpt = None,
    no_rerank: NoRerankOpt = False,
    debug: DebugOpt = False,
) -> None:
    """Answer one question from your course material, with page-level citations."""
    settings: Settings = ctx.obj
    try:
        pipeline = _load_pipeline(settings)
        with console.status("Thinking (first question also loads the models)..."):
            result = pipeline.ask(
                question, mode=mode, retrieval_mode=retrieval, rerank=False if no_rerank else None
            )
    except LectureLensError as exc:
        raise _fail(exc) from exc
    _render_result(result, debug)


@app.command()
def chat(
    ctx: typer.Context,
    mode: ModeOpt = None,
    retrieval: RetrievalOpt = None,
    no_rerank: NoRerankOpt = False,
    debug: DebugOpt = False,
) -> None:
    """Interactive tutor with follow-up questions. Type /clear to reset, /exit to quit."""
    settings: Settings = ctx.obj
    try:
        pipeline = _load_pipeline(settings)
    except LectureLensError as exc:
        raise _fail(exc) from exc
    console.print("[bold]LectureLens chat[/] - ask about your notes. /clear resets, /exit quits.")
    history: list[ChatTurn] = []
    while True:
        try:
            question = console.input("\n[bold cyan]You:[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in EXIT_COMMANDS:
            break
        if question.lower() == CLEAR_COMMAND:
            history.clear()
            console.print("[dim]History cleared.[/]")
            continue
        if not question:
            continue
        try:
            with console.status("Thinking..."):
                result = pipeline.ask(
                    question,
                    history=history,
                    mode=mode,
                    retrieval_mode=retrieval,
                    rerank=False if no_rerank else None,
                )
        except LectureLensError as exc:
            err_console.print(f"[bold red]Error:[/] {exc}")
            continue
        _render_result(result, debug)
        history.append(ChatTurn(question=question, answer=result.response.answer))


@app.command()
def models(
    ctx: typer.Context,
    show_all: Annotated[
        bool, typer.Option("--all", help="Also show models that cannot generate text.")
    ] = False,
) -> None:
    """List model IDs available to your API key (use them in config.yaml)."""
    from lecturelens.llm.client import list_models  # lazy: keep SDK import off other commands

    settings: Settings = ctx.obj
    try:
        found = list_models(settings.llm, settings.require_api_key(), generate_only=not show_all)
    except LectureLensError as exc:
        raise _fail(exc) from exc

    configured = {settings.llm.model, settings.llm.fallback_model, settings.llm.judge_model}
    table = Table(title=f"Models available to {settings.llm.api_key_env}")
    id_width = max((len(m.model_id) for m in found), default=0)
    table.add_column("Model ID", style="bold", no_wrap=True, min_width=id_width)
    table.add_column("Display name")
    table.add_column("Input tokens", justify="right")
    table.add_column("Output tokens", justify="right")
    table.add_column("In config", justify="center")
    for m in found:
        table.add_row(
            m.model_id,
            m.display_name,
            f"{m.input_token_limit:,}" if m.input_token_limit else "-",
            f"{m.output_token_limit:,}" if m.output_token_limit else "-",
            "yes" if m.model_id in configured else "",
        )
    console.print(table)

    available = {m.model_id for m in found}
    for role, model_id in (
        ("llm.model", settings.llm.model),
        ("llm.fallback_model", settings.llm.fallback_model),
        ("llm.judge_model", settings.llm.judge_model),
    ):
        if model_id and model_id not in available:
            console.print(f"[yellow]Warning:[/] {role} = {model_id!r} is not in this list.")


def run() -> None:
    """Run the CLI; unexpected errors print one friendly line, the traceback goes to the log."""
    try:
        app(prog_name="lecturelens")
    except Exception as exc:  # last-resort guard (NFR-3: users never see a stack trace)
        logger.exception("Unhandled error")
        err_console.print(
            f"[bold red]Unexpected error:[/] {type(exc).__name__}: {exc}\n"
            f"Full traceback written to {APP_LOG_NAME} in app.log_dir."
        )
        raise SystemExit(1) from None
