"""Typer command-line interface: ``python -m lecturelens <command>``.

This is the only module that prints to the console (via Rich). Commands are added phase by
phase: P0 ships ``models``; ingest, stats, ask, chat, quiz, summarize, flashcards and eval follow.
"""

import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from lecturelens.config import DEFAULT_CONFIG_PATH, Settings, load_settings
from lecturelens.errors import LectureLensError
from lecturelens.logging_utils import APP_LOG_NAME, setup_logging

logger = logging.getLogger(__name__)
console = Console()
err_console = Console(stderr=True)

app = typer.Typer(
    name="lecturelens",
    help="LectureLens: a citation-grounded RAG study tutor for your course notes.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


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
    try:
        settings = load_settings(config)
    except LectureLensError as exc:
        raise _fail(exc) from exc
    setup_logging(settings.app.log_level, settings.app.log_dir, [settings.llm.api_key_env])
    ctx.obj = settings


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
