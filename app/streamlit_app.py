"""LectureLens web UI (SPEC 11.1). Run from the repo root: ``streamlit run app/streamlit_app.py``.

This file only lays out widgets and calls the ``lecturelens`` package; all retrieval, prompting
and validation logic lives in the package (see ``lecturelens.services.Services``).
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import streamlit as st

from lecturelens.config import Settings, load_settings
from lecturelens.errors import LectureLensError
from lecturelens.evaluation.report import headline_table
from lecturelens.evaluation.runner import EvalReport
from lecturelens.ingestion.loader import discover_files
from lecturelens.logging_utils import setup_logging
from lecturelens.schemas import AskResult, ChatTurn, RetrievedChunk
from lecturelens.services import Services
from lecturelens.tools.base import Scope
from lecturelens.tools.flashcards import source_label, to_csv

logger = logging.getLogger("lecturelens.app")
T = TypeVar("T")

ANSWER_MODES = {"concise": "Concise", "detailed": "Detailed", "eli5": "Explain like I'm new"}
RETRIEVAL_MODES = {"hybrid": "Hybrid (BM25 + dense)", "dense": "Dense only", "bm25": "BM25 only"}
CONFIDENCE_COLOR = {"high": "green", "medium": "orange", "low": "red"}
OPTION_LETTERS = "ABCD"
RESULTS_JSON = "results.json"
UPLOAD_TYPES = ["pdf", "md", "txt"]


# ---------------------------------------------------------------- shared helpers


@st.cache_resource
def get_services() -> Services:
    """One Services container per server process (models load once, on first use)."""
    settings = load_settings()
    setup_logging(settings.app.log_level, settings.app.log_dir, [settings.llm.api_key_env])
    return Services(settings)


def run_safely(action: Callable[[], T], spinner: str) -> T | None:
    """Run ``action`` with a spinner; show a friendly error instead of a stack trace."""
    try:
        with st.spinner(spinner):
            return action()
    except LectureLensError as exc:
        st.error(str(exc))
    except Exception:  # noqa: BLE001 - NFR-3: the UI never shows a stack trace
        logger.exception("Unexpected error in the UI")
        st.error("Something went wrong. Details were written to logs/lecturelens.log.")
    return None


def refs(ids: list[int], sources: list[RetrievedChunk]) -> str:
    """``S1 notes.pdf p.3, S2 ...`` for a list of 1-based source ids."""
    return ", ".join(f"S{i} {source_label(sources[i - 1])}" for i in ids)


def source_cards(sources: list[RetrievedChunk], cited: set[int] | None = None) -> None:
    """One expander per source with file, page, section, excerpt and retrieval scores."""
    for i, src in enumerate(sources, start=1):
        chunk = src.chunk
        marker = "📌 " if cited and i in cited else ""
        title = f"{marker}S{i} · {chunk.file_name} · p.{chunk.page}"
        with st.expander(title):
            if chunk.heading_path:
                st.caption(chunk.heading_path)
            st.markdown(chunk.text)
            ranks = ", ".join(f"{name} #{rank}" for name, rank in src.ranks.items())
            st.caption(f"score {src.score:.3f} ({src.source}) · ranks: {ranks or '-'}")


def scope_picker(key: str, services: Services) -> Scope | None:
    """Topic text box or document picker; returns the chosen scope (or None if incomplete)."""
    files = services.stats().files
    kind = st.radio("Cover", ["Topic", "Document"], horizontal=True, key=f"{key}_kind")
    if kind == "Topic":
        topic = st.text_input("Topic", placeholder="e.g. finite-state transducers", key=f"{key}_t")
        return Scope(topic=topic) if topic.strip() else None
    if not files:
        st.info("No documents indexed yet; add some in the Library tab.")
        return None
    names = {f.file_name: f.doc_id for f in files}
    choice = st.selectbox("Document", list(names), key=f"{key}_doc")
    return Scope(doc_id=names[choice])


def show_notes(notes: list[str]) -> None:
    for note in notes:
        st.caption(f"ℹ️ {note}")


# ---------------------------------------------------------------- sidebar


def sidebar(services: Services) -> dict:
    """Index status and answer settings; returns the options the Ask tab uses."""
    settings: Settings = services.settings
    stats = services.stats()
    st.sidebar.title("📚 LectureLens")
    st.sidebar.caption("Answers from your course notes, with page-level citations.")
    c1, c2 = st.sidebar.columns(2)
    c1.metric("Documents", stats.n_docs)
    c2.metric("Chunks", stats.n_chunks)
    st.sidebar.caption(f"Embeddings: {stats.embedding_model or settings.embeddings.model}")

    st.sidebar.subheader("Answer settings")
    mode = st.sidebar.selectbox(
        "Answer style",
        list(ANSWER_MODES),
        index=list(ANSWER_MODES).index(settings.generation.default_mode),
        format_func=ANSWER_MODES.get,
    )
    retrieval = st.sidebar.selectbox(
        "Retrieval",
        list(RETRIEVAL_MODES),
        index=list(RETRIEVAL_MODES).index(settings.retrieval.mode),
        format_func=RETRIEVAL_MODES.get,
    )
    rerank = st.sidebar.toggle("Cross-encoder rerank", value=settings.retrieval.rerank)
    names = {f.file_name: f.doc_id for f in stats.files}
    chosen = st.sidebar.multiselect("Only search these documents", list(names))
    if st.sidebar.button("Clear chat", width="stretch"):
        st.session_state.chat = []
    st.sidebar.caption("See the About tab for how it works.")
    return {
        "mode": mode,
        "retrieval_mode": retrieval,
        "rerank": rerank,
        "doc_filter": [names[n] for n in chosen] or None,
    }


# ---------------------------------------------------------------- tabs


def library_tab(services: Services) -> None:
    settings = services.settings
    st.subheader("Your course material")
    uploads = st.file_uploader(
        "Add PDFs, Markdown or text files", type=UPLOAD_TYPES, accept_multiple_files=True
    )
    col1, col2 = st.columns(2)
    if col1.button("Index now", type="primary", disabled=not uploads):
        data_dir = Path(settings.app.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        for upload in uploads or []:
            (data_dir / Path(upload.name).name).write_bytes(upload.getvalue())
        index_folder(services, data_dir)
    if col2.button("Load sample notes"):
        index_folder(services, Path(settings.app.sample_dir))

    stats = services.stats()
    if not stats.files:
        st.info("The library is empty. Upload your lecture PDFs or load the sample notes.")
        return
    st.dataframe(
        [
            {
                "File": f.file_name,
                "Pages": f.n_pages,
                "Chunks": f.n_chunks,
                "Indexed at (UTC)": f.indexed_at,
            }
            for f in stats.files
        ],
        hide_index=True,
        width="stretch",
    )
    with st.expander("Remove a document from the index"):
        names = {f.file_name: f.doc_id for f in stats.files}
        victim = st.selectbox("Document", list(names), key="delete_doc")
        st.caption("The file itself is not deleted; ingesting its folder again re-adds it.")
        if st.button("Remove from index"):
            name = run_safely(lambda: services.delete_document(names[victim]), "Removing...")
            if name:
                st.toast(f"Removed {name}")
                st.rerun()


def index_folder(services: Services, folder: Path) -> None:
    """Ingest a folder with a progress bar, then report what happened."""
    total = max(len(discover_files(folder, services.settings.ingestion.extensions)), 1)
    bar = st.progress(0.0, text="Loading the embedding model (first run downloads it)...")
    done = {"n": 0}

    def on_file(path: Path) -> None:
        bar.progress(done["n"] / total, text=f"Indexing {path.name}...")
        done["n"] += 1

    report = run_safely(lambda: services.ingest(folder, on_file=on_file), "Indexing...")
    bar.empty()
    if report is None:
        return
    st.success(
        f"{len(report.indexed)} file(s) indexed, {report.new_chunks} new chunks "
        f"({report.seconds:.0f}s). {len(report.skipped)} unchanged."
    )
    for name, error in report.failed.items():
        st.warning(f"{name}: {error}")


def render_answer(result: AskResult) -> None:
    """Confidence badge, answer, source cards and a debug expander."""
    response = result.response
    st.badge(f"confidence: {response.confidence}", color=CONFIDENCE_COLOR[response.confidence])
    st.markdown(response.answer)
    if result.sources:
        source_cards(result.sources, set(response.cited_sources))
    with st.expander("Debug"):
        st.write(f"**Standalone question:** {result.standalone_question}")
        st.json({"timings_ms": result.timings_ms, "usage": result.usage})


def ask_tab(services: Services, options: dict) -> None:
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for turn in st.session_state.chat:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            render_answer(turn["result"])
    question = st.chat_input("Ask about your course material")
    if not question:
        return
    with st.chat_message("user"):
        st.markdown(question)
    history = [
        ChatTurn(question=t["question"], answer=t["result"].response.answer)
        for t in st.session_state.chat
    ]
    with st.chat_message("assistant"):
        result = run_safely(
            lambda: services.pipeline.ask(question, history=history, **options), "Thinking..."
        )
        if result is not None:
            render_answer(result)
            st.session_state.chat.append({"question": question, "result": result})


def quiz_tab(services: Services) -> None:
    cfg = services.settings.quiz
    with st.form("quiz_form"):
        scope = scope_picker("quiz", services)
        c1, c2 = st.columns(2)
        n = c1.number_input("Questions", 1, cfg.max_n, cfg.default_n)
        difficulty = c2.selectbox(
            "Difficulty",
            ["mixed", "easy", "medium", "hard"],
            index=["mixed", "easy", "medium", "hard"].index(cfg.default_difficulty),
        )
        generate = st.form_submit_button("Generate quiz", type="primary")
    if generate:
        if scope is None:
            st.warning("Enter a topic or pick a document first.")
        else:
            result = run_safely(
                lambda: services.quiz.generate(scope, n=int(n), difficulty=difficulty),
                "Writing your quiz...",
            )
            if result is not None:
                st.session_state.quiz = result
                st.session_state.quiz_submitted = False
    result = st.session_state.get("quiz")
    if result is None:
        return
    st.subheader(f"Quiz: {result.scope_label}")
    show_notes(result.notes)
    submitted = st.session_state.get("quiz_submitted", False)
    score = 0
    for i, q in enumerate(result.output.questions):
        st.markdown(f"**{i + 1}. {q.question}**")
        labels = [f"{OPTION_LETTERS[j]}) {opt}" for j, opt in enumerate(q.options)]
        picked = st.radio(
            "Your answer",
            range(len(labels)),
            format_func=labels.__getitem__,
            index=None,
            key=f"quiz_q{i}",
            disabled=submitted,
            label_visibility="collapsed",
        )
        if submitted:
            if picked == q.correct_index:
                score += 1
                st.success(f"Correct. {q.explanation}")
            else:
                st.error(f"Answer: {OPTION_LETTERS[q.correct_index]}. {q.explanation}")
            st.caption(
                f"Sources: {refs(q.source_ids, result.sources)} · {q.difficulty}, {q.bloom_level}"
            )
    c1, c2 = st.columns(2)
    if not submitted and c1.button("Submit answers", type="primary"):
        st.session_state.quiz_submitted = True
        st.rerun()
    if submitted:
        st.metric("Score", f"{score} / {len(result.output.questions)}")
    if c2.button("New quiz"):
        for key in [k for k in st.session_state if str(k).startswith("quiz")]:
            del st.session_state[key]
        st.rerun()


def study_tab(services: Services) -> None:
    scope = scope_picker("study", services)
    c1, c2, c3 = st.columns([1, 1, 1])
    n_cards = c3.number_input("Cards", 1, 30, services.settings.flashcards.default_n)
    if c1.button("Summarise", type="primary", disabled=scope is None):
        st.session_state.summary = run_safely(
            lambda: services.summarizer.summarize(scope), "Summarising..."
        )
    if c2.button("Make flashcards", disabled=scope is None):
        st.session_state.cards = run_safely(
            lambda: services.flashcards.generate(scope, n=int(n_cards)), "Writing flashcards..."
        )
    summary = st.session_state.get("summary")
    if summary is not None:
        s = summary.output
        st.subheader(s.title)
        st.markdown(s.overview)
        st.markdown("**Key points**")
        for point in s.key_points:
            st.markdown(
                f"- {point.point}  \n  <small>{refs(point.source_ids, summary.sources)}</small>",
                unsafe_allow_html=True,
            )
        if s.glossary:
            st.markdown("**Glossary**")
            st.dataframe(
                [
                    {
                        "Term": g.term,
                        "Definition": g.definition,
                        "Sources": refs(g.source_ids, summary.sources),
                    }
                    for g in s.glossary
                ],
                hide_index=True,
                width="stretch",
            )
        with st.expander("Sources"):
            source_cards(summary.sources)
    cards = st.session_state.get("cards")
    if cards is not None:
        st.subheader(f"Flashcards: {cards.scope_label}")
        show_notes(cards.notes)
        st.download_button(
            "Download CSV (Anki import)",
            to_csv(cards.output, cards.sources),
            file_name="lecturelens_flashcards.csv",
            mime="text/csv",
        )
        for card in cards.output.cards:
            with st.expander(card.front):
                st.markdown(card.back)
                st.caption(refs(card.source_ids, cards.sources))


def about_tab(services: Services) -> None:
    settings = services.settings
    st.subheader("How LectureLens works")
    st.markdown(
        "1. **Ingest:** PDFs are parsed page by page (OCR for scanned pages), cleaned, and split "
        "into ~400-token chunks that never cross a page, so every chunk has one page number.\n"
        "2. **Retrieve:** each question is searched with BM25 (exact terms) and dense "
        "embeddings (meaning), fused with Reciprocal Rank Fusion and reranked by a "
        "cross-encoder.\n"
        "3. **Answer:** Gemini answers *only* from the numbered excerpts and must cite them as "
        "`[S1]`, `[S2]`... Citations that do not exist are removed; uncited answers get one "
        "repair attempt; if nothing relevant is found, no LLM call is made at all.\n"
        "4. **Follow-ups** are rewritten into standalone questions before searching."
    )
    architecture = Path("docs/architecture.png")
    if architecture.is_file():
        st.image(str(architecture))
    results = Path(settings.eval.output_dir) / RESULTS_JSON
    st.subheader("Latest evaluation")
    if results.is_file():
        report = EvalReport.model_validate_json(results.read_text(encoding="utf-8"))
        st.caption(
            f"{report.n_questions} questions · {report.created_at} · prompt file "
            f"v{report.prompt_version} · full report: eval/results/report.md"
        )
        st.markdown("\n".join(headline_table(report)))
    else:
        st.info("No evaluation results yet. Run `python -m lecturelens eval`.")


# ---------------------------------------------------------------- page


def main() -> None:
    st.set_page_config(page_title="LectureLens", page_icon="📚", layout="wide")
    try:
        services = get_services()
        options = sidebar(services)
    except LectureLensError as exc:
        st.error(f"LectureLens could not start: {exc}")
        st.stop()
    library, ask, quiz, study, about = st.tabs(
        ["📁 Library", "💬 Ask", "📝 Quiz", "🗂️ Summary & Flashcards", "ℹ️ About"]
    )
    with library:
        library_tab(services)
    with ask:
        ask_tab(services, options)
    with quiz:
        quiz_tab(services)
    with study:
        study_tab(services)
    with about:
        about_tab(services)


main()
