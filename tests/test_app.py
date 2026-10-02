"""Headless tests of the Streamlit app with AppTest (no browser, no models, no network)."""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from lecturelens import services as services_mod
from lecturelens.config import Settings
from lecturelens.indexing.indexer import Indexer
from lecturelens.schemas import AnswerResponse, AskResult, Quiz, QuizQuestion, Usage
from lecturelens.tools.base import StudyResult

from .conftest import ROOT

APP = str(ROOT / "app" / "streamlit_app.py")
TIMEOUT_S = 60


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):  # noqa: ANN202
    """Run the app from the repo root against a temp index, without an API key."""
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("LECTURELENS__APP__INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("LECTURELENS__APP__LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("GEMINI_API_KEY", "")
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def _run() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT_S)
    at.run()
    assert not at.exception, at.exception
    return at


def test_empty_library_renders_without_models_or_key() -> None:
    at = _run()
    assert [t.label for t in at.tabs] == [
        "📁 Library",
        "💬 Ask",
        "📝 Quiz",
        "🗂️ Summary & Flashcards",
        "ℹ️ About",
    ]
    assert any("library is empty" in i.value for i in at.info)
    assert [m.value for m in at.sidebar.metric] == ["0", "0"]


def _stub_services(monkeypatch: pytest.MonkeyPatch, indexer: Indexer, **features) -> None:
    """Replace Services so the app uses an offline index and scripted features."""

    class StubServices(services_mod.Services):
        def __init__(self, settings: Settings) -> None:
            super().__init__(indexer.settings)
            self.__dict__.update(
                embedder=indexer.embedder, store=indexer.store, indexer=indexer, **features
            )

    import lecturelens.services

    monkeypatch.setattr(lecturelens.services, "Services", StubServices)


def test_ask_renders_answer_badge_and_sources(
    monkeypatch: pytest.MonkeyPatch, tiny_index: Indexer
) -> None:
    source = tiny_index.store.all_chunks()[0]

    class Pipeline:
        def ask(self, question: str, **kwargs) -> AskResult:
            from lecturelens.schemas import RetrievedChunk

            return AskResult(
                question=question,
                standalone_question=question,
                response=AnswerResponse(
                    answer="Tokens are units of text [S1].",
                    cited_sources=[1],
                    answerable=True,
                    confidence="high",
                ),
                sources=[
                    RetrievedChunk(
                        chunk=source,
                        score=2.5,
                        source="rerank",
                        ranks={"dense": 1, "bm25": 2, "rerank": 1},
                    )
                ],
                timings_ms={"total_ms": 5.0},
                usage={"llm_calls": 1},
            )

    _stub_services(monkeypatch, tiny_index, pipeline=Pipeline())
    at = _run()
    assert [m.value for m in at.sidebar.metric] == ["3", str(tiny_index.store.count())]
    at.chat_input[0].set_value("What is a token?").run()
    assert not at.exception, at.exception
    texts = [m.value for m in at.markdown]
    assert "Tokens are units of text [S1]." in texts
    assert any(f"S1 · {source.file_name}" in e.label for e in at.expander)


def test_ask_without_api_key_shows_friendly_error(
    monkeypatch: pytest.MonkeyPatch, tiny_index: Indexer
) -> None:
    _stub_services(monkeypatch, tiny_index)  # real pipeline -> LLM client needs the key
    at = _run()
    at.chat_input[0].set_value("What is a token?").run()
    assert not at.exception
    assert any("GEMINI_API_KEY is not set" in e.value for e in at.error)


def test_quiz_generate_submit_and_score(
    monkeypatch: pytest.MonkeyPatch, tiny_index: Indexer
) -> None:
    question = QuizQuestion(
        question="What does BPE merge?",
        options=["frequent pairs", "rare pairs", "sentences", "pages"],
        correct_index=0,
        explanation="Most frequent adjacent pair.",
        source_ids=[1],
        difficulty="easy",
        bloom_level="remember",
    )
    source = tiny_index.store.all_chunks()[0]

    class QuizTool:
        def generate(self, scope, n=None, difficulty=None):  # noqa: ANN001, ANN201
            from lecturelens.schemas import RetrievedChunk

            return StudyResult(
                Quiz(topic="BPE", questions=[question]),
                [RetrievedChunk(chunk=source, score=1.0, source="hybrid")],
                Usage(),
                "BPE",
            )

    _stub_services(monkeypatch, tiny_index, quiz=QuizTool())
    at = _run()
    at.text_input(key="quiz_t").set_value("BPE")
    at.button(key="FormSubmitter:quiz_form-Generate quiz").click().run()
    assert not at.exception, at.exception
    at.radio(key="quiz_q0").set_value(0).run()
    next(b for b in at.button if b.label == "Submit answers").click().run()
    assert not at.exception, at.exception
    assert any(m.value == "1 / 1" for m in at.metric)
    assert any("Most frequent adjacent pair." in s.value for s in at.success)
