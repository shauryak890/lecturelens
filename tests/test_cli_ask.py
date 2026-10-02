"""CLI `ask` and `chat` with a stubbed pipeline (no models, no network)."""

import pytest
from typer.testing import CliRunner

from lecturelens.cli import app
from lecturelens.errors import LLMError
from lecturelens.rag import pipeline as pipeline_mod
from lecturelens.schemas import AnswerResponse, AskResult, Chunk, RetrievedChunk

from .conftest import CONFIG_PATH


class StubPipeline:
    """Records questions and returns a fixed cited answer."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[dict] = []

    def ask(self, question: str, **kwargs) -> AskResult:
        if "history" in kwargs:  # snapshot: the CLI keeps appending to the same list
            kwargs["history"] = list(kwargs["history"])
        self.calls.append({"question": question, **kwargs})
        if self.fail:
            raise LLMError("Gemini API 503: overloaded")
        chunk = Chunk(
            chunk_id="d:14:0",
            doc_id="d",
            file_name="lecture3.pdf",
            page=14,
            heading_path="Unit 2 > BPE",
            chunk_index=0,
            text="BPE merges pairs.",
            n_tokens=3,
        )
        return AskResult(
            question=question,
            standalone_question=question,
            response=AnswerResponse(
                answer="BPE merges the most frequent pair [S1].",
                cited_sources=[1],
                answerable=True,
                confidence="high",
            ),
            sources=[
                RetrievedChunk(
                    chunk=chunk,
                    score=3.2,
                    source="rerank",
                    ranks={"dense": 2, "bm25": 1, "rerank": 1},
                )
            ],
            timings_ms={"total_ms": 12.0},
            usage={"prompt_tokens": 900, "output_tokens": 40, "llm_calls": 1, "cache_hits": 0},
        )


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> StubPipeline:
    stub = StubPipeline()
    monkeypatch.setattr(pipeline_mod, "build_pipeline", lambda settings: stub)
    return stub


def _run(*args: str, input: str | None = None):  # "input" mirrors CliRunner
    return CliRunner().invoke(app, ["--config", str(CONFIG_PATH), *args], input=input)


def test_ask_prints_answer_sources_and_debug(stub: StubPipeline) -> None:
    result = _run(
        "ask", "How does BPE work?", "--mode", "eli5", "-r", "bm25", "--no-rerank", "--debug"
    )
    assert result.exit_code == 0, result.output
    assert "BPE merges the most frequent pair [S1]" in result.output
    assert "lecture3.pdf" in result.output and "14" in result.output
    assert "dense:2 bm25:1 rerank:1" in result.output
    assert "prompt_tokens 900" in result.output
    assert stub.calls == [
        {
            "question": "How does BPE work?",
            "mode": "eli5",
            "retrieval_mode": "bm25",
            "rerank": False,
        }
    ]


def test_ask_rejects_unknown_mode(stub: StubPipeline) -> None:
    assert _run("ask", "q", "--mode", "pirate").exit_code == 2


def test_ask_error_is_friendly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline_mod, "build_pipeline", lambda settings: StubPipeline(fail=True))
    result = _run("ask", "q")
    assert result.exit_code == 1
    assert "503" in result.output and "Traceback" not in result.output


def test_chat_keeps_history_and_clears_it(stub: StubPipeline) -> None:
    result = _run("chat", input="What is BPE?\nand its merges?\n/clear\nnew topic\n/exit\n")
    assert result.exit_code == 0, result.output
    histories = [len(call["history"]) for call in stub.calls]
    assert histories == [0, 1, 0]  # second question sees the first turn; /clear resets
    assert stub.calls[1]["history"][0].question == "What is BPE?"


def test_brackets_in_model_text_are_not_eaten_as_markup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: Rich read "[i]" in "D[i][0] = i" as an italic tag and dropped it."""
    from lecturelens import services as services_mod
    from lecturelens.schemas import Flashcard, FlashcardSet, Usage
    from lecturelens.tools.base import StudyResult

    card = Flashcard(front="MED table init?", back="D[i][0] = i and D[0][j] = j", source_ids=[1])
    chunk = StubPipeline().ask("q").sources[0]

    class StubTools:
        def generate(self, scope, n=None):
            return StudyResult(FlashcardSet(cards=[card]), [chunk], Usage(), "MED [notes]")

    class StubServices:
        def __init__(self, settings) -> None:
            self.flashcards = StubTools()

    monkeypatch.setattr(services_mod, "Services", StubServices)
    result = _run("flashcards", "--topic", "MED")
    assert result.exit_code == 0, result.output
    assert "D[i][0] = i" in result.output and "D[0][j] = j" in result.output
    assert "MED [notes]" in result.output
