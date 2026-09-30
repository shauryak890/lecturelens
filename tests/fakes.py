"""Offline stand-ins for heavy models and the LLM API (SPEC 13): deterministic, no network."""

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel

from lecturelens.config import LLMCfg
from lecturelens.llm.client import LLMBackend, LLMClient, RateLimiter, RawResponse
from lecturelens.llm.prompts import PromptRegistry
from lecturelens.schemas import Usage

_TOKEN = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    """Bag-of-words hashing embedder: texts sharing words get similar unit vectors."""

    def __init__(self, dim: int = 64, model_name: str = "hash-embedder") -> None:
        self.dim = dim
        self.model_name = model_name

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            vec[int.from_bytes(digest, "little") % self.dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.stack([self._vector(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


class WhitespaceCounter:
    """Token counter where every whitespace-separated word is one token."""

    def count(self, text: str) -> int:
        return len(text.split())


class FakeReranker:
    """Scores a passage by how many distinct query words it contains (minus ``offset``)."""

    def __init__(self, offset: float = 0.0) -> None:
        self.offset = offset
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, passages: list[str]) -> list[float]:
        self.calls.append((query, passages))
        words = set(_TOKEN.findall(query.lower()))
        return [len(words & set(_TOKEN.findall(p.lower()))) - self.offset for p in passages]


@dataclass
class LLMCall:
    """One call recorded by FakeLLMClient."""

    task: str
    system: str | None
    user: str
    schema: type[BaseModel]
    model: str | None


class FakeLLMClient(LLMClient):
    """LLMClient whose API is replaced by scripted responses per task.

    Prompts are still rendered by the real PromptRegistry (so missing template variables fail
    the test), but ``generate_json`` returns the next scripted object for the task, or raises
    it if it is an exception. The last scripted response repeats once the script runs out.
    """

    def __init__(
        self,
        cfg: LLMCfg,
        registry: PromptRegistry,
        responses: dict[str, list[BaseModel | Exception]] | None = None,
    ) -> None:
        super().__init__(cfg, registry, LLMBackend(), limiter=RateLimiter(10**6))
        self.responses = {task: list(items) for task, items in (responses or {}).items()}
        self.calls: list[LLMCall] = []
        self._served: dict[str, int] = defaultdict(int)

    def tasks_called(self) -> list[str]:
        return [c.task for c in self.calls]

    def generate_json(
        self,
        *,
        task,
        system,
        user,
        schema,
        temperature,
        max_output_tokens=None,
        model=None,
        allow_repair=True,
    ):  # untyped on purpose: same keywords as LLMClient.generate_json
        self.calls.append(LLMCall(task, system, user, schema, model))
        script = self.responses.get(task)
        if not script:
            raise AssertionError(f"FakeLLMClient: no scripted response for task {task!r}")
        index = min(self._served[task], len(script) - 1)
        self._served[task] += 1
        item = script[index]
        if isinstance(item, Exception):
            raise item
        assert isinstance(item, schema), (
            f"scripted {type(item).__name__} is not a {schema.__name__}"
        )
        return item, Usage(prompt_tokens=len(user.split()), output_tokens=10, api_calls=1)


@dataclass
class FakeBackend(LLMBackend):
    """Backend that plays back scripted outcomes (RawResponse or exception), one per request."""

    script: list[RawResponse | Exception]
    calls: list[dict] = field(default_factory=list)

    def generate(self, **kwargs) -> RawResponse:  # same keywords as LLMBackend.generate
        self.calls.append(kwargs)
        if not self.script:
            raise AssertionError("FakeBackend: unexpected extra request")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
