# LectureLens - citation-grounded RAG tutor for your course notes

> Ask your lecture PDFs a question, get an answer with page-level citations,
> and generate quizzes, summaries and flashcards from the same material.

DSE4150 (Natural Language Processing) course project. The full specification is in
[docs/SPEC.pdf](docs/SPEC.pdf); design decisions are logged in [docs/DECISIONS.md](docs/DECISIONS.md).

## Progress

| Phase | Scope | Status |
|---|---|---|
| P0 Setup | Scaffold, config + prompt files, schemas, settings loader, PromptRegistry, `models` command | Done |
| P1 Ingestion + indexing | Loader, cleaner, chunker, embedder, ChromaDB, BM25, manifest, `ingest`/`stats`, sample notes | Done |
| P2 Retrieval + RAG | RRF, reranker, hybrid retriever, LLM client, citations, pipeline, `ask`/`chat` | Not started |
| P3 Tools + UI | Quiz, summary, flashcards, Streamlit app | Not started |
| P4 Evaluation + polish | Metrics, judge, ablations, report, README results | Not started |

## Quickstart (Windows PowerShell; use `source .venv/bin/activate` on macOS/Linux)

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
copy .env.example .env      # then put your GEMINI_API_KEY in .env
python -m lecturelens models
python -m lecturelens ingest --sample   # or put PDFs in data/raw and run: ingest
python -m lecturelens stats
```

The first `ingest` downloads the bge-small embedding model (~130 MB) once; later runs load it
from the local Hugging Face cache and skip unchanged files.

## Key files

- [prompts/prompts.yaml](prompts/prompts.yaml): every LLM prompt, versioned, with per-task temperature and output schema.
- [config/config.yaml](config/config.yaml): every tunable setting. Override any key with an env var
  `LECTURELENS__SECTION__KEY`, for example `LECTURELENS__LLM__MODEL=gemini-2.5-flash`.

## Testing

```
pytest -q
ruff check . && ruff format --check .
```

## Licence

MIT (see [LICENSE](LICENSE)). Note: the PDF parser dependency `pymupdf4llm` is AGPL-3.0.
