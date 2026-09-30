# CLAUDE.md - instructions for Claude Code
## Project
LectureLens: a citation-grounded RAG study tutor over course PDFs (NLP course project, DSE4150).
Full specification: docs/SPEC.pdf. Follow it. If something is ambiguous, prefer the simpler option
and note it in docs/DECISIONS.md.
## Hard requirements (graded)
- Python 3.11, package under src/lecturelens, runnable as `python -m lecturelens`.
- ALL prompts live in prompts/prompts.yaml. No prompt strings in Python code.
- ALL settings live in config/config.yaml. No magic numbers in code.
- API key only from env (GEMINI_API_KEY via .env). Never print or log it.
- LLM outputs are JSON validated with Pydantic models in schemas.py.
- Do NOT use LangChain or LlamaIndex. Write retrieval, fusion, prompting yourself.
- Tests must run offline: use FakeLLMClient and HashEmbedder fixtures.
## Conventions
- Type hints + Google-style docstrings on public functions. ruff line length 100.
- Custom exceptions in lecturelens/errors.py. Use logging, not print (Rich only in cli.py).
- Keep Streamlit file thin: it only calls the package.
- Before using the google-genai SDK, check the installed version and current docs; keep all SDK
 calls inside llm/client.py.
## Commands
- Install: pip install -r requirements.txt -r requirements-dev.txt && pip install -e .
- Test: pytest -q Lint: ruff check . && ruff format --check .
- Ingest sample: python -m lecturelens ingest --sample
- App: streamlit run app/streamlit_app.py
## Workflow
Work phase by phase (P0-P4 in SPEC section 15). At the end of each phase: run tests + ruff,
update README progress, and make a git commit with a descriptive message.