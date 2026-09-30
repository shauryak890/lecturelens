# Design decisions

Places where the spec was ambiguous or left a choice open, and what was chosen. Per CLAUDE.md the
simpler option was preferred.

## P0 - Scaffold

- **Spec file name.** The spec arrived as `docs/LectureLens_NLP_Project_Spec.pdf`; renamed to
  `docs/SPEC.pdf` as CLAUDE.md and SPEC section 16.1 expect.
- **Scaffold scope.** Subpackages from SPEC section 5 (`ingestion/`, `indexing/`, `retrieval/`,
  `llm/`, `rag/`, `tools/`, `evaluation/`) exist with `__init__.py` only. Their modules get written
  in the phase that implements them, so there are no empty placeholder files. `app/`, the
  Makefile (optional in the spec) and the CI workflow (P4) are deferred.
- **`models` command and the SDK.** CLAUDE.md requires all google-genai calls to live in
  `llm/client.py`, so P0 adds `list_models()` there. The full `LLMClient` comes in P2. The command
  lists models that support `generateContent` (`--all` shows every model) and warns when a model
  ID in config.yaml is not available to the key.
- **Model IDs.** `llm.model`, `fallback_model` and `judge_model` are copied unchanged from SPEC
  section 10. The spec calls them placeholders: check them with `python -m lecturelens models`.
- **Env override syntax.** `LECTURELENS__A__B=value` sets `a.b`. Keys are case-insensitive
  (Windows upper-cases env var names). Values are parsed as YAML scalars, so `false` becomes a
  bool and `7` an int. Unknown sections or keys are errors, not silently ignored.
- **Strict config.** Every config model uses `extra="forbid"`, so a typo such as `llm.modle`
  fails at startup. Settings are immutable. Ablations use `Settings.with_overrides()`, which
  returns a new validated copy.
- **Secrets.** `Settings` holds only the *name* of the API key variable
  (`llm.api_key_env`). The key itself is read by `Settings.require_api_key()` when an LLM command
  runs, so `snapshot()` cannot leak it. A logging filter also redacts the key's value from every
  log record.
- **Relative paths.** Paths in config.yaml are relative to the working directory. Run commands
  from the repository root.
- **Prompt schema key.** `schema` in prompts.yaml maps to `TaskSpec.schema_name`, because a field
  named `schema` would shadow a Pydantic `BaseModel` attribute.
- **Long prompt lines.** Several lines in SPEC section 9 were wrapped only by the PDF layout (the
  few-shot `Output:` JSON and the `detailed`/`eli5` mode strings). They are single lines in
  prompts.yaml, so each example is one valid JSON object. A test checks that it validates
  against the task's schema.
- **Python version.** Development uses Python 3.11 (CLAUDE.md). `requires-python = ">=3.10"` and
  ruff `target-version = "py310"` follow SPEC section 14 / NFR-6.
- **google-genai client lifetime.** `genai.Client.__del__` closes the shared HTTP transport, so
  a client must stay referenced for as long as its `models` API is used. Calling
  `genai.Client(...).models.list()` on a temporary failed with "Cannot send a request, as the
  client has been closed". `list_models()` uses `with genai.Client(...) as client:`. The P2
  `LLMClient` must keep one client as an attribute for its whole lifetime.
- **No stack traces on the console (NFR-3).** `cli.run()` (used by `python -m lecturelens` and
  the `lecturelens` script) catches unexpected errors and prints one line. The console log
  handler drops tracebacks; the full traceback goes to `logs/lecturelens.log`.

## P1 - Ingestion and indexing

- **Header/footer removal needs at least 3 pages.** SPEC 7.1 drops lines that appear on more
  than 50% of pages. On a 1-2 page document every line qualifies, so notes would be erased.
  New config key `ingestion.header_footer_min_pages: 3`. As the spec says, digits are masked
  first, so "Page 3 of 20" and "Page 4 of 20" count as one repeated line.
- **Chunking details (SPEC 7.2).**
  - Pieces are split at headings, then paragraphs, then single lines, then sentences, then
    words. The single-line level is an addition: bullet lists have no blank lines between
    items. A single "word" over the limit is halved by characters.
  - Pieces are packed greedily across headings within a page. Short sections (typical of
    slides) are therefore merged rather than becoming tiny chunks.
  - A chunk's `heading_path` is the section that contributes most of its tokens.
  - Overlap takes the last ~`chunk_overlap_tokens` tokens of the previous chunk, cut at word
    boundaries. It is not just the last whole piece, which could be a 200-token paragraph.
  - A trailing chunk with fewer than `min_chunk_tokens` *new* tokens is merged into its
    predecessor only if the result still fits `chunk_size_tokens`.
  - The heading path carries over to following pages of the same document (continuation
    slides) and resets per document. `chunk_index` counts within a page, so ids are
    `doc_id:page:index`.
- **Token counting** uses the `tokenizers` library with bge-small's own tokenizer (much
  lighter than importing transformers). Truncation is disabled so long texts are counted
  fully, not capped at 512.
- **Context header** (`"{file} | {heading path}"`) is added only to the text that is embedded
  and BM25-indexed (`chunker.indexed_text`). Users and the LLM see the plain chunk text.
- **BM25 uses the Lucene IDF**, `log(1 + (N - n + 0.5) / (n + 0.5))`, via a small subclass of
  rank-bm25's `BM25Okapi`. The classic IDF is 0 for a term in half the chunks and negative
  above that. rank-bm25 floors negatives at `epsilon * mean IDF`, which is ~0 on small
  corpora, so real matches scored 0. `BM25Index.build(ids, texts)` takes texts rather than
  chunks, so the index does not need to know about context headers.
- **Stemming** (`retrieval.bm25.stemming`, off by default) uses `snowballstemmer`, a small
  pure-Python dependency with no data downloads, imported only when enabled.
- **`indexing/indexer.py`** is an addition to the SPEC 5 tree. It orchestrates ingest so the
  CLI (and the P3 Streamlit Library tab) contain no pipeline logic. `build_indexer(settings)`
  is the single factory.
- **Manifest.** It records the embedding model, dimension and all chunking-related settings.
  A change to any of them raises `IndexMismatchError` asking for `ingest --rebuild`, so
  incompatible chunks are never mixed. Other behaviour:
  - Files are keyed by absolute path.
  - A changed file's old chunks are replaced.
  - A second file with identical content is reported as a duplicate and indexed once.
  - A file that fails to parse is reported and the rest of the folder is still indexed.
  - The manifest is saved after every file, so an interrupted ingest resumes.
  - Limitation: files deleted from disk are not pruned automatically (a delete action comes
    with the P3 Library tab).
  - BM25 settings are stored in `bm25.pkl`; if they change, BM25 is silently rebuilt (cheap,
    no embeddings needed).
- **Lazy model loading.** The embedder (torch + bge, ~15 s to load) and the tokenizer load
  on first use. Re-ingesting unchanged files takes ~2 s instead of ~17 s. Because of this the
  model name is checked up front and the vector dimension just before the first embedding.
  P2's retriever must check the manifest (model name and dimension) before querying.
- **GeminiEmbedder is deferred.** It is optional in SPEC 6.4, and `embeddings.provider:
  gemini` raises a clear `ConfigError`. Local embeddings cost no API quota (NFR-1).
- **Test fakes** (`HashEmbedder`, `WhitespaceCounter`) live in `tests/fakes.py`, not in the
  package. Tests use a real Chroma store in a temp dir, so they stay offline and fast (~5 s).
- **Chroma** runs with telemetry off and its built-in embedder disabled
  (`embedding_function=None`). Vectors are always passed in explicitly, using cosine space.
