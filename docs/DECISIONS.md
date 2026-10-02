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

### P1 follow-up after ingesting the real course PDFs

- **Repeated lines glued inside other lines.** On one page of a downloaded PDF, a per-page
  download watermark (a person's name and email address) came out on the same line as other
  text. Exact line matching missed it and it was indexed. Repeated lines of at least
  `ingestion.header_footer_substring_min_chars` (20) visible characters are now also removed
  when they appear *inside* another line of the same document. The length minimum stops a
  short repeated line (a bare page number, "Q&A") from cutting words out of real text.
  Digits in line keys are now masked with a NUL character instead of "#", so a repeated
  Markdown heading cannot turn its "#" into a digit wildcard.
- **OCR for scanned pages (RapidOCR).** Pages 8-14 of one lecture-notes PDF were scanned
  images, and pymupdf4llm skips OCR *silently* when no engine is installed. We added the
  `rapidocr` package (pip only, ONNX models, no system install), which pymupdf4llm detects
  automatically. pymupdf4llm decides per page, so text pages are not slowed down. It costs
  ~3-4 s per scanned page on CPU. Toggle it with `ingestion.use_ocr`; changing the toggle
  changes extracted text, so it requires `ingest --rebuild`. Tesseract was the alternative
  but needs a system install on every machine, including the examiner's.
- **Library console output.** PyMuPDF messages go to the `lecturelens.pymupdf` logger, the
  RapidOCR logger is raised to WARNING, and stray `print()`s during PDF parsing are captured
  and logged at DEBUG. The CLI progress display stays clean and the details are in
  `logs/lecturelens.log`.

## P2 - Retrieval and grounded answering

- **Gemini "thinking" tokens count against `max_output_tokens`.** Measured live on
  gemini-3.8-flash: with a 150-token cap the model spent 142 tokens thinking and returned 4
  tokens of broken JSON. New config keys:
  - `llm.thinking_level: low` keeps thinking short (the model rejects `minimal`).
  - `llm.thinking_token_allowance: 1024` is added to every task's `max_output_tokens`, so the
    prompt file's numbers keep meaning "visible answer length".
  - If a model rejects the thinking level (HTTP 400 mentioning thinking), the backend
    retries that model once without it and remembers not to send it again.
  - Empty output with finish reason `MAX_TOKENS` raises `LLMOutputError`, whose message
    names the allowance. A repair call cannot fix missing content.
- **Automatic function calling is disabled** in every request. We use no tools, and the SDK
  otherwise prints a notice for each call.
- **SDK boundary.** All google-genai code is in `llm/client.py`, behind a one-method
  `LLMBackend` interface (`GeminiBackend`). Tests inject `FakeBackend`, so the *real*
  retry/fallback/cache/repair logic runs offline. `FakeLLMClient` (tests/fakes.py)
  subclasses `LLMClient`: it renders the real prompts, then returns scripted objects.
- **Rate limiter.** It enforces a sliding 60 s window rather than the spec's token bucket.
  A bucket of size N can allow up to 2N requests across a window boundary; a window never
  exceeds `requests_per_minute`.
- **Retries** use tenacity: `stop_after_attempt(max_retries)` and
  `wait_exponential_jitter(initial=backoff_base_s)`, i.e. about 2, 4, 8 s plus jitter,
  retrying only on 429, 5xx and network timeouts. After the last retry the fallback model
  is tried once. Auth errors (401/403, or 400 "API key not valid") and unknown models fail
  immediately with a hint.
- **Repair of invalid JSON** is a prompt-file task (`repair_json`, schema supplied by the
  caller). It sends the validation error and the raw text back, at most once. A second
  failure raises `LLMOutputError`.
- **Prompt file 1.1.0** adds `formats` (excerpt label, history turn, repair problem
  descriptions) and `messages` (not found, empty index, blocked). Every string sent to the
  model or shown as a fixed answer therefore lives in prompts.yaml.
- **Cache.** It stores the *validated* JSON, keyed by model, system, user, temperature,
  schema and prompt version. Entries that no longer validate (schema changed) are ignored.
  Only temperatures up to `llm.cache.max_cacheable_temperature` are cached.
- **Token budget** (`retrieval.max_context_tokens`) is applied in one place,
  `rag.context.build_context`, so `[S{i}]` always equals `sources[i-1]`. The first chunk is
  always kept.
- **Abstention without an LLM call** happens when the index is empty or when reranking is
  on and every candidate scores below `retrieval.min_rerank_score`. With reranking off there
  is no score threshold (BM25 and cosine scores have no calibrated cut-off), so the LLM
  decides and its `answerable=false` path is measured in the eval.
- **The reranker scores the same text that was indexed** (with the "file | heading"
  context header).
- **Citations.** Grouped forms such as `[S1, S3]` are normalised to `[S1][S3]`. Invalid ids
  are stripped from text and list, and `cited_sources` becomes the union of valid ids listed
  and valid ids used in the text. An "answerable" reply left with no valid citation gets one
  `repair_answer` call; if it is still uncited, confidence is set to low.
- **Blocked or empty responses** (safety filters) become the fixed "blocked" message with
  `answerable=false` instead of an error.
- **`AskResult.usage`** has `prompt_tokens`, `output_tokens`, `llm_calls` (HTTP requests,
  including retries and repairs) and `cache_hits`. `timings_ms` has condense, retrieve,
  generate and total.
- **Query-time index check.** `build_retriever` refuses an index built with a different
  embedding model. Models (embedder, cross-encoder) load lazily on the first question.
- **`llm.provider: openai_compatible`** (optional in SPEC 8.5) is not implemented and raises
  a clear `ConfigError`.
- **CLI.**
  - `ask`/`chat` accept `--mode`, `--retrieval`, `--no-rerank` and `--debug`. Chat keeps
    history (`/clear`, `/exit`).
  - Console streams replace characters they cannot encode. A cp1252 Windows console
    otherwise crashed printing "⇒" in an answer.

### Evaluation dataset (`eval/qa_dataset.jsonl`)

- **Written by Claude Code at the student's request** (SPEC 15 assigns it to the student). It
  must be reviewed before the P4 evaluation. Every answerable question was written from the
  full text of its gold page(s) as extracted into the index, and every reference answer
  paraphrases those pages.
- **Mix (30 questions):** 7 factual, 7 conceptual, 4 comparison, 3 keyword-heavy, 2 follow-up
  pairs (4 items, the second turn carries `history`) and 5 unanswerable. The unanswerable
  topics are RLHF, Viterbi/HMM, BLEU, beam search and LDA. Each has zero keyword hits in the
  whole index, including the sample notes.
- **Gold pages use the file name and 1-based page** as shown in citations. A question can
  have several gold pages when the material repeats (e.g. e-insertion appears on two PPT1
  slides and one PPT-2 slide). q10 also counts the sample note `01_tokenization.md`, which
  genuinely answers it.
- **Label sanity check** (retrieval only, no LLM, default hybrid + rerank, 23 questions
  without history): hit@1 16/23 and hit@5 23/23, so every gold page is retrievable. 3 of the
  5 unanswerable questions already abstain without an LLM call (all rerank scores below -5).

### Heading noise from PDF extraction (after P2)

- **Inline HTML is stripped in `normalize_text`.** pymupdf4llm emits `<mark>` (510 times in
  the course PDFs), `<br>` (499), `<u>` (92), `<sup>` (32) and HTML comments around OCR'd
  picture text. These polluted chunk text (sent to the LLM) and heading paths.
  - Formatting tags are removed and their text kept.
  - `<sup>n</sup>` becomes `^n` and `<sub>i</sub>` becomes `_i`, so `a^n b^n` keeps its
    meaning; `<br>` becomes a space.
  - Only a whitelist of *bare* tags is matched (span/font only with `=` attributes). The
    n-gram markers `<s>`/`</s>`, fastText n-grams such as `<wh` and maths such as
    `x <b and y> c` are content and survive.
  - Side effect: prompt tokens per answer dropped by ~10% on the tokenization deck.
- **Document titles are left out of heading paths.** A deck's title slide had the only
  level-1 heading (`# ... CS224N/Ling284`), with every slide title at level 2. Because
  headings carry over between pages, that title prefixed the path of all 76 slides. The OCR'd
  Studocu banner (`# studocu`) did the same.
  - Rule (`chunker.document_title_line`): in a multi-page document, if the shallowest
    heading occurs exactly once and on the first page, it is the document title. It stays in
    the page text but opens no section.
  - Single-page notes keep their top heading ("Unit 2 > ..."), as do documents with several
    top-level headings (the ToC PDF has 46).
- **Index format version.** `INGEST_FORMAT_VERSION` (indexer.py) is part of the chunking
  settings recorded in the manifest, so an index built by older cleaning/chunking *code* is
  detected and `ingest` asks for `--rebuild`. Config values alone cannot reveal code changes.
  Bump it whenever cleaning or chunking output changes (now 2).
- **Known remaining noise** (no general rule fixes these without heuristics that would break
  other documents):
  - pymupdf4llm sometimes tags a sentence as a heading, and it then roots later pages.
  - OCR marks table lines such as "min(4, 2, 3) = 2" as headings.
  - Decks that mix `##` and `###` slide titles nest a slide under the previous slide's title.
- **Result after `ingest --rebuild`:** 0 tags in text or headings. One figure-only page
  (ToC p.41, OCR text "Figure 6 a a ob q1 b") now falls under `min_page_chars` and is dropped;
  the HTML comment markers had kept it alive. Eval retrieval check unchanged (hit@1 16/23,
  hit@5 23/23).

## P3 - Study tools and UI

- **Scope.** Every tool takes a `Scope`: a topic (the best chunks are retrieved with the
  normal hybrid pipeline) or a document. Document chunks are sampled *evenly* from first to
  last, so a summary covers the whole lecture rather than the first few slides. The scope
  label (the topic, or the document's file name) is passed into the prompt template variable
  each task already uses (`{topic}` or `{scope}`), so no prompt text lives in Python.
- **One shared helper.** `StudyTool._retrieve_and_generate()` gathers chunks, builds the
  numbered context within `retrieval.max_context_tokens`, runs the prompt-file task and
  returns the sources. Quiz, summary and flashcards differ only in their post-checks.
- **Quiz post-checks** (deterministic, after the single LLM call):
  - Each question must have 4 distinct non-empty options and no "all/none of the above".
  - At least one valid source id is required; ungrounded questions are dropped, not shown.
  - Near-duplicates are removed by word-set Jaccard >= `quiz.dedup_similarity` (0.8, new
    config key).
  - The quiz is capped at `quiz.max_n`.
  - Options are shuffled with `random.Random(quiz.seed)`: the answer is not always "A", and
    the same quiz is reproducible.
  - Dropped questions are reported as notes in the UI and CLI.
- **Summary and flashcards** strip source ids that do not exist. Flashcards with empty sides
  or duplicate fronts are dropped. The CSV has `front, back, source` columns (Anki: Import >
  Text file, comma-separated, first row is a header). `source` is "file p.N" joined by "; ".
- **`lecturelens.services.Services`** is the single factory (SPEC 14). Components are
  `cached_property` and built on first use: the app opens and the Library works without an
  API key or any model loaded. The embedder and Chroma store are *shared* by indexing and
  retrieval, so the model loads once and new documents are searchable immediately. After an
  ingest or delete, the in-memory BM25 index is reloaded.
- **Deleting a document** removes it from the index (chunks, manifest, BM25), not from disk.
  Ingesting its folder again re-adds it. That keeps the UI from deleting the user's files.
- **CLI.** `quiz` is interactive by default (`--show-answers` prints the key, `--out` saves
  JSON), `summarize`, and `flashcards --csv`. `--doc` accepts a file name or any unique part
  of it. All model and document text is escaped before Rich prints it: "D[i][0]" in a
  generated flashcard was rendered as "D[0]" because Rich read "[i]" as an italic tag.
- **Streamlit app** (`app/streamlit_app.py`) only lays out widgets and calls `Services`.
  - Uploaded files are saved into `app.data_dir` (git-ignored) and that folder is ingested,
    so the CLI and the UI share one library.
  - `Services` is held with `st.cache_resource`. Chat history, the current quiz and results
    live in `st.session_state`.
  - Every action goes through `run_safely`, which shows `st.error` with the package's
    friendly message (e.g. a missing `GEMINI_API_KEY`) and logs anything unexpected; the UI
    never shows a stack trace (NFR-3).
  - The Ask tab uses the sidebar's answer style, retrieval mode, rerank toggle and document
    filter. Source cards are expanders; cited sources are pinned.
- **App tests** use Streamlit's `AppTest` (headless): an empty library, an answer rendered
  with badge and source cards (stubbed pipeline), a missing API key shown as an error, and
  quiz generate/submit/score. They run offline in a few seconds.

## P4 - Evaluation and polish

- **Relevance is page-level.** A retrieved chunk is relevant when its (file, page) is a gold
  page. Hit@k, Recall@k, nDCG@k (binary) and MRR count each gold page once, even if two chunks
  of the same page are retrieved. Metrics are averaged over the 25 answerable questions.
- **Retrieval ablations make no LLM calls.** The only exception is condensing the two
  follow-up questions once (cached). Without an API key, a follow-up's query falls back to
  "previous question + follow-up", and the report says so.
- **The chunk sweep parses each file once** (`load_pages` + `clean_pages`, including OCR),
  then re-chunks and re-embeds at each size into a temporary Chroma store. It does not run
  `ingest` three times. The configured index is never touched.
- **Minimal-prompt ablation.** A `tutor_minimal` system prompt and an `answer_minimal` task
  share the answer template through YAML anchors, so only the system prompt differs.
  `RAGPipeline(answer_task=...)` selects it. It runs on 10 questions
  (`eval.minimal_prompt_questions`): 5 unanswerable plus the first 5 answerable first-turn
  questions, to protect the daily quota (SPEC 12.3).
- **Generation metrics.**
  - Faithfulness is judged only for questions the system answered (`answerable=true` with
    sources); abstentions have no claims to verify.
  - Citation validity is measured *before* repair: `AskResult` now records
    `removed_citations` and `repaired`.
  - Abstention precision/recall treat "not answerable" as the positive class.
- **The judge uses `llm.judge_model`** (gemini-3.1-flash-lite) at temperature 0, so verdicts
  are cached and reproducible. If its claim and verdict lists differ in length, the overlap
  is used and a warning is logged.
- **Report.** `report.md` and `results.json` are stamped with the date, prompt version,
  models and a full config snapshot. Success and failure examples are selected
  automatically; the failure text states the observable cause (retrieval miss, wrong
  abstention, unsupported claims). The report can be re-rendered from `results.json`
  without new API calls. Chunk sizes within 0.01 MRR at equal Hit@5 count as a tie, and the
  configured size is kept.
- **Reranking is off by default** (`retrieval.rerank: false`, changed after the evaluation at
  the student's request). The SPEC's default was hybrid + rerank. On the course PDFs the
  ms-marco cross-encoder had these effects:
  - It lowered Hit@1 from 0.80 to 0.68, Hit@5 from 1.00 to 0.96 and MRR from 0.873 to 0.790.
  - It added ~1.1 s per query on CPU.
  - It caused the only generation failure: q25's gold slide fell out of the top 5, so the
    system wrongly abstained.
  With it off, the full re-run scored faithfulness 1.00, relevancy 5.00, abstention accuracy
  100% and zero failures. Trade-off: off-syllabus questions always reach the LLM (no
  rerank-score threshold), and the LLM declined all 5. Reranking remains available via config,
  `--rerank` (the CLI flag became `--rerank/--no-rerank`) and the sidebar toggle. Two tests
  that cover rerank-threshold abstention now opt into reranking explicitly.
- **Dependencies.** The direct dependencies in requirements*.txt are pinned to the tested
  versions. Transitive ones are not (torch wheels differ per platform). CI installs CPU-only
  PyTorch first so it does not download CUDA wheels.
- **README screenshots** are not included yet: the in-app browser pane renders the UI too
  small for usable images. The student adds three (Ask with source cards, Quiz after
  submission, Library) under `docs/screenshots/`.
- **Architecture diagram** is a Mermaid flowchart in the README (rendered by GitHub) instead
  of `docs/architecture.png`, so it stays editable and versioned as text.
