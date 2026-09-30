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
