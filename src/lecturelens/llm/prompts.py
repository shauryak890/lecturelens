"""PromptRegistry: load prompts/prompts.yaml, validate it, and render templates.

Templates use ``str.format_map`` placeholders (``{name}``); literal braces are doubled. Rendering
uses a strict mapping, so a missing variable raises :class:`PromptError` instead of silently
producing a broken prompt.
"""

import string
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from lecturelens.errors import PromptError
from lecturelens.schemas import get_schema

DEFAULT_PROMPTS_PATH = Path("prompts/prompts.yaml")


class TaskSpec(BaseModel):
    """One entry under ``tasks:`` in the prompt file."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    name: str = ""
    system: str | None
    temperature: float = Field(ge=0, le=2)
    max_output_tokens: int = Field(gt=0)
    schema_name: str | None = Field(alias="schema")  # None: schema supplied by the caller
    template: str
    modes: dict[str, str] = {}

    @property
    def response_model(self) -> type[BaseModel]:
        """The Pydantic class the LLM output must validate against.

        Raises:
            PromptError: If the task declares no schema (e.g. ``repair_json``).
        """
        if self.schema_name is None:
            raise PromptError(f"Task {self.name!r} has no fixed schema; the caller supplies one")
        return get_schema(self.schema_name)


class PromptFile(BaseModel):
    """Top-level structure of prompts.yaml."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(min_length=1)
    changelog: list[str] = []
    formats: dict[str, str] = {}
    messages: dict[str, str] = {}
    system: dict[str, str]
    tasks: dict[str, TaskSpec]

    @model_validator(mode="after")
    def _check_references(self) -> "PromptFile":
        for name, task in self.tasks.items():
            if task.system is not None and task.system not in self.system:
                raise ValueError(f"task {name!r} uses unknown system prompt {task.system!r}")
            try:
                if task.schema_name is not None:
                    get_schema(task.schema_name)
            except PromptError as exc:
                raise ValueError(f"task {name!r}: {exc}") from exc
        return self


class _StrictVars(dict):
    """format_map mapping that raises on missing variables."""

    def __missing__(self, key: str) -> Any:
        raise PromptError(f"Missing prompt variable {{{key}}}")


def _render(text: str, where: str, variables: dict[str, Any]) -> str:
    try:
        return text.format_map(_StrictVars(variables))
    except PromptError as exc:
        raise PromptError(f"{exc} while rendering {where}") from None
    except (ValueError, IndexError, AttributeError) as exc:
        raise PromptError(f"Malformed template in {where}: {exc}") from exc


def placeholders(text: str) -> set[str]:
    """Return the ``{name}`` placeholders used in a template (doubled braces excluded)."""
    return {field for _, field, _, _ in string.Formatter().parse(text) if field}


class PromptRegistry:
    """Loads and renders the versioned prompt file.

    Attributes:
        version: The prompt file version, logged with every LLM call.
        path: Where the prompts were loaded from.
    """

    def __init__(self, path: str | Path = DEFAULT_PROMPTS_PATH) -> None:
        """Load and validate the prompt file.

        Args:
            path: Path to prompts.yaml.

        Raises:
            PromptError: If the file is missing, not valid YAML, or structurally invalid.
        """
        self.path = Path(path)
        self._file = self._load(self.path)
        self.version = self._file.version

    @staticmethod
    def _load(path: Path) -> PromptFile:
        if not path.is_file():
            raise PromptError(f"Prompt file not found: {path}")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise PromptError(f"Invalid YAML in {path}: {exc}") from exc
        if not isinstance(raw, dict) or not isinstance(raw.get("tasks"), dict):
            raise PromptError(f"{path} must define a 'tasks' mapping")
        for name, task in raw["tasks"].items():
            if isinstance(task, dict):
                task["name"] = name
        try:
            return PromptFile.model_validate(raw)
        except ValidationError as exc:
            raise PromptError(f"Invalid prompt file {path}: {exc}") from exc

    @property
    def task_names(self) -> list[str]:
        """Names of all tasks in the prompt file."""
        return list(self._file.tasks)

    @property
    def system_names(self) -> list[str]:
        """Names of all system prompts in the prompt file."""
        return list(self._file.system)

    def task(self, name: str) -> TaskSpec:
        """Return the spec for task ``name``.

        Raises:
            PromptError: If the task does not exist.
        """
        try:
            return self._file.tasks[name]
        except KeyError:
            raise PromptError(
                f"Unknown prompt task {name!r}. Known tasks: {', '.join(self.task_names)}"
            ) from None

    def system(self, name: str, **variables: Any) -> str:
        """Render the system prompt ``name``.

        Raises:
            PromptError: If the system prompt does not exist or a variable is missing.
        """
        if name not in self._file.system:
            raise PromptError(
                f"Unknown system prompt {name!r}. Known: {', '.join(self.system_names)}"
            )
        return _render(self._file.system[name], f"system.{name}", variables)

    def format(self, name: str, **variables: Any) -> str:
        """Render the building block ``formats.<name>`` (e.g. the excerpt label).

        Raises:
            PromptError: If the format does not exist or a variable is missing.
        """
        if name not in self._file.formats:
            raise PromptError(f"Unknown format {name!r}. Known: {', '.join(self._file.formats)}")
        return _render(self._file.formats[name], f"formats.{name}", variables)

    def message(self, name: str) -> str:
        """Return the fixed user-facing message ``messages.<name>``.

        Raises:
            PromptError: If the message does not exist.
        """
        try:
            return self._file.messages[name]
        except KeyError:
            known = ", ".join(self._file.messages)
            raise PromptError(f"Unknown message {name!r}. Known: {known}") from None

    def system_for(self, task: str, **variables: Any) -> str | None:
        """Render the system prompt used by ``task``, or ``None`` if it has none."""
        spec = self.task(task)
        return None if spec.system is None else self.system(spec.system, **variables)

    def required_vars(self, task: str) -> set[str]:
        """Return the variables the template of ``task`` needs."""
        return placeholders(self.task(task).template)

    def mode_instruction(self, task: str, mode: str, **variables: Any) -> str:
        """Render the style instruction for ``mode`` (e.g. the answer task's ``concise``).

        Raises:
            PromptError: If the task has no such mode or a variable is missing.
        """
        spec = self.task(task)
        if mode not in spec.modes:
            known = ", ".join(spec.modes) or "none"
            raise PromptError(f"Task {task!r} has no mode {mode!r}. Known modes: {known}")
        return _render(spec.modes[mode], f"tasks.{task}.modes.{mode}", variables)

    def render(self, task: str, **variables: Any) -> tuple[str, TaskSpec]:
        """Render the user prompt of ``task``.

        Args:
            task: Task name, e.g. ``"answer"``.
            **variables: Values for the template placeholders. Extra values are ignored.

        Returns:
            The rendered user prompt and the task spec (temperature, schema, token limit).

        Raises:
            PromptError: If the task is unknown or a placeholder has no value.
        """
        spec = self.task(task)
        return _render(spec.template, f"tasks.{task}", variables), spec
