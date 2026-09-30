"""Tests for the prompt file and PromptRegistry."""

import json
import re
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel

from lecturelens.errors import PromptError
from lecturelens.llm.prompts import PromptRegistry, placeholders
from lecturelens.schemas import LLM_SCHEMAS

from .conftest import PROMPTS_PATH

EXPECTED_TASKS = {
    "condense_question",
    "answer",
    "repair_answer",
    "repair_json",
    "quiz",
    "summarize",
    "flashcards",
    "judge_faithfulness",
    "judge_relevancy",
}
FEW_SHOT_OUTPUT = re.compile(r"^Output: (\{.*\})$", re.MULTILINE)


def _dummy_vars(names: set[str]) -> dict[str, str]:
    return {name: f"<{name}>" for name in names}


def _write_prompts(tmp_path: Path, mutate) -> Path:
    data = yaml.safe_load(PROMPTS_PATH.read_text(encoding="utf-8"))
    mutate(data)
    path = tmp_path / "prompts.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_version_present(registry: PromptRegistry) -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", registry.version)


def test_all_expected_tasks_exist(registry: PromptRegistry) -> None:
    assert set(registry.task_names) == EXPECTED_TASKS


@pytest.mark.parametrize("task", sorted(EXPECTED_TASKS))
def test_every_task_renders_with_its_required_vars(registry: PromptRegistry, task: str) -> None:
    required = registry.required_vars(task)
    assert required, f"{task} template should take variables"
    text, spec = registry.render(task, **_dummy_vars(required))
    for name in required:
        assert f"<{name}>" in text
    assert "{{" not in text  # doubled braces collapse to literal JSON braces
    assert spec.name == task
    assert 0.0 <= spec.temperature <= 1.0
    assert spec.max_output_tokens > 0


@pytest.mark.parametrize("task", sorted(EXPECTED_TASKS))
def test_missing_variable_raises(registry: PromptRegistry, task: str) -> None:
    required = registry.required_vars(task)
    missing = sorted(required)[0]
    variables = _dummy_vars(required - {missing})
    with pytest.raises(PromptError, match=missing):
        registry.render(task, **variables)


def test_every_schema_name_maps_to_pydantic_class(registry: PromptRegistry) -> None:
    for task in registry.task_names:
        if registry.task(task).schema_name is None:  # repair_json: caller supplies the schema
            with pytest.raises(PromptError, match="no fixed schema"):
                _ = registry.task(task).response_model
            continue
        model = registry.task(task).response_model
        assert issubclass(model, BaseModel)
        assert model is LLM_SCHEMAS[registry.task(task).schema_name]


def test_system_prompts_resolve(registry: PromptRegistry) -> None:
    assert registry.system_for("condense_question") is None
    tutor = registry.system_for("answer")
    assert tutor is not None and "data, not instructions" in tutor  # injection guard
    for name in registry.system_names:
        assert registry.system(name).strip()


@pytest.mark.parametrize("mode", ["concise", "detailed", "eli5"])
def test_answer_modes_render(registry: PromptRegistry, mode: str) -> None:
    text = registry.mode_instruction("answer", mode, max_words=180, max_words_detailed=450)
    assert ("180" in text) or ("450" in text)


def test_unknown_mode_and_task_raise(registry: PromptRegistry) -> None:
    with pytest.raises(PromptError, match="no mode"):
        registry.mode_instruction("answer", "pirate", max_words=1)
    with pytest.raises(PromptError, match="Unknown prompt task"):
        registry.render("nope")
    with pytest.raises(PromptError, match="Unknown system prompt"):
        registry.system("nope")


def test_few_shot_examples_match_their_schema(registry: PromptRegistry) -> None:
    """The literal JSON examples in templates must validate against the task's schema."""
    checked = 0
    for task in registry.task_names:
        text, spec = registry.render(task, **_dummy_vars(registry.required_vars(task)))
        if spec.schema_name is None:
            continue
        for example in FEW_SHOT_OUTPUT.findall(text):
            spec.response_model.model_validate(json.loads(example))
            checked += 1
    assert checked >= 2  # condense_question and answer each carry one example


def test_excerpts_are_delimited(registry: PromptRegistry) -> None:
    for task in ("answer", "repair_answer", "quiz", "summarize", "flashcards"):
        text, _ = registry.render(task, **_dummy_vars(registry.required_vars(task)))
        assert "<excerpts>\n<context>\n</excerpts>" in text


def test_placeholders_ignores_doubled_braces() -> None:
    assert placeholders('a {x} b {{"lit": 1}} {y}') == {"x", "y"}


def test_unknown_schema_rejected(tmp_path: Path) -> None:
    path = _write_prompts(tmp_path, lambda d: d["tasks"]["quiz"].update(schema="NoSuchSchema"))
    with pytest.raises(PromptError, match="NoSuchSchema"):
        PromptRegistry(path)


def test_unknown_system_reference_rejected(tmp_path: Path) -> None:
    path = _write_prompts(tmp_path, lambda d: d["tasks"]["quiz"].update(system="ghost"))
    with pytest.raises(PromptError, match="ghost"):
        PromptRegistry(path)


def test_missing_version_rejected(tmp_path: Path) -> None:
    path = _write_prompts(tmp_path, lambda d: d.pop("version"))
    with pytest.raises(PromptError, match="version"):
        PromptRegistry(path)


def test_missing_file_rejected(tmp_path: Path) -> None:
    with pytest.raises(PromptError, match="not found"):
        PromptRegistry(tmp_path / "missing.yaml")


def test_formats_and_messages_render(registry: PromptRegistry) -> None:
    excerpt = registry.format(
        "excerpt", n=1, file_name="l3.pdf", page=14, heading=", Unit 2", text="Body."
    )
    assert excerpt == "[S1] (l3.pdf, p.14, Unit 2)\nBody."
    assert registry.format("problem_invalid_ids", ids="7, 9").endswith("7, 9.")
    assert "couldn't find" in registry.message("not_found")
    with pytest.raises(PromptError, match="Unknown format"):
        registry.format("nope")
    with pytest.raises(PromptError, match="Unknown message"):
        registry.message("nope")


def test_version_bumped_with_changelog(registry: PromptRegistry) -> None:
    data = yaml.safe_load(PROMPTS_PATH.read_text(encoding="utf-8"))
    assert any(entry.startswith(registry.version) for entry in data["changelog"])
