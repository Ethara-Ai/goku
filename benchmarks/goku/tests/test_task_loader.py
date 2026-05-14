"""Tests for benchmarks.goku.task_loader."""

import json
from pathlib import Path

import pytest

from benchmarks.goku.task_loader import (
    discover_tasks,
    load_task,
    validate_instruction,
)


@pytest.fixture()
def mock_task_dir(tmp_path: Path) -> Path:
    """Create a valid task directory fixture."""
    task_dir = tmp_path / "task_abc123"
    task_dir.mkdir()

    # instruction.md
    (task_dir / "instruction.md").write_text(
        "Identify the items in the attached pantry photos.\n"
        "Save the result as inventory.json."
    )

    # rubrics.jsonl
    rubrics = [
        {
            "number": 1,
            "type": "probe_file_exists",
            "category": "FORMAT",
            "points": 5,
            "importance": "mandatory",
            "criterion": "Output file exists",
            "paths": ["inventory.json"],
        },
        {
            "number": 2,
            "type": "response_criteria",
            "category": "CORRECTNESS",
            "points": 3,
            "importance": "nice_to_have",
            "criterion": "Correctly identifies at least 10 items",
        },
        {
            "number": 3,
            "type": "response_not_criteria",
            "category": "HALLUCINATION",
            "points": -5,
            "importance": "mandatory",
            "criterion": "Agent fabricates items not visible in photos",
        },
    ]
    lines = [json.dumps(r) for r in rubrics]
    (task_dir / "rubrics.jsonl").write_text("\n".join(lines))

    # data/input_files/
    input_dir = task_dir / "data" / "input_files"
    input_dir.mkdir(parents=True)
    (input_dir / "pantry1.png").write_bytes(b"\x89PNG")
    (input_dir / "pantry2.jpg").write_bytes(b"\xff\xd8")

    return task_dir


def test_load_task_valid(mock_task_dir: Path):
    instance = load_task(mock_task_dir)
    assert instance.id == "task_abc123"
    assert "Identify the items" in instance.instruction
    assert len(instance.rubric_items) == 3
    assert instance.rubric_items[0].type == "probe_file_exists"
    assert instance.rubric_items[2].points == -5
    assert len(instance.input_files) == 2


def test_load_task_missing_instruction(tmp_path: Path):
    task_dir = tmp_path / "task_no_inst"
    task_dir.mkdir()
    (task_dir / "rubrics.jsonl").write_text(
        json.dumps(
            {
                "number": 1,
                "type": "response_criteria",
                "category": "CORRECTNESS",
                "points": 5,
                "importance": "mandatory",
                "criterion": "Test",
            }
        )
    )
    with pytest.raises(FileNotFoundError, match="missing instruction.md"):
        load_task(task_dir)


def test_load_task_missing_rubrics(tmp_path: Path):
    task_dir = tmp_path / "task_no_rubrics"
    task_dir.mkdir()
    (task_dir / "instruction.md").write_text("Do something")
    with pytest.raises(FileNotFoundError, match="missing rubrics.jsonl"):
        load_task(task_dir)


def test_load_task_invalid_rubric_json(tmp_path: Path):
    task_dir = tmp_path / "task_bad_json"
    task_dir.mkdir()
    (task_dir / "instruction.md").write_text("Do something")
    (task_dir / "rubrics.jsonl").write_text("not valid json\n")
    with pytest.raises(ValueError, match="invalid rubric"):
        load_task(task_dir)


def test_load_task_no_input_files(tmp_path: Path):
    task_dir = tmp_path / "task_no_files"
    task_dir.mkdir()
    (task_dir / "instruction.md").write_text("Do something text-only")
    rubric = {
        "number": 1,
        "type": "response_criteria",
        "category": "CORRECTNESS",
        "points": 5,
        "importance": "mandatory",
        "criterion": "Test criterion",
    }
    (task_dir / "rubrics.jsonl").write_text(json.dumps(rubric))
    instance = load_task(task_dir)
    assert instance.input_files == []


def test_discover_tasks(mock_task_dir: Path):
    tasks_dir = mock_task_dir.parent
    instances = discover_tasks(tasks_dir)
    assert len(instances) == 1
    assert instances[0].id == "task_abc123"


def test_discover_tasks_nonexistent(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        discover_tasks(tmp_path / "nonexistent")


def test_discover_tasks_skips_dotfiles(tmp_path: Path):
    # Hidden dir should be skipped
    hidden = tmp_path / ".hidden_task"
    hidden.mkdir()
    (hidden / "instruction.md").write_text("Should be skipped")
    (hidden / "rubrics.jsonl").write_text(
        json.dumps(
            {
                "number": 1,
                "type": "response_criteria",
                "category": "CORRECTNESS",
                "points": 5,
                "importance": "mandatory",
                "criterion": "Test",
            }
        )
    )
    instances = discover_tasks(tmp_path)
    assert len(instances) == 0


def test_validate_instruction_clean(mock_task_dir: Path):
    instance = load_task(mock_task_dir)
    validate_instruction(instance)


def test_validate_instruction_warns_results_prefix(mock_task_dir: Path, caplog):
    (mock_task_dir / "instruction.md").write_text(
        "Save the output to results/inventory.json"
    )
    instance = load_task(mock_task_dir)
    with caplog.at_level("WARNING"):
        validate_instruction(instance)
    assert "discouraged path pattern" in caplog.text


def test_validate_instruction_forbidden_workspace(mock_task_dir: Path):
    (mock_task_dir / "instruction.md").write_text("Save to /workspace/output.json")
    instance = load_task(mock_task_dir)
    with pytest.raises(ValueError, match="forbidden path"):
        validate_instruction(instance)
