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


# ─────────────────────────────────────────────────────────────────────────────
# task_category + per-category file validation
# ─────────────────────────────────────────────────────────────────────────────


def _write_rubrics_with_category(
    task_dir: Path, category: str | None, rubric_items: list[dict]
) -> None:
    """Helper to write rubrics.jsonl with an optional category header line."""
    lines = []
    if category is not None:
        lines.append(json.dumps({"task_category": category}))
    for r in rubric_items:
        lines.append(json.dumps(r))
    (task_dir / "rubrics.jsonl").write_text("\n".join(lines) + "\n")


@pytest.fixture()
def minimal_task(tmp_path: Path) -> Path:
    """A bare-minimum task (instruction.md + rubrics.jsonl + empty input dir)."""
    td = tmp_path / "task_xyz"
    td.mkdir()
    (td / "instruction.md").write_text("Do the thing.")
    _write_rubrics_with_category(td, None, [
        {"number": 1, "type": "probe_file_exists", "category": "FORMAT",
         "points": 5, "importance": "mandatory", "criterion": "x",
         "paths": ["x.json"]},
    ])
    (td / "data" / "input_files").mkdir(parents=True)
    return td


def test_task_category_declared_via_header(minimal_task: Path):
    """An explicit task_category header on rubrics.jsonl flows through to the instance."""
    _write_rubrics_with_category(minimal_task, "pdf", [
        {"number": 1, "type": "probe_file_exists", "category": "FORMAT",
         "points": 5, "importance": "mandatory", "criterion": "x",
         "paths": ["x.json"]},
    ])
    instance = load_task(minimal_task)
    assert instance.task_category == "pdf"


def test_task_category_inferred_from_input_extensions(minimal_task: Path):
    """No header → loader infers category from file extensions."""
    img = minimal_task / "data" / "input_files" / "photo.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)  # tiny PNG-like
    instance = load_task(minimal_task)
    assert instance.task_category == "image"


def test_task_category_inferred_mixed_when_extensions_disagree(
    minimal_task: Path,
):
    """A task with both an image and a PDF input → category = 'mixed'."""
    (minimal_task / "data" / "input_files" / "photo.png").write_bytes(b"\x89PNG\r\n\x00")
    (minimal_task / "data" / "input_files" / "doc.pdf").write_bytes(b"%PDF-1.4\n%fake")
    instance = load_task(minimal_task)
    assert instance.task_category == "mixed"


def test_declared_category_strictly_rejects_wrong_extension(minimal_task: Path):
    """task_category='pdf' but a video input → loader fails loud."""
    _write_rubrics_with_category(minimal_task, "pdf", [
        {"number": 1, "type": "probe_file_exists", "category": "FORMAT",
         "points": 5, "importance": "mandatory", "criterion": "x",
         "paths": ["x.json"]},
    ])
    (minimal_task / "data" / "input_files" / "clip.mp4").write_bytes(b"fake mp4")
    with pytest.raises(ValueError, match="not allowed for task_category"):
        load_task(minimal_task)


def test_inferred_category_warns_but_does_not_raise(
    minimal_task: Path, caplog
):
    """Legacy task with oversized image but no explicit category → warns only."""
    # Image that's just over the 5 MB cap
    big = minimal_task / "data" / "input_files" / "huge.png"
    big.write_bytes(b"\x89PNG\r\n" + b"\x00" * (5_000_001))
    with caplog.at_level("WARNING"):
        instance = load_task(minimal_task)
    assert instance.task_category == "image"
    assert any("exceeds image cap" in r.message for r in caplog.records)


def test_declared_pdf_oversized_fails_loud(minimal_task: Path):
    """task_category='pdf' with an oversized PDF → ValueError at task load."""
    _write_rubrics_with_category(minimal_task, "pdf", [
        {"number": 1, "type": "probe_file_exists", "category": "FORMAT",
         "points": 5, "importance": "mandatory", "criterion": "x",
         "paths": ["x.json"]},
    ])
    big = minimal_task / "data" / "input_files" / "huge.pdf"
    big.write_bytes(b"%PDF-1.4\n" + b"\x00" * (30_000_001))
    with pytest.raises(ValueError, match="exceeds PDF cap"):
        load_task(minimal_task)


def test_invalid_task_category_value(minimal_task: Path):
    """task_category='bogus' → ValueError at parse time."""
    _write_rubrics_with_category(minimal_task, "bogus", [
        {"number": 1, "type": "probe_file_exists", "category": "FORMAT",
         "points": 5, "importance": "mandatory", "criterion": "x",
         "paths": ["x.json"]},
    ])
    with pytest.raises(ValueError, match="invalid task_category"):
        load_task(minimal_task)
