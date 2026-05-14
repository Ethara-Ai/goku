"""Task discovery and loading for the Goku benchmark."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from benchmarks.goku.models import GokuEvalInstance, RubricItem


logger = logging.getLogger(__name__)

# Patterns that MUST NOT appear in instruction.md (hard errors)
FORBIDDEN_PATH_PATTERNS = [
    r"/workspace/",
    r"/home/",
    r"[A-Z]:\\",
]

# Patterns that SHOULD NOT appear but are tolerated with a warning
# (annotators commonly use results/ as the doc allows naming specific paths)
WARN_PATH_PATTERNS = [
    r"results/",
]


def discover_tasks(
    tasks_dir: Path, strict: bool = True
) -> list[GokuEvalInstance]:
    """Find all task directories and parse instruction.md + rubrics.jsonl.

    Args:
        tasks_dir: Path to the tasks/ directory containing task subdirectories.
        strict: If True (default), any malformed task raises a ValueError
            so the operator notices immediately. If False, broken tasks are
            logged and skipped (legacy behavior — use only for debugging).

    Returns:
        List of GokuEvalInstance objects, sorted by task key.

    Raises:
        FileNotFoundError: If tasks_dir does not exist.
        ValueError: In strict mode, if any task fails to load or validate.
            The error names the offending task directory and includes the
            underlying parse error.
    """
    if not tasks_dir.exists():
        raise FileNotFoundError(f"Tasks directory not found: {tasks_dir}")

    tasks: list[GokuEvalInstance] = []
    failures: list[tuple[str, str]] = []  # (task_dir_name, error_message)
    for task_dir in sorted(tasks_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        if task_dir.name.startswith("."):
            continue
        try:
            instance = load_task(task_dir)
            validate_instruction(instance)
            tasks.append(instance)
        except Exception as exc:
            logger.exception("Failed to load task from %s", task_dir)
            failures.append((task_dir.name, f"{type(exc).__name__}: {exc}"))

    if failures and strict:
        details = "\n".join(f"  - {name}: {err}" for name, err in failures)
        raise ValueError(
            f"discover_tasks: {len(failures)} task(s) failed to load from "
            f"{tasks_dir}:\n{details}\n"
            f"Fix the offending task(s) — or pass strict=False to skip "
            f"broken tasks (NOT recommended for production runs)."
        )

    return tasks


def load_task(task_dir: Path) -> GokuEvalInstance:
    """Load a single task from its directory.

    Expected structure:
        task_dir/
        ├── instruction.md
        ├── rubrics.jsonl
        └── data/input_files/   (optional, contains media)

    Args:
        task_dir: Path to the task directory.

    Returns:
        A GokuEvalInstance with parsed instruction, rubrics, and input file paths.

    Raises:
        FileNotFoundError: If instruction.md or rubrics.jsonl is missing.
        ValueError: If rubrics.jsonl contains invalid JSON.
    """
    task_key = task_dir.name

    instruction_path = task_dir / "instruction.md"
    if not instruction_path.exists():
        raise FileNotFoundError(
            f"Task {task_key}: missing instruction.md at {instruction_path}"
        )
    instruction = instruction_path.read_text(encoding="utf-8").strip()

    rubrics_path = task_dir / "rubrics.jsonl"
    if not rubrics_path.exists():
        raise FileNotFoundError(
            f"Task {task_key}: missing rubrics.jsonl at {rubrics_path}"
        )
    rubric_items: list[RubricItem] = []
    for line_num, line in enumerate(
        rubrics_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            rubric_items.append(RubricItem(**data))
        except (json.JSONDecodeError, Exception) as e:
            raise ValueError(
                f"Task {task_key}: invalid rubric at line {line_num}: {e}"
            ) from e

    input_files: list[str] = []
    input_dir = task_dir / "data" / "input_files"
    if input_dir.exists():
        for f in sorted(input_dir.iterdir()):
            if f.is_file() and not f.name.startswith("."):
                input_files.append(str(f.resolve()))

    return GokuEvalInstance(
        id=task_key,
        instruction=instruction,
        rubric_items=rubric_items,
        input_files=input_files,
    )


def validate_instruction(instance: GokuEvalInstance) -> None:
    """Validate that instruction.md uses bare filenames only.

    Per doc L114, instruction.md must not contain harness-specific paths like
    /workspace/, /home/, or absolute Windows paths.

    Raises:
        ValueError: If hard-forbidden path patterns are found in instruction.md.
    """
    for pattern in FORBIDDEN_PATH_PATTERNS:
        matches = re.findall(pattern, instance.instruction)
        if matches:
            raise ValueError(
                f"Task {instance.id}: instruction.md contains forbidden path "
                f"pattern '{pattern}' — matches: {matches}. "
                f"Per doc L114, instruction.md must use bare filenames only."
            )

    for pattern in WARN_PATH_PATTERNS:
        matches = re.findall(pattern, instance.instruction)
        if matches:
            logger.warning(
                "Task %s: instruction.md contains discouraged path pattern "
                "'%s' — matches: %s. Consider using bare filenames only.",
                instance.id,
                pattern,
                matches,
            )
