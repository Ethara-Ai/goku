"""Task discovery and loading for the Goku benchmark."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from benchmarks.goku.media_render import (
    MAX_IMAGE_BYTES,
    MAX_PDF_BYTES,
    MAX_VIDEO_BYTES,
    MAX_VIDEO_DURATION_SEC,
    _probe_duration_seconds,
)
from benchmarks.goku.models import GokuEvalInstance, RubricItem, TaskCategory


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


# Per-category file extension whitelists. A task tagged `pdf` must contain
# only PDFs in its data/input_files/; `image` only image extensions; `video`
# only video extensions. Loader fails loud on category violations.
_CATEGORY_EXTENSIONS = {
    "pdf":   {".pdf"},
    "image": {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"},
    "video": {".mp4", ".mov", ".webm", ".avi", ".mkv"},
}


def _infer_task_category(input_files: list[str]) -> TaskCategory:
    """Auto-detect a task's category from its input file extensions.

    Used as a fallback when a task's rubrics.jsonl has no explicit
    ``task_category`` header line. Returns "mixed" for any mixed set —
    callers should warn or convert to a strict category.
    """
    if not input_files:
        return "mixed"
    suffixes = {Path(f).suffix.lower() for f in input_files}
    for category, allowed in _CATEGORY_EXTENSIONS.items():
        if suffixes.issubset(allowed):
            return category  # type: ignore[return-value]
    return "mixed"


def _validate_input_files_for_category(
    task_key: str,
    category: TaskCategory,
    input_files: list[str],
    *,
    strict: bool,
) -> None:
    """Validate input files against per-category caps.

    Args:
        task_key: Task directory name (used in error messages).
        category: The resolved task category.
        input_files: Absolute paths to files in ``data/input_files/``.
        strict: If True, violations raise ValueError (use for NEW tasks with
            an explicit ``task_category`` header). If False, violations log
            a WARNING only (used for legacy tasks where the category was
            inferred from file extensions) so existing batches don't break.

    The intent is to catch annotator mistakes at task-discovery time — long
    before the harness sends anything to a model — so the operator sees a
    clear error instead of a silent truncation or downstream API rejection.
    """
    if category == "mixed":
        # Legacy/unrestricted bucket; only do size checks, no extension lock.
        allowed_exts = None
    else:
        allowed_exts = _CATEGORY_EXTENSIONS[category]

    errors: list[str] = []
    for f_str in input_files:
        f = Path(f_str)
        suffix = f.suffix.lower()
        size = f.stat().st_size if f.is_file() else 0

        # 1. Extension matches declared category
        if allowed_exts is not None and suffix not in allowed_exts:
            errors.append(
                f"  - {f.name}: extension {suffix!r} not allowed for "
                f"task_category={category!r} (allowed: {sorted(allowed_exts)})"
            )
            continue

        # 2. Per-file size cap
        if suffix == ".pdf" and size > MAX_PDF_BYTES:
            errors.append(
                f"  - {f.name}: {size:,} bytes exceeds PDF cap of "
                f"{MAX_PDF_BYTES:,} bytes ({MAX_PDF_BYTES // 1_000_000} MB)"
            )
        elif suffix in _CATEGORY_EXTENSIONS["image"] and size > MAX_IMAGE_BYTES:
            errors.append(
                f"  - {f.name}: {size:,} bytes exceeds image cap of "
                f"{MAX_IMAGE_BYTES:,} bytes ({MAX_IMAGE_BYTES // 1_000_000} MB)"
            )
        elif suffix in _CATEGORY_EXTENSIONS["video"]:
            if size > MAX_VIDEO_BYTES:
                errors.append(
                    f"  - {f.name}: {size:,} bytes exceeds video cap of "
                    f"{MAX_VIDEO_BYTES:,} bytes "
                    f"({MAX_VIDEO_BYTES // 1_000_000} MB)"
                )
            # 3. Video duration cap (requires ffprobe — soft-skip if absent)
            duration = _probe_duration_seconds(f)
            if duration is not None and duration > MAX_VIDEO_DURATION_SEC:
                errors.append(
                    f"  - {f.name}: duration {duration:.0f}s exceeds video "
                    f"cap of {MAX_VIDEO_DURATION_SEC}s "
                    f"({MAX_VIDEO_DURATION_SEC // 60} min)"
                )

    if not errors:
        return

    joined = "\n".join(errors)
    msg = (
        f"Task {task_key}: input_files violate per-category limits "
        f"(category={category!r}):\n{joined}\n"
        f"Fix the offending file(s) or change the task_category header in "
        f"rubrics.jsonl. See annotator_guide.md for the limits."
    )
    if strict:
        raise ValueError(msg)
    logger.warning(
        "%s\n(category was inferred, not explicitly declared — downgrading "
        "to warning so legacy tasks still load. Add `task_category` header "
        "to rubrics.jsonl to enforce strict limits.)", msg,
    )


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
    # The optional first JSON object of rubrics.jsonl may be a HEADER record
    # (no "number" field) — used to attach task-level metadata like
    # ``task_category``. We pop it from the rubric stream before validating
    # rubric items so it doesn't get parsed as a malformed rubric.
    rubric_items: list[RubricItem] = []
    declared_category: TaskCategory | None = None
    raw_lines = rubrics_path.read_text(encoding="utf-8").splitlines()
    for line_num, line in enumerate(raw_lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Task {task_key}: invalid rubric (JSON parse failed) at "
                f"line {line_num}: {e}"
            ) from e
        # Header record? (no "number" — distinguishes from rubric items)
        if isinstance(data, dict) and "number" not in data:
            if "task_category" in data:
                cat = data["task_category"]
                if cat not in {"pdf", "image", "video", "mixed"}:
                    raise ValueError(
                        f"Task {task_key}: invalid task_category {cat!r} at "
                        f"line {line_num}; must be 'pdf', 'image', 'video', "
                        f"or 'mixed'."
                    )
                declared_category = cat  # type: ignore[assignment]
            continue
        try:
            rubric_items.append(RubricItem(**data))
        except Exception as e:
            raise ValueError(
                f"Task {task_key}: invalid rubric at line {line_num}: {e}"
            ) from e

    input_files: list[str] = []
    input_dir = task_dir / "data" / "input_files"
    if input_dir.exists():
        for f in sorted(input_dir.iterdir()):
            if f.is_file() and not f.name.startswith("."):
                input_files.append(str(f.resolve()))

    # Resolve task_category: explicit header wins; otherwise infer from
    # file extensions. Legacy tasks (no header, no inputs) fall through
    # to "mixed".
    task_category: TaskCategory = declared_category or _infer_task_category(
        input_files
    )

    # Per-category file validation. Strict (raises) when the category was
    # explicitly declared via the rubrics.jsonl header — that's an annotator
    # contract and violations are bugs. Lenient (warns) when the category
    # was inferred from extensions — legacy tasks pre-dating the category
    # split shouldn't fail to load just because the harness now has limits.
    _validate_input_files_for_category(
        task_key,
        task_category,
        input_files,
        strict=(declared_category is not None),
    )

    return GokuEvalInstance(
        id=task_key,
        instruction=instruction,
        rubric_items=rubric_items,
        input_files=input_files,
        task_category=task_category,
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
