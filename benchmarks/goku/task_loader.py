"""Task discovery and loading for the Goku benchmark."""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from pathlib import Path

from benchmarks.goku.media_render import (
    MAX_IMAGE_BYTES,
    MAX_PDF_BYTES,
    MAX_VIDEO_BYTES,
    MAX_VIDEO_DURATION_SEC,
    _probe_duration_seconds,
)
from benchmarks.goku.models import GokuEvalInstance, RubricItem, TaskCategory


# Per DIU Goku doc L116-117: factuality rubrics MUST cite which input_file /
# page / quote supports the criterion. The combination of category + type
# below is what we consider "factuality" for this validation.
FACTUALITY_CATEGORIES = {"CORRECTNESS", "MM_REASONING"}
FACTUALITY_LLM_TYPES = {"response_criteria", "response_not_criteria"}


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


# raw_shell content that ALWAYS exits 0 (rubric trivially passes) or ALWAYS
# exits non-zero (rubric trivially fails). Annotator placeholders typically
# look like these; flag so the rubric is meaningful before scoring runs.
_TRIVIAL_PASS_COMMANDS = frozenset({"true", ":", "/bin/true", "echo ok"})
_TRIVIAL_FAIL_COMMANDS = frozenset({"false", "/bin/false", "exit 1", "exit 2"})

# Patterns inside raw_shell that violate the spec or threaten the harness.
# Hard-forbidden: harness paths (Tab 2 L115), network egress, deletion of
# absolute roots. Soft-warn: non-rooted rm -rf, sudo.
_HARD_FORBIDDEN_RAW_SHELL_PATTERNS = [
    (r"/workspace/", "harness path — use bare filenames (DIU L115)"),
    (r"/home/", "absolute home path — use bare filenames (DIU L115)"),
    (r"\bcurl\b", "network egress — rubrics must score local artifacts only"),
    (r"\bwget\b", "network egress — rubrics must score local artifacts only"),
    (r"\brm\s+-rf?\s+/(?!\w)", "deletion of an absolute root — destructive"),
]
_SOFT_WARN_RAW_SHELL_PATTERNS = [
    (r"\brm\s+-rf?\b", "raw_shell contains `rm -rf` — destructive op in scoring sandbox"),
    (r"\bsudo\b", "raw_shell contains `sudo` — won't work in the eval sandbox"),
]


def _validate_raw_shell_one(
    item: RubricItem, task_key: str,
) -> list[str]:
    """Validate ONE shell_succeeds_real rubric's raw_shell content.

    Returns a list of warning strings (caller logs them). Raises ValueError
    on hard errors (empty command, syntax error, hard-forbidden pattern) —
    those make the rubric unusable and the operator must fix them.
    """
    if item.type != "shell_succeeds_real":
        return []

    cmd = (item.raw_shell or "").strip()
    if not cmd:
        raise ValueError(
            f"Task {task_key}: rubric #{item.number} has "
            f"type=shell_succeeds_real but raw_shell is empty or missing"
        )

    warnings: list[str] = []

    if shutil.which("bash"):
        try:
            result = subprocess.run(
                ["bash", "-n", "-c", cmd],
                capture_output=True, text=True, timeout=10,
            )
        except subprocess.TimeoutExpired:
            warnings.append(
                f"rubric #{item.number} raw_shell `bash -n` timed out (>10s)"
            )
        else:
            if result.returncode != 0:
                raise ValueError(
                    f"Task {task_key}: rubric #{item.number} raw_shell has "
                    f"bash syntax error: "
                    f"{(result.stderr or '').strip()[:300]}"
                )

    for pat, reason in _HARD_FORBIDDEN_RAW_SHELL_PATTERNS:
        if re.search(pat, cmd):
            raise ValueError(
                f"Task {task_key}: rubric #{item.number} raw_shell contains "
                f"forbidden pattern {pat!r} ({reason}). Fix the rubric before "
                f"running this task."
            )

    for pat, reason in _SOFT_WARN_RAW_SHELL_PATTERNS:
        if re.search(pat, cmd):
            warnings.append(f"rubric #{item.number}: {reason}")

    if cmd in _TRIVIAL_PASS_COMMANDS:
        warnings.append(
            f"rubric #{item.number}: raw_shell is {cmd!r} — task will ALWAYS "
            f"pass; rubric has no real assertion. Looks like an annotator "
            f"placeholder."
        )
    elif cmd in _TRIVIAL_FAIL_COMMANDS:
        warnings.append(
            f"rubric #{item.number}: raw_shell is {cmd!r} — task will ALWAYS "
            f"fail. Looks like an annotator placeholder."
        )

    has_python = bool(re.search(r"\bpython3?\b\s+-c|\bpython3?\b\s*<<", cmd))
    if has_python and "assert" not in cmd and "raise" not in cmd and "sys.exit" not in cmd:
        warnings.append(
            f"rubric #{item.number}: raw_shell runs python but contains no "
            f"`assert`, `raise`, or `sys.exit` — the rubric will pass for any "
            f"non-crashing script. Was that intended?"
        )

    return warnings


def validate_raw_shell_rubrics(
    rubric_items: list[RubricItem], task_key: str,
) -> None:
    """Run _validate_raw_shell_one over every rubric in a task, collecting
    warnings and surfacing hard errors. Called from load_task()."""
    for item in rubric_items:
        for w in _validate_raw_shell_one(item, task_key):
            logger.warning("Task %s: %s", task_key, w)


# Phrases at the START of a `response_not_criteria` criterion that signal
# the rubric author may have written a double-negative. For `response_not_
# criteria`, the criterion text describes the HALLUCINATION the judge
# should detect. "The agent does not claim X" works as a criterion (judge
# looks for the absence of X), but only when interpreted as "the indicator
# is: agent did not claim X" — that's the spec example pattern.
#
# Where it breaks: when the negation flips the polarity of an IMAGE/OUTPUT
# property check, e.g. "the generated images do not appear as flat 2D".
# The judge reads it literally, finds the agent DID produce 2D images,
# concludes the literal criterion ("do not appear 2D") is FALSE, returns
# criteria_met=False — which the harness maps to "no hallucination" → no
# penalty. The annotator's INTENT was the opposite.
#
# We can't tell from the criterion text alone which case it is. We just
# warn so the annotator reviews — the spec example pattern stays valid,
# the inverted-polarity case gets surfaced.
_NEGATIVE_CRITERION_DOUBLE_NEG_HEADS = (
    "the generated",       # "the generated images do not appear..."
    "the produced",
    "the output",
    "the resulting",
    "the rendered",
    "the saved",
)
_NEGATION_TOKENS = (" do not ", " does not ", " is not ", " are not ")


def _looks_like_double_negative_response_not_criteria(
    item: RubricItem,
) -> str | None:
    """Heuristic: flag suspicious double-negative criteria on
    response_not_criteria rubrics. Returns a reason string if suspicious,
    else None.

    Sensitivity tuning: we only trip when the criterion subject is an
    OUTPUT artifact ("the generated images", "the produced file") AND
    the criterion contains a negation. Subject = agent action ("the
    agent does not claim X") matches the spec's example pattern and is
    intentionally not flagged.
    """
    if item.type != "response_not_criteria":
        return None
    text = (item.criterion or "").lower()
    starts_with_output_subject = any(
        text.lstrip().startswith(h) for h in _NEGATIVE_CRITERION_DOUBLE_NEG_HEADS
    )
    if not starts_with_output_subject:
        return None
    if not any(tok in text for tok in _NEGATION_TOKENS):
        return None
    return (
        f"rubric #{item.number} is `response_not_criteria` but the criterion "
        f"reads as 'the [output] does NOT [bad thing]' — likely double-negative. "
        f"For response_not_criteria, the criterion text should describe the "
        f"HALLUCINATION to detect (the bad state). Rewrite as 'the [output] "
        f"[is/appears as] [bad thing]' so the judge fires when the bad state "
        f"is present. See annotator_guide.md (response_not_criteria polarity)."
    )


def validate_negative_criteria_polarity(
    rubric_items: list[RubricItem], task_key: str,
) -> None:
    """Warn on response_not_criteria rubrics that LOOK like they have
    inverted polarity (double-negation). Called from load_task().

    Warning-only — we can't auto-fix annotator-authored rubrics. The
    annotator reviews each warning and either confirms intent (e.g., the
    spec's 'agent does not claim X' pattern) or rewrites for clarity.
    """
    for item in rubric_items:
        reason = _looks_like_double_negative_response_not_criteria(item)
        if reason:
            logger.warning("Task %s: %s", task_key, reason)


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
        # Header records MUST carry `"kind": "header"`. Anything else
        # without `"number"` is a malformed rubric item and falls through
        # to RubricItem(...) below, which raises a clear error rather than
        # silently dropping the line.
        if isinstance(data, dict) and data.get("kind") == "header":
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
                f"Task {task_key}: invalid rubric at line {line_num}: {e}. "
                f"Note: header records must carry `\"kind\": \"header\"`."
            ) from e
        last = rubric_items[-1]
        if (
            last.category in FACTUALITY_CATEGORIES
            and last.type in FACTUALITY_LLM_TYPES
            and last.source is None
        ):
            logger.warning(
                "Task %s: rubric #%d is a factuality item "
                "(category=%s, type=%s) but has no `source` field. Per DIU "
                "Goku doc L116-117, factuality rubrics MUST cite which "
                "input_file / page / quote supports the criterion. "
                "Backfill the `source` field before next vendor delivery.",
                task_key, last.number, last.category, last.type,
            )
        # V3 (2026-05-26) policy: rubric points must be POSITIVE. Old
        # hallucination rubrics used negative points (-5) with type
        # `response_not_criteria`; new spec says rewrite as positive
        # `response_criteria` whose criterion asserts what the agent
        # MUST satisfy ("agent only identifies items actually visible,
        # does not fabricate ..."). Existing negative-point rubrics
        # still score correctly (backward compat) but are flagged so
        # authors can migrate.
        if last.points is not None and last.points < 0:
            suggested_type = (
                "response_criteria"
                if last.type == "response_not_criteria" else last.type
            )
            logger.warning(
                "Task %s: rubric #%d has negative points (%d). Per V3 spec "
                "(2026-05-26), all rubrics must use POSITIVE point values. "
                "Convert: type %s → %s, points %d → %d, and rewrite the "
                "criterion as a positive assertion the agent must satisfy. "
                "Scoring still works for back-compat; please migrate.",
                task_key, last.number, last.points,
                last.type, suggested_type, last.points, abs(last.points),
            )

    if not rubric_items:
        raise ValueError(
            f"Task {task_key}: rubrics.jsonl contains no rubric items "
            f"(only header lines or blanks). Per spec, every task must "
            f"have at least one rubric item."
        )

    validate_raw_shell_rubrics(rubric_items, task_key)
    validate_negative_criteria_polarity(rubric_items, task_key)

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


PROBE_TYPES_WITH_PATHS = frozenset({
    "probe_file_exists",
    "probe_file_contains",
    "probe_dir_exists",
})


def validate_instruction(instance: GokuEvalInstance) -> None:
    """Validate that instruction.md uses bare filenames only.

    Per doc L114, instruction.md must not contain harness-specific paths like
    /workspace/, /home/, or absolute Windows paths.

    Additionally, per doc L196-200: a `paths:` check in a rubric is only valid
    when the prompt names that exact filename. We emit a WARNING (not an error)
    when a probe rubric references a filename the instruction doesn't mention,
    so annotators can fix the mismatch before promoting to a hard error.

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

    for item in instance.rubric_items:
        if item.type not in PROBE_TYPES_WITH_PATHS:
            continue
        candidates = list(item.paths or [])
        if item.path:
            candidates.append(item.path)
        for p in candidates:
            bare = Path(p).name
            if bare not in instance.instruction:
                logger.warning(
                    "Task %s: rubric #%d (type=%s) references file %r but the "
                    "instruction doesn't mention that filename. Per DIU Goku "
                    "doc L196-200, paths: checks are only valid when the "
                    "prompt names the file. Use response_criteria instead, or "
                    "add the filename to the prompt.",
                    instance.id, item.number, item.type, p,
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
