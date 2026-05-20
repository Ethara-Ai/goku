"""Deterministic scorers for Goku rubric types.

Implements 6 rubric types that can be evaluated without an LLM:
  - probe_file_exists
  - probe_file_contains
  - probe_dir_exists
  - shell_succeeds_real
  - response_contains
  - response_regex_present
"""

from __future__ import annotations

import logging
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

from benchmarks.goku.models import RubricItem, ScorerResult


logger = logging.getLogger(__name__)

DETERMINISTIC_TYPES = frozenset(
    {
        "probe_file_exists",
        "probe_file_contains",
        "probe_dir_exists",
        "shell_succeeds_real",
        "response_contains",
        "response_regex_present",
    }
)


def score_deterministic(
    item: RubricItem,
    output_dir: Path,
    response: str,
) -> ScorerResult:
    """Score a single deterministic rubric item.

    Args:
        item: The rubric item to evaluate.
        output_dir: Path to the agent's output directory (downloaded files).
        response: The agent's final text response.

    Returns:
        A ScorerResult with pass/fail, rationale, and points awarded.

    Raises:
        ValueError: If item.type is not a deterministic type.
    """
    if item.type not in DETERMINISTIC_TYPES:
        raise ValueError(
            f"Rubric item #{item.number}: type '{item.type}' is not deterministic. "
            f"Expected one of: {sorted(DETERMINISTIC_TYPES)}"
        )

    scorer_fn = _SCORERS[item.type]
    passed, rationale = scorer_fn(item, output_dir, response)

    # Calculate points awarded
    if item.points > 0:
        points_awarded = item.points if passed else 0
    else:
        # Negative items: points deducted only if criterion IS matched
        points_awarded = item.points if passed else 0

    return ScorerResult(
        number=item.number,
        passed=passed,
        judge_rationale=rationale,
        points_awarded=points_awarded,
    )


def _score_probe_file_exists(
    item: RubricItem, output_dir: Path, _response: str
) -> tuple[bool, str]:
    """Check that all files in item.paths exist under output_dir.

    Searches recursively — paths are bare filenames per doc spec, so they
    may be in subdirectories (e.g. avatars/option-1.webp).
    """
    if not item.paths:
        return False, "No paths specified in rubric item"

    missing: list[str] = []
    found: list[str] = []
    for p in item.paths:
        # Try direct path first
        full_path = output_dir / p
        if full_path.exists() and full_path.is_file():
            size = full_path.stat().st_size
            found.append(f"{p} ({size} bytes)")
            continue
        # Search recursively for bare filename
        matches = list(output_dir.rglob(p))
        file_matches = [m for m in matches if m.is_file()]
        if file_matches:
            size = file_matches[0].stat().st_size
            rel = file_matches[0].relative_to(output_dir)
            found.append(f"{rel} ({size} bytes)")
        else:
            missing.append(p)

    if missing:
        return False, f"Missing files: {missing}. Found: {found}"
    return True, f"All files exist: {found}"


def _score_probe_file_contains(
    item: RubricItem, output_dir: Path, _response: str
) -> tuple[bool, str]:
    """Check that a file contains a pattern (regex).

    Accepts path from either ``item.paths[0]`` (preferred, matches doc spec)
    or ``item.path`` (legacy/convenience).
    """
    # Resolve path: prefer paths[0] (doc spec), fall back to path
    file_path: str | None = None
    if item.paths:
        file_path = item.paths[0]
    elif item.path:
        file_path = item.path

    if not file_path:
        return False, "No path specified in rubric item (neither paths nor path)"
    if not item.pattern:
        return False, "No pattern specified in rubric item"

    full_path = output_dir / file_path
    if not full_path.exists():
        matches = list(output_dir.rglob(file_path))
        file_matches = [m for m in matches if m.is_file()]
        if file_matches:
            full_path = file_matches[0]
        else:
            return False, f"File not found: {file_path}"

    try:
        content = full_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False, f"File {file_path} is not valid UTF-8 text"

    flags = re.IGNORECASE if item.ignore_case else 0
    match = re.search(item.pattern, content, flags)
    if match:
        return True, f"Pattern '{item.pattern}' found in {file_path}: '{match.group()}'"
    return False, f"Pattern '{item.pattern}' not found in {file_path}"


def _score_probe_dir_exists(
    item: RubricItem, output_dir: Path, _response: str
) -> tuple[bool, str]:
    """Check that all directories in item.paths exist under output_dir."""
    if not item.paths:
        return False, "No paths specified in rubric item"

    missing: list[str] = []
    found: list[str] = []
    for p in item.paths:
        full_path = output_dir / p
        if full_path.exists() and full_path.is_dir():
            found.append(p)
        else:
            missing.append(p)

    if missing:
        return False, f"Missing directories: {missing}. Found: {found}"
    return True, f"All directories exist: {found}"


# Lines that pollute subprocess stderr because our harness's `benchmarks/`
# package executes `sitecustomize` and Modal-sandbox banners on every Python
# subprocess. We strip them so the rationale shows the actual error.
_SHELL_STDERR_NOISE_MARKERS = (
    "sitecustomize imported",
    "modal sitecustomize",
    "modal-client",
    "run_instance_modal",
    "OpenHands SDK v",
    "injected modal",
    "applied sandbox timing",
    "applied runtime debug",
    "Report a bug:",
    "Get help:",
    "Scale up:",
    "Set OPENHANDS",
)


def _clean_shell_stderr(err: str, budget: int = 400) -> str:
    """Strip harness boot-noise from a subprocess's stderr and return the
    last ``budget`` characters of what remains.

    Strategy (in order):
      1. If a Python ``Traceback (most recent call last)`` appears, take
         everything from that line onward — that's the real error, and it
         contains the AssertionError / KeyError / etc. that the rubric
         actually wanted to surface.
      2. Otherwise, filter out lines matching the known noise markers and
         the OpenHands banner box, then take the last ``budget`` chars.
      3. Return "(empty)" if nothing useful remains.
    """
    if not err.strip():
        return "(empty)"

    lines = err.splitlines()

    # 1. Prefer the traceback if one is present.
    for i, ln in enumerate(lines):
        if "Traceback (most recent call last)" in ln:
            tail = "\n".join(lines[i:]).strip()
            return tail[-budget:] if len(tail) > budget else tail

    # 2. No traceback — filter known noise and take the tail.
    cleaned: list[str] = []
    for ln in lines:
        if any(m in ln for m in _SHELL_STDERR_NOISE_MARKERS):
            continue
        if ln.startswith(("+--", "| ")) or ln.strip() == "|":
            continue
        cleaned.append(ln)
    joined = "\n".join(cleaned).strip()
    if not joined:
        return "(empty)"
    return joined[-budget:] if len(joined) > budget else joined


def _score_shell_succeeds_real(
    item: RubricItem, output_dir: Path, _response: str
) -> tuple[bool, str]:
    """Run a shell command and check it exits with code 0."""
    if not item.raw_shell:
        return False, "No raw_shell command specified in rubric item"

    try:
        result = subprocess.run(
            item.raw_shell,
            shell=True,
            cwd=str(output_dir),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            stdout_preview = result.stdout[:200] if result.stdout else "(empty)"
            return True, f"Shell command exited 0. stdout: {stdout_preview}"
        stderr_preview = _clean_shell_stderr(result.stderr or "", budget=400)
        return (
            False,
            f"Shell command exited {result.returncode}. stderr: {stderr_preview}",
        )
    except subprocess.TimeoutExpired:
        return False, "Shell command timed out after 30 seconds"
    except Exception as e:
        return False, f"Shell command failed: {e}"


def _score_response_contains(
    item: RubricItem, _output_dir: Path, response: str
) -> tuple[bool, str]:
    """Check that all needles appear as substrings in the response."""
    if not item.needles:
        return False, "No needles specified in rubric item"

    response_lower = response.lower()
    missing: list[str] = []
    found: list[str] = []
    for needle in item.needles:
        if needle.lower() in response_lower:
            found.append(needle)
        else:
            missing.append(needle)

    if missing:
        return False, f"Missing needles: {missing}. Found: {found}"
    return True, f"All needles found: {found}"


def _score_response_regex_present(
    item: RubricItem, _output_dir: Path, response: str
) -> tuple[bool, str]:
    """Check that a regex pattern matches somewhere in the response."""
    if not item.pattern:
        return False, "No pattern specified in rubric item"

    match = re.search(item.pattern, response)
    if match:
        return True, f"Regex '{item.pattern}' matched: '{match.group()}'"
    return False, f"Regex '{item.pattern}' not found in response"


# Dispatcher mapping type → scorer function
_SCORERS: dict[
    str,
    Callable[[RubricItem, Path, str], tuple[bool, str]],
] = {
    "probe_file_exists": _score_probe_file_exists,
    "probe_file_contains": _score_probe_file_contains,
    "probe_dir_exists": _score_probe_dir_exists,
    "shell_succeeds_real": _score_shell_succeeds_real,
    "response_contains": _score_response_contains,
    "response_regex_present": _score_response_regex_present,
}
