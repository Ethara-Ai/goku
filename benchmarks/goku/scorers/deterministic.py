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
import multiprocessing as mp
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


def _resolves_within(child: Path, parent: Path) -> bool:
    """True iff ``child`` resolves to a path under ``parent``. Used to keep
    probe checks honest when the agent creates symlinks pointing outside
    the sandbox (e.g. ``output/expected.json -> /etc/passwd``).

    Catches ValueError (not under parent), OSError (resolve failure), and
    RuntimeError (symlink loop on platforms that raise instead of failing
    silently). All paths to "outside / inaccessible" return False.
    """
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError, RuntimeError):
        return False


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
        full_path = output_dir / p
        if (
            full_path.exists()
            and full_path.is_file()
            and not full_path.is_symlink()
            and _resolves_within(full_path, output_dir)
        ):
            size = full_path.stat().st_size
            found.append(f"{p} ({size} bytes)")
            continue
        matches = list(output_dir.rglob(p))
        file_matches = [
            m
            for m in matches
            if m.is_file() and not m.is_symlink() and _resolves_within(m, output_dir)
        ]
        if file_matches:
            size = file_matches[0].stat().st_size
            rel = file_matches[0].relative_to(output_dir)
            found.append(f"{rel} ({size} bytes)")
        else:
            missing.append(p)

    if missing:
        return False, f"Missing files: {missing}. Found: {found}"
    return True, f"All files exist: {found}"


# Cap the matched-span preview the worker hands back through the queue.
# The rationale only ever shows a short snippet, and — critically — a small
# payload can never exceed the OS pipe buffer (~64 KB) that backs a
# multiprocessing.Queue. A full m.group() on a greedy pattern over a large
# file can be megabytes; that overflows the pipe, the child's feeder thread
# blocks flushing it, the child never exits, and proc.join(timeout) below
# then misreports a *valid* match as a timeout (scoring it as a FAIL). Keeping
# the payload tiny removes that deadlock at the source while leaving the
# matched/not-matched signal (payload is None ⇔ no match) unchanged.
_MATCH_PREVIEW_CAP = 200

# Preferred regex backend: the third-party ``regex`` module enforces a wall-
# clock ``timeout`` INSIDE its C matcher, on the calling thread — no
# subprocess, no fork. We fall back to the process-isolation path below only
# when ``regex`` isn't importable.
#
# Why this matters for heavy/concurrent work: the fork-based fallback spawns a
# child per call. In the live harness the scorer runs on a worker thread while
# other threads (litellm, the judge council) are busy allocating and logging.
# A child forked at that instant inherits whatever malloc/glibc and logging
# locks those threads were holding — locked, with no owner thread in the child
# — and can deadlock until the timeout fires, at which point a *valid* match is
# killed and misscored as a FAIL. Measured: 150 forks under malloc+logging
# contention → 21 hung past 7s, ~1200 ms/call average. The same workload via
# ``regex`` → 0 hangs, ~0.01 ms/call. ``regex`` is also far more
# backtracking-resistant, so genuine ReDoS rarely even reaches the timeout.
try:
    import regex as _regex  # noqa: N812  (third-party, drop-in superset of re)
except ImportError:  # pragma: no cover - regex is a transitive dependency
    _regex = None


def _search_with_timeout(
    pattern: str, text: str, flags: int = 0, timeout: float = 30.0
) -> tuple[bool | None, str]:
    """Search ``text`` for ``pattern`` with a hard wall-clock ``timeout``.

    Returns ``(matched, info)`` where ``matched`` is ``True``/``False`` for a
    decided check and ``None`` when the match could not be evaluated (timeout
    or invalid pattern); callers treat ``None`` as a failed — not hung —
    check. ``info`` is a short (≤``_MATCH_PREVIEW_CAP``) preview / reason
    string used only for the human-readable rationale.

    Primary path uses the ``regex`` module's in-thread C-level timeout
    (thread-safe, no process churn). Falls back to a forked ``re.search`` only
    when ``regex`` is unavailable. See the module note above for why forking
    per call is avoided under concurrency.
    """
    if _regex is not None:
        try:
            m = _regex.search(pattern, text, flags, timeout=timeout)
        except _regex.error as exc:
            return None, f"invalid regex: {str(exc)[:_MATCH_PREVIEW_CAP]}"
        except TimeoutError:
            return None, f"regex timed out after {timeout:g}s"
        if m is None:
            return False, ""
        return True, m.group()[:_MATCH_PREVIEW_CAP]
    return _search_with_timeout_subprocess(pattern, text, flags, timeout)


def _regex_search_worker(pattern: str, text: str, flags: int, q) -> None:
    try:
        m = re.search(pattern, text, flags)
        # Truncate the preview, but preserve the None-means-no-match contract.
        q.put(("ok", m.group()[:_MATCH_PREVIEW_CAP] if m else None))
    except re.error as exc:
        q.put(("error", str(exc)[:_MATCH_PREVIEW_CAP]))


def _search_with_timeout_subprocess(
    pattern: str, text: str, flags: int = 0, timeout: float = 30.0
) -> tuple[bool | None, str]:
    """Fallback: run ``re.search`` in a separate process so a pathological
    (ReDoS) or malformed pattern can be hard-killed instead of hanging the
    scorer. Used only when the ``regex`` module is not importable.

    Python's ``re`` runs in C and ignores thread-based timeouts, so the match
    must be executed in a child process that can be terminated. ``fork`` is
    used (not ``spawn``) to avoid re-importing the heavy benchmarks/SDK stack
    on every call; the OS still hard-kills a stuck C-level match on
    ``terminate()`` via SIGTERM. The worker caps the payload it returns
    (``_MATCH_PREVIEW_CAP``) so it can never exceed the queue pipe buffer.
    """
    try:
        ctx = mp.get_context("fork")
    except ValueError:  # platform without fork (e.g. Windows) — fall back
        ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=_regex_search_worker, args=(pattern, text, flags, q))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return None, f"regex timed out after {timeout:g}s"
    try:
        status, payload = q.get_nowait()
    except Exception:
        return None, "regex evaluation produced no result"
    if status == "error":
        return None, f"invalid regex: {payload}"
    return (payload is not None), payload


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
    if not (
        full_path.exists()
        and full_path.is_file()
        and not full_path.is_symlink()
        and _resolves_within(full_path, output_dir)
    ):
        matches = list(output_dir.rglob(file_path))
        file_matches = [
            m
            for m in matches
            if m.is_file() and not m.is_symlink() and _resolves_within(m, output_dir)
        ]
        if file_matches:
            full_path = file_matches[0]
        else:
            return False, f"File not found: {file_path}"

    try:
        content = full_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False, f"File {file_path} is not valid UTF-8 text"

    flags = re.IGNORECASE if item.ignore_case else 0
    matched, info = _search_with_timeout(item.pattern, content, flags, timeout=30)
    if matched is None:
        return False, f"Pattern '{item.pattern}' on {file_path}: {info}"
    if matched:
        return True, f"Pattern '{item.pattern}' found in {file_path}: '{info}'"
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
        if (
            full_path.exists()
            and full_path.is_dir()
            and not full_path.is_symlink()
            and _resolves_within(full_path, output_dir)
        ):
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


# Upper bound on how many candidate subdirectories the FileNotFoundError
# fallback will re-run the shell command in. Each retry runs the (possibly
# 30 s) command once, so an uncapped fan-out over an organize/sort task that
# copied an identically-named file into many subfolders would turn a single
# rubric into an N × 30 s stall. 8 comfortably covers the real case (a file
# saved one level deeper than the rubric expected) while bounding the worst
# case. Candidates are tried shallowest-first so that real case is reached
# before the cap.
_MAX_SUBDIR_FALLBACK_RETRIES = 8


def _score_shell_succeeds_real(
    item: RubricItem, output_dir: Path, _response: str
) -> tuple[bool, str]:
    """Run a shell command and check it exits with code 0.

    Fallback for agent-created subdirectories (2026-05-23 / P0 fix)
    ----------------------------------------------------------------
    Some agents (notably claude-opus 4.7) reliably save their outputs into a
    self-named subdirectory like ``/workspace/results/`` or
    ``/workspace/project/``. Our ``_download_outputs()`` then copies that
    subdir down to ``task/results/<subdir>/`` on the host. Rubrics written
    with bare-path references (``open('foo.json')``) then fail with
    ``FileNotFoundError`` even though the file IS present, just one level
    deeper than the rubric expects.

    To avoid silently penalising the model for an organisational choice,
    when the direct execution fails with a ``FileNotFoundError`` for a
    bare filename, we search ``output_dir`` recursively for that filename
    and retry the same shell command from each candidate parent dir.
    Existing PASSES are unchanged (direct execution wins early). Existing
    fails that aren't FileNotFoundError are unchanged. Only newly-correct
    PASSES emerge — never new fails.
    """
    if not item.raw_shell:
        return False, "No raw_shell command specified in rubric item"
    raw_shell = item.raw_shell

    def _run(cwd: Path):
        return subprocess.run(
            ["bash", "-c", raw_shell],
            shell=False,
            cwd=str(cwd),
            env={"PATH": "/usr/bin:/bin", "LANG": "C"},
            capture_output=True,
            text=True,
            timeout=30,
        )

    try:
        result = _run(output_dir)
    except subprocess.TimeoutExpired:
        return False, "Shell command timed out after 30 seconds"
    except Exception as e:
        return False, f"Shell command failed: {e}"

    if result.returncode == 0:
        stdout_preview = result.stdout[:200] if result.stdout else "(empty)"
        return True, f"Shell command exited 0. stdout: {stdout_preview}"

    # Subdir fallback: only triggers on FileNotFoundError for a bare filename.
    # Restrict to bare names (no path separators) so the rubric author's
    # intent — "the file lives directly under cwd" — is what we're rescuing.
    stderr_full = result.stderr or ""
    m = re.search(r"FileNotFoundError.*?'([^/']+\.[A-Za-z0-9]+)'", stderr_full)
    fallback_attempt = None  # (subdir, retry_result) for diagnostic if all fail
    truncated = False  # True if more rescue candidates existed than the cap
    missing_name = ""  # set below when a FileNotFoundError name is detected
    if m:
        missing_name = m.group(1)
        try:
            candidates = sorted(output_dir.rglob(missing_name))
        except OSError:
            candidates = []
        # Only real files in a *sub*directory are rescue candidates. Try the
        # shallowest first (the intended case is a file saved one level
        # deeper) and cap the number of command re-runs — see
        # _MAX_SUBDIR_FALLBACK_RETRIES. Capping never introduces a new fail
        # versus direct execution; it only bounds how many fail→pass rescues
        # we attempt, and the most likely rescue ranks first.
        eligible = [
            hit for hit in candidates if hit.is_file() and hit.parent != output_dir
        ]
        eligible.sort(key=lambda h: (len(h.parts), str(h)))
        truncated = len(eligible) > _MAX_SUBDIR_FALLBACK_RETRIES
        for hit in eligible[:_MAX_SUBDIR_FALLBACK_RETRIES]:
            # Ensure the subdir is a descendant of output_dir (guard against
            # symlinks pointing outside the task dir).
            try:
                rel_parent = hit.parent.relative_to(output_dir)
            except ValueError:
                continue
            try:
                retry = _run(hit.parent)
            except (subprocess.TimeoutExpired, OSError):
                continue
            if retry.returncode == 0:
                stdout_preview = retry.stdout[:200] if retry.stdout else "(empty)"
                return True, (
                    f"Shell command exited 0 (resolved via subdir '{rel_parent}/' "
                    f"— file '{missing_name}' was saved one level deeper "
                    f"than the rubric expected). stdout: {stdout_preview}"
                )
            # Remember the first non-zero retry so we can surface a clearer
            # diagnostic — the file was found, but the rubric still failed.
            if fallback_attempt is None:
                fallback_attempt = (rel_parent, missing_name, retry)

    # If we capped the candidate fan-out, say so — a rescue could be hiding
    # in an untried subdir, so the operator knows the FAIL may be incomplete.
    cap_note = (
        f" (note: >{_MAX_SUBDIR_FALLBACK_RETRIES} candidate subdirs contained "
        f"'{missing_name if m else ''}'; only the {_MAX_SUBDIR_FALLBACK_RETRIES} "
        f"shallowest were retried — raise the cap if a deeper one is expected to pass)"
        if truncated
        else ""
    )

    if fallback_attempt is not None:
        rel_parent, missing_name, retry = fallback_attempt
        retry_stderr = _clean_shell_stderr(retry.stderr or "", budget=400)
        return (
            False,
            f"Shell command exited {retry.returncode}. File '{missing_name}' "
            f"WAS found in subdir '{rel_parent}/' (rubric expected it at cwd), "
            f"but rubric still failed on its actual check.{cap_note} "
            f"stderr: {retry_stderr}",
        )
    stderr_preview = _clean_shell_stderr(result.stderr or "", budget=400)
    return (
        False,
        f"Shell command exited {result.returncode}.{cap_note} stderr: {stderr_preview}",
    )


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

    matched, info = _search_with_timeout(item.pattern, response, 0, timeout=30)
    if matched is None:
        return False, f"Regex '{item.pattern}': {info}"
    if matched:
        return True, f"Regex '{item.pattern}' matched: '{info}'"
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
