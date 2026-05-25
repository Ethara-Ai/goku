"""Strip a task's entries from harness resume-state files so it can be re-inferred.

Background:
  The OpenHands evaluation harness uses TWO files to decide whether an instance
  has "already completed" and should be skipped on re-run:

    1. <model_dir>/output.jsonl
       - one line per (instance, attempt) tuple with the full record
       - read by _get_completed_instances() in benchmarks/utils/evaluation.py

    2. <model_dir>/output.critic_attempt_<N>.jsonl
       - tracks which instances are completed within a given critic attempt
       - read by _get_instances_for_attempt() in the same file

  If either file still contains the target instance_id, the harness silently
  filters it out at discovery time and prints "No instances to process". The
  outer wrapper sees no scores, retries, and exits in seconds. This bit us hard
  during the task_ff5c9742 re-inference — both files needed to be stripped to
  unblock re-inference.

  This script strips a given task_key from BOTH file types across every
  matching model directory under eval_outputs/. Conversations tarballs and
  per-task subdirectories are also archived so a fresh download has clean
  ground state.

Usage:
  python clean_resume_state.py \\
      --output-base eval_outputs \\
      --tasks task_ff5c9742c645e2cf,task_abc123... \\
      [--models claude-opus-4.7,gpt-5.5,gemini-3.1] \\
      [--dry-run]

Exit codes:
  0  success (entries stripped or already clean)
  1  partial — some files couldn't be processed (logs the details)
  2  bad arguments / missing paths
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path


logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )


def strip_task_from_jsonl(
    jsonl_path: Path, task_keys: set[str], *, dry_run: bool, archive_suffix: str
) -> tuple[int, int]:
    """Remove any line whose `instance_id` matches any task_key in the set.

    Returns:
        (lines_kept, lines_removed)
    """
    if not jsonl_path.is_file():
        return (0, 0)

    keep: list[str] = []
    removed = 0
    for raw_line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip("\n")
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            # Malformed line — preserve it; cleanup is conservative
            keep.append(line)
            continue
        if isinstance(data, dict) and data.get("instance_id") in task_keys:
            removed += 1
            continue
        keep.append(line)

    if removed == 0:
        return (len(keep), 0)

    if dry_run:
        logger.info(
            "[dry-run] %s — would remove %d entries, keep %d",
            jsonl_path, removed, len(keep),
        )
        return (len(keep), removed)

    # Archive original before overwriting (idempotent — skip if archive exists).
    # Captures the state at this exact rerun moment; useful for forensic
    # comparison but lossy across multiple rerun cycles.
    archive = jsonl_path.with_suffix(jsonl_path.suffix + archive_suffix)
    if not archive.exists():
        shutil.copy2(jsonl_path, archive)

    # Cumulative ledger (2026-05-23 / P1 fix). Across many `--rerun` cycles
    # individual archive snapshots become sparse — each only captures the
    # state at one moment. If task A's entry is stripped in cycle 1, then
    # cycle 2's archive sees an output.jsonl that no longer mentions A.
    # Re-scoring task A months later then needs to find SOME archive that
    # was taken between A's most recent inference and its first strip.
    #
    # The cumulative ledger ``output.jsonl.ever_seen`` accumulates EVERY
    # entry ever observed (de-duped by instance_id + attempt) and is never
    # stripped. It's the authoritative recovery source for ``rescore.py``
    # when the live ``output.jsonl`` has had entries removed.
    ever_seen_path = jsonl_path.with_suffix(jsonl_path.suffix + ".ever_seen")
    _append_to_ever_seen(jsonl_path, ever_seen_path)

    new_content = "\n".join(keep) + ("\n" if keep else "")
    jsonl_path.write_text(new_content, encoding="utf-8")
    logger.info(
        "stripped %d entry(ies) from %s (kept %d, archive: %s)",
        removed, jsonl_path, len(keep), archive.name,
    )
    return (len(keep), removed)


def _append_to_ever_seen(live_path: Path, ever_seen_path: Path) -> int:
    """Merge any entries from ``live_path`` not already in ``ever_seen_path``
    into ``ever_seen_path``. De-dupes by ``(instance_id, attempt)``.

    Idempotent: running twice has no effect after the first call.
    Returns the number of newly-added entries.
    """
    # Build the set of (instance_id, attempt) already in ever_seen.
    seen: set[tuple[str, int]] = set()
    if ever_seen_path.is_file():
        for line in ever_seen_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(d, dict):
                key = (str(d.get("instance_id", "")), int(d.get("attempt", 0) or 0))
                seen.add(key)

    # Walk live file and append novel entries.
    new_lines: list[str] = []
    if live_path.is_file():
        for raw_line in live_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                # Preserve malformed lines too — append unconditionally if
                # they don't match anything we've seen exactly.
                if line not in {x.strip() for x in (ever_seen_path.read_text().splitlines() if ever_seen_path.is_file() else [])}:
                    new_lines.append(line)
                continue
            if not isinstance(d, dict):
                continue
            key = (str(d.get("instance_id", "")), int(d.get("attempt", 0) or 0))
            if key in seen:
                continue
            seen.add(key)
            new_lines.append(line)

    if new_lines:
        with ever_seen_path.open("a", encoding="utf-8") as f:
            for ln in new_lines:
                f.write(ln + "\n")
        logger.info(
            "ever_seen ledger: appended %d new entry(ies) to %s",
            len(new_lines), ever_seen_path.name,
        )
    return len(new_lines)


def archive_task_artifacts(
    model_dir: Path, task_keys: set[str], *, dry_run: bool, archive_suffix: str
) -> int:
    """Archive per-task subdirectory + conversations tarball if present.

    Returns:
        number of artifacts archived
    """
    archived = 0
    for task_key in task_keys:
        # Per-task subdir (contains scores.jsonl, results/, logs/, etc.)
        task_subdir = model_dir / task_key
        if task_subdir.exists():
            target = task_subdir.with_name(task_subdir.name + archive_suffix)
            if not target.exists():
                if dry_run:
                    logger.info("[dry-run] would archive %s", task_subdir)
                else:
                    shutil.move(str(task_subdir), str(target))
                    logger.info("archived %s → %s", task_subdir.name, target.name)
                archived += 1

        # Conversations tarball
        conv_tar = model_dir / "conversations" / f"{task_key}.tar.gz"
        if conv_tar.is_file():
            target = conv_tar.with_name(conv_tar.name + archive_suffix)
            if not target.exists():
                if dry_run:
                    logger.info("[dry-run] would archive %s", conv_tar)
                else:
                    shutil.move(str(conv_tar), str(target))
                    logger.info("archived %s → %s", conv_tar.name, target.name)
                archived += 1
    return archived


def find_model_dirs(
    output_base: Path, model_filters: list[str] | None
) -> list[Path]:
    """Find every `<base>/run_*/run_<N>/goku/<model_slug>_sdk_..._maxiter_*/` dir.

    Filters by `model_filters` substrings (case-insensitive) if provided.
    """
    dirs: list[Path] = []
    for run_outer in sorted(output_base.glob("run_*")):
        if not run_outer.is_dir():
            continue
        for goku_dir in run_outer.rglob("goku"):
            if not goku_dir.is_dir():
                continue
            for model_dir in sorted(goku_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                if "_sdk_" not in model_dir.name:
                    continue
                if model_filters:
                    name_l = model_dir.name.lower()
                    if not any(f.lower() in name_l for f in model_filters):
                        continue
                dirs.append(model_dir)
    return dirs


def clean(
    output_base: Path,
    task_keys: set[str],
    *,
    model_filters: list[str] | None = None,
    dry_run: bool = False,
) -> tuple[int, int, int]:
    """Strip task_keys from all matching model directories under output_base.

    Returns:
        (model_dirs_processed, jsonl_lines_removed, artifacts_archived)
    """
    if not output_base.is_dir():
        raise FileNotFoundError(f"output_base not found: {output_base}")

    archive_suffix = f".archive_pre_rerun_{time.strftime('%Y%m%d_%H%M%S')}"
    model_dirs = find_model_dirs(output_base, model_filters)
    logger.info(
        "Found %d model dir(s) under %s%s",
        len(model_dirs), output_base,
        f" (filter: {model_filters})" if model_filters else "",
    )

    total_removed = 0
    total_archived = 0
    for model_dir in model_dirs:
        for jsonl_name in ("output.jsonl",):
            _, removed = strip_task_from_jsonl(
                model_dir / jsonl_name, task_keys,
                dry_run=dry_run, archive_suffix=archive_suffix,
            )
            total_removed += removed
        # Critic attempts can be multiple files (attempt_1, attempt_2, ...)
        for critic_file in model_dir.glob("output.critic_attempt_*.jsonl"):
            _, removed = strip_task_from_jsonl(
                critic_file, task_keys,
                dry_run=dry_run, archive_suffix=archive_suffix,
            )
            total_removed += removed

        total_archived += archive_task_artifacts(
            model_dir, task_keys,
            dry_run=dry_run, archive_suffix=archive_suffix,
        )

    return (len(model_dirs), total_removed, total_archived)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Clean harness resume-state for one or more tasks so they can be "
            "re-inferred. Strips entries from output.jsonl + "
            "output.critic_attempt_*.jsonl across every matching model dir, "
            "and archives per-task subdirs and conversations tarballs."
        ),
    )
    p.add_argument(
        "--output-base",
        type=Path,
        required=True,
        help="Path to eval_outputs/ root (contains run_1/, run_2/, ...).",
    )
    p.add_argument(
        "--tasks",
        type=str,
        required=True,
        help="Comma-separated task keys to clean (e.g. task_ff5c9742c645e2cf).",
    )
    p.add_argument(
        "--models",
        type=str,
        default="",
        help=(
            "Comma-separated model-dir substrings to limit to "
            "(e.g. claude-opus-4.7,gpt-5.5). Default: all model dirs."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be cleaned without modifying anything.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    _setup_logging(args.verbose)

    task_keys = {t.strip() for t in args.tasks.split(",") if t.strip()}
    if not task_keys:
        logger.error("No task keys provided.")
        return 2

    model_filters = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models else None
    )

    if not args.output_base.is_dir():
        logger.error("output_base not found: %s", args.output_base)
        return 2

    try:
        n_dirs, n_removed, n_archived = clean(
            args.output_base, task_keys,
            model_filters=model_filters, dry_run=args.dry_run,
        )
    except Exception as exc:
        logger.exception("Cleanup failed: %s", exc)
        return 1

    verb = "would clean" if args.dry_run else "cleaned"
    logger.info(
        "Done: %s %d model dir(s); %d JSONL entries removed; %d artifact(s) archived.",
        verb, n_dirs, n_removed, n_archived,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
