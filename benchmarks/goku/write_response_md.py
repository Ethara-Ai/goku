"""One-shot CLI: walk a delivery folder and write `results/response.md`
for every existing (task, model, run) directory.

Use this to backfill an existing delivery that was exported before the
response-md feature landed in `export_delivery_format()`, or whenever the
source `output.jsonl` has changed (e.g. after re-inference) and you want
to refresh response.md without re-exporting the whole delivery.

The source `output.jsonl` lives in `eval_outputs/`, NOT in `delivery/`
(delivery deliberately excludes the trajectory per doc Tab 2). You must
point at both roots.

Idempotent — re-running rewrites each response.md deterministically from
the unchanged source trajectory.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from benchmarks.goku.config import get_model_display_name
from benchmarks.goku.response_extractor import (
    extract_final_response_from_jsonl,
    write_response_md,
)


logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )


def discover_run_dirs(delivery_root: Path) -> list[tuple[str, str, str, Path]]:
    """Find every (task, model, run_label, run_dir) under a delivery root.

    Accepts either:
      * a path to `MM Agentic Pilot Samples-<DATE>/tasks/`
      * a path to `MM Agentic Pilot Samples-<DATE>/` (we'll descend into `tasks/`)

    Returns a sorted list of tuples for deterministic processing.
    """
    if (delivery_root / "tasks").is_dir():
        tasks_dir = delivery_root / "tasks"
    elif delivery_root.name == "tasks":
        tasks_dir = delivery_root
    else:
        tasks_dir = delivery_root

    discovered: list[tuple[str, str, str, Path]] = []
    if not tasks_dir.is_dir():
        return discovered

    for task_dir in sorted(tasks_dir.iterdir()):
        if not task_dir.is_dir() or not task_dir.name.startswith("task_"):
            continue
        runs_dir = task_dir / "runs"
        if not runs_dir.is_dir():
            continue
        for model_dir in sorted(runs_dir.iterdir()):
            if not model_dir.is_dir() or model_dir.name.startswith("_"):
                continue
            children = sorted(model_dir.iterdir())
            run_subdirs = [c for c in children if c.is_dir() and c.name.startswith("run_")]
            if run_subdirs:
                for run_dir in run_subdirs:
                    discovered.append((task_dir.name, model_dir.name, run_dir.name, run_dir))
            else:
                discovered.append((task_dir.name, model_dir.name, "", model_dir))
    return discovered


def build_source_index(
    eval_outputs_root: Path,
) -> dict[tuple[str, str, str], Path]:
    """Index every (task_key, delivery_model_name, run_label) → output.jsonl.

    Walks `eval_outputs_root/run_<N>/...**/output.jsonl`, parses each line
    to collect instance_ids, and maps the parent slug back to the delivery
    display name via `get_model_display_name`.
    """
    index: dict[tuple[str, str, str], Path] = {}
    if not eval_outputs_root.is_dir():
        logger.warning("eval_outputs_root not found: %s", eval_outputs_root)
        return index

    for run_dir in sorted(eval_outputs_root.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue
        run_label = run_dir.name  # "run_1" or "run_2"

        for outj in run_dir.rglob("output.jsonl"):
            slug_dir = outj.parent.name
            # Strip the trailing OpenHands suffix to isolate the model slug
            slug = slug_dir.split("_sdk_")[0] if "_sdk_" in slug_dir else slug_dir
            display = get_model_display_name(slug)

            # Parse each line once to harvest every instance_id
            try:
                text = outj.read_text(encoding="utf-8")
            except OSError as exc:
                logger.debug("Skip unreadable %s: %s", outj, exc)
                continue

            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                iid = data.get("instance_id") if isinstance(data, dict) else None
                if iid:
                    index[(iid, display, run_label)] = outj
    return index


def backfill(
    delivery_root: Path,
    eval_outputs_root: Path,
    *,
    tasks_filter: set[str] | None = None,
    models_filter: set[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> tuple[int, int, int]:
    """Walk delivery, write/overwrite response.md for each run.

    Returns:
        (written, skipped_existing, skipped_no_source) tuple.
    """
    targets = discover_run_dirs(delivery_root)
    if tasks_filter:
        targets = [t for t in targets if t[0] in tasks_filter]
    if models_filter:
        targets = [t for t in targets if t[1] in models_filter]

    logger.info("Indexing source trajectories under %s …", eval_outputs_root)
    index = build_source_index(eval_outputs_root)
    logger.info("Indexed %d source (task, model, run) tuples", len(index))

    written = skipped_existing = skipped_no_source = 0

    for task_key, model_name, run_label, run_dir in targets:
        label = f"{task_key}/{model_name}" + (f"/{run_label}" if run_label else "")
        results_dir = run_dir / "results"
        response_md = results_dir / "response.md"

        if response_md.exists() and not force:
            logger.debug("[skip-existing] %s → response.md already present", label)
            skipped_existing += 1
            continue

        source_outj = index.get((task_key, model_name, run_label))
        if source_outj is None:
            logger.warning(
                "[skip-no-source] %s → no matching output.jsonl in %s",
                label, eval_outputs_root,
            )
            skipped_no_source += 1
            continue

        text = extract_final_response_from_jsonl(
            source_outj, instance_id=task_key
        )
        n_chars = len(text)
        if dry_run:
            logger.info(
                "[dry-run] %s → would write response.md (%d chars) from %s",
                label, n_chars, source_outj,
            )
        else:
            write_response_md(text, response_md)
            logger.info(
                "[ok] %s → response.md written (%d chars)", label, n_chars
            )
        written += 1

    return written, skipped_existing, skipped_no_source


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Backfill response.md (model's final natural-language response) "
            "into every runs/<model>/run_N/results/ folder under a delivery "
            "directory, reading source trajectories from eval_outputs/. "
            "Idempotent and safe to re-run."
        ),
    )
    p.add_argument(
        "delivery_root",
        type=Path,
        help=(
            "Path to a delivery folder (e.g. "
            "'delivery/MM Agentic Pilot Samples-2026-05-15/'). "
            "Pointing at the inner tasks/ dir also works."
        ),
    )
    p.add_argument(
        "--eval-outputs-root",
        type=Path,
        required=True,
        help=(
            "Path to the eval_outputs/ directory containing run_<N>/ "
            "subdirectories with the source output.jsonl trajectories."
        ),
    )
    p.add_argument("--tasks", default="",
                   help="Comma-separated task keys to limit to (default: all).")
    p.add_argument("--models", default="",
                   help="Comma-separated model dir names to limit to (default: all).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing response.md files.")
    p.add_argument("--dry-run", action="store_true",
                   help="List what would be written without writing anything.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    _setup_logging(args.verbose)

    if not args.delivery_root.exists():
        logger.error("delivery_root not found: %s", args.delivery_root)
        return 2
    if not args.eval_outputs_root.exists():
        logger.error("eval_outputs_root not found: %s", args.eval_outputs_root)
        return 2

    tasks_filter = (
        {t.strip() for t in args.tasks.split(",") if t.strip()}
        if args.tasks else None
    )
    models_filter = (
        {m.strip() for m in args.models.split(",") if m.strip()}
        if args.models else None
    )

    written, skip_e, skip_n = backfill(
        args.delivery_root,
        args.eval_outputs_root,
        tasks_filter=tasks_filter,
        models_filter=models_filter,
        force=args.force,
        dry_run=args.dry_run,
    )
    logger.info(
        "Done: %d %s, %d skipped (already had response.md, use --force to redo), "
        "%d skipped (no matching output.jsonl in eval_outputs).",
        written, "would-write" if args.dry_run else "written",
        skip_e, skip_n,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
