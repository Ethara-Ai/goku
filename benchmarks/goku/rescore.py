"""Re-score existing Goku runs against current rubrics — no inference re-run.

Use case: an annotator edits ``dataset/<task>/rubrics.jsonl`` after a batch
has already executed. They want the new rubrics applied to the existing agent
outputs without paying the cost of re-running inference.

Pipeline:
  1. Discover (task, model, run) tuples by walking ``eval_outputs/`` for
     ``scores.jsonl`` files.
  2. For each, load:
       - The current rubric items from ``dataset/<task>/rubrics.jsonl``
       - The agent's previously-saved response and history from
         ``output.jsonl`` (matching by ``instance_id``)
       - The agent's saved output files from the sibling ``results/`` dir
  3. Run scoring (deterministic + LLM judge) against the new rubrics.
  4. Overwrite ``scores.jsonl`` (optionally backing the original up first).
  5. Re-export the delivery folder so packaged scores reflect the rerun.

What this does NOT do:
  - Call the agent LLM again (no agent inference cost).
  - Modify ``output.jsonl`` (agent output is the source of truth — unchanged).
  - Touch ``bash_events/`` (preserved as-is).

It DOES re-invoke the judge LLM for every ``response_criteria`` and
``response_not_criteria`` rubric item, so there is a judge API cost
proportional to (LLM-judged items per task) × (tasks) × (models) × (runs).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

from benchmarks.goku.models import RubricItem, ScorerResult
from benchmarks.goku.scorers.deterministic import (
    DETERMINISTIC_TYPES,
    score_deterministic,
)
from benchmarks.goku.scorers.llm_judge import LLM_JUDGE_TYPES, score_llm_judge
from benchmarks.goku.scoring import compute_task_score, write_scores_jsonl
from benchmarks.goku.task_loader import load_task
from benchmarks.utils.llm_config import load_llm_config


load_dotenv()
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — extract agent context from saved output.jsonl
# ─────────────────────────────────────────────────────────────────────────────

def extract_response_from_history(history: list) -> str:
    """Walk the saved history list (latest-first) for the final agent text.

    Handles both the MessageEvent shape (`llm_message.content[0].text`) and
    the FinishAction shape (`action.message`). Returns "" if nothing matches.
    """
    if not isinstance(history, list):
        return ""
    for event in reversed(history):
        if not isinstance(event, dict):
            continue
        if event.get("source") != "agent":
            continue
        # MessageEvent shape
        llm_message = event.get("llm_message")
        if isinstance(llm_message, dict):
            content = llm_message.get("content")
            if isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict):
                    text = first.get("text")
                    if text:
                        return text
        # FinishAction shape
        action = event.get("action")
        if isinstance(action, dict):
            msg = action.get("message")
            if msg:
                return msg
    return ""


def format_trajectory(history: list) -> str:
    """Best-effort trajectory string for the LLM judge's context window."""
    if not isinstance(history, list):
        return ""
    lines: list[str] = []
    for i, event in enumerate(history):
        if not isinstance(event, dict):
            continue
        kind = event.get("kind") or event.get("__type__", "Event")
        lines.append(f"[{i}] {kind}")
        action = event.get("action")
        if isinstance(action, dict):
            action_type = action.get("__type__") or action.get("type", "Action")
            lines.append(f"    Action: {action_type}")
            if action.get("command"):
                lines.append(f"    Command: {str(action['command'])[:200]}")
            if action.get("message"):
                lines.append(f"    Message: {str(action['message'])[:200]}")
        if len(lines) > 500:
            lines.append("... (truncated)")
            break
    return "\n".join(lines)


def collect_file_contents(results_dir: Path) -> str:
    """Read the agent's saved output files for LLM judge context.

    Mirrors ``GokuEvaluation._collect_file_contents`` but excludes
    ``bash_events/`` (bash trace files are agent debugging, not artifacts).
    """
    if not results_dir.exists():
        return "(no output files)"
    contents: list[str] = []
    for f in sorted(results_dir.rglob("*")):
        if not f.is_file():
            continue
        try:
            rel = f.relative_to(results_dir)
        except ValueError:
            continue
        if rel.parts and rel.parts[0] == "bash_events":
            continue
        if f.stat().st_size > 50_000:
            contents.append(f"--- {f.name} --- (binary, {f.stat().st_size} bytes)")
            continue
        try:
            text = f.read_text(encoding="utf-8")
            contents.append(f"--- {f.name} ---\n{text[:20000]}")
        except UnicodeDecodeError:
            contents.append(f"--- {f.name} --- (binary, {f.stat().st_size} bytes)")
    return "\n\n".join(contents) if contents else "(no output files)"


# ─────────────────────────────────────────────────────────────────────────────
# Core: rescore a single (task, model, run)
# ─────────────────────────────────────────────────────────────────────────────

def rescore_single(
    *,
    task_dir: Path,
    rubric_items: list[RubricItem],
    response_text: str,
    file_contents: str,
    trajectory: str,
    judge_model: str,
    judge_api_key: str | None,
    judge_region: str | None,
    skip_llm_judge: bool,
) -> tuple[list[ScorerResult], list[RubricItem]]:
    """Run all rubric items for one task and return per-item results."""
    results: list[ScorerResult] = []
    output_dir = task_dir / "results"
    for item in rubric_items:
        if item.type in DETERMINISTIC_TYPES:
            results.append(score_deterministic(item, output_dir, response_text))
        elif item.type in LLM_JUDGE_TYPES:
            if skip_llm_judge:
                # Preserve a placeholder so the rubric structure stays intact.
                results.append(ScorerResult(
                    number=item.number,
                    passed=False,
                    judge_rationale="(skipped — --skip-llm-judge)",
                    points_awarded=0,
                ))
            else:
                results.append(score_llm_judge(
                    item=item,
                    response=response_text,
                    file_contents=file_contents,
                    trajectory=trajectory,
                    judge_model=judge_model,
                    judge_api_key=judge_api_key,
                    aws_region_name=judge_region,
                ))
        else:
            raise ValueError(
                f"Unknown rubric type: {item.type} for item #{item.number}"
            )
    return results, rubric_items


# ─────────────────────────────────────────────────────────────────────────────
# Discovery + driver
# ─────────────────────────────────────────────────────────────────────────────

def discover_targets(
    output_base: Path,
    *,
    task_filter: set[str] | None,
    model_filter: list[str] | None,
) -> list[tuple[str, Path]]:
    """Find every per-task scores.jsonl under output_base.

    Returns a list of (task_key, scores_file_path) tuples, sorted by path.
    """
    if not output_base.exists():
        raise FileNotFoundError(f"Output directory not found: {output_base}")
    targets: list[tuple[str, Path]] = []
    for scores_file in sorted(output_base.rglob("scores.jsonl")):
        if "_archive_" in str(scores_file):
            continue
        task_key = scores_file.parent.name
        if not task_key.startswith("task_"):
            continue
        if task_filter is not None and task_key not in task_filter:
            continue
        if model_filter is not None:
            # The model dir name is two levels up: .../<model_dir>/<task>/scores.jsonl
            model_dir_name = scores_file.parent.parent.name
            if not any(m in model_dir_name for m in model_filter):
                continue
        targets.append((task_key, scores_file))
    return targets


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOGLEVEL", "INFO"),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Re-score existing Goku runs against current rubrics. "
            "Does not re-run agent inference; only re-runs scoring "
            "(deterministic + LLM judge) on the saved agent outputs."
        )
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Path to eval_outputs/ (or whatever your --output-dir was).",
    )
    parser.add_argument(
        "--tasks-dir", required=True,
        help="Path to dataset/ — current rubrics are loaded from here.",
    )
    parser.add_argument(
        "--tasks", default=None,
        help="Comma-separated task keys (default: rescore all tasks found).",
    )
    parser.add_argument(
        "--models", default=None,
        help=(
            "Comma-separated substrings to filter which model dirs to "
            "rescore (default: all). Substring match against the model "
            "dir name."
        ),
    )
    parser.add_argument(
        "--judge-llm-config", default=None,
        help=(
            "Path to LLM config JSON for the judge model. Falls back to "
            "GOKU_JUDGE_MODEL / AWS_BEARER_TOKEN_BEDROCK / AWS_REGION_NAME "
            "env vars if omitted."
        ),
    )
    parser.add_argument(
        "--backup", action="store_true",
        help=(
            "Before overwriting scores.jsonl, copy it to "
            "scores.before-rescore.jsonl (skipped if backup already exists)."
        ),
    )
    parser.add_argument(
        "--skip-llm-judge", action="store_true",
        help=(
            "Skip LLM-judged rubric items (response_criteria / "
            "response_not_criteria). Deterministic items still rescored. "
            "Useful for cheap dry-runs that don't incur judge API cost."
        ),
    )
    parser.add_argument(
        "--export-delivery", default=None,
        help=(
            "After rescoring, export the delivery folder structure to this "
            "path (e.g. delivery/). Skipped if omitted."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List what would be rescored, then exit without modifying anything.",
    )
    args = parser.parse_args()

    # ---- 1. Resolve filters ----
    task_filter = (
        {t.strip() for t in args.tasks.split(",") if t.strip()}
        if args.tasks else None
    )
    model_filter = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models else None
    )

    # ---- 2. Resolve judge config ----
    judge_model: str
    judge_api_key: str | None
    judge_region: str | None
    if args.judge_llm_config:
        judge_llm = load_llm_config(args.judge_llm_config)
        judge_model = judge_llm.model
        raw_key = judge_llm.api_key
        if raw_key is None:
            judge_api_key = None
        elif hasattr(raw_key, "get_secret_value"):
            judge_api_key = raw_key.get_secret_value()  # type: ignore[union-attr]
        else:
            judge_api_key = str(raw_key)
        judge_region = getattr(judge_llm, "aws_region_name", None)
    else:
        judge_model = os.getenv(
            "GOKU_JUDGE_MODEL", "bedrock/converse/moonshotai.kimi-k2.5"
        )
        judge_api_key = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
        judge_region = os.getenv("AWS_REGION_NAME")

    # ---- 3. Discover targets ----
    output_base = Path(args.output_dir)
    tasks_dir = Path(args.tasks_dir)
    targets = discover_targets(
        output_base, task_filter=task_filter, model_filter=model_filter
    )
    logger.info(
        "Found %d (task, model, run) tuples to rescore in %s",
        len(targets), output_base,
    )

    if args.dry_run:
        for task_key, scores_file in targets:
            print(f"  would rescore: {task_key}  <-  {scores_file}")
        return

    if not targets:
        logger.warning("No targets found — exiting.")
        return

    # ---- 4. Re-score each target ----
    task_cache: dict[str, object] = {}
    n_ok = 0
    n_skip = 0
    n_fail = 0
    for task_key, scores_file in targets:
        # Load task rubrics (cached per task_key)
        if task_key not in task_cache:
            task_path = tasks_dir / task_key
            if not task_path.is_dir():
                logger.warning(
                    "Task %s not found in %s — skipping", task_key, tasks_dir
                )
                task_cache[task_key] = None
            else:
                try:
                    task_cache[task_key] = load_task(task_path)
                except Exception as e:
                    logger.error("Failed to load task %s: %s", task_key, e)
                    task_cache[task_key] = None
        task = task_cache[task_key]
        if task is None:
            n_skip += 1
            continue

        # Find the model-level output.jsonl that contains this task's record
        model_output_jsonl = scores_file.parent.parent / "output.jsonl"
        if not model_output_jsonl.is_file():
            logger.warning("No output.jsonl beside %s — skipping", scores_file)
            n_skip += 1
            continue

        agent_data = None
        try:
            with open(model_output_jsonl, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if d.get("instance_id") == task_key:
                        agent_data = d
                        break
        except OSError as e:
            logger.error("Could not read %s: %s", model_output_jsonl, e)
            n_fail += 1
            continue

        if agent_data is None:
            logger.warning(
                "No output.jsonl entry for %s in %s — skipping",
                task_key, model_output_jsonl,
            )
            n_skip += 1
            continue

        # Reconstruct judge context from saved data
        history = agent_data.get("history") or []
        response_text = extract_response_from_history(history)
        trajectory = format_trajectory(history)
        file_contents = collect_file_contents(scores_file.parent / "results")

        # Optionally back up the original scores.jsonl
        if args.backup:
            backup = scores_file.with_name("scores.before-rescore.jsonl")
            if not backup.exists():
                shutil.copy2(scores_file, backup)

        # Score
        try:
            results, rubric_items = rescore_single(
                task_dir=scores_file.parent,
                rubric_items=task.rubric_items,  # type: ignore[union-attr]
                response_text=response_text,
                file_contents=file_contents,
                trajectory=trajectory,
                judge_model=judge_model,
                judge_api_key=judge_api_key,
                judge_region=judge_region,
                skip_llm_judge=args.skip_llm_judge,
            )
        except Exception:
            logger.exception("Scoring failed for %s in %s", task_key, scores_file)
            n_fail += 1
            continue

        task_score = compute_task_score(results, rubric_items)
        write_scores_jsonl(task_score, scores_file, rubric_items=rubric_items)
        logger.info(
            "Rescored %s in %s: passed=%s, per_task_score=%.4f, awarded=%d/%d",
            task_key, scores_file.parent.parent.name[:30],
            task_score.passed, task_score.per_task_score,
            task_score.awarded, task_score.max_total,
        )
        n_ok += 1

    logger.info("Rescore complete: %d ok, %d skipped, %d failed", n_ok, n_skip, n_fail)

    # ---- 5. Optionally re-export delivery ----
    if args.export_delivery:
        from datetime import date

        from benchmarks.goku.eval_infer import export_delivery_format

        delivery_root = (
            Path(args.export_delivery)
            / f"MM Agentic Pilot Samples-{date.today().isoformat()}"
        )
        # We don't know which --models slugs the user wants to export; reuse
        # the model_filter (or fall back to discovering all model dir names).
        if model_filter:
            export_models = model_filter
        else:
            export_models = sorted({
                t[1].parent.parent.name for t in targets
            })
        export_delivery_format(
            output_base_dir=output_base,
            tasks_source_dir=tasks_dir,
            delivery_dir=delivery_root,
            model_ids=export_models,
            n_runs=3,  # arbitrary; export walks all run_* anyway
        )
        logger.info("Delivery exported to %s", delivery_root)


if __name__ == "__main__":
    sys.exit(main())
