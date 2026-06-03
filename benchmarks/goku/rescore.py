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
from benchmarks.goku.scorers.llm_judge import (
    LLM_JUDGE_TYPES,
    score_llm_judge,
    score_llm_judge_council,
)
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


def collect_file_contents(results_dir: Path) -> tuple[str, list[str]]:
    """Re-judge path: like run_infer's collector but excludes ``bash_events/``
    since those are debugging traces, not agent artifacts."""
    from benchmarks.goku.judge_context import collect_file_contents as _impl
    return _impl(results_dir, exclude_top_dirs={"bash_events"})


def update_output_jsonl_test_result(
    output_jsonl: Path,
    instance_id: str,
    task_score,  # TaskScore — typed weakly to avoid a circular import at module top
) -> bool:
    """Rewrite the ``test_result`` aggregates for ``instance_id`` in ``output_jsonl``.

    Why this exists
    ---------------
    Pre-fix, rescore.py only rewrote ``scores.jsonl`` — the per-item file
    consumed by the delivery export. The benchmark report (``eval_infer.
    load_scores_from_runs``) instead reads aggregates from ``output.jsonl``'s
    ``test_result`` block. Without this update, ``goku-eval`` after a rescore
    silently reports the OLD pre-rescore numbers and the corrected scores
    never reach ``mean_per_task_score`` / ``pass_rate`` / the JSON report.

    Behavior
    --------
      * Locates the line whose ``instance_id`` matches and overwrites only
        the aggregate fields (``awarded``, ``max_total``, ``raw_score``,
        ``per_task_score``, ``passed``, ``judge_cost_usd``). All other
        fields on the line (history, metrics, instance data, instruction,
        custom ``test_result`` keys) are preserved.
      * Writes atomically via tempfile + ``os.replace`` so a crash mid-write
        can't leave a partial file. The rest of the file is unchanged.
      * No-op when the file is missing, the line isn't found, or the value
        round-trip would be identical.

    Returns ``True`` if a matching line was rewritten, else ``False``.
    """
    if not output_jsonl.is_file():
        return False
    try:
        lines = output_jsonl.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False

    updated = False
    new_lines: list[str] = []
    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip():
            new_lines.append(line)
            continue
        try:
            d = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            new_lines.append(line)
            continue
        if not isinstance(d, dict) or d.get("instance_id") != instance_id:
            new_lines.append(line)
            continue
        tr = d.get("test_result")
        if not isinstance(tr, dict):
            tr = {}
        # Preserve unrelated test_result keys; only overwrite the aggregates
        # rescore.py freshly computes. Round to match write_scores_jsonl.
        tr.update({
            "awarded": task_score.awarded,
            "max_total": task_score.max_total,
            "raw_score": round(task_score.raw_score, 4),
            "per_task_score": round(task_score.per_task_score, 4),
            "passed": task_score.passed,
            "judge_cost_usd": round(task_score.judge_cost_usd, 6),
        })
        d["test_result"] = tr
        new_lines.append(json.dumps(d, ensure_ascii=False))
        updated = True

    if not updated:
        return False

    # Atomic write: temp file in same dir → os.replace. Same-filesystem
    # rename is atomic on POSIX, so concurrent readers see either the old
    # file or the new one, never a partial.
    import tempfile
    fd, tmp_path = tempfile.mkstemp(
        dir=str(output_jsonl.parent),
        prefix=f".{output_jsonl.name}.tmp.",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tf:
            tf.write("\n".join(new_lines))
            if new_lines:
                tf.write("\n")
        os.replace(tmp_path, str(output_jsonl))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return True


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
    judge_council_models: list[str] | None = None,
    judge_council_api_keys: list[str | None] | None = None,
    judge_council_regions: list[str | None] | None = None,
    input_image_paths: list[str] | None = None,
    output_media_paths: list[str] | None = None,
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
            elif judge_council_models:
                # Council mode — multi-judge majority vote.
                results.append(score_llm_judge_council(
                    item=item,
                    response=response_text,
                    file_contents=file_contents,
                    trajectory=trajectory,
                    judge_models=judge_council_models,
                    judge_api_keys=judge_council_api_keys,
                    aws_region_names=judge_council_regions,
                    input_image_paths=input_image_paths or [],
                    output_media_paths=output_media_paths or [],
                    task_key=task_dir.name,
                    # Defense against confabulation/inconsistency at the
                    # individual-judge level. Conditional retry — only
                    # fires when the judge's response trips suspicion
                    # checks (cited fake filename or rationale/boolean
                    # mismatch). Clean responses pay zero extra cost.
                    enable_per_judge_voting=True,
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
                    input_image_paths=input_image_paths or [],
                    output_media_paths=output_media_paths or [],
                    # task_dir.name is the task_xxx hash (e.g. task_abc...).
                    # Used by the judge's many-image S3-URL short-circuit
                    # so all rubric calls reuse one upload via cache.
                    task_key=task_dir.name,
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
    # Use the shared archive-detection helper from eval_infer so all
    # discovery sites in the harness use the same predicate. Previous
    # `_archive_` substring missed the actual `.archive_pre_rerun_`
    # naming used by clean_resume_state.py.
    from benchmarks.goku.eval_infer import _is_archive_path
    for scores_file in sorted(output_base.rglob("scores.jsonl")):
        if _is_archive_path(scores_file):
            continue
        task_key = scores_file.parent.name
        if not task_key.startswith("task_"):
            continue
        if task_filter is not None and task_key not in task_filter:
            continue
        if model_filter is not None:
            model_dir_name = scores_file.parent.parent.name
            if model_dir_name not in model_filter:
                continue
        targets.append((task_key, scores_file))
    return targets


def main() -> None:
    # Eagerly apply httpx_patches only. The judge talks to providers
    # directly (LiteLLM), not via the docker agent_server, so the SDK
    # patch IS theoretically safe here for judge calls. But to keep the
    # behavior simple + consistent with run_infer (avoid the container
    # serialization landmine), we leave native PDF on the judge to the
    # `_build_media_blocks` direct-block construction in llm_judge.py —
    # which already builds the correct per-provider PDF block shape
    # without going through the SDK DocumentContent class.
    from benchmarks.utils import httpx_patches
    httpx_patches.apply()

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
            "Comma-separated EXACT model dir names to rescore "
            "(default: all). Use the full slug (e.g. "
            "'claude-opus-4.7_sdk_v1', not 'opus')."
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
        "--judge-llm-configs", default=None,
        help=(
            "Comma-separated paths to LLM config JSONs for a judge COUNCIL "
            "(e.g., .llm_config/claude-sonnet-4.6.json,.llm_config/gpt-5.json,"
            ".llm_config/gemini-3.5-flash.json). When ≥2 configs are given, "
            "each LLM-judged rubric is re-scored by all judges in parallel "
            "and the verdict is majority-vote. Mutually exclusive with "
            "--judge-llm-config."
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
    if args.judge_llm_config and args.judge_llm_configs:
        parser.error(
            "Specify either --judge-llm-config (single) OR "
            "--judge-llm-configs (council), not both."
        )

    judge_model: str
    judge_api_key: str | None
    judge_region: str | None
    judge_council_models: list[str] | None = None
    judge_council_api_keys: list[str | None] | None = None
    judge_council_regions: list[str | None] | None = None

    def _resolve_key(llm) -> str | None:
        k = getattr(llm, "api_key", None)
        if k is None:
            return None
        if hasattr(k, "get_secret_value"):
            return k.get_secret_value()
        return str(k)

    if args.judge_llm_configs:
        # Council mode
        paths = [p.strip() for p in args.judge_llm_configs.split(",") if p.strip()]
        if len(paths) < 2:
            parser.error(
                f"--judge-llm-configs requires ≥2 paths; got {len(paths)}. "
                f"Use --judge-llm-config for single-judge scoring."
            )
        council_llms = [load_llm_config(p) for p in paths]
        judge_council_models = [j.model for j in council_llms]
        judge_council_api_keys = [_resolve_key(j) for j in council_llms]
        judge_council_regions = [
            getattr(j, "aws_region_name", None) for j in council_llms
        ]
        # Sentinel values for the single-judge path (not used in council mode)
        judge_model = "<council>"
        judge_api_key = None
        judge_region = None
    elif args.judge_llm_config:
        judge_llm = load_llm_config(args.judge_llm_config)
        judge_model = judge_llm.model
        judge_api_key = _resolve_key(judge_llm)
        judge_region = getattr(judge_llm, "aws_region_name", None)
    else:
        judge_model = os.getenv(
            "GOKU_JUDGE_MODEL", "gemini/gemini-3.5-flash"
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

        # Find the model-level output.jsonl that contains this task's record.
        # On miss, fall back to the cumulative ``output.jsonl.ever_seen``
        # ledger that clean_resume_state.py maintains across --rerun cycles
        # (2026-05-23 / P1 fix). Without that fallback, repeated --rerun
        # cleanups make an entry permanently unrescore-able.
        model_dir = scores_file.parent.parent
        model_output_jsonl = model_dir / "output.jsonl"
        ever_seen_jsonl = model_dir / "output.jsonl.ever_seen"

        def _find_entry(jsonl_path: Path) -> dict | None:
            if not jsonl_path.is_file():
                return None
            try:
                with open(jsonl_path, encoding="utf-8") as f:
                    for raw in f:
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            obj = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if obj.get("instance_id") == task_key:
                            return obj
            except OSError:
                return None
            return None

        agent_data = _find_entry(model_output_jsonl)
        if agent_data is None:
            agent_data = _find_entry(ever_seen_jsonl)
            if agent_data is not None:
                logger.info(
                    "Recovered %s entry from %s (live output.jsonl no longer "
                    "has it — likely stripped by prior --rerun)",
                    task_key, ever_seen_jsonl.name,
                )

        if agent_data is None:
            both = model_output_jsonl.name
            if ever_seen_jsonl.is_file():
                both += f" or {ever_seen_jsonl.name}"
            logger.warning(
                "No entry for %s in %s — skipping",
                task_key, both,
            )
            n_skip += 1
            continue

        # Reconstruct judge context from saved data
        history = agent_data.get("history") or []
        response_text = extract_response_from_history(history)
        trajectory = format_trajectory(history)
        file_contents, output_media_paths = collect_file_contents(
            scores_file.parent / "results"
        )

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
                judge_council_models=judge_council_models,
                judge_council_api_keys=judge_council_api_keys,
                judge_council_regions=judge_council_regions,
                # task.input_files is populated by load_task() from
                # dataset/<task>/data/input_files/ — absolute paths to
                # the task's input media that the judge needs for visual
                # grounding (otherwise it has only the agent's claims).
                input_image_paths=task.input_files,  # type: ignore[union-attr]
                # Agent-produced media (PDFs/images/videos saved into the
                # agent's results/ directory). Without this the judge sees
                # only "(binary, N bytes)" placeholders for them and has
                # to bluff about their content.
                output_media_paths=output_media_paths,
            )
        except Exception:
            logger.exception("Scoring failed for %s in %s", task_key, scores_file)
            n_fail += 1
            continue

        task_score = compute_task_score(results, rubric_items)
        write_scores_jsonl(task_score, scores_file, rubric_items=rubric_items)
        # Propagate the rescored aggregates back into the model-level
        # output.jsonl. Without this, the downstream benchmark report
        # (eval_infer.load_scores_from_runs reads test_result from
        # output.jsonl) silently shows stale pre-rescore numbers.
        if not update_output_jsonl_test_result(
            model_output_jsonl, task_key, task_score,
        ):
            logger.warning(
                "Updated %s but could not propagate to %s — benchmark "
                "report may show stale aggregates for this instance.",
                scores_file, model_output_jsonl,
            )
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
