"""Goku benchmark evaluation report generator.

Reads output.jsonl files from completed runs and generates:
  - Per-model benchmark reports (mean scores, pass rates, pass@N)
  - Cross-model calibration checks
  - Summary markdown report
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

from benchmarks.goku.benchmark_report import generate_report
from benchmarks.goku.calibration import check_calibration
from benchmarks.goku.models import BenchmarkReport, TaskScore


logger = logging.getLogger(__name__)


def load_scores_from_runs(
    output_base_dir: Path,
    model_id: str,
    n_runs: int = 3,
    claimed_paths: set[Path] | None = None,
) -> tuple[dict[str, list[TaskScore]], dict[str, list[dict]]]:
    """Load TaskScore objects (and per-run cost/token metrics) from output.jsonl.

    Expected directory structure:
        output_base_dir/
        ├── run_1/goku/<model_slug>/output.jsonl
        ├── run_2/goku/<model_slug>/output.jsonl
        └── run_3/goku/<model_slug>/output.jsonl

    Filters output.jsonl files to only include those under paths matching
    the model_id (slug form: slashes replaced with underscores).

    Args:
        output_base_dir: Base directory containing run_N/ subdirs.
        model_id: Model identifier to locate output dirs.
        n_runs: Number of runs to look for.
        claimed_paths: Optional set of output.jsonl paths already counted by
            a previous call. Matching files are skipped, and newly-loaded
            paths are added to the set in place. Used to deduplicate when
            multiple `model_id` substrings might match the same output dir
            (e.g. when passing both a new-form and legacy slug for the same
            logical model).

    Returns:
        A tuple of:
          - task_scores: {task_key: [TaskScore, ...]}, one entry per run
          - per_task_metrics: {task_key: [{"cost_usd": float,
                                            "prompt_tokens": int,
                                            "completion_tokens": int,
                                            "cache_read_tokens": int,
                                            "cache_write_tokens": int}, ...]}
            (same ordering as task_scores; entries are zero if metrics absent)
    """
    task_scores: dict[str, list[TaskScore]] = {}
    per_task_metrics: dict[str, list[dict]] = {}
    model_slug = model_id.replace("/", "_")
    if claimed_paths is None:
        claimed_paths = set()

    for run_num in range(1, n_runs + 1):
        run_dir = output_base_dir / f"run_{run_num}"

        output_files = list(run_dir.rglob("output.jsonl"))
        if not output_files:
            logger.warning(f"No output.jsonl found in {run_dir}")
            continue

        for output_file in output_files:
            if model_slug not in str(output_file):
                continue
            resolved = output_file.resolve()
            if resolved in claimed_paths:
                continue
            claimed_paths.add(resolved)

            with open(output_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    instance_id = data.get("instance_id", "")
                    test_result = data.get("test_result", {})

                    if not instance_id or not test_result:
                        continue

                    score = TaskScore(
                        awarded=test_result.get("awarded", 0),
                        max_total=test_result.get("max_total", 0),
                        raw_score=test_result.get("raw_score", 0.0),
                        per_task_score=test_result.get("per_task_score", 0.0),
                        passed=test_result.get("passed", False),
                        items=[],
                    )

                    if instance_id not in task_scores:
                        task_scores[instance_id] = []
                        per_task_metrics[instance_id] = []
                    task_scores[instance_id].append(score)

                    # Pull cost + token usage from the metrics field (populated
                    # by OpenHands/LiteLLM during inference).
                    metrics = data.get("metrics") or {}
                    usage = metrics.get("accumulated_token_usage") or {}

                    # Judge cost: prefer test_result.judge_cost_usd (new runs).
                    # Fall back to reading the per-task scores.jsonl summary
                    # line for backward compat with older runs.
                    judge_cost = float(test_result.get("judge_cost_usd") or 0.0)
                    if judge_cost == 0.0:
                        scores_path = (
                            output_file.parent / instance_id / "scores.jsonl"
                        )
                        if scores_path.is_file():
                            try:
                                with open(scores_path, encoding="utf-8") as sf:
                                    for sline in sf:
                                        sline = sline.strip()
                                        if not sline:
                                            continue
                                        try:
                                            srow = json.loads(sline)
                                        except json.JSONDecodeError:
                                            continue
                                        if isinstance(srow, dict) and "judge_cost_usd" in srow:
                                            judge_cost = float(
                                                srow["judge_cost_usd"] or 0.0
                                            )
                                            break
                            except OSError:
                                pass

                    per_task_metrics[instance_id].append({
                        "cost_usd": float(metrics.get("accumulated_cost") or 0.0),
                        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                        "completion_tokens": int(usage.get("completion_tokens") or 0),
                        "cache_read_tokens": int(usage.get("cache_read_tokens") or 0),
                        "cache_write_tokens": int(usage.get("cache_write_tokens") or 0),
                        "judge_cost_usd": judge_cost,
                        "has_metrics": bool(metrics),
                    })

    return task_scores, per_task_metrics


def generate_markdown_report(
    reports: list[BenchmarkReport],
    calibration_results: list[dict],
) -> str:
    """Generate a human-readable markdown report.

    Args:
        reports: List of BenchmarkReport (one per model).
        calibration_results: Calibration check results.

    Returns:
        Formatted markdown string.
    """
    lines: list[str] = ["# Goku Benchmark Report\n"]

    # Model comparison table
    lines.append("## Model Comparison\n")
    lines.append(
        "| Model | Score | Raw | Pass Rate | pass@3 | pass^3 | Tasks | "
        "Agent $ | Judge $ | Total $ | $/run |"
    )
    lines.append(
        "|-------|-------|-----|-----------|--------|--------|-------|"
        "---------|---------|---------|-------|"
    )
    grand_agent_cost = 0.0
    grand_judge_cost = 0.0
    for r in reports:
        agent_cost = r.total_cost_usd
        judge_cost = r.total_judge_cost_usd
        total = agent_cost + judge_cost
        grand_agent_cost += agent_cost
        grand_judge_cost += judge_cost
        agent_str = f"${agent_cost:.2f}" if agent_cost > 0 else "—"
        judge_str = f"${judge_cost:.2f}" if judge_cost > 0 else "—"
        total_str = f"${total:.2f}" if total > 0 else "—"
        per_run_str = (
            f"${r.mean_cost_per_run_usd:.3f}"
            if r.mean_cost_per_run_usd > 0
            else "—"
        )
        lines.append(
            f"| {r.model_id} | {r.mean_per_task_score:.4f} | "
            f"{r.mean_raw_score:.4f} | {r.pass_rate:.4f} | "
            f"{r.pass_at_3:.4f} | {r.pass_hat_3:.4f} | "
            f"{r.total_tasks} | {agent_str} | {judge_str} | {total_str} | "
            f"{per_run_str} |"
        )
    if grand_agent_cost > 0 or grand_judge_cost > 0:
        grand_total = grand_agent_cost + grand_judge_cost
        lines.append(
            f"| **TOTAL** | | | | | | | **${grand_agent_cost:.2f}** | "
            f"**${grand_judge_cost:.2f}** | **${grand_total:.2f}** | |"
        )

    # Token usage breakdown (only if at least one model has metrics)
    has_tokens = any(
        r.total_prompt_tokens > 0 or r.total_completion_tokens > 0
        for r in reports
    )
    if has_tokens:
        lines.append("\n## Token Usage\n")
        lines.append(
            "| Model | Prompt | Completion | Cache Read | Cache Write | Runs w/ Metrics |"
        )
        lines.append(
            "|-------|--------|------------|------------|-------------|-----------------|"
        )
        for r in reports:
            lines.append(
                f"| {r.model_id} | {r.total_prompt_tokens:,} | "
                f"{r.total_completion_tokens:,} | {r.total_cache_read_tokens:,} | "
                f"{r.total_cache_write_tokens:,} | {r.total_runs_with_metrics} |"
            )

    # Calibration
    lines.append("\n## Calibration\n")
    too_easy = [r for r in calibration_results if r["flag"] == "too_easy"]
    too_hard = [r for r in calibration_results if r["flag"] == "too_hard"]
    well_cal = [r for r in calibration_results if r["flag"] == "well_calibrated"]

    lines.append(f"- Well-calibrated: {len(well_cal)} tasks")
    lines.append(f"- Too easy: {len(too_easy)} tasks")
    lines.append(f"- Too hard: {len(too_hard)} tasks")

    if too_easy:
        lines.append("\n### Too Easy Tasks")
        for r in too_easy:
            lines.append(f"- **{r['task_key']}**: {r['model_scores']}")

    if too_hard:
        lines.append("\n### Too Hard Tasks")
        for r in too_hard:
            lines.append(f"- **{r['task_key']}**: {r['model_scores']}")

    return "\n".join(lines)


def export_delivery_format(
    output_base_dir: Path,
    tasks_source_dir: Path,
    delivery_dir: Path,
    model_ids: list[str],
    n_runs: int = 3,
) -> None:
    """Build the full delivery package per the doc spec.

    Produces (per DIU Goku doc.md Tab 2 folder tree):
        delivery_dir/
        └── tasks/<task_key>/
            ├── instruction.md          (from tasks_source_dir)
            ├── rubrics.jsonl           (from tasks_source_dir)
            ├── data/input_files/       (from tasks_source_dir)
            └── runs/<display_name>/[run_N/]
                ├── scores.jsonl
                └── results/
                    ├── response.md     (model's final response)
                    └── <artifacts>     (agent-produced files)

    Deliberately NOT shipped (kept in eval_outputs/ for debugging only):
      - output.jsonl       — full OpenHands trajectory; not in doc spec
      - results/bash_events/ — raw tool-call event log; not in doc spec
    """
    from benchmarks.goku.config import get_model_display_name

    task_keys_seen: set[str] = set()
    # Dedup: each scores.jsonl is exported under at most one model_id.
    # Earlier --models entries claim a matching path first. Prevents
    # double-write when a legacy and new slug both substring-match the
    # same path.
    claimed_scores_paths: set[Path] = set()

    for model_id in model_ids:
        model_slug = model_id.replace("/", "_")
        display_name = get_model_display_name(model_id)

        for run_num in range(1, n_runs + 1):
            run_dir = output_base_dir / f"run_{run_num}"
            scores_files = list(run_dir.rglob("scores.jsonl"))

            for scores_file in scores_files:
                if model_slug not in str(scores_file):
                    continue
                resolved_scores = scores_file.resolve()
                if resolved_scores in claimed_scores_paths:
                    continue
                claimed_scores_paths.add(resolved_scores)

                task_key = scores_file.parent.name
                task_src = tasks_source_dir / task_key

                if task_key not in task_keys_seen and task_src.is_dir():
                    task_keys_seen.add(task_key)
                    task_dest = delivery_dir / "tasks" / task_key

                    for name in ("instruction.md", "rubrics.jsonl"):
                        src = task_src / name
                        if src.exists():
                            task_dest.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src, task_dest / name)

                    src_data = task_src / "data" / "input_files"
                    if src_data.is_dir():
                        dest_data = task_dest / "data" / "input_files"
                        if dest_data.exists():
                            shutil.rmtree(dest_data)
                        shutil.copytree(src_data, dest_data)

                run_label = f"run_{run_num}" if n_runs > 1 else ""
                model_run_dir = (
                    delivery_dir / "tasks" / task_key / "runs" / display_name
                )
                if run_label:
                    model_run_dir = model_run_dir / run_label

                model_run_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(scores_file, model_run_dir / "scores.jsonl")

                # Source trajectory lives in eval_outputs/, NOT in delivery.
                # We only read it to extract response.md (below); the raw
                # trajectory is intentionally not shipped — per doc Tab 2
                # the delivery contains only scores + response + artifacts.
                source_output_jsonl = scores_file.parent.parent / "output.jsonl"

                # Copy agent-produced artifacts into delivery results/,
                # EXCLUDING bash_events/ (internal tool-call log) and
                # __pycache__/ (Python bytecode cache from any helper
                # scripts the agent wrote) — neither is part of the
                # doc-specified deliverable.
                agent_results = scores_file.parent / "results"
                if agent_results.is_dir():
                    dest_results = model_run_dir / "results"
                    if dest_results.exists():
                        shutil.rmtree(dest_results)
                    shutil.copytree(
                        agent_results,
                        dest_results,
                        ignore=shutil.ignore_patterns(
                            "bash_events", "__pycache__", "*.pyc"
                        ),
                    )
                    # Defensive flatten: if the agent (or a legacy
                    # spec-violating prompt) created a nested `results/`
                    # subdir, lift its contents up one level. Collisions
                    # with siblings at the parent are left in place.
                    nested = dest_results / "results"
                    if nested.is_dir():
                        for child in list(nested.iterdir()):
                            target = dest_results / child.name
                            if not target.exists():
                                shutil.move(str(child), str(target))
                        try:
                            nested.rmdir()
                        except OSError:
                            pass  # leave it if non-empty due to collisions

                # Per doc Tab 2 (folder tree), each per-run results/ carries
                # the model's final natural-language response as response.md
                # alongside any artifacts. Extracted from the SOURCE
                # eval_outputs/output.jsonl (never written to delivery).
                from benchmarks.goku.response_extractor import (
                    extract_final_response_from_jsonl,
                    write_response_md,
                )
                results_dir = model_run_dir / "results"
                results_dir.mkdir(parents=True, exist_ok=True)
                final_response = extract_final_response_from_jsonl(
                    source_output_jsonl, instance_id=task_key
                )
                write_response_md(final_response, results_dir / "response.md")

                logger.info(
                    f"Exported {task_key} / {display_name}"
                    + (f" / {run_label}" if run_label else "")
                )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Goku benchmark evaluation report"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Base output directory containing run_N/ subdirectories",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="Model identifiers to include in report",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per model (default: 3)",
    )
    parser.add_argument(
        "--report-file",
        type=str,
        default=None,
        help="Output file for markdown report (default: stdout)",
    )
    parser.add_argument(
        "--export-delivery",
        type=str,
        default=None,
        help="Export to delivery folder structure at this path",
    )
    parser.add_argument(
        "--tasks-dir",
        type=str,
        default=None,
        help="Source tasks directory (for --export-delivery to copy instruction/rubrics/data)",
    )
    args = parser.parse_args()

    output_base = Path(args.output_dir)
    reports: list[BenchmarkReport] = []
    per_model_scores: dict[str, dict[str, float]] = {}

    # Shared dedup set: each output.jsonl is counted under at most one
    # model_id. Earlier --models entries claim a matching path first.
    claimed_paths: set[Path] = set()
    # Track which display name each model_id maps to so we can merge reports
    # that resolve to the same canonical model (e.g. legacy + new slug for
    # the same logical model).
    reports_by_display: dict[str, BenchmarkReport] = {}

    for model_id in args.models:
        logger.info(f"Loading scores for model: {model_id}")
        task_scores, per_task_metrics = load_scores_from_runs(
            output_base, model_id, n_runs=args.runs, claimed_paths=claimed_paths
        )

        if not task_scores:
            logger.warning(f"No scores found for model {model_id}")
            continue

        # Use the resolved display name as the canonical id for the report,
        # so that a legacy slug + new slug pair for the same logical model
        # produce a single combined report row.
        from benchmarks.goku.config import get_model_display_name
        display_id = get_model_display_name(model_id)
        if display_id in reports_by_display:
            logger.info(
                "Skipping duplicate report for display name '%s' "
                "(already populated by an earlier --models entry).",
                display_id,
            )
            continue

        report = generate_report(
            task_scores=task_scores,
            model_id=display_id,
            n_runs=args.runs,
            per_task_metrics=per_task_metrics,
        )
        reports.append(report)
        reports_by_display[display_id] = report

        # Collect per-task mean scores for calibration
        model_scores: dict[str, float] = {}
        for task_key, scores in task_scores.items():
            model_scores[task_key] = (
                sum(s.per_task_score for s in scores) / len(scores) if scores else 0.0
            )
        per_model_scores[model_id] = model_scores

    # Cross-model calibration
    calibration_results = check_calibration(per_model_scores)

    # Generate markdown
    md = generate_markdown_report(reports, calibration_results)

    if args.report_file:
        report_path = Path(args.report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(md, encoding="utf-8")
        logger.info(f"Report written to {report_path}")
    else:
        sys.stdout.write(md)

    # Also write JSON reports
    for report in reports:
        json_path = output_base / f"report_{report.model_id.replace('/', '_')}.json"
        json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        logger.info(f"JSON report written to {json_path}")

    # Export delivery format if requested
    if args.export_delivery:
        from datetime import date

        tasks_source = Path(args.tasks_dir) if args.tasks_dir else Path("dataset")
        delivery_root = (
            Path(args.export_delivery)
            / f"MM Agentic Pilot Samples-{date.today().isoformat()}"
        )
        export_delivery_format(
            output_base_dir=output_base,
            tasks_source_dir=tasks_source,
            delivery_dir=delivery_root,
            model_ids=args.models,
            n_runs=args.runs,
        )
        logger.info(f"Delivery package at {delivery_root}")

    logger.info("Goku evaluation report complete.")


if __name__ == "__main__":
    main()
