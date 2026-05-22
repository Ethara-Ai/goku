"""Benchmark-level report generation for the Goku evaluation.

Computes aggregate metrics across all tasks for a single model:
  - mean_per_task_score: primary headline metric
  - mean_raw_score: can be negative (before clipping)
  - pass_rate: fraction of tasks that passed (pass@1)
  - pass_at_3: unbiased Codex estimator (lenient)
  - pass_hat_3: strict: all 3 runs must pass
"""

from __future__ import annotations

import logging
import math

from benchmarks.goku.models import BenchmarkReport, RubricItem, TaskScore


TAB3_TARGET_THRESHOLD = 0.7  # per DIU Goku doc Tab 3 May 15 addendum
FORMAT_CATEGORY = "FORMAT"


logger = logging.getLogger(__name__)


def pass_at_n(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k given n total runs with c passes.

    From the Codex paper (Chen et al., 2021):
        pass@k = 1 - C(n-c, k) / C(n, k)

    Args:
        n: Total number of runs attempted.
        c: Number of runs that passed.
        k: k in pass@k (how many chances).

    Returns:
        Probability that at least 1 of k random samples passes.
        Returns 1.0 if c >= n (all passed), 0.0 if c == 0.
    """
    if c == 0:
        return 0.0
    if c >= n:
        return 1.0
    if k > n:
        k = n
    # Use log-space to avoid overflow with large combinatorials
    # pass@k = 1 - prod_{i=0}^{k-1} (n-c-i) / (n-i)
    log_prod = 0.0
    for i in range(k):
        if n - c - i <= 0:
            return 1.0
        log_prod += math.log(n - c - i) - math.log(n - i)
    return 1.0 - math.exp(log_prod)


def pass_hat_n(n: int, c: int) -> bool:
    """Strict pass^N: all N runs must pass.

    Args:
        n: Total number of runs.
        c: Number of runs that passed.

    Returns:
        True only if c == n (all runs passed).
    """
    return c == n


def _compute_category_breakdown(
    task_scores: dict[str, list[TaskScore]],
    task_rubric_items: dict[str, list[RubricItem]],
) -> tuple[dict[str, float], float, bool]:
    """Return (mean_score_by_category, mean_non_format_score, tab3_target_hit).

    Each per-category mean is the average of (awarded/max_total) across
    rubric items in that category, weighted equally per item, then averaged
    across tasks and runs.
    """
    cat_awarded: dict[str, float] = {}
    cat_max: dict[str, float] = {}

    for task_key, scores in task_scores.items():
        rubric_items = task_rubric_items.get(task_key)
        if not rubric_items or not scores:
            continue
        items_by_number = {ri.number: ri for ri in rubric_items}
        for ts in scores:
            for res in ts.items:
                ri = items_by_number.get(res.number)
                # Per spec: per-category mean is built from POSITIVE items only.
                # Negative items are penalty-only and contribute to raw_score,
                # not to per-category correctness percentage. Mirrors the spec's
                # `max_total = sum(positive points only)` rule (Tab 2 L218).
                if ri is None or ri.points <= 0:
                    continue
                cat = ri.category
                cat_max[cat] = cat_max.get(cat, 0.0) + ri.points
                cat_awarded[cat] = cat_awarded.get(cat, 0.0) + (
                    ri.points if res.passed else 0
                )

    mean_by_cat: dict[str, float] = {}
    for cat, mx in cat_max.items():
        if mx > 0:
            mean_by_cat[cat] = round(min(1.0, max(0.0, cat_awarded.get(cat, 0.0) / mx)), 4)

    non_format_awarded = sum(
        v for c, v in cat_awarded.items() if c != FORMAT_CATEGORY
    )
    non_format_max = sum(v for c, v in cat_max.items() if c != FORMAT_CATEGORY)
    if non_format_max > 0:
        mean_non_format = round(
            min(1.0, max(0.0, non_format_awarded / non_format_max)), 4
        )
        tab3_hit = mean_non_format <= TAB3_TARGET_THRESHOLD
    else:
        mean_non_format = 0.0
        tab3_hit = False
    return mean_by_cat, mean_non_format, tab3_hit


def generate_report(
    task_scores: dict[str, list[TaskScore]],
    model_id: str,
    n_runs: int = 3,
    total_cost_usd: float | None = None,
    per_task_metrics: dict[str, list[dict]] | None = None,
    task_rubric_items: dict[str, list[RubricItem]] | None = None,
) -> BenchmarkReport:
    """Generate a benchmark-level report for one model.

    Args:
        task_scores: Mapping of task_key -> list of TaskScore (one per run).
        model_id: Identifier for the model being evaluated.
        n_runs: Expected number of runs per task.
        total_cost_usd: Override the aggregated cost (e.g. to inject a value
            computed elsewhere). If None and per_task_metrics is provided,
            the cost is summed from the metrics. If both are None, defaults to 0.
        per_task_metrics: Optional mapping of task_key -> list of per-run dicts
            with keys `cost_usd`, `prompt_tokens`, `completion_tokens`,
            `cache_read_tokens`, `cache_write_tokens`, `has_metrics`. When
            provided, token totals and cost are aggregated from this.

    Returns:
        A BenchmarkReport with aggregate metrics.
    """
    # Aggregate cost + token usage if metrics are provided
    agg_cost = 0.0
    agg_prompt = 0
    agg_completion = 0
    agg_cache_read = 0
    agg_cache_write = 0
    agg_judge_cost = 0.0
    n_runs_with_metrics = 0
    if per_task_metrics:
        for runs in per_task_metrics.values():
            for m in runs:
                agg_cost += float(m.get("cost_usd", 0.0))
                agg_prompt += int(m.get("prompt_tokens", 0))
                agg_completion += int(m.get("completion_tokens", 0))
                agg_cache_read += int(m.get("cache_read_tokens", 0))
                agg_cache_write += int(m.get("cache_write_tokens", 0))
                agg_judge_cost += float(m.get("judge_cost_usd", 0.0))
                if m.get("has_metrics"):
                    n_runs_with_metrics += 1
    resolved_cost: float = (
        float(total_cost_usd) if total_cost_usd is not None else agg_cost
    )

    if not task_scores:
        return BenchmarkReport(
            model_id=model_id,
            mean_per_task_score=0.0,
            mean_raw_score=0.0,
            pass_rate=0.0,
            pass_at_3=0.0,
            pass_hat_3=0.0,
            total_tasks=0,
            total_cost_usd=resolved_cost,
            total_prompt_tokens=agg_prompt,
            total_completion_tokens=agg_completion,
            total_cache_read_tokens=agg_cache_read,
            total_cache_write_tokens=agg_cache_write,
            mean_cost_per_run_usd=0.0,
            total_runs_with_metrics=n_runs_with_metrics,
            total_judge_cost_usd=round(agg_judge_cost, 4),
        )

    total_tasks = len(task_scores)
    all_per_task_scores: list[float] = []
    all_raw_scores: list[float] = []
    pass_counts: list[tuple[int, int]] = []  # (n_runs, n_passed) per task

    for task_key, scores in task_scores.items():
        if not scores:
            logger.warning("Task %s has no scores", task_key)
            continue

        # Per-task: mean across runs
        task_mean_pts = sum(s.per_task_score for s in scores) / len(scores)
        task_mean_raw = sum(s.raw_score for s in scores) / len(scores)
        all_per_task_scores.append(task_mean_pts)
        all_raw_scores.append(task_mean_raw)

        # Pass counting for pass@N
        n = len(scores)
        c = sum(1 for s in scores if s.passed)
        pass_counts.append((n, c))

    # Aggregate metrics
    mean_per_task_score = (
        sum(all_per_task_scores) / len(all_per_task_scores)
        if all_per_task_scores
        else 0.0
    )
    mean_raw_score = (
        sum(all_raw_scores) / len(all_raw_scores) if all_raw_scores else 0.0
    )

    # pass_rate = mean per-run pass rate across tasks
    # (fraction of individual runs that passed, averaged over tasks)
    if pass_counts:
        per_task_pass_rates = [c / n if n > 0 else 0.0 for n, c in pass_counts]
        pass_rate = sum(per_task_pass_rates) / len(per_task_pass_rates)
    else:
        pass_rate = 0.0

    # pass@3 (unbiased estimator averaged across tasks)
    pass_at_3_scores = [pass_at_n(n, c, min(3, n)) for n, c in pass_counts]
    avg_pass_at_3 = (
        sum(pass_at_3_scores) / len(pass_at_3_scores) if pass_at_3_scores else 0.0
    )

    # pass^3 (strict: fraction of tasks where ALL runs passed)
    pass_hat_3_rate = (
        sum(1 for n, c in pass_counts if pass_hat_n(n, c)) / len(pass_counts)
        if pass_counts
        else 0.0
    )

    mean_cost = (
        agg_cost / n_runs_with_metrics if n_runs_with_metrics > 0 else 0.0
    )

    if task_rubric_items:
        mean_by_cat, mean_non_format, tab3_hit = _compute_category_breakdown(
            task_scores, task_rubric_items
        )
    else:
        mean_by_cat, mean_non_format, tab3_hit = {}, 0.0, False

    return BenchmarkReport(
        model_id=model_id,
        mean_per_task_score=round(mean_per_task_score, 4),
        mean_raw_score=round(mean_raw_score, 4),
        pass_rate=round(pass_rate, 4),
        pass_at_3=round(avg_pass_at_3, 4),
        pass_hat_3=round(pass_hat_3_rate, 4),
        total_tasks=total_tasks,
        total_cost_usd=round(resolved_cost, 4),
        total_prompt_tokens=agg_prompt,
        total_completion_tokens=agg_completion,
        total_cache_read_tokens=agg_cache_read,
        total_cache_write_tokens=agg_cache_write,
        mean_cost_per_run_usd=round(mean_cost, 4),
        total_runs_with_metrics=n_runs_with_metrics,
        total_judge_cost_usd=round(agg_judge_cost, 4),
        mean_score_by_category=mean_by_cat,
        mean_non_format_score=mean_non_format,
        tab3_difficulty_target_hit=tab3_hit,
    )
