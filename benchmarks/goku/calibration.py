"""Calibration checks for Goku benchmark tasks.

Flags tasks that may be too easy or too hard based on model scores:
  - too_easy: ALL models score > 0.9
  - too_hard: ALL models score < 0.3
  - well_calibrated: at least 1 model > 0.7, no model scores 1.0 trivially
"""

from __future__ import annotations

import logging

from benchmarks.goku.config import CALIBRATION_DEFAULTS


logger = logging.getLogger(__name__)


def check_calibration(
    per_model_scores: dict[str, dict[str, float]],
    too_easy_threshold: float | None = None,
    too_hard_threshold: float | None = None,
    target_min_score: float | None = None,
) -> list[dict[str, object]]:
    """Check calibration of tasks across models.

    Args:
        per_model_scores: Mapping of model_id -> {task_key: per_task_score}.
        too_easy_threshold: All models above this = too easy (default 0.9).
        too_hard_threshold: All models below this = too hard (default 0.3).
        target_min_score: At least 1 model should exceed this (default 0.7).

    Returns:
        List of calibration flags, one per task:
        [{"task_key": str, "flag": str, "model_scores": dict}, ...]
        flag is one of: "too_easy", "too_hard", "well_calibrated"
    """
    if too_easy_threshold is None:
        too_easy_threshold = CALIBRATION_DEFAULTS["too_easy_threshold"]
    if too_hard_threshold is None:
        too_hard_threshold = CALIBRATION_DEFAULTS["too_hard_threshold"]
    if target_min_score is None:
        target_min_score = CALIBRATION_DEFAULTS["target_min_score"]

    # Collect all task keys across all models
    all_task_keys: set[str] = set()
    for model_scores in per_model_scores.values():
        all_task_keys.update(model_scores.keys())

    results: list[dict[str, object]] = []

    for task_key in sorted(all_task_keys):
        # Gather scores for this task across all models
        task_model_scores: dict[str, float] = {}
        for model_id, model_scores in per_model_scores.items():
            if task_key in model_scores:
                task_model_scores[model_id] = model_scores[task_key]

        if not task_model_scores:
            continue

        scores = list(task_model_scores.values())

        # Determine calibration flag
        if all(s > too_easy_threshold for s in scores):
            flag = "too_easy"
        elif all(s < too_hard_threshold for s in scores):
            flag = "too_hard"
        else:
            flag = "well_calibrated"

        # Additional warning: no model exceeds target minimum
        if max(scores) < target_min_score:
            logger.warning(
                "Task %s: no model exceeds target minimum %.2f. "
                "Scores: %s. Consider adjusting rubrics.",
                task_key,
                target_min_score,
                task_model_scores,
            )

        results.append(
            {
                "task_key": task_key,
                "flag": flag,
                "model_scores": task_model_scores,
            }
        )

    # Log summary
    too_easy_count = sum(1 for r in results if r["flag"] == "too_easy")
    too_hard_count = sum(1 for r in results if r["flag"] == "too_hard")
    well_calibrated_count = sum(1 for r in results if r["flag"] == "well_calibrated")
    logger.info(
        "Calibration: %d well-calibrated, %d too-easy, %d too-hard (of %d tasks)",
        well_calibrated_count,
        too_easy_count,
        too_hard_count,
        len(results),
    )

    return results
