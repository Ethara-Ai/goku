"""Per-task score aggregation for the Goku benchmark.

Implements the scoring formula from the project spec:
  awarded = sum(passed positive points) - sum(triggered negative |points|)
  max_total = sum(all positive points)
  raw_score = awarded / max_total
  per_task_score = clip(raw_score, 0, 1)
  pass = all mandatory positives pass AND no mandatory negatives triggered
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from benchmarks.goku.models import RubricItem, ScorerResult, TaskScore


logger = logging.getLogger(__name__)


def compute_task_score(
    results: list[ScorerResult],
    rubric_items: list[RubricItem],
) -> TaskScore:
    """Compute aggregate score for a single task run.

    Args:
        results: List of ScorerResult, one per rubric item.
        rubric_items: Corresponding list of RubricItem definitions.

    Returns:
        A TaskScore with awarded, max_total, raw_score, per_task_score,
        and pass/fail determination.

    Raises:
        ValueError: If results and rubric_items have different lengths.
    """
    if len(results) != len(rubric_items):
        raise ValueError(
            f"Results ({len(results)}) and rubric items ({len(rubric_items)}) "
            f"must have the same length"
        )

    awarded = 0
    max_total = 0

    for result, item in zip(results, rubric_items):
        if item.points > 0:
            max_total += item.points
            if result.passed:
                awarded += item.points
        elif item.points < 0:
            # NOTE on response_not_criteria:
            # result.passed = True means the NEGATIVE criterion WAS detected
            # (i.e., the agent DID hallucinate). This triggers the penalty.
            # result.passed = False means the agent did NOT hallucinate (good).
            # The naming is intentional: "passed" = "criterion matched" for ALL types.
            if result.passed:
                awarded -= abs(item.points)

    # Compute scores
    raw_score = awarded / max_total if max_total > 0 else 0.0
    per_task_score = max(0.0, min(1.0, raw_score))

    # Pass/fail: mandatory gating
    passed = True
    for result, item in zip(results, rubric_items):
        if item.importance != "mandatory":
            continue
        if item.points > 0 and not result.passed:
            # Mandatory positive item was NOT achieved
            passed = False
            break
        if item.points < 0 and result.passed:
            # Mandatory negative criterion WAS triggered (hallucination)
            passed = False
            break

    # Sum judge LLM cost across all items (0 for purely deterministic tasks).
    judge_cost_usd = sum(getattr(r, "judge_cost_usd", 0.0) or 0.0 for r in results)

    return TaskScore(
        awarded=awarded,
        max_total=max_total,
        raw_score=raw_score,
        per_task_score=per_task_score,
        passed=passed,
        items=results,
        judge_cost_usd=judge_cost_usd,
    )


def write_scores_jsonl(
    score: TaskScore, output_path: Path, rubric_items: list[RubricItem] | None = None
) -> None:
    """Write scores.jsonl in the format specified by the project doc.

    Format:
        {"number": 1, "passed": true, "judge_rationale": "..."}
        {"number": 2, "passed": false, "judge_rationale": "..."}
        ...
        {"pass": true}
        {"per_task_score": 0.82}
        {"awarded": 24, "max_total": 29, "raw_score": 0.83}

    Note on negative items (response_not_criteria):
        Internally, result.passed=True means "criterion matched" (hallucination detected).
        In scores.jsonl output, we invert this for negative items so that:
          passed=false means the task FAILED on this item (hallucination present).
          passed=true means the task PASSED on this item (no hallucination).
        This matches the spec example (doc L147).

    Args:
        score: The TaskScore to write.
        output_path: Path to write scores.jsonl to.
        rubric_items: Optional rubric items for determining negative item semantics.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Track whether any rubric was council-judged. If so, we'll also write a
    # sidecar audit file alongside scores.jsonl to preserve per-judge data,
    # but the primary scores.jsonl stays in CANONICAL CLIENT FORMAT —
    # {number, passed, judge_rationale} only, no council metadata. The
    # client-facing spec (DIU Goku doc §3.1) defines this schema; council
    # metadata in scores.jsonl would surprise downstream consumers.
    any_council = any(
        item.per_judge_verdicts is not None for item in score.items
    )

    with open(output_path, "w", encoding="utf-8") as f:
        # Per-item rows — canonical schema only.
        for i, item_result in enumerate(score.items):
            # For negative items, invert 'passed' in output to match spec semantics:
            # spec: passed=false when hallucination detected (task failed on this item)
            # internal: passed=True when criterion matched (hallucination detected)
            display_passed = item_result.passed
            if rubric_items and i < len(rubric_items) and rubric_items[i].points < 0:
                display_passed = not item_result.passed

            # For council-judged rubrics, pick a representative MAJORITY
            # judge's rationale instead of the bracketed council summary
            # (which mentions all 3 judges by name — not canonical client
            # format). Falls back to item_result.judge_rationale for
            # single-judge / deterministic / all-failed-judge cases.
            chosen_rationale = item_result.judge_rationale
            if item_result.per_judge_verdicts:
                majority_rationales = [
                    v.judge_rationale for v in item_result.per_judge_verdicts
                    if v.passed == item_result.passed and v.error is None
                ]
                if majority_rationales:
                    chosen_rationale = majority_rationales[0]

            row: dict = {
                "number": item_result.number,
                "passed": display_passed,
                "judge_rationale": chosen_rationale,
            }
            f.write(json.dumps(row) + "\n")

        # Summary rows
        f.write(json.dumps({"pass": score.passed}) + "\n")
        f.write(json.dumps({"per_task_score": round(score.per_task_score, 4)}) + "\n")
        f.write(
            json.dumps(
                {
                    "awarded": score.awarded,
                    "max_total": score.max_total,
                    "raw_score": round(score.raw_score, 4),
                }
            )
            + "\n"
        )
        # Judge cost summary (always emitted; 0 if no LLM-judged items)
        f.write(
            json.dumps({"judge_cost_usd": round(score.judge_cost_usd, 6)}) + "\n"
        )

    logger.info("Wrote scores to %s", output_path)

    # Sidecar audit file: scores.council_audit.jsonl alongside scores.jsonl,
    # written ONLY when at least one rubric was scored by the judge council.
    # Holds per-judge verdicts + vote/consensus/disagreement metadata that
    # used to live in scores.jsonl. Kept locally for our debugging /
    # rubric-refinement workflow; NOT shipped in delivery (the delivery
    # exporter doesn't pick up `scores.council_audit.jsonl`). If audit file
    # ends up in delivery, it's because someone copied it manually.
    if any_council:
        audit_path = output_path.with_name("scores.council_audit.jsonl")
        with open(audit_path, "w", encoding="utf-8") as af:
            for i, item_result in enumerate(score.items):
                if item_result.per_judge_verdicts is None:
                    continue
                display_passed = item_result.passed
                if (rubric_items and i < len(rubric_items)
                        and rubric_items[i].points < 0):
                    display_passed = not item_result.passed
                row = {
                    "number": item_result.number,
                    "passed": display_passed,
                    "vote": item_result.vote,
                    "consensus": item_result.consensus,
                    "disagreement": item_result.disagreement,
                    "per_judge_verdicts": [
                        {
                            "judge_model": v.judge_model,
                            "passed": (
                                (not v.passed)
                                if (rubric_items and i < len(rubric_items)
                                    and rubric_items[i].points < 0)
                                else v.passed
                            ),
                            "judge_rationale": v.judge_rationale,
                            "judge_cost_usd": round(v.judge_cost_usd, 6),
                            **({"error": v.error} if v.error else {}),
                        }
                        for v in item_result.per_judge_verdicts
                    ],
                }
                af.write(json.dumps(row) + "\n")
        logger.info("Wrote council audit sidecar to %s", audit_path)
