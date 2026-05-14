"""Tests for benchmarks.goku.scoring."""

import json
from pathlib import Path
from typing import Literal

import pytest

from benchmarks.goku.models import RubricItem, ScorerResult, TaskScore
from benchmarks.goku.scoring import compute_task_score, write_scores_jsonl


def _item(
    number: int,
    points: int,
    importance: Literal["mandatory", "nice_to_have"] = "mandatory",
) -> RubricItem:
    """Helper to make a minimal rubric item."""
    return RubricItem(
        number=number,
        type="response_criteria",
        category="CORRECTNESS",
        points=points,
        importance=importance,
        criterion=f"Criterion #{number}",
    )


def _result(number: int, passed: bool, points: int) -> ScorerResult:
    """Helper to make a scorer result."""
    return ScorerResult(
        number=number,
        passed=passed,
        judge_rationale=f"Item #{number}: {'passed' if passed else 'failed'}",
        points_awarded=points,
    )


class TestComputeTaskScore:
    def test_all_positive_all_pass(self):
        items = [_item(1, 5), _item(2, 3)]
        results = [_result(1, True, 5), _result(2, True, 3)]
        score = compute_task_score(results, items)
        assert score.awarded == 8
        assert score.max_total == 8
        assert score.raw_score == 1.0
        assert score.per_task_score == 1.0
        assert score.passed is True

    def test_all_positive_all_fail(self):
        items = [_item(1, 5), _item(2, 3)]
        results = [_result(1, False, 0), _result(2, False, 0)]
        score = compute_task_score(results, items)
        assert score.awarded == 0
        assert score.max_total == 8
        assert score.raw_score == 0.0
        assert score.per_task_score == 0.0
        assert score.passed is False

    def test_mixed_pass_fail(self):
        items = [_item(1, 5), _item(2, 3)]
        results = [_result(1, True, 5), _result(2, False, 0)]
        score = compute_task_score(results, items)
        assert score.awarded == 5
        assert score.max_total == 8
        assert score.per_task_score == pytest.approx(0.625)
        # Mandatory item #2 failed → overall fails
        assert score.passed is False

    def test_negative_item_penalty(self):
        items = [_item(1, 5), _item(2, -5, "mandatory")]
        # Item 2 passed = hallucination detected
        results = [_result(1, True, 5), _result(2, True, -5)]
        score = compute_task_score(results, items)
        # awarded = 5 - 5 = 0, max_total = 5 (only positive counted)
        assert score.awarded == 0
        assert score.max_total == 5
        assert score.raw_score == 0.0
        assert score.per_task_score == 0.0
        # Mandatory negative triggered → fails
        assert score.passed is False

    def test_negative_item_no_penalty(self):
        items = [_item(1, 5), _item(2, -5, "mandatory")]
        # Item 2 not passed = no hallucination (good)
        results = [_result(1, True, 5), _result(2, False, 0)]
        score = compute_task_score(results, items)
        assert score.awarded == 5
        assert score.max_total == 5
        assert score.raw_score == 1.0
        assert score.passed is True

    def test_nice_to_have_does_not_gate_pass(self):
        items = [_item(1, 5, "mandatory"), _item(2, 3, "nice_to_have")]
        results = [_result(1, True, 5), _result(2, False, 0)]
        score = compute_task_score(results, items)
        # Mandatory passed, nice_to_have failed → still passes
        assert score.passed is True
        assert score.awarded == 5
        assert score.max_total == 8
        assert score.per_task_score == pytest.approx(0.625)

    def test_per_task_score_clipped_to_zero(self):
        """When penalties exceed positives, score clips to 0."""
        items = [_item(1, 3), _item(2, -5, "mandatory")]
        results = [_result(1, True, 3), _result(2, True, -5)]
        score = compute_task_score(results, items)
        # awarded = 3 - 5 = -2, max_total = 3, raw = -0.667
        assert score.raw_score == pytest.approx(-2 / 3, abs=0.01)
        assert score.per_task_score == 0.0  # Clipped

    def test_empty_rubric(self):
        score = compute_task_score([], [])
        assert score.awarded == 0
        assert score.max_total == 0
        assert score.raw_score == 0.0
        assert score.per_task_score == 0.0
        assert score.passed is True  # No mandatory items → vacuously passes

    def test_mismatched_lengths_raises(self):
        items = [_item(1, 5)]
        results = [_result(1, True, 5), _result(2, True, 3)]
        with pytest.raises(ValueError, match="same length"):
            compute_task_score(results, items)

    def test_worked_example_from_doc(self):
        """Validates the scoring formula from doc line 233-240."""
        # Scenario: 4 rubric items
        # +5 mandatory pass, +3 mandatory pass, +3 nice_to_have fail, -5 mandatory not triggered
        items = [
            _item(1, 5, "mandatory"),
            _item(2, 3, "mandatory"),
            _item(3, 3, "nice_to_have"),
            _item(4, -5, "mandatory"),
        ]
        results = [
            _result(1, True, 5),
            _result(2, True, 3),
            _result(3, False, 0),
            _result(4, False, 0),  # Not triggered = no penalty
        ]
        score = compute_task_score(results, items)
        # awarded = 5 + 3 = 8, max_total = 5 + 3 + 3 = 11
        assert score.awarded == 8
        assert score.max_total == 11
        assert score.raw_score == pytest.approx(8 / 11, abs=0.01)
        assert score.per_task_score == pytest.approx(8 / 11, abs=0.01)
        assert score.passed is True


class TestWriteScoresJsonl:
    def test_writes_correct_format(self, tmp_path: Path):
        rubric_items = [_item(1, 5), _item(2, 3)]
        items_results = [
            ScorerResult(
                number=1,
                passed=True,
                judge_rationale="File exists",
                points_awarded=5,
            ),
            ScorerResult(
                number=2,
                passed=False,
                judge_rationale="Pattern missing",
                points_awarded=0,
            ),
        ]
        score = TaskScore(
            awarded=5,
            max_total=8,
            raw_score=0.625,
            per_task_score=0.625,
            passed=False,
            items=items_results,
        )

        output_path = tmp_path / "scores.jsonl"
        write_scores_jsonl(score, output_path, rubric_items=rubric_items)

        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 6  # 2 items + 4 summary rows (incl. judge_cost_usd)

        row1 = json.loads(lines[0])
        assert row1["number"] == 1
        assert row1["passed"] is True
        assert "judge_rationale" in row1

        row2 = json.loads(lines[1])
        assert row2["number"] == 2
        assert row2["passed"] is False

        pass_row = json.loads(lines[2])
        assert pass_row == {"pass": False}

        score_row = json.loads(lines[3])
        assert score_row == {"per_task_score": 0.625}

        detail_row = json.loads(lines[4])
        assert detail_row["awarded"] == 5
        assert detail_row["max_total"] == 8
        assert detail_row["raw_score"] == 0.625

        # Judge cost summary line — 0 for purely deterministic tasks
        judge_cost_row = json.loads(lines[5])
        assert "judge_cost_usd" in judge_cost_row
        assert judge_cost_row["judge_cost_usd"] == 0.0

    def test_negative_item_passed_inverted_in_output(self, tmp_path: Path):
        """Spec L147: hallucination detected → passed=false in scores.jsonl."""
        rubric_items = [_item(1, 5), _item(2, -5, "mandatory")]
        items_results = [
            ScorerResult(number=1, passed=True, judge_rationale="OK", points_awarded=5),
            ScorerResult(
                number=2,
                passed=True,
                judge_rationale="Hallucination detected",
                points_awarded=-5,
            ),
        ]
        score = TaskScore(
            awarded=0,
            max_total=5,
            raw_score=0.0,
            per_task_score=0.0,
            passed=False,
            items=items_results,
        )

        output_path = tmp_path / "scores.jsonl"
        write_scores_jsonl(score, output_path, rubric_items=rubric_items)

        lines = output_path.read_text().strip().split("\n")
        row2 = json.loads(lines[1])
        assert row2["number"] == 2
        assert row2["passed"] is False

    def test_creates_parent_dirs(self, tmp_path: Path):
        score = TaskScore(
            awarded=0,
            max_total=5,
            raw_score=0.0,
            per_task_score=0.0,
            passed=False,
            items=[],
        )
        output_path = tmp_path / "nested" / "dir" / "scores.jsonl"
        write_scores_jsonl(score, output_path)
        assert output_path.exists()
