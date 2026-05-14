"""Tests for benchmarks.goku.benchmark_report."""

import pytest

from benchmarks.goku.benchmark_report import (
    generate_report,
    pass_at_n,
    pass_hat_n,
)
from benchmarks.goku.models import TaskScore


def _score(per_task: float, raw: float, passed: bool) -> TaskScore:
    """Helper to create a TaskScore."""
    return TaskScore(
        awarded=int(per_task * 10),
        max_total=10,
        raw_score=raw,
        per_task_score=per_task,
        passed=passed,
        items=[],
    )


class TestPassAtN:
    def test_all_pass(self):
        assert pass_at_n(3, 3, 3) == 1.0

    def test_none_pass(self):
        assert pass_at_n(3, 0, 3) == 0.0

    def test_one_of_three_passes(self):
        # pass@3 with 1/3 passes: 1 - C(2,3)/C(3,3) = 1 - 0 = 1.0
        # Actually: 1 - prod(2-i)/(3-i) for i=0..2 = 1 - (2/3 * 1/2 * 0/1)
        # n=3, c=1, k=3: at i=2, n-c-i = 3-1-2 = 0 → return 1.0
        result = pass_at_n(3, 1, 3)
        assert result == 1.0

    def test_one_of_three_pass_at_1(self):
        # pass@1: 1 - (n-c)/(n) = 1 - 2/3 ≈ 0.333
        result = pass_at_n(3, 1, 1)
        assert result == pytest.approx(1 / 3, abs=0.01)

    def test_two_of_three_pass_at_1(self):
        # pass@1: 1 - 1/3 ≈ 0.667
        result = pass_at_n(3, 2, 1)
        assert result == pytest.approx(2 / 3, abs=0.01)

    def test_k_greater_than_n(self):
        # k clamped to n
        result = pass_at_n(3, 2, 5)
        assert result == 1.0  # At least 1 pass in 3 tries


class TestPassHatN:
    def test_all_pass(self):
        assert pass_hat_n(3, 3) is True

    def test_not_all_pass(self):
        assert pass_hat_n(3, 2) is False

    def test_none_pass(self):
        assert pass_hat_n(3, 0) is False


class TestGenerateReport:
    def test_empty_scores(self):
        report = generate_report({}, "test-model")
        assert report.total_tasks == 0
        assert report.mean_per_task_score == 0.0
        assert report.pass_rate == 0.0

    def test_single_task_all_pass(self):
        task_scores = {
            "task_1": [
                _score(0.8, 0.8, True),
                _score(0.9, 0.9, True),
                _score(0.85, 0.85, True),
            ],
        }
        report = generate_report(task_scores, "claude-opus-4.7")
        assert report.model_id == "claude-opus-4.7"
        assert report.total_tasks == 1
        assert report.mean_per_task_score == pytest.approx(0.85, abs=0.01)
        assert report.pass_rate == 1.0
        assert report.pass_hat_3 == 1.0  # All 3 passed

    def test_multiple_tasks_mixed(self):
        task_scores = {
            "task_1": [
                _score(0.9, 0.9, True),
                _score(0.8, 0.8, True),
                _score(0.7, 0.7, True),
            ],
            "task_2": [
                _score(0.3, 0.3, False),
                _score(0.4, 0.4, False),
                _score(0.5, 0.5, True),
            ],
        }
        report = generate_report(task_scores, "gpt-5.5")
        assert report.total_tasks == 2
        # Task 1 mean: 0.8, Task 2 mean: 0.4 → overall: 0.6
        assert report.mean_per_task_score == pytest.approx(0.6, abs=0.01)
        # pass_rate: task_1 = 3/3 = 1.0, task_2 = 1/3 = 0.333 → mean = 0.667
        assert report.pass_rate == pytest.approx(2 / 3, abs=0.01)
        # pass^3: task_1 all pass (3/3), task_2 not all (1/3) → 0.5
        assert report.pass_hat_3 == 0.5

    def test_cost_tracking(self):
        task_scores = {"task_1": [_score(0.5, 0.5, True)]}
        report = generate_report(task_scores, "model", total_cost_usd=12.50)
        assert report.total_cost_usd == 12.50
