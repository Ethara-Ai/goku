"""Tests for benchmarks.goku.models."""

import pytest
from pydantic import ValidationError

from benchmarks.goku.models import (
    BenchmarkReport,
    GokuEvalInstance,
    RubricItem,
    ScorerResult,
    TaskScore,
)


class TestRubricItem:
    """Tests for RubricItem model validation."""

    def test_valid_probe_file_exists(self):
        item = RubricItem(
            number=1,
            type="probe_file_exists",
            category="FORMAT",
            points=5,
            importance="mandatory",
            criterion="Output file must exist",
            paths=["output.json"],
        )
        assert item.type == "probe_file_exists"
        assert item.points == 5

    def test_valid_response_criteria(self):
        item = RubricItem(
            number=2,
            type="response_criteria",
            category="MM_REASONING",
            points=3,
            importance="nice_to_have",
            criterion="Agent correctly identifies the object",
        )
        assert item.type == "response_criteria"
        assert item.category == "MM_REASONING"

    def test_valid_response_not_criteria(self):
        item = RubricItem(
            number=3,
            type="response_not_criteria",
            category="HALLUCINATION",
            points=-5,
            importance="mandatory",
            criterion="Agent fabricates information not in the image",
        )
        assert item.points == -5

    def test_all_eight_types(self):
        types = [
            "probe_file_exists",
            "probe_file_contains",
            "probe_dir_exists",
            "shell_succeeds_real",
            "response_contains",
            "response_regex_present",
            "response_criteria",
            "response_not_criteria",
        ]
        for t in types:
            item = RubricItem(
                number=1,
                type=t,
                category="CORRECTNESS",
                points=5,
                importance="mandatory",
                criterion="Test criterion",
            )
            assert item.type == t

    def test_all_six_categories(self):
        cats = [
            "CORRECTNESS",
            "FORMAT",
            "BEHAVIOR",
            "MM_REASONING",
            "HALLUCINATION",
            "STYLE",
        ]
        for c in cats:
            item = RubricItem(
                number=1,
                type="response_criteria",
                category=c,
                points=3,
                importance="mandatory",
                criterion="Test",
            )
            assert item.category == c

    def test_invalid_type_rejected(self):
        with pytest.raises(ValidationError):
            RubricItem(
                number=1,
                type="invalid_type",
                category="CORRECTNESS",
                points=5,
                importance="mandatory",
                criterion="Test",
            )

    def test_invalid_category_rejected(self):
        with pytest.raises(ValidationError):
            RubricItem(
                number=1,
                type="response_criteria",
                category="INVALID",
                points=5,
                importance="mandatory",
                criterion="Test",
            )

    def test_number_must_be_positive(self):
        with pytest.raises(ValidationError):
            RubricItem(
                number=0,
                type="response_criteria",
                category="CORRECTNESS",
                points=5,
                importance="mandatory",
                criterion="Test",
            )

    def test_empty_criterion_rejected(self):
        with pytest.raises(ValidationError):
            RubricItem(
                number=1,
                type="response_criteria",
                category="CORRECTNESS",
                points=5,
                importance="mandatory",
                criterion="",
            )

    def test_source_field_accepts_dict(self):
        item = RubricItem(
            number=1,
            type="response_criteria",
            category="CORRECTNESS",
            points=5,
            importance="mandatory",
            criterion="Test",
            source={"asset": "image1.png", "section": "top-left"},
        )
        assert item.source == {"asset": "image1.png", "section": "top-left"}

    def test_source_field_accepts_list(self):
        item = RubricItem(
            number=1,
            type="response_criteria",
            category="CORRECTNESS",
            points=5,
            importance="mandatory",
            criterion="Test",
            source=[
                {"asset": "img1.png", "quote": "text"},
                {"asset": "img2.png", "quote": "other"},
            ],
        )
        assert isinstance(item.source, list)
        assert len(item.source) == 2


class TestScorerResult:
    """Tests for ScorerResult model."""

    def test_basic_creation(self):
        result = ScorerResult(
            number=1,
            passed=True,
            judge_rationale="File exists at expected path",
            points_awarded=5,
        )
        assert result.passed is True
        assert result.points_awarded == 5

    def test_serialization_matches_scores_jsonl(self):
        result = ScorerResult(
            number=3,
            passed=False,
            judge_rationale="Pattern not found",
            points_awarded=0,
        )
        d = result.model_dump()
        assert "number" in d
        assert "passed" in d
        assert "judge_rationale" in d
        assert "points_awarded" in d


class TestTaskScore:
    """Tests for TaskScore model."""

    def test_per_task_score_clamped(self):
        score = TaskScore(
            awarded=10,
            max_total=10,
            raw_score=1.0,
            per_task_score=1.0,
            passed=True,
            items=[],
        )
        assert score.per_task_score == 1.0

    def test_per_task_score_zero(self):
        score = TaskScore(
            awarded=0,
            max_total=10,
            raw_score=0.0,
            per_task_score=0.0,
            passed=False,
            items=[],
        )
        assert score.per_task_score == 0.0

    def test_per_task_score_rejects_negative(self):
        with pytest.raises(ValidationError):
            TaskScore(
                awarded=-5,
                max_total=10,
                raw_score=-0.5,
                per_task_score=-0.5,
                passed=False,
                items=[],
            )

    def test_per_task_score_rejects_above_one(self):
        with pytest.raises(ValidationError):
            TaskScore(
                awarded=15,
                max_total=10,
                raw_score=1.5,
                per_task_score=1.5,
                passed=True,
                items=[],
            )


class TestGokuEvalInstance:
    """Tests for GokuEvalInstance model."""

    def test_basic_creation(self):
        inst = GokuEvalInstance(
            id="task_e25b6d",
            instruction="Identify items in the pantry",
            rubric_items=[],
            input_files=["/path/to/pantry1.png"],
        )
        assert inst.id == "task_e25b6d"
        assert len(inst.input_files) == 1

    def test_default_input_files(self):
        inst = GokuEvalInstance(
            id="task_abc",
            instruction="Do something",
            rubric_items=[],
        )
        assert inst.input_files == []


class TestBenchmarkReport:
    """Tests for BenchmarkReport model."""

    def test_basic_creation(self):
        report = BenchmarkReport(
            model_id="claude-opus-4.7",
            mean_per_task_score=0.75,
            mean_raw_score=0.68,
            pass_rate=0.80,
            pass_at_3=0.90,
            pass_hat_3=0.60,
            total_tasks=14,
            total_cost_usd=25.50,
        )
        assert report.total_tasks == 14
        assert report.mean_per_task_score == 0.75
