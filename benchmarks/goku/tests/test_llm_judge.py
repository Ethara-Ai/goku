"""Tests for benchmarks.goku.scorers.llm_judge."""

import json
from unittest.mock import MagicMock, patch

import pytest

from benchmarks.goku.models import RubricItem, RubricType
from benchmarks.goku.scorers.llm_judge import score_llm_judge


def _make_item(item_type: RubricType, points: int = 5) -> RubricItem:
    """Helper to create a judge-type rubric item."""
    return RubricItem(
        number=1,
        type=item_type,
        category="CORRECTNESS" if points > 0 else "HALLUCINATION",
        points=points,
        importance="mandatory",
        criterion="Test criterion for evaluation",
    )


class TestResponseCriteria:
    @patch("benchmarks.goku.scorers.llm_judge.litellm.completion")
    def test_criteria_met(self, mock_completion):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {"criteria_met": True, "reasoning": "Criterion clearly satisfied"}
        )
        mock_completion.return_value = mock_response

        item = _make_item("response_criteria", 5)
        result = score_llm_judge(
            item=item,
            response="The agent identified 12 items correctly.",
            file_contents='--- inventory.json ---\n{"items": [...]}',
            trajectory="[0] ActionEvent\n    Action: bash\n    Command: ...",
        )

        assert result.passed is True
        assert result.points_awarded == 5
        assert "satisfied" in result.judge_rationale

    @patch("benchmarks.goku.scorers.llm_judge.litellm.completion")
    def test_criteria_not_met(self, mock_completion):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {"criteria_met": False, "reasoning": "Criterion not satisfied"}
        )
        mock_completion.return_value = mock_response

        item = _make_item("response_criteria", 3)
        result = score_llm_judge(
            item=item,
            response="Incomplete response",
            file_contents="(no output files)",
            trajectory="",
        )

        assert result.passed is False
        assert result.points_awarded == 0


class TestResponseNotCriteria:
    @patch("benchmarks.goku.scorers.llm_judge.litellm.completion")
    def test_hallucination_detected(self, mock_completion):
        """criteria_met=True means hallucination IS present → penalty."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {"criteria_met": True, "reasoning": "Agent fabricated data"}
        )
        mock_completion.return_value = mock_response

        item = _make_item("response_not_criteria", -5)
        result = score_llm_judge(
            item=item,
            response="The item is from 1850 and made by Tiffany & Co.",
            file_contents="",
            trajectory="",
        )

        # passed=True means criterion matched = hallucination detected
        assert result.passed is True
        assert result.points_awarded == -5

    @patch("benchmarks.goku.scorers.llm_judge.litellm.completion")
    def test_no_hallucination(self, mock_completion):
        """criteria_met=False means no hallucination → no penalty."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {"criteria_met": False, "reasoning": "No hallucination detected"}
        )
        mock_completion.return_value = mock_response

        item = _make_item("response_not_criteria", -5)
        result = score_llm_judge(
            item=item,
            response="Accurate information from the image",
            file_contents="",
            trajectory="",
        )

        assert result.passed is False
        assert result.points_awarded == 0


class TestErrorHandling:
    @patch("benchmarks.goku.scorers.llm_judge.litellm.completion")
    def test_invalid_json_response(self, mock_completion):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Not valid JSON at all"
        mock_completion.return_value = mock_response

        item = _make_item("response_criteria", 5)
        result = score_llm_judge(
            item=item,
            response="Test",
            file_contents="",
            trajectory="",
        )

        # Should fail gracefully
        assert result.passed is False
        assert result.points_awarded == 0
        assert "invalid JSON" in result.judge_rationale

    @patch("benchmarks.goku.scorers.llm_judge.litellm.completion")
    def test_api_call_failure(self, mock_completion):
        mock_completion.side_effect = Exception("API timeout")

        item = _make_item("response_criteria", 5)
        result = score_llm_judge(
            item=item,
            response="Test",
            file_contents="",
            trajectory="",
        )

        assert result.passed is False
        assert result.points_awarded == 0
        assert "failed" in result.judge_rationale

    def test_rejects_deterministic_type(self):
        item = RubricItem(
            number=1,
            type="probe_file_exists",
            category="FORMAT",
            points=5,
            importance="mandatory",
            criterion="Test",
            paths=["test.json"],
        )
        with pytest.raises(ValueError, match="not LLM-judged"):
            score_llm_judge(
                item=item,
                response="Test",
                file_contents="",
                trajectory="",
            )


class TestPromptConstruction:
    @patch("benchmarks.goku.scorers.llm_judge.litellm.completion")
    def test_uses_correct_model(self, mock_completion):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {"criteria_met": True, "reasoning": "OK"}
        )
        mock_completion.return_value = mock_response

        item = _make_item("response_criteria", 5)
        score_llm_judge(
            item=item,
            response="Test",
            file_contents="",
            trajectory="",
            judge_model="custom/model",
            judge_api_key="test-key",
        )

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["model"] == "custom/model"
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["temperature"] == 0.0

    @patch("benchmarks.goku.scorers.llm_judge.litellm.completion")
    def test_truncates_long_inputs(self, mock_completion):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {"criteria_met": True, "reasoning": "OK"}
        )
        mock_completion.return_value = mock_response

        item = _make_item("response_criteria", 5)
        # Pass very long inputs
        score_llm_judge(
            item=item,
            response="x" * 20000,
            file_contents="y" * 20000,
            trajectory="z" * 10000,
        )

        # Verify the prompt was constructed (call should succeed)
        assert mock_completion.called
        call_kwargs = mock_completion.call_args[1]
        prompt_content = call_kwargs["messages"][0]["content"]
        # Should be truncated (32000 + 32000 + 16000 = 80000 max for content portions)
        assert len(prompt_content) < 85000
