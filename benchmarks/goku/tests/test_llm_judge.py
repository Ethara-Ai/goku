"""Tests for benchmarks.goku.scorers.llm_judge."""

import base64
import json
from pathlib import Path
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
        # No images passed → content stays a string (text-only payload).
        assert isinstance(prompt_content, str)
        # Should be truncated (32000 + 32000 + 16000 = 80000 max for content portions)
        assert len(prompt_content) < 85000


# Minimal 1×1 PNG (transparent), valid bytes — used to verify the judge
# encodes images into image_url blocks without depending on real fixtures.
_TINY_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


class TestMultimodalPayload:
    """Regression tests for the multimodal judge fix.

    Before this fix, the judge received only text, so any rubric that
    required visual grounding (e.g. response_not_criteria HALLUCINATION
    items) was effectively the judge bluffing. These tests pin the new
    behavior: when image paths are supplied, the messages array carries
    real image_url content blocks.
    """

    @patch("benchmarks.goku.scorers.llm_judge.litellm.completion")
    def test_image_paths_produce_multimodal_content_array(
        self, mock_completion, tmp_path
    ):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {"criteria_met": False, "reasoning": "OK"}
        )
        mock_completion.return_value = mock_response

        img1 = tmp_path / "fixture.png"
        img1.write_bytes(_TINY_PNG_BYTES)

        item = _make_item("response_not_criteria", -5)
        score_llm_judge(
            item=item,
            response="Agent claims X is visible",
            file_contents="",
            trajectory="",
            input_image_paths=[str(img1)],
        )

        call_kwargs = mock_completion.call_args[1]
        content = call_kwargs["messages"][0]["content"]
        assert isinstance(content, list), \
            "With images, content must be a multimodal list, not a string"
        assert content[0]["type"] == "text"
        # Find the image block
        image_blocks = [b for b in content if b.get("type") == "image_url"]
        assert len(image_blocks) == 1, "Expected exactly one image block"
        url = image_blocks[0]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        # Confirm the base64 round-trips back to the original bytes
        decoded = base64.b64decode(url.split(",", 1)[1])
        assert decoded == _TINY_PNG_BYTES

    @patch("benchmarks.goku.scorers.llm_judge.litellm.completion")
    def test_no_image_paths_keeps_text_only_payload(self, mock_completion):
        """Backwards-compat: text-only tasks must not regress to multimodal."""
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
            # input_image_paths omitted entirely
        )
        content = mock_completion.call_args[1]["messages"][0]["content"]
        assert isinstance(content, str), \
            "Without images, content must remain a string for back-compat"

    @patch("benchmarks.goku.scorers.llm_judge.litellm.completion")
    def test_skips_non_image_files(self, mock_completion, tmp_path):
        """PDFs, MP4s, missing files, and unknown extensions are silently skipped."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {"criteria_met": False, "reasoning": "OK"}
        )
        mock_completion.return_value = mock_response

        valid_png = tmp_path / "ok.png"
        valid_png.write_bytes(_TINY_PNG_BYTES)
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        missing = tmp_path / "does_not_exist.png"

        item = _make_item("response_criteria", 5)
        score_llm_judge(
            item=item,
            response="Test",
            file_contents="",
            trajectory="",
            input_image_paths=[str(missing), str(pdf), str(valid_png)],
        )
        content = mock_completion.call_args[1]["messages"][0]["content"]
        # Only the valid PNG should produce an image block.
        image_blocks = [
            b for b in content if isinstance(b, dict) and b.get("type") == "image_url"
        ]
        assert len(image_blocks) == 1

    @patch("benchmarks.goku.scorers.llm_judge.litellm.completion")
    def test_prompt_notes_image_presence(self, mock_completion, tmp_path):
        """The prompt should tell the judge whether images are attached."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {"criteria_met": False, "reasoning": "OK"}
        )
        mock_completion.return_value = mock_response

        img = tmp_path / "f.png"
        img.write_bytes(_TINY_PNG_BYTES)

        item = _make_item("response_not_criteria", -5)
        score_llm_judge(
            item=item, response="r", file_contents="", trajectory="",
            input_image_paths=[str(img)],
        )
        with_img_text = mock_completion.call_args[1]["messages"][0]["content"][0]["text"]
        assert "attached" in with_img_text.lower()

        score_llm_judge(
            item=item, response="r", file_contents="", trajectory="",
        )
        without_img_text = mock_completion.call_args[1]["messages"][0]["content"]
        assert "no input images" in without_img_text.lower()
