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
    def test_pdf_uses_document_block_for_anthropic_judge(
        self, mock_completion, tmp_path
    ):
        """Claude-family judge (native PDF) → Anthropic-style ``document`` block."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {"criteria_met": False, "reasoning": "OK"}
        )
        mock_completion.return_value = mock_response

        pdf = tmp_path / "spec.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake content")

        item = _make_item("response_criteria", 5)
        score_llm_judge(
            item=item, response="r", file_contents="", trajectory="",
            input_image_paths=[str(pdf)],
            judge_model="bedrock/converse/anthropic.claude-opus-4-7",
        )
        content = mock_completion.call_args[1]["messages"][0]["content"]
        assert isinstance(content, list)
        doc_blocks = [
            b for b in content if isinstance(b, dict) and b.get("type") == "document"
        ]
        assert len(doc_blocks) == 1
        src = doc_blocks[0]["source"]
        assert src["type"] == "base64"
        assert src["media_type"] == "application/pdf"
        assert base64.b64decode(src["data"]) == b"%PDF-1.4 fake content"

    @patch("benchmarks.goku.scorers.llm_judge.pdf_to_page_images")
    @patch("benchmarks.goku.scorers.llm_judge.litellm.completion")
    def test_pdf_uses_openai_file_block_for_gpt_judge(
        self, mock_completion, mock_pdf_render, tmp_path
    ):
        """GPT/Gemini judge (native PDF) → OpenAI-style ``file`` block."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {"criteria_met": True, "reasoning": "OK"}
        )
        mock_completion.return_value = mock_response

        pdf = tmp_path / "spec.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake content")

        item = _make_item("response_criteria", 5)
        score_llm_judge(
            item=item, response="r", file_contents="", trajectory="",
            input_image_paths=[str(pdf)],
            judge_model="openai/gpt-5.5",
        )
        # GPT path uses native 'file' block, NOT the rendering fallback.
        mock_pdf_render.assert_not_called()
        content = mock_completion.call_args[1]["messages"][0]["content"]
        file_blocks = [
            b for b in content if isinstance(b, dict) and b.get("type") == "file"
        ]
        assert len(file_blocks) == 1
        assert file_blocks[0]["file"]["filename"] == "spec.pdf"
        assert file_blocks[0]["file"]["file_data"].startswith(
            "data:application/pdf;base64,"
        )

    @patch("benchmarks.goku.scorers.llm_judge.pdf_to_page_images")
    @patch("benchmarks.goku.scorers.llm_judge.litellm.completion")
    def test_pdf_renders_to_images_for_kimi_judge(
        self, mock_completion, mock_pdf_render, tmp_path
    ):
        """Kimi judge (no native PDF) → pypdfium2 renders pages → image_url blocks."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {"criteria_met": True, "reasoning": "OK"}
        )
        mock_completion.return_value = mock_response

        # Mock the renderer to return two fake page PNGs (we don't want the
        # test to depend on a real PDF parser succeeding).
        page1 = tmp_path / "page_001.png"
        page2 = tmp_path / "page_002.png"
        page1.write_bytes(_TINY_PNG_BYTES)
        page2.write_bytes(_TINY_PNG_BYTES)
        mock_pdf_render.return_value = [page1, page2]

        pdf = tmp_path / "spec.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake content")

        item = _make_item("response_not_criteria", -5)
        score_llm_judge(
            item=item, response="r", file_contents="", trajectory="",
            input_image_paths=[str(pdf)],
            judge_model="bedrock/converse/moonshotai.kimi-k2.5",
        )
        mock_pdf_render.assert_called_once()
        content = mock_completion.call_args[1]["messages"][0]["content"]
        image_blocks = [
            b for b in content if isinstance(b, dict) and b.get("type") == "image_url"
        ]
        # 2 pages → 2 image_url blocks; no document/file block on this path.
        assert len(image_blocks) == 2
        assert not any(b.get("type") in ("document", "file") for b in content)

    @patch("benchmarks.goku.scorers.llm_judge.video_to_keyframes")
    @patch("benchmarks.goku.scorers.llm_judge.litellm.completion")
    def test_video_renders_to_keyframes_uniformly(
        self, mock_completion, mock_keyframes, tmp_path
    ):
        """Videos uniformly extract keyframes → image_url blocks for any judge."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {"criteria_met": False, "reasoning": "OK"}
        )
        mock_completion.return_value = mock_response

        frame_paths = []
        for i in range(3):
            fp = tmp_path / f"frame_{i+1:03d}.png"
            fp.write_bytes(_TINY_PNG_BYTES)
            frame_paths.append(fp)
        mock_keyframes.return_value = frame_paths

        video = tmp_path / "demo.mp4"
        video.write_bytes(b"\x00\x00\x00\x18ftypmp42 fake mp4 header")

        item = _make_item("response_not_criteria", -5)
        score_llm_judge(
            item=item, response="r", file_contents="", trajectory="",
            input_image_paths=[str(video)],
            judge_model="bedrock/converse/moonshotai.kimi-k2.5",
        )
        mock_keyframes.assert_called_once()
        content = mock_completion.call_args[1]["messages"][0]["content"]
        image_blocks = [
            b for b in content if isinstance(b, dict) and b.get("type") == "image_url"
        ]
        assert len(image_blocks) == 3  # 3 keyframes

    @patch("benchmarks.goku.scorers.llm_judge.video_to_keyframes")
    @patch("benchmarks.goku.scorers.llm_judge.litellm.completion")
    def test_video_failure_is_flagged_in_rationale(
        self, mock_completion, mock_keyframes, tmp_path
    ):
        """If keyframe extraction fails (corrupt video etc.), the gap surfaces in the rationale."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {"criteria_met": False, "reasoning": "verdict-text"}
        )
        mock_completion.return_value = mock_response

        # Simulate ffmpeg failure (corrupt file, missing binary, etc.)
        mock_keyframes.side_effect = RuntimeError("ffmpeg exit 1: bad data")

        video = tmp_path / "demo.mp4"
        video.write_bytes(b"\x00\x00\x00\x18ftypmp42 fake mp4 header")

        item = _make_item("response_not_criteria", -5)
        result = score_llm_judge(
            item=item, response="r", file_contents="", trajectory="",
            input_image_paths=[str(video)],
            judge_model="bedrock/converse/moonshotai.kimi-k2.5",
        )
        # Video produced no blocks; the warning surfaces in the rationale.
        content = mock_completion.call_args[1]["messages"][0]["content"]
        assert isinstance(content, str), "no media → text-only payload"
        assert "JUDGE MEDIA WARNINGS" in result.judge_rationale
        assert "demo.mp4" in result.judge_rationale

    @patch("benchmarks.goku.scorers.llm_judge.litellm.completion")
    def test_skips_unknown_and_missing_files(self, mock_completion, tmp_path):
        """Missing files and unknown extensions are flagged, not silently dropped."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {"criteria_met": False, "reasoning": "OK"}
        )
        mock_completion.return_value = mock_response

        valid_png = tmp_path / "ok.png"
        valid_png.write_bytes(_TINY_PNG_BYTES)
        unknown = tmp_path / "weird.xyz"
        unknown.write_bytes(b"junk")
        missing = tmp_path / "does_not_exist.png"

        item = _make_item("response_criteria", 5)
        result = score_llm_judge(
            item=item, response="r", file_contents="", trajectory="",
            input_image_paths=[str(missing), str(unknown), str(valid_png)],
        )
        content = mock_completion.call_args[1]["messages"][0]["content"]
        image_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "image_url"]
        # Only the valid PNG goes through.
        assert len(image_blocks) == 1
        # Unknown extension is flagged (missing is silently dropped — no
        # meaningful information to surface; the agent already failed
        # to produce the file in the first place).
        assert "weird.xyz" in result.judge_rationale

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
        assert "no attached media" in without_img_text.lower()


class TestInputOutputMediaSeparation:
    """Critical invariant tests: the judge must NEVER conflate INPUT media
    (task fixture) with OUTPUT media (agent's work product).

    Wrong conflation would cause:
      * Hallucination rubrics to fail false-positively when the agent's
        OUTPUT image legitimately depicts something different from INPUT.
      * Output-correctness rubrics ("did the agent's PDF contain X?") to be
        graded against the INPUT PDF instead of the OUTPUT PDF.
    """

    @patch("benchmarks.goku.scorers.llm_judge.litellm.completion")
    def test_input_and_output_media_both_attached_with_labels(
        self, mock_completion, tmp_path
    ):
        """Both INPUT and OUTPUT media reach the judge with explicit labels."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {"criteria_met": True, "reasoning": "OK"}
        )
        mock_completion.return_value = mock_response

        input_img = tmp_path / "fixture.png"
        input_img.write_bytes(_TINY_PNG_BYTES)
        output_img = tmp_path / "agent_output.png"
        output_img.write_bytes(_TINY_PNG_BYTES)

        item = _make_item("response_criteria", 5)
        score_llm_judge(
            item=item, response="r", file_contents="", trajectory="",
            input_image_paths=[str(input_img)],
            output_media_paths=[str(output_img)],
        )
        content = mock_completion.call_args[1]["messages"][0]["content"]
        assert isinstance(content, list)

        # Find the text labels (must be separate text blocks, not embedded
        # in the main prompt — that way they sit adjacent to the relevant
        # media blocks).
        text_blocks = [
            b["text"] for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        label_text = "\n".join(text_blocks).lower()
        assert "=== input media" in label_text, (
            "INPUT MEDIA section header missing — judge would conflate "
            "input and output media"
        )
        assert "=== output media" in label_text, (
            "OUTPUT MEDIA section header missing — judge would conflate "
            "input and output media"
        )
        # Both images make it through.
        image_blocks = [
            b for b in content if isinstance(b, dict) and b.get("type") == "image_url"
        ]
        assert len(image_blocks) == 2

    @patch("benchmarks.goku.scorers.llm_judge.litellm.completion")
    def test_input_label_appears_BEFORE_input_blocks(
        self, mock_completion, tmp_path
    ):
        """The INPUT header must appear in the content stream BEFORE the
        input image blocks (and similarly OUTPUT header before output
        blocks), so the judge sees each label adjacent to the media it
        labels. If the order is wrong the judge might attach labels to the
        wrong section."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {"criteria_met": True, "reasoning": "OK"}
        )
        mock_completion.return_value = mock_response

        in1 = tmp_path / "in1.png"
        in1.write_bytes(_TINY_PNG_BYTES)
        in2 = tmp_path / "in2.png"
        in2.write_bytes(_TINY_PNG_BYTES)
        out1 = tmp_path / "out1.png"
        out1.write_bytes(_TINY_PNG_BYTES)

        item = _make_item("response_criteria", 5)
        score_llm_judge(
            item=item, response="r", file_contents="", trajectory="",
            input_image_paths=[str(in1), str(in2)],
            output_media_paths=[str(out1)],
        )
        content = mock_completion.call_args[1]["messages"][0]["content"]

        # Find indices of the section headers and image blocks
        input_header_idx = None
        output_header_idx = None
        first_image_after_input = None
        first_image_after_output = None
        for i, b in enumerate(content):
            if not isinstance(b, dict):
                continue
            text = b.get("text", "")
            if "=== INPUT MEDIA" in text:
                input_header_idx = i
            elif "=== OUTPUT MEDIA" in text:
                output_header_idx = i
            elif b.get("type") == "image_url":
                if input_header_idx is not None and first_image_after_input is None and (output_header_idx is None or i < output_header_idx):
                    first_image_after_input = i
                elif output_header_idx is not None and first_image_after_output is None:
                    first_image_after_output = i

        assert input_header_idx is not None
        assert output_header_idx is not None
        # Input header comes before output header
        assert input_header_idx < output_header_idx
        # The first image after the INPUT header is one of the input images
        assert first_image_after_input is not None
        assert first_image_after_input > input_header_idx
        assert first_image_after_input < output_header_idx
        # And after the OUTPUT header comes an output image
        assert first_image_after_output is not None
        assert first_image_after_output > output_header_idx

    @patch("benchmarks.goku.scorers.llm_judge.litellm.completion")
    def test_only_input_media_no_output_section_header(
        self, mock_completion, tmp_path
    ):
        """If a task has no output media, the dedicated OUTPUT section-header
        block must not appear in the content array (only the input one).

        We match the unique parenthetical of the section header — the prompt
        body's `media_note` instructions ALSO mention the label strings to
        teach the judge what they mean, so a coarse substring match would
        false-positive.
        """
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {"criteria_met": True, "reasoning": "OK"}
        )
        mock_completion.return_value = mock_response

        in1 = tmp_path / "in1.png"
        in1.write_bytes(_TINY_PNG_BYTES)
        item = _make_item("response_criteria", 5)
        score_llm_judge(
            item=item, response="r", file_contents="", trajectory="",
            input_image_paths=[str(in1)],
            # No output_media_paths
        )
        content = mock_completion.call_args[1]["messages"][0]["content"]
        input_header_blocks = [
            b for b in content
            if isinstance(b, dict) and b.get("type") == "text"
            and "=== INPUT MEDIA (the task fixture" in b.get("text", "")
        ]
        output_header_blocks = [
            b for b in content
            if isinstance(b, dict) and b.get("type") == "text"
            and "=== OUTPUT MEDIA (files the agent" in b.get("text", "")
        ]
        assert len(input_header_blocks) == 1
        assert len(output_header_blocks) == 0

    @patch("benchmarks.goku.scorers.llm_judge.litellm.completion")
    def test_only_output_media_no_input_section_header(
        self, mock_completion, tmp_path
    ):
        """Symmetric: only OUTPUT media → no INPUT section-header block."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {"criteria_met": True, "reasoning": "OK"}
        )
        mock_completion.return_value = mock_response

        out1 = tmp_path / "out1.png"
        out1.write_bytes(_TINY_PNG_BYTES)
        item = _make_item("response_criteria", 5)
        score_llm_judge(
            item=item, response="r", file_contents="", trajectory="",
            output_media_paths=[str(out1)],
            # No input_image_paths
        )
        content = mock_completion.call_args[1]["messages"][0]["content"]
        input_header_blocks = [
            b for b in content
            if isinstance(b, dict) and b.get("type") == "text"
            and "=== INPUT MEDIA (the task fixture" in b.get("text", "")
        ]
        output_header_blocks = [
            b for b in content
            if isinstance(b, dict) and b.get("type") == "text"
            and "=== OUTPUT MEDIA (files the agent" in b.get("text", "")
        ]
        assert len(input_header_blocks) == 0
        assert len(output_header_blocks) == 1

    @patch("benchmarks.goku.scorers.llm_judge.litellm.completion")
    def test_input_and_output_warnings_distinguished_in_rationale(
        self, mock_completion, tmp_path
    ):
        """If files drop from BOTH input AND output, the rationale must say
        which section each came from — otherwise the operator can't tell
        whether to fix a dataset file or an agent-produced file."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {"criteria_met": False, "reasoning": "test verdict"}
        )
        mock_completion.return_value = mock_response

        bad_input = tmp_path / "bad_input.xyz"
        bad_input.write_bytes(b"junk")
        bad_output = tmp_path / "bad_output.xyz"
        bad_output.write_bytes(b"junk")

        item = _make_item("response_not_criteria", -5)
        result = score_llm_judge(
            item=item, response="r", file_contents="", trajectory="",
            input_image_paths=[str(bad_input)],
            output_media_paths=[str(bad_output)],
        )
        # Warnings appear in the rationale, labeled INPUT / OUTPUT
        assert "INPUT:" in result.judge_rationale
        assert "bad_input.xyz" in result.judge_rationale
        assert "OUTPUT:" in result.judge_rationale
        assert "bad_output.xyz" in result.judge_rationale
