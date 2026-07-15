"""Unit tests for benchmarks.goku.claudecode_runner (Claude Code backend).

Covers the pure logic that does not touch Docker or the OAuth bridge:
  1.  _build_prompt includes a media manifest when input files are present
  2.  _build_prompt omits the manifest for text-only tasks
  3.  _parse_stream_json extracts the final ``result`` text + usage/cost
  4.  _parse_stream_json renders tool_use / tool_result into the trajectory
  5.  _parse_stream_json falls back to the last assistant text when there is
      no ``result`` line (e.g. a mid-stream timeout)
  6.  _parse_stream_json raises ClaudeCodeUnavailableError on unparseable output
"""

from __future__ import annotations

import json

from benchmarks.goku.claudecode_runner import (
    ClaudeCodeUnavailableError,
    _build_prompt,
    _parse_stream_json,
)


def _stream(lines: list[dict]) -> str:
    return "\n".join(json.dumps(line) for line in lines)


FULL_STREAM = [
    {"type": "system", "subtype": "init", "session_id": "s1"},
    {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "name": "Read",
                    "input": {"file_path": "/workspace/chart.png"},
                }
            ],
        },
    },
    {
        "type": "user",
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "content": "image 1024x768"}],
        },
    },
    {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "The chart shows Q3 revenue up 12%."}],
        },
    },
    {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "Final: Q3 revenue up 12%.",
        "num_turns": 3,
        "total_cost_usd": 0.0421,
        "usage": {
            "input_tokens": 1500,
            "output_tokens": 220,
            "cache_read_input_tokens": 800,
            "cache_creation_input_tokens": 100,
        },
    },
]


def test_build_prompt_includes_media_manifest():
    prompt = _build_prompt("Do the task.", ["chart.png", "data.pdf"])
    assert "/workspace/chart.png" in prompt
    assert "/workspace/data.pdf" in prompt
    assert "Output requirements" in prompt


def test_build_prompt_text_only_has_no_manifest():
    prompt = _build_prompt("No media task.", [])
    assert "Task input files" not in prompt
    assert "Output requirements" in prompt


def test_parse_stream_json_extracts_result_and_usage():
    res = _parse_stream_json(_stream(FULL_STREAM), "task_x")
    assert res.response_text == "Final: Q3 revenue up 12%."
    assert res.is_error is False
    assert res.metrics["input_tokens"] == 1500
    assert res.metrics["output_tokens"] == 220
    assert res.metrics["cache_read_tokens"] == 800
    assert res.metrics["cache_write_tokens"] == 100
    assert res.metrics["cost_usd"] == 0.0421
    assert res.metrics["num_turns"] == 3


def test_parse_stream_json_renders_trajectory():
    res = _parse_stream_json(_stream(FULL_STREAM), "task_x")
    assert "tool_use Read" in res.trajectory
    assert "tool_result" in res.trajectory
    assert "assistant:" in res.trajectory


def test_parse_stream_json_falls_back_to_last_assistant_text():
    # No trailing "result" line — e.g. a mid-stream timeout.
    res = _parse_stream_json(_stream(FULL_STREAM[:4]), "task_y")
    assert res.response_text == "The chart shows Q3 revenue up 12%."


def test_parse_stream_json_raises_on_unparseable_output():
    try:
        _parse_stream_json("garbage, no json here", "task_z")
    except ClaudeCodeUnavailableError:
        return
    raise AssertionError("expected ClaudeCodeUnavailableError")
