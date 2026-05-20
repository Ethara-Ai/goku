"""Unit tests for benchmarks.goku.response_extractor.

Covers every edge case enumerated in the design doc:
  1.  FinishAction with non-empty action.message → use it
  2.  FinishAction with empty action.message but tool_call.arguments (JSON
      string) carries message → use the parsed message
  3.  FinishAction with empty action.message but tool_call.arguments (dict)
      carries message → use the dict message
  4.  FinishAction present but all message fields empty → fall back to
      last-agent-text scan
  5.  No FinishAction at all → fall back to last-agent-text scan
  6.  thought as list[{type:'text', text:...}] → extract text blocks only
  7.  thought as plain string → use as-is
  8.  content field instead of thought → handled identically
  9.  Empty history → returns ""
  10. Missing output.jsonl file → returns ""
  11. output.jsonl has malformed JSON lines → skips them
  12. Multiple records, filter by instance_id → returns matching one
  13. write_response_md with empty text writes the stub
  14. write_response_md with empty text and fallback_stub=False writes
      empty file
  15. write_response_md creates parent directories
  16. Unicode content (emoji) round-trips through UTF-8
  17. Non-FinishAction agent events with tool_use blocks should not be
      mistaken for the final response when a FinishAction exists later
  18. FinishAction earlier in history is correctly preferred over an
      earlier abnormal-fallback text (LAST FinishAction wins)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.goku.response_extractor import (
    NO_RESPONSE_STUB,
    extract_event_text,
    extract_final_response,
    extract_final_response_from_jsonl,
    write_response_md,
)


# ---------- helpers ----------


def _agent_event(**kwargs: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"source": "agent"}
    base.update(kwargs)
    return base


def _finish_event(message: str | None, args_message: str | None = None,
                  args_as_dict: bool = False):
    """Construct a FinishAction event mimicking OpenHands SDK shape."""
    event = _agent_event(
        action={"kind": "FinishAction", "message": message or ""},
        tool_name="finish",
    )
    if args_message is not None:
        if args_as_dict:
            event["tool_call"] = {"arguments": {"message": args_message}}
        else:
            event["tool_call"] = {
                "arguments": json.dumps({"message": args_message})
            }
    return event


def _thought_event(thought):
    return _agent_event(thought=thought, action={"kind": "ActionEvent"})


# ---------- extract_event_text ----------


def test_extract_event_text_thought_list_blocks():
    e = _thought_event([
        {"type": "text", "text": "Hello"},
        {"type": "text", "text": "World"},
    ])
    assert extract_event_text(e) == "Hello\n\nWorld"


def test_extract_event_text_thought_string():
    e = _thought_event("Plain string thought")
    assert extract_event_text(e) == "Plain string thought"


def test_extract_event_text_content_field():
    e = _agent_event(content="From content field")
    assert extract_event_text(e) == "From content field"


def test_extract_event_text_content_list_blocks():
    e = _agent_event(content=[
        {"type": "text", "text": "Block A"},
        {"type": "tool_use", "name": "bash"},  # should be skipped
        {"type": "text", "text": "Block B"},
    ])
    assert extract_event_text(e) == "Block A\n\nBlock B"


def test_extract_event_text_empty_and_whitespace_skipped():
    e = _thought_event([
        {"type": "text", "text": ""},
        {"type": "text", "text": "   "},
        {"type": "text", "text": "real"},
    ])
    assert extract_event_text(e) == "real"


def test_extract_event_text_no_text_fields():
    e = _agent_event(action={"kind": "CmdRunAction"})
    assert extract_event_text(e) == ""


# ---------- extract_final_response: FinishAction path ----------


def test_finish_with_message_wins():
    """Case 1: FinishAction.action.message is the canonical path."""
    history = [
        _thought_event("earlier ramble"),
        _finish_event("Here is the final answer."),
    ]
    assert extract_final_response({"history": history}) == "Here is the final answer."


def test_finish_message_empty_falls_back_to_args_json_string():
    """Case 2: action.message empty but tool_call.arguments JSON has it."""
    history = [
        _thought_event("earlier"),
        _finish_event(message="", args_message="from arguments"),
    ]
    assert extract_final_response({"history": history}) == "from arguments"


def test_finish_message_empty_falls_back_to_args_dict():
    """Case 3: tool_call.arguments may be a dict, not a JSON string."""
    history = [
        _thought_event("earlier"),
        _finish_event(message="", args_message="from dict args",
                      args_as_dict=True),
    ]
    assert extract_final_response({"history": history}) == "from dict args"


def test_finish_with_no_message_falls_back_to_history_text():
    """Case 4: FinishAction with everything empty → fall back."""
    history = [
        _thought_event("This is the real answer text."),
        _finish_event(message=""),  # finish exists but empty
    ]
    assert extract_final_response({"history": history}) == "This is the real answer text."


def test_no_finish_falls_back_to_history_text():
    """Case 5: agent terminated abnormally (no finish call)."""
    history = [
        _thought_event("step 1"),
        _thought_event("step 2 — partial answer here"),
    ]
    assert extract_final_response({"history": history}) == "step 2 — partial answer here"


def test_last_finish_wins_over_earlier_one():
    """Case 18: multiple finishes (rare) — latest is authoritative."""
    history = [
        _finish_event("first finish — should be ignored"),
        _thought_event("intermediate"),
        _finish_event("LAST finish wins"),
    ]
    assert extract_final_response({"history": history}) == "LAST finish wins"


def test_finish_arguments_invalid_json_string_does_not_crash():
    """tool_call.arguments may be malformed JSON — must not raise."""
    history = [
        _thought_event("fallback content"),
        _agent_event(
            action={"kind": "FinishAction", "message": ""},
            tool_call={"arguments": "{not-valid-json"},
        ),
    ]
    # Falls through to thought-text scan
    assert extract_final_response({"history": history}) == "fallback content"


# ---------- extract_final_response: empty/odd inputs ----------


def test_empty_history():
    """Case 9: empty history → returns empty string."""
    assert extract_final_response({"history": []}) == ""


def test_missing_history_key():
    assert extract_final_response({}) == ""


def test_history_not_a_list():
    assert extract_final_response({"history": "not a list"}) == ""


def test_non_agent_events_skipped():
    """User/environment events with text should not be picked up."""
    history = [
        {"source": "user", "content": "the user's question"},
        {"source": "environment", "content": "tool output"},
    ]
    assert extract_final_response({"history": history}) == ""


def test_non_dict_history_entries_tolerated():
    history = [None, "weird", 42, _finish_event("real message")]
    assert extract_final_response({"history": history}) == "real message"


# ---------- extract_final_response_from_jsonl ----------


def test_extract_from_jsonl_filters_by_instance_id(tmp_path: Path):
    """Case 12: multiple records, return the one matching instance_id."""
    p = tmp_path / "output.jsonl"
    lines = [
        {"instance_id": "task_a",
         "history": [_finish_event("answer A")]},
        {"instance_id": "task_b",
         "history": [_finish_event("answer B")]},
    ]
    p.write_text("\n".join(json.dumps(l) for l in lines))
    assert extract_final_response_from_jsonl(p, "task_b") == "answer B"
    assert extract_final_response_from_jsonl(p, "task_a") == "answer A"


def test_extract_from_jsonl_no_match_returns_empty(tmp_path: Path):
    p = tmp_path / "output.jsonl"
    p.write_text(json.dumps(
        {"instance_id": "task_x", "history": [_finish_event("x")]}
    ))
    assert extract_final_response_from_jsonl(p, "task_other") == ""


def test_extract_from_jsonl_no_instance_id_returns_first(tmp_path: Path):
    p = tmp_path / "output.jsonl"
    lines = [
        {"history": [_finish_event("first")]},
        {"history": [_finish_event("second")]},
    ]
    p.write_text("\n".join(json.dumps(l) for l in lines))
    assert extract_final_response_from_jsonl(p) == "first"


def test_extract_from_jsonl_missing_file():
    """Case 10: missing file → empty string, no exception."""
    assert extract_final_response_from_jsonl(Path("/no/such/path.jsonl")) == ""


def test_extract_from_jsonl_skips_malformed_lines(tmp_path: Path):
    """Case 11: malformed lines must not abort the search."""
    p = tmp_path / "output.jsonl"
    p.write_text(
        "{not valid json\n"
        + "\n"
        + json.dumps({"instance_id": "t", "history": [_finish_event("ok")]})
        + "\n"
    )
    assert extract_final_response_from_jsonl(p, "t") == "ok"


def test_extract_from_jsonl_unicode_roundtrip(tmp_path: Path):
    """Case 16: emoji / non-ASCII content must survive UTF-8 round-trip."""
    text = "Avatars: 🔵 cyan, 🟡 gold, ⚪ silver, 🟣 cosmic — pick one"
    p = tmp_path / "output.jsonl"
    p.write_text(json.dumps({
        "instance_id": "t",
        "history": [_finish_event(text)],
    }), encoding="utf-8")
    assert extract_final_response_from_jsonl(p, "t") == text


# ---------- write_response_md ----------


def test_write_response_md_writes_content(tmp_path: Path):
    dest = tmp_path / "results" / "response.md"
    out = write_response_md("# Hello\n\nbody", dest)
    assert out == dest
    assert dest.read_text(encoding="utf-8") == "# Hello\n\nbody\n"


def test_write_response_md_adds_trailing_newline_only_if_missing(tmp_path: Path):
    dest = tmp_path / "response.md"
    write_response_md("already ends in newline\n", dest)
    assert dest.read_text(encoding="utf-8") == "already ends in newline\n"


def test_write_response_md_empty_writes_stub_by_default(tmp_path: Path):
    """Case 13: empty text → stub note (so reviewer sees the 'why')."""
    dest = tmp_path / "response.md"
    write_response_md("", dest)
    assert dest.read_text(encoding="utf-8") == NO_RESPONSE_STUB


def test_write_response_md_empty_without_stub(tmp_path: Path):
    """Case 14: explicit opt-out of stub → empty file."""
    dest = tmp_path / "response.md"
    write_response_md("", dest, fallback_stub=False)
    assert dest.read_text(encoding="utf-8") == ""


def test_write_response_md_creates_parent_dirs(tmp_path: Path):
    """Case 15: results/ folder may not exist yet."""
    dest = tmp_path / "deeply" / "nested" / "results" / "response.md"
    write_response_md("hi", dest)
    assert dest.exists()


def test_write_response_md_overwrites_existing(tmp_path: Path):
    dest = tmp_path / "response.md"
    dest.write_text("old content")
    write_response_md("new content", dest)
    assert dest.read_text(encoding="utf-8") == "new content\n"


# ---------- Integration: realistic OpenHands shape ----------


def test_realistic_openhands_finish_shape():
    """Mirror the actual shape observed in our production output.jsonl
    (claude-opus-4.7 run_1 of task_c6f4581ec2f2c0dc):
        event.source = 'agent'
        event.thought = [{'type': 'text', 'text': ''}]   # empty
        event.action = {'kind': 'FinishAction', 'message': '<the answer>'}
        event.tool_name = 'finish'
        event.tool_call.arguments = '{"message": "<same answer>"}'  # JSON
    """
    record = {
        "instance_id": "task_xyz",
        "history": [
            _thought_event([{"type": "text", "text": "early reasoning"}]),
            {
                "source": "agent",
                "thought": [{"type": "text", "text": ""}],
                "action": {
                    "kind": "FinishAction",
                    "message": "All four avatars saved.\n\n- option-1.webp 🔵",
                },
                "tool_name": "finish",
                "tool_call": {
                    "arguments": json.dumps({
                        "message": "All four avatars saved.\n\n- option-1.webp 🔵"
                    })
                },
            },
            {"source": "environment", "kind": "ConversationStateUpdateEvent"},
        ],
    }
    assert extract_final_response(record) == "All four avatars saved.\n\n- option-1.webp 🔵"
