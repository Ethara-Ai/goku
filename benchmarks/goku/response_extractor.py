"""Extract the agent's final natural-language response from an OpenHands
output.jsonl record and write it as a markdown file in the delivery folder.

Per `DIU Goku doc.md` Tab 2 (delivery folder tree), each per-run results/
directory should contain a markdown document with the model's final
response alongside any artifacts the agent produced.

The canonical place to find that response is the `FinishAction` event
emitted by the agent via the `finish` tool. We fall back to the last
agent text event if the agent terminated abnormally (timeout, hard error)
without invoking finish.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


NO_RESPONSE_STUB = (
    "*(No final response — agent did not emit a closing message "
    "via the `finish` tool.)*\n"
)


def extract_event_text(event: dict[str, Any]) -> str:
    """Pull any natural-language text out of a single agent history event.

    Handles all of OpenHands SDK's content shapes:
      - thought: list of {type, text} blocks (Anthropic-style content blocks)
      - thought: plain string
      - content: list of blocks
      - content: plain string
    Non-text blocks (tool_use, image, etc.) are skipped.
    """
    chunks: list[str] = []

    for field in ("thought", "content"):
        v = event.get(field)
        if isinstance(v, list):
            for blk in v:
                if isinstance(blk, dict) and blk.get("type") == "text":
                    t = blk.get("text") or ""
                    if isinstance(t, str) and t.strip():
                        chunks.append(t.strip())
        elif isinstance(v, str) and v.strip():
            chunks.append(v.strip())

    return "\n\n".join(chunks).strip()


def extract_final_response(record: dict[str, Any]) -> str:
    """Find the agent's final natural-language response in a single
    OpenHands output record.

    Priority:
      1. Last FinishAction event's `action.message` (canonical).
      2. Last FinishAction event's `tool_call.arguments.message`
         (parsed as JSON if string, used directly if dict).
      3. Last agent event with non-empty thought/content text
         (abnormal-termination fallback).

    Returns the extracted text, or "" if nothing usable is found.
    """
    history = record.get("history") or []
    if not isinstance(history, list):
        return ""

    # Pass 1: FinishAction — the canonical end-of-task event
    for event in reversed(history):
        if not isinstance(event, dict):
            continue
        if event.get("source") != "agent":
            continue
        action = event.get("action")
        if not isinstance(action, dict):
            continue
        if action.get("kind") != "FinishAction":
            continue

        msg = action.get("message")
        if isinstance(msg, str) and msg.strip():
            return msg.strip()

        # Fallback within finish event: tool_call.arguments may carry the
        # message even if action.message is empty (rare but observed).
        tool_call = event.get("tool_call")
        if isinstance(tool_call, dict):
            args = tool_call.get("arguments")
            parsed_args: dict[str, Any] | None = None
            if isinstance(args, dict):
                parsed_args = args
            elif isinstance(args, str) and args:
                try:
                    candidate = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    candidate = None
                if isinstance(candidate, dict):
                    parsed_args = candidate
            if parsed_args is not None:
                fb = parsed_args.get("message")
                if isinstance(fb, str) and fb.strip():
                    return fb.strip()

        # Found FinishAction but no usable message — stop the FinishAction
        # search; do NOT keep looking for an earlier one (the LATEST
        # finish is authoritative, even if empty).
        break

    # Pass 2: last agent event with any text content (abnormal-termination
    # fallback — used when no FinishAction was emitted).
    for event in reversed(history):
        if not isinstance(event, dict):
            continue
        if event.get("source") != "agent":
            continue
        text = extract_event_text(event)
        if text:
            return text

    return ""


def extract_final_response_from_jsonl(
    output_jsonl: Path,
    instance_id: str | None = None,
) -> str:
    """Read output.jsonl, find the matching record, return final response.

    Args:
        output_jsonl: Path to an OpenHands-style output.jsonl. Multiple
            records (one per instance_id) may share the file.
        instance_id: If provided, only consider records matching this id.
            If None, the first parseable record wins.

    Returns:
        Extracted final response text, or "" if not found or file missing.
    """
    if not output_jsonl.is_file():
        return ""
    try:
        text = output_jsonl.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not read %s: %s", output_jsonl, exc)
        return ""

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue  # tolerate malformed lines
        if not isinstance(record, dict):
            continue
        if instance_id is not None and record.get("instance_id") != instance_id:
            continue
        return extract_final_response(record)

    return ""


def write_response_md(
    text: str,
    dest: Path,
    *,
    fallback_stub: bool = True,
) -> Path:
    """Write the response text to a markdown file at `dest`.

    The file content is the response text verbatim (preserving the agent's
    own markdown formatting). Per client clarification, no metadata header
    or scaffolding is added.

    Args:
        text: The extracted final response (may be empty string).
        dest: Destination file path (e.g. results/response.md).
        fallback_stub: If True and `text` is empty, write the stub note so
            human reviewers can distinguish "agent had no response" from
            "extraction silently failed". If False, write an empty file.

    Returns:
        The destination path.
    """
    if text:
        content = text if text.endswith("\n") else text + "\n"
    elif fallback_stub:
        content = NO_RESPONSE_STUB
    else:
        content = ""

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    return dest
