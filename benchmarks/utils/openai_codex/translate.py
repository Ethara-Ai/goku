"""Chat Completions <-> Responses API translation for the Codex bridge.

The ChatGPT/Codex backend only speaks the Responses API. Many OpenAI clients
(litellm without responses-mode registration, the harness preflight probe, the
openai SDK's `.chat.completions`) speak Chat Completions. This module translates
both ways so the bridge is a drop-in `/v1/chat/completions` endpoint too:

    chat request  --chat_to_responses-->  responses request  (to codex backend)
    responses SSE --responses_sse_to_chat_sse--> chat SSE      (streaming client)
    responses obj --responses_to_chat-->  chat.completion obj  (unary client)

This translator is function-calling- and image-aware so it can drive an agentic,
multimodal client (goku's OpenHands agent over gpt-5.5), not only text/edit-block
pipelines: assistant ``tool_calls`` round-trip as Responses ``function_call``
items, ``role:"tool"`` results as ``function_call_output`` items, and user
``image_url`` parts as Responses ``input_image`` parts. Sampling params the codex
backend rejects are still dropped upstream by the bridge.
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional


def _content_to_text(content: Any) -> str:
    """Flatten a chat message `content` (str or list of parts) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") in ("text", "input_text", "output_text"):
                    out.append(part.get("text", ""))
                elif "text" in part:
                    out.append(part["text"])
        return "".join(out)
    return "" if content is None else str(content)


def _to_input_parts(content: Any) -> list[dict]:
    """Convert Chat `content` (str or multimodal part list) into Responses input
    content parts, PRESERVING images (goku is a multimodal benchmark).

    - text/input_text parts -> {"type":"input_text","text":...}
    - image_url parts       -> {"type":"input_image","image_url":<url>}
      (accepts both the OpenAI object form {"image_url":{"url":...}} and a bare
      string, and passes an already-Responses-shaped input_image through.)
    """
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    parts: list[dict] = []
    if isinstance(content, list):
        for p in content:
            if not isinstance(p, dict):
                continue
            ptype = p.get("type")
            if ptype in ("text", "input_text", "output_text"):
                parts.append({"type": "input_text", "text": p.get("text", "")})
            elif ptype == "image_url":
                iu = p.get("image_url")
                url = iu.get("url") if isinstance(iu, dict) else iu
                if url:
                    parts.append({"type": "input_image", "image_url": url})
            elif ptype == "input_image":
                parts.append(p)  # already Responses-shaped
            elif "text" in p:
                parts.append({"type": "input_text", "text": p["text"]})
    if not parts:
        parts.append({"type": "input_text", "text": ""})
    return parts


def chat_to_responses(chat: dict) -> dict:
    """Translate a Chat Completions request body into a Responses request body.

    - system/developer messages are concatenated into `instructions`.
    - user messages -> input `message` items (text + images preserved).
    - assistant messages -> an output_text `message` (if it has text) plus one
      `function_call` item per `tool_calls` entry.
    - `role:"tool"` results -> `function_call_output` items (keyed by tool_call_id).
    - max_tokens/max_completion_tokens -> max_output_tokens.
    - stream, tools, tool_choice, reasoning pass through.
    """
    out: dict = {"model": chat.get("model")}

    instructions: list[str] = []
    input_items: list[dict] = []
    for msg in chat.get("messages", []) or []:
        role = msg.get("role")
        if role in ("system", "developer"):
            text = _content_to_text(msg.get("content"))
            if text:
                instructions.append(text)
        elif role == "assistant":
            text = _content_to_text(msg.get("content"))
            if text:
                input_items.append({"type": "message", "role": "assistant",
                                    "content": [{"type": "output_text", "text": text}]})
            for tc in msg.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                input_items.append({
                    "type": "function_call",
                    "call_id": tc.get("id") or tc.get("call_id") or "",
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments", "") or "",
                })
        elif role == "tool":
            # Structured tool result -> function_call_output (matched by call id).
            input_items.append({
                "type": "function_call_output",
                "call_id": msg.get("tool_call_id") or msg.get("call_id") or "",
                "output": _content_to_text(msg.get("content")),
            })
        else:  # user (default) — preserve text AND images
            input_items.append({"type": "message", "role": "user",
                                "content": _to_input_parts(msg.get("content"))})

    if instructions:
        out["instructions"] = "\n\n".join(instructions)
    out["input"] = input_items

    mt = chat.get("max_completion_tokens") or chat.get("max_tokens")
    if mt:
        out["max_output_tokens"] = mt
    if chat.get("stream"):
        out["stream"] = True
    if chat.get("tools"):
        out["tools"] = chat["tools"]
    if chat.get("tool_choice") is not None:
        out["tool_choice"] = chat["tool_choice"]
    # Pass a reasoning hint through if the caller set one (litellm uses this).
    if isinstance(chat.get("reasoning"), dict):
        out["reasoning"] = chat["reasoning"]
    return out


def _extract_text_from_output(output: list) -> str:
    """Concatenate assistant text from a Responses `output` array."""
    text = ""
    for item in output or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            for c in item.get("content", []) or []:
                if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                    text += c.get("text", "")
    return text


def _usage_to_chat(usage: Optional[dict]) -> Optional[dict]:
    if not isinstance(usage, dict):
        return None
    prompt = usage.get("input_tokens", 0) or 0
    completion = usage.get("output_tokens", 0) or 0
    out = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": usage.get("total_tokens", prompt + completion),
    }
    # Preserve reasoning-token detail so cost/thinking capture sees it.
    otd = usage.get("output_tokens_details") or {}
    if isinstance(otd, dict) and otd.get("reasoning_tokens") is not None:
        out["completion_tokens_details"] = {"reasoning_tokens": otd["reasoning_tokens"]}
    itd = usage.get("input_tokens_details") or {}
    if isinstance(itd, dict) and itd.get("cached_tokens") is not None:
        out["prompt_tokens_details"] = {"cached_tokens": itd["cached_tokens"]}
    return out


def _extract_tool_calls(output: list) -> list:
    """Map Responses `function_call` output items to Chat `tool_calls`.

    The rust/coding pipeline uses text edit-blocks (no function calling), but the
    /chat/completions endpoint is a general drop-in; represent tool calls if a
    client uses them rather than silently dropping them.
    """
    tool_calls = []
    for item in output or []:
        if isinstance(item, dict) and item.get("type") == "function_call":
            tool_calls.append({
                "id": item.get("call_id") or item.get("id") or f"call_{len(tool_calls)}",
                "type": "function",
                "function": {"name": item.get("name", ""),
                             "arguments": item.get("arguments", "") or ""},
            })
    return tool_calls


def responses_to_chat(resp: dict, model: str, created: int) -> dict:
    """Build a Chat Completions response object from a final Responses object."""
    output = resp.get("output", []) or []
    text = _extract_text_from_output(output)
    tool_calls = _extract_tool_calls(output)
    status = resp.get("status")
    if tool_calls:
        finish = "tool_calls"
    elif status == "incomplete":
        finish = "length"
    else:
        finish = "stop"
    # content is null only for a tool-call-only message; otherwise a string
    # (empty string for an empty/refused response) so litellm parsing is happy.
    message: dict = {"role": "assistant",
                     "content": (text if text else (None if tool_calls else ""))}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": resp.get("id", "chatcmpl-codex"),
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish,
        }],
        "usage": _usage_to_chat(resp.get("usage")) or {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _sse(obj: dict) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


def iter_responses_sse_as_chat(lines: list[bytes], model: str, created: int):
    """Translate Responses SSE lines into Chat Completions SSE chunks.

    Emits an initial role chunk, a content chunk per `response.output_text.delta`,
    then a final chunk with finish_reason + usage and the `[DONE]` sentinel.
    """
    cid = "chatcmpl-codex"
    base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model}
    yield _sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})

    usage_chat: Optional[dict] = None
    for line in lines:
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload in (b"", b"[DONE]"):
            continue
        try:
            evt = json.loads(payload)
        except json.JSONDecodeError:
            continue
        etype = evt.get("type", "")
        if etype == "response.output_text.delta":
            delta = evt.get("delta", "")
            if delta:
                yield _sse({**base, "choices": [{"index": 0, "delta": {"content": delta},
                                                 "finish_reason": None}]})
        elif etype in ("response.completed", "response.incomplete"):
            usage_chat = _usage_to_chat((evt.get("response") or {}).get("usage"))

    final = {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    if usage_chat is not None:
        final["usage"] = usage_chat
    yield _sse(final)
    yield b"data: [DONE]\n\n"


def _sse_event_from_line(line: str):
    """Parse one SSE `data:` line into an event dict, or None."""
    line = line.strip()
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if payload in ("", "[DONE]"):
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def _chat_chunk_bytes(base: dict, delta: dict, finish=None, usage=None) -> bytes:
    choice = {"index": 0, "delta": delta, "finish_reason": finish}
    obj = {**base, "choices": [choice]}
    if usage is not None:
        obj["usage"] = usage
    return _sse(obj)


async def aiter_responses_sse_as_chat(aline_iter, model: str, created: int):
    """Async, INCREMENTAL translation of a Responses SSE line-stream into
    Chat-Completions SSE chunks — yields as each upstream event arrives so a
    log/liveness watchdog keeps seeing progress (no whole-turn buffering).

    Function calls are surfaced: when a ``function_call`` output item completes
    (``response.output_item.done``) it is emitted as a Chat ``tool_calls`` delta
    chunk, and the terminal ``finish_reason`` becomes ``tool_calls`` — without
    this an agentic client (OpenHands) would never see the model's tool calls."""
    cid = "chatcmpl-codex"
    base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model}
    yield _chat_chunk_bytes(base, {"role": "assistant"})

    usage_chat = None
    tool_idx = 0  # sequential index for Chat tool_calls deltas
    async for raw in aline_iter:
        line = raw.decode("utf-8", "ignore") if isinstance(raw, (bytes, bytearray)) else raw
        evt = _sse_event_from_line(line)
        if evt is None:
            continue
        etype = evt.get("type", "")
        if etype == "response.output_text.delta":
            delta = evt.get("delta", "")
            if delta:
                yield _chat_chunk_bytes(base, {"content": delta})
        elif etype == "response.output_item.done" and isinstance(evt.get("item"), dict) \
                and evt["item"].get("type") == "function_call":
            item = evt["item"]
            yield _chat_chunk_bytes(base, {"tool_calls": [{
                "index": tool_idx,
                "id": item.get("call_id") or item.get("id") or f"call_{tool_idx}",
                "type": "function",
                "function": {"name": item.get("name", ""),
                             "arguments": item.get("arguments", "") or ""},
            }]})
            tool_idx += 1
        elif etype in ("response.completed", "response.incomplete"):
            usage_chat = _usage_to_chat((evt.get("response") or {}).get("usage"))
        elif etype in ("response.failed", "error"):
            # Surface a terminal error to the client as a final empty chunk; the
            # bridge already logged the detail.
            break
    finish = "tool_calls" if tool_idx > 0 else "stop"
    yield _chat_chunk_bytes(base, {}, finish=finish, usage=usage_chat)
    yield b"data: [DONE]\n\n"


def now_ts(clock: Any = time.time) -> int:
    return int(clock())
