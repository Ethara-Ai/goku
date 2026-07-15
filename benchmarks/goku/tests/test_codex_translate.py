"""Unit tests for the Codex bridge's Chat<->Responses translation.

These guard the function-calling + multimodal round-trip that goku's OpenHands
agent depends on when driving gpt-5.5 through the ChatGPT-subscription bridge
(benchmarks.utils.openai_codex). The upstream vendored translator flattened
tool calls to text and dropped images; goku's enhanced version must preserve
both, in the request, the streaming response, and the unary response.
"""

from __future__ import annotations

import asyncio
import json

from benchmarks.utils.openai_codex import translate as x


def test_chat_to_responses_preserves_images_and_function_calls():
    chat = {
        "model": "openai/gpt-5.5",
        "tools": [{"type": "function", "function": {"name": "bash"}}],
        "messages": [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": [
                {"type": "text", "text": "Look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,ABC"}},
            ]},
            {"role": "assistant", "content": "running ls", "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "bash", "arguments": "{\"cmd\":\"ls\"}"}},
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "file1.txt"},
        ],
    }
    out = x.chat_to_responses(chat)
    items = out["input"]

    assert out["instructions"] == "You are an agent."
    assert out["tools"] == chat["tools"]

    # image preserved as input_image
    user = items[0]
    assert user["role"] == "user"
    kinds = [p["type"] for p in user["content"]]
    assert "input_text" in kinds and "input_image" in kinds
    assert user["content"][1]["image_url"] == "data:image/png;base64,ABC"

    # assistant tool_call -> function_call item
    fc = next(i for i in items if i.get("type") == "function_call")
    assert fc["name"] == "bash" and fc["call_id"] == "call_1"

    # tool result -> function_call_output item
    fco = next(i for i in items if i.get("type") == "function_call_output")
    assert fco["call_id"] == "call_1" and fco["output"] == "file1.txt"


def test_streaming_translation_emits_text_and_tool_calls():
    lines = [
        b'data: {"type":"response.output_text.delta","delta":"Running "}',
        b'data: {"type":"response.output_item.done","item":{"type":"function_call",'
        b'"call_id":"call_9","name":"bash","arguments":"{}"}}',
        b'data: {"type":"response.completed","response":{"usage":'
        b'{"input_tokens":10,"output_tokens":5}}}',
    ]

    async def collect() -> str:
        async def ait():
            for line in lines:
                yield line
        chunks = []
        async for c in x.aiter_responses_sse_as_chat(ait(), "openai/gpt-5.5", 123):
            chunks.append(c.decode())
        return "".join(chunks)

    s = asyncio.new_event_loop().run_until_complete(collect())
    assert '"content": "Running "' in s
    assert '"tool_calls"' in s and '"name": "bash"' in s
    assert '"finish_reason": "tool_calls"' in s
    assert "[DONE]" in s


def test_unary_responses_to_chat_maps_tool_calls_and_usage():
    resp = {
        "id": "resp_1", "status": "completed",
        "output": [{"type": "function_call", "call_id": "c2",
                    "name": "edit", "arguments": "{}"}],
        "usage": {"input_tokens": 3, "output_tokens": 1},
    }
    chat = x.responses_to_chat(resp, "openai/gpt-5.5", 1)
    choice = chat["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "edit"
    assert chat["usage"]["prompt_tokens"] == 3


def test_chat_to_responses_string_content_still_works():
    out = x.chat_to_responses({"model": "openai/gpt-5.5",
                               "messages": [{"role": "user", "content": "hi"}]})
    item = out["input"][0]
    assert item["role"] == "user"
    assert item["content"] == [{"type": "input_text", "text": "hi"}]
    # round-trips through json cleanly
    json.dumps(out)
