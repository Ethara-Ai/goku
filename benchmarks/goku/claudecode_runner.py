"""Claude Code agent backend for the Goku benchmark.

Drives trajectory generation with the official ``@anthropic-ai/claude-code`` CLI
running *inside* goku's existing Docker workspace container, authenticated
against the user's Claude Code **subscription** via the OAuth bridge
(``benchmarks.utils.claude_oauth``) — not a metered API key.

Design mirrors WildClawBench's ``claudecode`` backend, adapted to goku:

  * goku's ``prepare_workspace`` already creates the container and uploads all
    task media into ``/workspace``. The CLI reads those files from disk with its
    native Read tool, so multimodal inputs need no base64 inlining.
  * We install the CLI into the running container (its base image ships Node),
    then ``docker exec`` ``claude -p`` in non-interactive/print mode with
    ``--output-format stream-json`` so we get the full message stream (the
    trajectory) plus a final ``result`` object (response text + usage).
  * The returned ``(response_text, trajectory, metrics)`` feed goku's unchanged
    deterministic + LLM-judge scoring path; output files are downloaded by the
    caller's existing ``_download_outputs``.

The container reaches the host-run bridge via ``ANTHROPIC_BASE_URL`` (typically
``http://host.docker.internal:<port>``) and presents ``ANTHROPIC_API_KEY`` =
the bridge's shared secret.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass, field
from typing import Any

from openhands.sdk import get_logger
from openhands.sdk.workspace import RemoteWorkspace


logger = get_logger(__name__)

CLAUDE_CODE_NPM_PKG = "@anthropic-ai/claude-code"
# Where the CLI writes its per-session transcript / config inside the container.
CONTAINER_HOME = "/root"


class ClaudeCodeUnavailableError(RuntimeError):
    """Raised when the backend cannot run — no local container, install failure,
    or the CLI produced no usable output. Propagated so the eval harness retries
    rather than scoring a blank as a wrong answer."""


@dataclass
class ClaudeCodeResult:
    response_text: str
    trajectory: str
    raw_events: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] | None = None
    is_error: bool = False


def _container_id(workspace: RemoteWorkspace) -> str:
    cid = getattr(workspace, "_container_id", None)
    if not cid:
        raise ClaudeCodeUnavailableError(
            "The claudecode backend requires a locally-backed Docker workspace "
            "(RemoteWorkspace._container_id), but none was found. It is not "
            "compatible with API/remote runtimes."
        )
    return cid


def ensure_cli_installed(workspace: RemoteWorkspace, *, timeout: float = 240.0) -> None:
    """Install the Claude Code CLI in the container if not already present.

    Idempotent: a no-op when ``claude`` is already on PATH. The workspace base
    image (nikolaik/python-nodejs) ships Node+npm, so a global install works.
    """
    cid = _container_id(workspace)
    check = subprocess.run(
        ["docker", "exec", "-u", "root", cid, "bash", "-lc", "command -v claude"],
        capture_output=True,
        text=True,
    )
    if check.returncode == 0 and check.stdout.strip():
        logger.info("Claude Code CLI already present: %s", check.stdout.strip())
        return

    logger.info("Installing %s in container %s (one-time npm cost)…", CLAUDE_CODE_NPM_PKG, cid[:12])
    install = subprocess.run(
        [
            "docker", "exec", "-u", "root", cid, "bash", "-lc",
            f"npm install -g {shlex.quote(CLAUDE_CODE_NPM_PKG)} --no-fund --no-audit",
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if install.returncode != 0:
        raise ClaudeCodeUnavailableError(
            f"Failed to install {CLAUDE_CODE_NPM_PKG} in container: "
            f"{(install.stderr or install.stdout)[-800:]}"
        )
    verify = subprocess.run(
        ["docker", "exec", "-u", "root", cid, "bash", "-lc", "command -v claude"],
        capture_output=True,
        text=True,
    )
    if verify.returncode != 0 or not verify.stdout.strip():
        raise ClaudeCodeUnavailableError(
            "Claude Code CLI installed but is not on PATH inside the container."
        )
    logger.info("Claude Code CLI installed: %s", verify.stdout.strip())


def _build_prompt(instruction: str, input_file_names: list[str]) -> str:
    if not input_file_names:
        media_block = ""
    else:
        manifest = "\n".join(f"  - /workspace/{name}" for name in input_file_names)
        media_block = (
            "\n\n## Task input files (already in your working directory)\n"
            "The following files were provided for this task. Read/inspect them "
            "directly from disk as needed (images, PDFs, videos, data files):\n"
            f"{manifest}\n"
        )
    return (
        f"{instruction}{media_block}\n\n"
        "## Output requirements\n"
        "Your working directory is /workspace. Write any files you are asked to "
        "produce into /workspace (or a subdirectory of it) — that is the only "
        "location collected for grading. When finished, end with a concise final "
        "message stating your answer / what you produced."
    )


def _cli_env(bridge_url: str, bridge_api_key: str) -> dict[str, str]:
    return {
        "ANTHROPIC_BASE_URL": bridge_url,
        "ANTHROPIC_API_KEY": bridge_api_key,
        "HOME": CONTAINER_HOME,
        # Keep the headless run quiet & non-interactive.
        "IS_SANDBOX": "1",
        "CI": "1",
        "DISABLE_AUTOUPDATER": "1",
        "DISABLE_TELEMETRY": "1",
        "DISABLE_ERROR_REPORTING": "1",
        "DISABLE_NON_ESSENTIAL_MODEL_CALLS": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }


def run_claudecode_task(
    *,
    workspace: RemoteWorkspace,
    instruction: str,
    input_file_names: list[str],
    instance_id: str,
    model: str,
    bridge_url: str,
    bridge_api_key: str,
    max_turns: int,
    timeout_seconds: int,
) -> ClaudeCodeResult:
    """Run one task with the Claude Code CLI in the container and parse results.

    ``input_file_names`` are basenames already uploaded to /workspace by the
    caller's ``prepare_workspace``.
    """
    cid = _container_id(workspace)
    ensure_cli_installed(workspace)

    prompt = _build_prompt(instruction, input_file_names)

    env_args: list[str] = []
    for key, value in _cli_env(bridge_url, bridge_api_key).items():
        env_args += ["-e", f"{key}={value}"]

    cli = [
        "claude",
        "-p",
        "--model", model,
        "--output-format", "stream-json",
        "--verbose",
        "--dangerously-skip-permissions",
        "--add-dir", "/workspace",
        "--max-turns", str(max_turns),
    ]
    cmd = [
        "docker", "exec", "-i", "-u", "root", "-w", "/workspace",
        *env_args, cid, *cli,
    ]

    logger.info(
        "[%s] Running Claude Code (model=%s, max_turns=%d, timeout=%ds) via %s",
        instance_id, model, max_turns, timeout_seconds, bridge_url,
    )
    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        # Partial stdout may still hold a usable trajectory/answer.
        partial = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        logger.warning("[%s] Claude Code timed out after %ds", instance_id, timeout_seconds)
        result = _parse_stream_json(partial, instance_id)
        result.is_error = True
        if not result.response_text:
            result.response_text = ""
        return result

    if proc.returncode != 0 and not proc.stdout.strip():
        raise ClaudeCodeUnavailableError(
            f"[{instance_id}] Claude Code exited {proc.returncode} with no output. "
            f"stderr: {(proc.stderr or '')[-800:]}"
        )
    if proc.returncode != 0:
        logger.warning(
            "[%s] Claude Code exited %d (parsing partial output). stderr: %s",
            instance_id, proc.returncode, (proc.stderr or "")[-400:],
        )

    return _parse_stream_json(proc.stdout, instance_id)


def _parse_stream_json(stdout: str, instance_id: str) -> ClaudeCodeResult:
    """Parse ``--output-format stream-json`` output into a ClaudeCodeResult.

    Each non-empty line is a JSON object. We collect assistant/user messages for
    the trajectory and pull the final ``result`` object for response text +
    usage. Falls back to the last assistant text block if no result line exists.
    """
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    result_obj = next(
        (e for e in reversed(events) if e.get("type") == "result"), None
    )

    response_text = ""
    is_error = False
    metrics: dict[str, Any] = {}
    if result_obj is not None:
        response_text = (result_obj.get("result") or "").strip()
        is_error = bool(result_obj.get("is_error"))
        usage = result_obj.get("usage") or {}
        metrics = {
            "input_tokens": int(usage.get("input_tokens", 0) or 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
            "cache_read_tokens": int(usage.get("cache_read_input_tokens", 0) or 0),
            "cache_write_tokens": int(usage.get("cache_creation_input_tokens", 0) or 0),
            "cost_usd": float(result_obj.get("total_cost_usd", 0.0) or 0.0),
            "num_turns": int(result_obj.get("num_turns", 0) or 0),
        }

    if not response_text:
        response_text = _last_assistant_text(events)

    if not events:
        raise ClaudeCodeUnavailableError(
            f"[{instance_id}] Claude Code produced no parseable stream-json output."
        )

    return ClaudeCodeResult(
        response_text=response_text,
        trajectory=_format_trajectory(events),
        raw_events=events,
        metrics=metrics or None,
        is_error=is_error,
    )


def _last_assistant_text(events: list[dict[str, Any]]) -> str:
    for event in reversed(events):
        if event.get("type") != "assistant":
            continue
        message = event.get("message") or {}
        for block in reversed(message.get("content") or []):
            if isinstance(block, dict) and block.get("type") == "text":
                return (block.get("text") or "").strip()
    return ""


def _format_trajectory(events: list[dict[str, Any]]) -> str:
    """Render the CLI message stream as a readable trajectory for the LLM judge.

    Shows assistant text + tool calls and (truncated) tool results, in order.
    """
    lines: list[str] = []
    for i, event in enumerate(events):
        etype = event.get("type")
        if etype == "system":
            subtype = event.get("subtype", "")
            lines.append(f"[{i}] system/{subtype}")
        elif etype == "assistant":
            message = event.get("message") or {}
            for block in message.get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    lines.append(f"[{i}] assistant: {(block.get('text') or '')[:500]}")
                elif block.get("type") == "tool_use":
                    tool = block.get("name", "tool")
                    tool_input = json.dumps(block.get("input", {}), default=str)[:300]
                    lines.append(f"[{i}] assistant tool_use {tool}: {tool_input}")
        elif etype == "user":
            message = event.get("message") or {}
            for block in message.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    content = block.get("content")
                    text = content if isinstance(content, str) else json.dumps(content, default=str)
                    lines.append(f"[{i}] tool_result: {text[:300]}")
        elif etype == "result":
            lines.append(f"[{i}] result: {(event.get('result') or '')[:300]}")
        if len(lines) > 800:
            lines.append("... (truncated)")
            break
    return "\n".join(lines)
