"""Shared agent runner — drives one task from input → response + output files.

Used by:
  - benchmarks.goku.run_infer.GokuEvaluation (the batch CLI / `goku-infer`)
  - benchmarks.goku.service (the HTTP service for mm_tasker)

Both callers share the same Docker workspace creation, file upload, agent
construction, conversation loop, and output download. Differences (where to
write scores, how to aggregate runs, what shape to return to the framework
or to the wire) live in each caller.

The functions in this module do NOT score rubrics — that's the caller's
responsibility, since scoring needs the rubric set and (for `response_criteria`)
a judge LLM that the runner doesn't need to know about.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import shlex
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from benchmarks.utils.constants import EVAL_AGENT_SERVER_IMAGE
from benchmarks.utils.fake_user_response import (
    run_conversation_with_fake_user_response,
)
from benchmarks.utils.image_utils import create_docker_workspace
from benchmarks.utils.litellm_proxy import build_eval_llm
from openhands.sdk import (
    Agent,
    Conversation,
    Event,
    ImageContent,
    Message,
    MessageEvent,
    TextContent,
    get_logger,
)
from openhands.sdk.event import ActionEvent
from openhands.sdk.tool.builtins.finish import FinishAction
from openhands.sdk.workspace import RemoteWorkspace
from openhands.tools.preset.default import get_default_tools


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Image preprocessing — keep agent uploads under Bedrock's per-image limits.
# Constants and helpers moved verbatim from the original run_infer.py.
# ---------------------------------------------------------------------------
MAX_IMAGE_DIMENSION = 7680  # Bedrock limit is 8000px; leave margin
MAX_IMAGE_BYTES = 3_500_000  # 3.5 MB on disk → ~4.7 MB base64 (under 5 MB API limit)

# Magic-byte signatures → (mime_type, PIL format name)
_IMAGE_SIGNATURES: list[tuple[bytes, str, str]] = [
    (b"RIFF", "image/webp", "WEBP"),  # WebP starts with RIFF...WEBP
    (b"\x89PNG", "image/png", "PNG"),
    (b"\xff\xd8\xff", "image/jpeg", "JPEG"),
    (b"GIF8", "image/gif", "GIF"),
]


def _detect_image_format(image_path: str) -> tuple[str, str]:
    """Detect actual image format from magic bytes, not file extension."""
    with open(image_path, "rb") as f:
        header = f.read(12)

    for sig, mime, fmt in _IMAGE_SIGNATURES:
        if header.startswith(sig):
            # Extra check: RIFF can be non-WebP (e.g., AVI)
            if sig == b"RIFF" and header[8:12] != b"WEBP":
                continue
            return mime, fmt

    ext_mime = mimetypes.guess_type(image_path)[0] or "image/png"
    ext_fmt = "JPEG" if ext_mime == "image/jpeg" else "PNG"
    return ext_mime, ext_fmt


def _resize_if_needed(image_path: str) -> str:
    """Return a path to a re-encoded/downscaled copy if needed; else original.

    Handles dimension > 7680px, file size > 3.5MB, and mime/extension mismatch.
    """
    from PIL import Image

    real_mime, real_fmt = _detect_image_format(image_path)
    img = Image.open(image_path)
    file_size = os.path.getsize(image_path)
    max_dim = max(img.size)

    needs_resize = max_dim > MAX_IMAGE_DIMENSION
    needs_compress = file_size > MAX_IMAGE_BYTES
    ext_mime = mimetypes.guess_type(image_path)[0] or "image/png"
    needs_reencode = ext_mime != real_mime

    if not needs_resize and not needs_compress and not needs_reencode:
        return image_path

    reasons: list[str] = []
    if needs_resize:
        scale = MAX_IMAGE_DIMENSION / max_dim
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        reasons.append(f"dim {max_dim}→{max(new_size)}px")
    elif needs_compress:
        scale = (MAX_IMAGE_BYTES / file_size) ** 0.5
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        reasons.append(f"size {file_size / 1e6:.1f}→~{MAX_IMAGE_BYTES / 1e6:.1f}MB")

    if needs_reencode:
        reasons.append(f"reencode {ext_mime}→{real_mime}")

    out_fmt = real_fmt if real_fmt in ("PNG", "JPEG") else "PNG"
    if out_fmt == "JPEG" and img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    suffix = ".jpg" if out_fmt == "JPEG" else ".png"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    quality = 85
    img.save(tmp_path, format=out_fmt, quality=quality, optimize=True)

    for _ in range(3):
        if os.path.getsize(tmp_path) <= MAX_IMAGE_BYTES:
            break
        quality = max(quality - 15, 30)
        scale_factor = 0.75
        new_w = int(img.size[0] * scale_factor)
        new_h = int(img.size[1] * scale_factor)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        img.save(tmp_path, format=out_fmt, quality=quality, optimize=True)
        reasons.append(f"recompress q={quality} {new_w}x{new_h}")

    logger.info(
        f"Processed {os.path.basename(image_path)}: {', '.join(reasons)} "
        f"[{os.path.getsize(tmp_path) / 1e6:.2f}MB]"
    )
    return tmp_path


def _image_to_base64_url(image_path: str) -> str:
    """Convert an image file to a base64 data URL, resizing/fixing if needed."""
    resized_path = _resize_if_needed(image_path)
    real_mime, _ = _detect_image_format(
        resized_path if resized_path != image_path else image_path
    )
    if resized_path != image_path:
        real_mime = "image/jpeg" if resized_path.endswith(".jpg") else "image/png"
    with open(resized_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    if resized_path != image_path:
        os.unlink(resized_path)
    return f"data:{real_mime};base64,{data}"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass
class AgentRunResult:
    """What `run_agent` returns. The caller is responsible for cleaning up
    `output_dir` (it's a tempfile.mkdtemp path)."""

    response_text: str
    output_dir: Path
    events: list[Event] = field(default_factory=list)
    metrics: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Workspace + file upload — moved from GokuEvaluation.prepare_workspace
# ---------------------------------------------------------------------------
def prepare_agent_workspace(
    input_files: list[str],
    *,
    forward_env: list[str] | None = None,
) -> RemoteWorkspace:
    """Create a Docker workspace, ensure /workspace/results exists, and upload
    every input file. Mirrors the original prepare_workspace logic — same
    resize-then-upload + base64-fallback paths, same retry-on-mkdir.

    Returns the RemoteWorkspace; caller manages its lifecycle (typically by
    handing it to a `Conversation(... delete_on_close=True)`).
    """
    import platform as _platform

    docker_platform = "linux/arm64" if _platform.machine() == "arm64" else "linux/amd64"

    workspace = create_docker_workspace(
        agent_server_image=EVAL_AGENT_SERVER_IMAGE,
        base_image="nikolaik/python-nodejs:python3.12-nodejs22",
        build_target="binary",
        forward_env=forward_env or [],
        platform=docker_platform,
    )

    # Create workspace directories and verify filesystem is writable. The
    # agent-server health check only confirms HTTP readiness, not filesystem
    # readiness. Retry mkdir to give the container a moment.
    for attempt in range(3):
        result = workspace.execute_command("mkdir -p /workspace/results")
        exit_code = getattr(result, "exit_code", -1)
        if exit_code == 0:
            break
        logger.warning(
            f"mkdir /workspace/results failed (attempt {attempt + 1}): "
            f"{getattr(result, 'stderr', '')}"
        )
        time.sleep(2)

    for file_path in input_files:
        if not os.path.exists(file_path):
            logger.warning(f"Input file not found: {file_path}")
            continue
        _upload_one_file(workspace, file_path)

    return workspace  # type: ignore[return-value]


def _upload_one_file(workspace: RemoteWorkspace, file_path: str) -> None:
    """Upload one file to /workspace/<basename>. Resizes oversized images;
    falls back to base64 + bash if the SDK's file_upload fails (some
    agent-server images have a bug where /api/file/upload returns 500)."""
    file_name = os.path.basename(file_path)
    logger.info(f"Uploading {file_name} to workspace")

    actual_path = _resize_if_needed(file_path)
    resized = actual_path != file_path

    upload_ok = False
    for _ in range(2):
        upload_result = workspace.file_upload(actual_path, f"/workspace/{file_name}")
        if getattr(upload_result, "success", False):
            upload_ok = True
            break
        time.sleep(1)

    if not upload_ok:
        logger.info(f"SDK upload failed for {file_name}, falling back to base64 + bash")
        with open(actual_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")

        # file_name flows from annotator-controlled media (or the wire) — must
        # be shell-quoted before interpolation. The encoded base64 chunk is
        # newline-free + single-quote-free by construction so does not require
        # quoting.
        tmp_path = shlex.quote(f"/tmp/{file_name}.b64")
        dest_path = shlex.quote(f"/workspace/{file_name}")

        chunk_size = 65536  # avoid bash ARG_MAX limits
        first = True
        for i in range(0, len(encoded), chunk_size):
            chunk = encoded[i : i + chunk_size]
            operator = ">" if first else ">>"
            cmd = f"echo -n '{chunk}' {operator} {tmp_path}"
            workspace.execute_command(cmd, timeout=30.0)
            first = False

        decode_cmd = f"base64 -d {tmp_path} > {dest_path} && rm {tmp_path}"
        decode_result = workspace.execute_command(decode_cmd, timeout=30.0)
        if getattr(decode_result, "exit_code", -1) == 0:
            logger.info(f"Successfully uploaded {file_name} via base64 fallback")
        else:
            logger.error(
                f"Base64 fallback upload failed for {file_name}: "
                f"{getattr(decode_result, 'stderr', '')}"
            )

    if resized:
        os.unlink(actual_path)


# ---------------------------------------------------------------------------
# Agent loop — moved from GokuEvaluation.evaluate_instance (first half)
# ---------------------------------------------------------------------------
def run_agent(
    *,
    workspace: RemoteWorkspace,
    agent_llm: Any,
    instruction: str,
    input_files: list[str],
    instance_id: str,
    max_iterations: int,
) -> AgentRunResult:
    """Build Agent + Conversation, send the (optionally multimodal) message,
    drive the conversation to FinishAction, extract the final text response,
    and download the agent's output files to a fresh tempdir.

    Workspace is consumed by the Conversation (delete_on_close=True), so the
    caller does NOT separately clean it up; the conversation handles that on
    teardown. The returned `output_dir`, however, is the caller's to remove.
    """
    image_urls = _collect_image_urls(input_files)

    tools = get_default_tools(enable_browser=False)
    agent = Agent(llm=agent_llm, tools=tools, system_prompt_kwargs={"cli_mode": True})

    conversation = Conversation(
        agent=agent,
        workspace=workspace,
        max_iteration_per_run=max_iterations,
        delete_on_close=True,
    )

    if image_urls:
        msg = Message(
            role="user",
            content=[
                TextContent(text=instruction),
                ImageContent(image_urls=image_urls),
            ],
        )
        conversation.send_message(msg)
    else:
        conversation.send_message(instruction)

    run_conversation_with_fake_user_response(conversation)

    events: Sequence[Event] = conversation.state.events  # type: ignore[attr-defined]
    response_text = extract_response(events)

    input_file_names = [os.path.basename(f) for f in input_files]
    output_dir = download_outputs(workspace, instance_id, input_file_names)

    metrics: dict[str, Any] | None
    try:
        metrics = conversation.conversation_stats.get_combined_metrics()  # type: ignore[attr-defined]
    except Exception:
        metrics = None

    return AgentRunResult(
        response_text=response_text,
        output_dir=output_dir,
        events=list(events),
        metrics=metrics,
    )


def run_single_task(
    *,
    llm: Any,
    instruction: str,
    input_files: list[str],
    instance_id: str,
    max_iterations: int,
    forward_env: list[str] | None = None,
) -> AgentRunResult:
    """One-shot convenience: build workspace, upload files, run agent.

    `llm` is the raw LLM config returned by `load_llm_config(...)`; this
    function applies `build_eval_llm` internally so callers (the service in
    particular) don't have to know about that transform.
    """
    workspace = prepare_agent_workspace(input_files, forward_env=forward_env)
    agent_llm = build_eval_llm(llm)
    return run_agent(
        workspace=workspace,
        agent_llm=agent_llm,
        instruction=instruction,
        input_files=input_files,
        instance_id=instance_id,
        max_iterations=max_iterations,
    )


# ---------------------------------------------------------------------------
# Output extraction helpers — moved from GokuEvaluation
# ---------------------------------------------------------------------------
def _collect_image_urls(input_files: list[str]) -> list[str]:
    urls: list[str] = []
    for file_path in input_files:
        if not os.path.exists(file_path):
            continue
        ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
        if ext in ("jpg", "jpeg", "png", "gif", "webp", "bmp"):
            urls.append(_image_to_base64_url(file_path))
    return urls


def extract_response(events: Sequence[Event]) -> str:
    """Extract the agent's final text response. Mirrors GAIA's
    _extract_answer_from_history with a retry for RemoteConversation race
    conditions."""
    max_retries = 10
    retry_delay = 0.5

    for attempt in range(max_retries):
        for event in reversed(events):
            if not hasattr(event, "source") or event.source != "agent":  # type: ignore[attr-defined]
                continue

            text: str | None = None
            if isinstance(event, MessageEvent):
                if event.llm_message and event.llm_message.content:  # type: ignore[attr-defined]
                    content = event.llm_message.content[0]  # type: ignore[attr-defined]
                    if isinstance(content, TextContent):
                        text = content.text
            elif isinstance(event, ActionEvent) and isinstance(
                event.action, FinishAction
            ):
                text = event.action.message

            if text:
                return text

        if attempt < max_retries - 1:
            time.sleep(retry_delay)

    logger.warning("Could not extract agent response from events")
    return ""


def download_outputs(
    workspace: RemoteWorkspace,
    instance_id: str,
    input_file_names: list[str] | None = None,
) -> Path:
    """Download all files under /workspace into a fresh local tempdir, then
    return that dir. Excludes the originally-uploaded input files."""
    output_dir = Path(tempfile.mkdtemp(prefix=f"goku_{instance_id}_"))
    exclude_names = set(input_file_names or [])

    try:
        result = workspace.execute_command(
            "find /workspace -type f "
            "! -path '/workspace/conversations/*' "
            "! -path '/workspace/.openhands/*' "
            "2>/dev/null | head -200"
        )
        stdout = getattr(result, "output", "") or getattr(result, "stdout", "") or ""
        exit_code = getattr(result, "exit_code", -1)
        if exit_code == 0 and stdout.strip():
            for remote_path in stdout.strip().split("\n"):
                remote_path = remote_path.strip()
                if not remote_path:
                    continue
                file_name = os.path.basename(remote_path)
                if file_name in exclude_names:
                    continue
                rel_path = remote_path.replace("/workspace/", "", 1)
                local_path = output_dir / rel_path
                local_path.parent.mkdir(parents=True, exist_ok=True)
                _download_single_file(workspace, remote_path, local_path)
    except Exception as e:
        logger.warning(f"Failed to list workspace files: {e}")

    return output_dir


def _download_single_file(
    workspace: RemoteWorkspace, remote_path: str, local_path: Path
) -> None:
    """Download a single file, falling back to base64 via bash."""
    try:
        workspace.file_download(remote_path, str(local_path))
        if local_path.exists() and local_path.stat().st_size > 0:
            return
    except Exception:
        pass

    try:
        # remote_path is built from filenames found inside the agent workspace
        # (which accepted annotator-controlled filenames at upload time).
        # Quote unconditionally — a literal single quote in a filename would
        # break the prior form.
        result = workspace.execute_command(
            f"base64 {shlex.quote(remote_path)}", timeout=30.0
        )
        stdout = getattr(result, "output", "") or getattr(result, "stdout", "") or ""
        if getattr(result, "exit_code", -1) == 0 and stdout.strip():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(base64.b64decode(stdout.strip()))
            logger.info(f"Downloaded {remote_path} via base64 fallback")
        else:
            logger.warning(f"Failed to download {remote_path}")
    except Exception as e:
        logger.warning(f"Base64 download failed for {remote_path}: {e}")


# ---------------------------------------------------------------------------
# Judge-context helpers — moved from GokuEvaluation, also useful from the
# service when wiring `/judge` server-side.
# ---------------------------------------------------------------------------
_CONTENT_MAX_FILES = 20
_CONTENT_MAX_BYTES = 200_000  # 200 KB total across all files


def collect_file_contents(output_dir: Path) -> str:
    """Read output files and return a single formatted string for an LLM
    judge. Binary or oversized files are summarized rather than inlined.
    Stops after 20 files or 200 KB of text to keep judge prompts bounded."""
    contents: list[str] = []
    if not output_dir.exists():
        return "(no output files)"

    total_bytes = 0
    files_read = 0
    for f in sorted(output_dir.rglob("*")):
        if not f.is_file():
            continue
        if files_read >= _CONTENT_MAX_FILES or total_bytes >= _CONTENT_MAX_BYTES:
            contents.append("(remaining files truncated — limit reached)")
            break
        if f.stat().st_size > 50_000:
            contents.append(f"--- {f.name} --- (binary, {f.stat().st_size} bytes)")
            files_read += 1
            continue
        try:
            text = f.read_text(encoding="utf-8")
            chunk = text[:20000]
            contents.append(f"--- {f.name} ---\n{chunk}")
            total_bytes += len(chunk)
            files_read += 1
        except UnicodeDecodeError:
            contents.append(f"--- {f.name} --- (binary, {f.stat().st_size} bytes)")
            files_read += 1

    return "\n\n".join(contents) if contents else "(no output files)"


def format_trajectory(events: Sequence[Event]) -> str:
    """Format conversation events as a trajectory string for LLM judge context."""
    lines: list[str] = []
    for i, event in enumerate(events):
        event_type = type(event).__name__
        lines.append(f"[{i}] {event_type}")
        if hasattr(event, "action"):
            action = event.action  # type: ignore[attr-defined]
            action_type = type(action).__name__
            lines.append(f"    Action: {action_type}")
            if hasattr(action, "command"):
                cmd = str(action.command)[:200]
                lines.append(f"    Command: {cmd}")
            if hasattr(action, "message") and action.message:
                msg = str(action.message)[:200]
                lines.append(f"    Message: {msg}")
        if len(lines) > 500:
            lines.append("... (truncated)")
            break
    return "\n".join(lines)
