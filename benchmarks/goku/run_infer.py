"""Goku benchmark inference runner.

Orchestrates the full evaluation pipeline:
  1. Discover tasks from local folder structure
  2. Create Docker workspace per task
  3. Upload input files, run agent with multimodal instruction
  4. Download outputs, score against rubrics (deterministic + LLM judge)
  5. Write scores.jsonl per task

Follows the Terra/GAIA Evaluation pattern exactly.
"""

import base64
import mimetypes
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import List, Sequence

from dotenv import load_dotenv

from benchmarks.goku.config import INFER_DEFAULTS
from benchmarks.goku.models import RubricItem
from benchmarks.goku.scorers.deterministic import (
    DETERMINISTIC_TYPES,
    score_deterministic,
)
from benchmarks.goku.scorers.llm_judge import LLM_JUDGE_TYPES, score_llm_judge
from benchmarks.goku.scoring import compute_task_score, write_scores_jsonl
from benchmarks.goku.task_loader import discover_tasks
from benchmarks.utils.args_parser import get_parser
from benchmarks.utils.constants import EVAL_AGENT_SERVER_IMAGE
from benchmarks.utils.critics import create_critic
from benchmarks.utils.evaluation import Evaluation
from benchmarks.utils.evaluation_utils import (
    construct_eval_output_dir,
    get_default_on_result_writer,
)
from benchmarks.utils.fake_user_response import (
    run_conversation_with_fake_user_response,
)
from benchmarks.utils.image_utils import create_docker_workspace
from benchmarks.utils.litellm_proxy import build_eval_llm
from benchmarks.utils.llm_config import load_llm_config
from benchmarks.utils.models import EvalInstance, EvalMetadata, EvalOutput
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


load_dotenv()

logger = get_logger(__name__)


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
    """Detect actual image format from magic bytes, not file extension.

    Returns (mime_type, pil_format).
    """
    with open(image_path, "rb") as f:
        header = f.read(12)

    for sig, mime, fmt in _IMAGE_SIGNATURES:
        if header.startswith(sig):
            # Extra check: RIFF can be non-WebP (e.g., AVI)
            if sig == b"RIFF" and header[8:12] != b"WEBP":
                continue
            return mime, fmt

    # Fall back to extension-based guess
    ext_mime = mimetypes.guess_type(image_path)[0] or "image/png"
    ext_fmt = "JPEG" if ext_mime == "image/jpeg" else "PNG"
    return ext_mime, ext_fmt


def _resize_if_needed(image_path: str) -> str:
    """Return path to a processed copy if image exceeds dimension or size limits.

    Handles:
      - Pixel dimension > MAX_IMAGE_DIMENSION (Bedrock 8000px limit)
      - File size > MAX_IMAGE_BYTES (Bedrock 5MB per-image API limit)
      - MIME-type mismatch (e.g., .png extension but actually WebP)
    """
    import tempfile

    from PIL import Image

    real_mime, real_fmt = _detect_image_format(image_path)
    img = Image.open(image_path)
    file_size = os.path.getsize(image_path)
    max_dim = max(img.size)

    needs_resize = max_dim > MAX_IMAGE_DIMENSION
    needs_compress = file_size > MAX_IMAGE_BYTES

    # WebP files with .png extension need re-encoding regardless of size
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
        # Progressive downscale until under limit
        scale = (MAX_IMAGE_BYTES / file_size) ** 0.5  # rough area estimate
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        reasons.append(f"size {file_size / 1e6:.1f}→~{MAX_IMAGE_BYTES / 1e6:.1f}MB")

    if needs_reencode:
        reasons.append(f"reencode {ext_mime}→{real_mime}")

    # Convert to RGB if saving as JPEG (drop alpha)
    out_fmt = real_fmt if real_fmt in ("PNG", "JPEG") else "PNG"
    if out_fmt == "JPEG" and img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    suffix = ".jpg" if out_fmt == "JPEG" else ".png"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    # Save with compression, then check if still over limit
    quality = 85
    img.save(tmp_path, format=out_fmt, quality=quality, optimize=True)

    # If still over limit, progressively reduce quality / dimensions
    for attempt in range(3):
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
    # If we re-encoded, the output is always PNG or JPEG
    if resized_path != image_path:
        real_mime = "image/jpeg" if resized_path.endswith(".jpg") else "image/png"
    with open(resized_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    if resized_path != image_path:
        os.unlink(resized_path)
    return f"data:{real_mime};base64,{data}"


class GokuEvaluation(Evaluation):
    """Goku multimodal agentic evaluation benchmark.

    Implements the Evaluation ABC:
      - prepare_instances(): discover task folders, parse rubrics
      - prepare_workspace(): create Docker container, upload input files
      - evaluate_instance(): run agent, download outputs, score rubrics
    """

    def prepare_instances(self) -> List[EvalInstance]:
        """Load task instances from local task folder structure."""
        logger.info("Setting up Goku evaluation data")

        assert self.metadata.details is not None
        tasks_dir = Path(self.metadata.details.get("tasks_dir", "tasks"))

        if not tasks_dir.is_absolute():
            tasks_dir = Path.cwd() / tasks_dir

        if not tasks_dir.exists():
            raise FileNotFoundError(f"Tasks directory not found: {tasks_dir}")

        goku_instances = discover_tasks(tasks_dir)
        logger.info(f"Discovered {len(goku_instances)} tasks from {tasks_dir}")

        # Filter completed instances
        completed_instances = self._get_completed_instances()
        if completed_instances:
            goku_instances = [
                inst for inst in goku_instances if inst.id not in completed_instances
            ]
            logger.info(f"Filtered out {len(completed_instances)} completed instances")

        # Filter by selected_instances_file
        if self.metadata.selected_instances_file:
            with open(self.metadata.selected_instances_file) as f:
                selected_ids = {line.strip() for line in f if line.strip()}
            goku_instances = [
                inst for inst in goku_instances if inst.id in selected_ids
            ]
            logger.info(f"Filtered to {len(goku_instances)} selected instances")
            self.metadata.eval_limit = len(goku_instances)
        elif self.metadata.eval_limit and self.metadata.eval_limit > 0:
            goku_instances = goku_instances[: self.metadata.eval_limit]
            logger.info(f"Limited to {len(goku_instances)} instances")

        # Convert to EvalInstance format (data dict carries Goku fields)
        instances: List[EvalInstance] = []
        for goku_inst in goku_instances:
            instances.append(
                EvalInstance(
                    id=goku_inst.id,
                    data={
                        "instruction": goku_inst.instruction,
                        "rubric_items": [
                            item.model_dump() for item in goku_inst.rubric_items
                        ],
                        "input_files": goku_inst.input_files,
                    },
                )
            )

        logger.info(f"Total instances to process: {len(instances)}")
        return instances

    def prepare_workspace(
        self,
        instance: EvalInstance,
        resource_factor: int = 1,
        forward_env: list[str] | None = None,
    ) -> RemoteWorkspace:
        """Create Docker workspace and upload input files."""
        logger.info(f"Preparing workspace for instance {instance.id}")

        import platform as _platform

        docker_platform = (
            "linux/arm64" if _platform.machine() == "arm64" else "linux/amd64"
        )

        workspace = create_docker_workspace(
            agent_server_image=EVAL_AGENT_SERVER_IMAGE,
            base_image="nikolaik/python-nodejs:python3.12-nodejs22",
            build_target="binary",
            forward_env=forward_env or [],
            platform=docker_platform,
        )

        # Create workspace directories and verify filesystem is writable.
        # The agent-server health check only confirms HTTP readiness, not
        # filesystem readiness.  Retry mkdir to give the container a moment.
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

        # Upload ALL input files to workspace (images also uploaded for agent access)
        # Use base64 + execute_command as a fallback because some agent-server
        # images have a bug where the /api/file/upload endpoint returns 500 due
        # to FastAPI redirect_slashes dropping query parameters.
        input_files: list[str] = instance.data.get("input_files", [])
        for file_path in input_files:
            if not os.path.exists(file_path):
                logger.warning(f"Input file not found: {file_path}")
                continue

            file_name = os.path.basename(file_path)
            logger.info(f"Uploading {file_name} to workspace")

            # Resize oversized images before upload so the agent-server doesn't
            # hit Bedrock's 8000px dimension limit when re-sending to the model.
            actual_path = _resize_if_needed(file_path)
            resized = actual_path != file_path

            upload_ok = False
            for attempt in range(2):
                upload_result = workspace.file_upload(
                    actual_path, f"/workspace/{file_name}"
                )
                if getattr(upload_result, "success", False):
                    upload_ok = True
                    break
                time.sleep(1)

            if not upload_ok:
                logger.info(
                    f"SDK upload failed for {file_name}, falling back to base64 + bash"
                )

                with open(actual_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("ascii")

                chunk_size = 65536  # avoid bash ARG_MAX limits
                first = True
                for i in range(0, len(encoded), chunk_size):
                    chunk = encoded[i : i + chunk_size]
                    operator = ">" if first else ">>"
                    cmd = f"echo -n '{chunk}' {operator} /tmp/{file_name}.b64"
                    workspace.execute_command(cmd, timeout=30.0)
                    first = False

                decode_cmd = (
                    f"base64 -d /tmp/{file_name}.b64 "
                    f"> /workspace/{file_name} && "
                    f"rm /tmp/{file_name}.b64"
                )
                decode_result = workspace.execute_command(decode_cmd, timeout=30.0)
                if getattr(decode_result, "exit_code", -1) == 0:
                    upload_ok = True
                    logger.info(
                        f"Successfully uploaded {file_name} via base64 fallback"
                    )
                else:
                    logger.error(
                        f"Base64 fallback upload failed for {file_name}: "
                        f"{getattr(decode_result, 'stderr', '')}"
                    )

            if resized:
                os.unlink(actual_path)

        return workspace  # type: ignore[return-value]

    def evaluate_instance(
        self, instance: EvalInstance, workspace: RemoteWorkspace
    ) -> EvalOutput:
        """Run agent on a Goku task and score the results."""
        logger.info(f"Evaluating instance {instance.id}")

        instruction: str = instance.data["instruction"]
        input_files: list[str] = instance.data.get("input_files", [])

        # Collect image URLs for multimodal message
        image_urls: list[str] = []
        for file_path in input_files:
            if not os.path.exists(file_path):
                continue
            extension = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
            if extension in ("jpg", "jpeg", "png", "gif", "webp", "bmp"):
                image_urls.append(_image_to_base64_url(file_path))

        # Create agent
        agent_llm = build_eval_llm(self.metadata.llm)
        tools = get_default_tools(enable_browser=False)

        agent = Agent(
            llm=agent_llm,
            tools=tools,
            system_prompt_kwargs={"cli_mode": True},
        )

        # Create conversation
        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            max_iteration_per_run=self.metadata.max_iterations,
            delete_on_close=True,
        )

        # Send multimodal message
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

        # Run conversation
        run_conversation_with_fake_user_response(conversation)

        # Extract response from conversation events
        events: Sequence[Event] = conversation.state.events  # type: ignore[attr-defined]
        response_text = self._extract_response(events)

        # Download output files from workspace
        input_file_names = [
            os.path.basename(f) for f in instance.data.get("input_files", [])
        ]
        output_dir = self._download_outputs(workspace, instance.id, input_file_names)

        # Collect file contents for LLM judge context
        file_contents = self._collect_file_contents(output_dir)

        # Collect trajectory for LLM judge context
        trajectory = self._format_trajectory(events)

        # Score all rubric items
        rubric_items = [
            RubricItem(**item_data) for item_data in instance.data["rubric_items"]
        ]

        results = []
        for item in rubric_items:
            if item.type in DETERMINISTIC_TYPES:
                result = score_deterministic(item, output_dir, response_text)
            elif item.type in LLM_JUDGE_TYPES:
                details = self.metadata.details or {}
                judge_model = details.get("judge_model") or os.getenv(
                    "GOKU_JUDGE_MODEL",
                    "bedrock/converse/moonshotai.kimi-k2.5",
                )
                judge_api_key = details.get("judge_api_key") or os.getenv(
                    "AWS_BEARER_TOKEN_BEDROCK"
                )
                # Region resolution order:
                #   1) explicit `aws_region_name` in --judge-llm-config JSON
                #   2) AWS_REGION_NAME env var
                #   3) For Bedrock ARN models, parse the region from the ARN
                #      itself (`arn:aws:bedrock:<region>:...`).
                #   4) Last-resort fallback: us-east-1, with a warning.
                judge_region = details.get("judge_region") or os.getenv(
                    "AWS_REGION_NAME"
                )
                if not judge_region and judge_model.startswith("bedrock/"):
                    import re as _re
                    m = _re.search(r"arn:aws:bedrock:([a-z0-9-]+):", judge_model)
                    if m:
                        judge_region = m.group(1)
                        logger.info(
                            "AWS_REGION_NAME not set; using region '%s' "
                            "parsed from Bedrock ARN.",
                            judge_region,
                        )
                if not judge_region:
                    judge_region = "us-east-1"
                    if judge_model.startswith("bedrock/"):
                        logger.warning(
                            "AWS_REGION_NAME not set and could not infer from "
                            "judge model; falling back to us-east-1. Set "
                            "AWS_REGION_NAME or aws_region_name in your "
                            "judge LLM config if Bedrock is in a different region."
                        )
                result = score_llm_judge(
                    item=item,
                    response=response_text,
                    file_contents=file_contents,
                    trajectory=trajectory,
                    judge_model=judge_model,
                    judge_api_key=judge_api_key,
                    aws_region_name=judge_region,
                )
            else:
                raise ValueError(
                    f"Unknown rubric type: {item.type} for item #{item.number}"
                )
            results.append(result)

        # Compute task score
        task_score = compute_task_score(results, rubric_items)

        # Persist results to eval output dir
        eval_task_dir = Path(self.metadata.eval_output_dir) / instance.id
        eval_task_dir.mkdir(parents=True, exist_ok=True)

        # Write scores.jsonl
        eval_scores_path = eval_task_dir / "scores.jsonl"
        write_scores_jsonl(task_score, eval_scores_path, rubric_items=rubric_items)

        # Copy agent output files to results/ subdir for delivery packaging
        eval_results_dir = eval_task_dir / "results"
        if eval_results_dir.exists():
            shutil.rmtree(eval_results_dir)
        shutil.copytree(output_dir, eval_results_dir, dirs_exist_ok=True)

        logger.info(
            f"Instance {instance.id}: per_task_score={task_score.per_task_score:.4f}, "
            f"passed={task_score.passed}, "
            f"awarded={task_score.awarded}/{task_score.max_total}"
        )

        # Return evaluation output
        return EvalOutput(
            instance_id=instance.id,
            attempt=self.current_attempt,
            test_result={
                "per_task_score": task_score.per_task_score,
                "raw_score": task_score.raw_score,
                "passed": task_score.passed,
                "awarded": task_score.awarded,
                "max_total": task_score.max_total,
                "judge_cost_usd": task_score.judge_cost_usd,
            },
            instruction=instruction,
            error=None,
            history=list(events),
            metrics=conversation.conversation_stats.get_combined_metrics(),  # type: ignore[attr-defined]
            instance=instance.data,
        )

    def _extract_response(self, events: Sequence[Event]) -> str:
        """Extract the agent's final text response from conversation events.

        Mirrors GAIA's _extract_answer_from_history pattern with retry for
        RemoteConversation race conditions.
        """
        max_retries = 10
        retry_delay = 0.5

        for attempt in range(max_retries):
            for event in reversed(events):
                # Check for agent-sourced events
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

    def _download_outputs(
        self,
        workspace: RemoteWorkspace,
        instance_id: str,
        input_file_names: list[str] | None = None,
    ) -> Path:
        """Download output files from workspace to local temp dir.

        Downloads all files recursively from /workspace/, excluding the
        original input files that were uploaded.
        """
        output_dir = Path(tempfile.mkdtemp(prefix=f"goku_{instance_id}_"))
        exclude_names = set(input_file_names or [])

        try:
            result = workspace.execute_command(
                "find /workspace -type f "
                "! -path '/workspace/conversations/*' "
                "! -path '/workspace/.openhands/*' "
                "2>/dev/null | head -200"
            )
            stdout = (
                getattr(result, "output", "") or getattr(result, "stdout", "") or ""
            )
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
                    self._download_single_file(workspace, remote_path, local_path)
        except Exception as e:
            logger.warning(f"Failed to list workspace files: {e}")

        return output_dir

    @staticmethod
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
            result = workspace.execute_command(f"base64 '{remote_path}'", timeout=30.0)
            stdout = (
                getattr(result, "output", "") or getattr(result, "stdout", "") or ""
            )
            if getattr(result, "exit_code", -1) == 0 and stdout.strip():
                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_bytes(base64.b64decode(stdout.strip()))
                logger.info(f"Downloaded {remote_path} via base64 fallback")
            else:
                logger.warning(f"Failed to download {remote_path}")
        except Exception as e:
            logger.warning(f"Base64 download failed for {remote_path}: {e}")

    def _collect_file_contents(self, output_dir: Path) -> str:
        """Read output files and return them as a formatted string for LLM judge."""
        contents: list[str] = []
        if not output_dir.exists():
            return "(no output files)"

        for f in sorted(output_dir.rglob("*")):
            if not f.is_file():
                continue
            if f.stat().st_size > 50_000:
                contents.append(f"--- {f.name} --- (binary, {f.stat().st_size} bytes)")
                continue
            try:
                text = f.read_text(encoding="utf-8")
                contents.append(f"--- {f.name} ---\n{text[:20000]}")
            except UnicodeDecodeError:
                contents.append(f"--- {f.name} --- (binary, {f.stat().st_size} bytes)")

        return "\n\n".join(contents) if contents else "(no output files)"

    def _format_trajectory(self, events: Sequence[Event]) -> str:
        """Format conversation events as trajectory string for LLM judge context."""
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


def main() -> None:
    """Main entry point for Goku evaluation."""
    parser = get_parser()
    parser.add_argument(
        "--tasks-dir",
        type=str,
        default="tasks",
        help="Path to the tasks directory (default: ./tasks)",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Run a single task by key (e.g., task_e25b6d)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of evaluation runs per task (default: 3)",
    )
    parser.add_argument(
        "--judge-llm-config",
        type=str,
        default=None,
        help="Path to LLM config JSON for the judge model (e.g., .llm_config/kimi-k2.5-judge.json)",
    )
    parser.set_defaults(**INFER_DEFAULTS)
    args = parser.parse_args()

    # Handle single task selection
    instance_select_file = None
    if args.task:
        instance_select_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        )
        instance_select_file.write(args.task + "\n")
        instance_select_file.close()
        args.select = instance_select_file.name
        args.n_limit = 1

    # Create critic (GAIA pattern — PassCritic by default)
    critic = create_critic(args)

    llm = load_llm_config(args.llm_config_path)
    logger.info("Using LLM config: %s", llm.model_dump_json(indent=2))

    # Resolve the display name used for output directory naming. The actual
    # LLM `model` identifier may be a long Bedrock ARN (e.g.
    # "bedrock/converse/arn:aws:bedrock:..."), which is required by LiteLLM
    # but produces awful directory names. Resolution order:
    #   1) Optional `display_name` field in the LLM config JSON
    #   2) The config filename stem (e.g. "claude-opus-4.7.json" -> "claude-opus-4.7")
    #   3) Fall back to model.replace("/", "_") (legacy behavior)
    def _resolve_model_display_name(config_path: str, llm_model: str) -> str:
        import json as _json
        try:
            with open(config_path, encoding="utf-8") as _f:
                _raw = _json.load(_f)
            explicit = _raw.get("display_name")
            if explicit and isinstance(explicit, str) and explicit.strip():
                return explicit.strip()
        except (OSError, _json.JSONDecodeError):
            pass
        stem = os.path.splitext(os.path.basename(config_path))[0]
        if stem:
            return stem
        return llm_model.replace("/", "_")

    model_display_name = _resolve_model_display_name(
        args.llm_config_path, llm.model
    )
    logger.info(
        "Output directory display name: %s (actual model: %s)",
        model_display_name,
        llm.model,
    )

    # Load judge LLM config if provided
    judge_llm = None
    if args.judge_llm_config:
        judge_llm = load_llm_config(args.judge_llm_config)
        logger.info("Using judge LLM config: %s", judge_llm.model)
    else:
        logger.info("No judge LLM config provided — will use env vars for judge")

    # Run evaluation for each run
    for run_num in range(1, args.runs + 1):
        logger.info(f"=== Run {run_num}/{args.runs} ===")

        # Standard OpenHands eval convention: dir layout includes a `goku/`
        # benchmark namespace + `_sdk_<sha>_maxiter_<N>` suffix so runs from
        # different SDK versions / iteration caps don't silently collide.
        # Cosmetic cleanup happens at delivery export time, not here.
        output_dir = construct_eval_output_dir(
            base_dir=os.path.join(args.output_dir, f"run_{run_num}"),
            dataset_name="goku",
            model_name=model_display_name,
            max_iterations=args.max_iterations,
            eval_note=args.note,
        )

        metadata = EvalMetadata(
            llm=llm,
            dataset="goku",
            dataset_split="test",
            max_iterations=args.max_iterations,
            eval_output_dir=output_dir,
            details={
                "tasks_dir": args.tasks_dir,
                "run_number": run_num,
                "judge_model": judge_llm.model if judge_llm else None,
                "judge_api_key": judge_llm.api_key if judge_llm else None,
                "judge_region": (judge_llm.aws_region_name if judge_llm else None),
                # Hint stored for the delivery exporter — original LLM
                # identifiers (metadata.llm.model, details.judge_model) stay
                # authoritative; the exporter uses these display names when
                # building the delivery package.
                "model_display_name": model_display_name,
                "judge_model_display_name": (
                    os.path.splitext(os.path.basename(args.judge_llm_config))[0]
                    if args.judge_llm_config else None
                ),
            },
            eval_limit=args.n_limit,
            n_critic_runs=getattr(args, "n_critic_runs", 1),
            critic=critic,
            selected_instances_file=getattr(args, "select", None),
            max_retries=args.max_retries,
            workspace_type=args.workspace,
        )

        evaluator = GokuEvaluation(metadata=metadata, num_workers=args.num_workers)

        evaluator.run(on_result=get_default_on_result_writer(evaluator.output_path))

    # Cleanup
    if instance_select_file:
        os.unlink(instance_select_file.name)

    logger.info("Goku evaluation complete.")


if __name__ == "__main__":
    main()
