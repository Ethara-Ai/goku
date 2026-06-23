"""Goku HTTP service — wire entry point for mm_tasker (and any future client).

Wraps the goku eval engine (agent runner + scorers) behind two HTTP routes
that mm_tasker calls:

    POST /run     Run an agent against (prompt + media), score the workspace
                  against rubrics, return response + scores in one trip.
    POST /judge   Thin LLM call. Used for response_criteria grading after
                  the agent has finished.

The agent runner is intentionally a seam (``_run_agent_in_workspace``) — it
ships with a deterministic mock so the service is testable end-to-end before
the real OpenHands/Docker harness is wired in. The mock is enabled via the
env var GOKU_SERVICE_MODE=mock; flip to GOKU_SERVICE_MODE=live to require a
real runner.

Run it
------
    pip install fastapi 'uvicorn[standard]' pydantic
    # Dev:
    uvicorn benchmarks.goku.service:app --reload --port 8000
    # Prod:
    uvicorn benchmarks.goku.service:app --host 0.0.0.0 --port 8000 --workers 4

Environment
-----------
    GOKU_SERVICE_TOKEN           Bearer token clients must send. Required.
    GOKU_SERVICE_MODE            'mock' (default) | 'live'
    GOKU_SERVICE_OUTPUT_DIR      Root directory the service writes task folders
                                 into. Default './eval_outputs'. Layout (matches
                                 the delivery export shape in the README):
                                   <root>/MM Agentic Pilot Samples-YYYY-MM-DD/
                                     └── tasks/<task_id>/
                                          ├── instruction.md
                                          ├── rubrics.jsonl
                                          ├── data/input_files/
                                          └── runs/<model>/run_<N>/
                                                ├── scores.jsonl
                                                └── results/<agent files>
                                 <N> comes from the request's `run_index`
                                 (mm_tasker's mm.tasker.run.run_index).
    GOKU_SERVICE_BUNDLE_NAME     Override for the dated bundle directory name.
                                 Default 'MM Agentic Pilot Samples-{today UTC}'.
                                 Pin this when you want a stable bundle name
                                 across days (e.g. for a delivery snapshot).
    GOKU_LLM_CONFIG_DIR          Directory the live runner scans for per-model
                                 LLM configs. Default '.llm_config'. Each JSON
                                 file is matched against the request's `model`
                                 field by filename stem, `display_name`, or
                                 `model` value.
    GOKU_SERVICE_MAX_ITERATIONS  Per-run cap on agent iterations in live mode.
                                 Default 30 (matches the CLI default).
    GOKU_ALLOW_SHELL_RUBRICS     '1' to enable shell_succeeds_real scoring.
                                 Inherited by benchmarks.goku.scorers.deterministic.
    GOKU_MAX_REQUEST_BYTES       Hard cap on request body size. Default 200 MB.
    GOKU_MAX_RUBRICS             Reject requests with more rubrics. Default 200.
    GOKU_IDEMPOTENCY_TTL_S       How long to remember idempotency_key results.
                                 Default 600 (10 min).
    GOKU_JOB_TTL_S               How long completed/failed job results are kept
                                 in the in-memory job store before expiry.
                                 Default 3600 (1 h). Must be ≥ the client's
                                 total polling budget (mm_tasker.run_timeout).

Wire contract
-------------
Defined inline below as Pydantic models. Matches custom_addons/mm_tasker/
models/agent_dispatcher.py (the client side).
"""

from __future__ import annotations

import base64
import hmac
import io
import json
import logging
import mimetypes
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError, field_validator


# Load .env BEFORE the config block below reads os.environ. Searches up from
# CWD by default, so running uvicorn from anywhere inside the repo works.
load_dotenv()

from benchmarks.goku.models import RubricItem, ScorerResult  # noqa: E402
from benchmarks.goku.scorers.deterministic import (  # noqa: E402
    DETERMINISTIC_TYPES,
    score_deterministic,
)


logger = logging.getLogger("goku.service")
logging.basicConfig(
    level=os.environ.get("GOKU_SERVICE_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


# ---------------------------------------------------------------------------
# Configuration (env-driven; immutable at process start)
# ---------------------------------------------------------------------------
SERVICE_TOKEN = os.environ.get("GOKU_SERVICE_TOKEN", "").strip()
SERVICE_MODE = os.environ.get("GOKU_SERVICE_MODE", "mock").strip().lower()
SERVICE_OUTPUT_DIR = Path(
    os.environ.get("GOKU_SERVICE_OUTPUT_DIR", "eval_outputs")
).resolve()
LLM_CONFIG_DIR = Path(os.environ.get("GOKU_LLM_CONFIG_DIR", ".llm_config")).resolve()
SERVICE_MAX_ITERATIONS = int(os.environ.get("GOKU_SERVICE_MAX_ITERATIONS", "30"))

# Default judge model — overridden per-request by GOKU_JUDGE_MODEL env var.
# Credentials are intentionally read per-request (not at startup) so that
# AWS bearer-token rotation is picked up without a service restart.
_DEFAULT_JUDGE_MODEL = "bedrock/converse/moonshotai.kimi-k2.5"

# Bundle name resolution: explicit env override wins; otherwise compute
# "MM Agentic Pilot Samples-<today UTC>" *per request* so each day's runs
# land in their own bundle by default. The env-pinned form is for
# deployments that want a stable name across days.
_BUNDLE_NAME_OVERRIDE = os.environ.get("GOKU_SERVICE_BUNDLE_NAME", "").strip()


def _current_bundle_dir() -> Path:
    """Return <SERVICE_OUTPUT_DIR>/<bundle name>. The bundle name is the env
    override if set, otherwise 'MM Agentic Pilot Samples-YYYY-MM-DD' using
    today's UTC date.
    """
    if _BUNDLE_NAME_OVERRIDE:
        name = _BUNDLE_NAME_OVERRIDE
    else:
        today = datetime.now(timezone.utc).date().isoformat()
        name = f"MM Agentic Pilot Samples-{today}"
    return SERVICE_OUTPUT_DIR / name


MAX_REQUEST_BYTES = int(
    os.environ.get("GOKU_MAX_REQUEST_BYTES", str(200 * 1024 * 1024))
)
MAX_RUBRICS = int(os.environ.get("GOKU_MAX_RUBRICS", "200"))
IDEMPOTENCY_TTL_S = int(os.environ.get("GOKU_IDEMPOTENCY_TTL_S", "600"))
JOB_TTL_S = int(os.environ.get("GOKU_JOB_TTL_S", "3600"))

if SERVICE_MODE == "mock":
    # Surface this prominently at startup. Defaulting to mock is convenient
    # for dev but is a footgun in prod — a missing GOKU_SERVICE_MODE env
    # should not silently serve fake responses.
    logger.warning(
        "GOKU_SERVICE_MODE=mock — /run will return a stubbed agent response. "
        "Set GOKU_SERVICE_MODE=live for real inference."
    )

# Rubric types the LLM judge handles. The /run endpoint deliberately skips
# these — mm_tasker's separate "Run Judge" flow hits /judge instead.
LLM_JUDGE_TYPES = frozenset({"response_criteria", "response_not_criteria"})


# ---------------------------------------------------------------------------
# Wire contract — Pydantic models
#
# These mirror the JSON shapes in mm_tasker/models/agent_dispatcher.py.
# Keep them aligned; if you change one side, change the other.
# ---------------------------------------------------------------------------
class WireMedia(BaseModel):
    """A single media file as sent over the wire."""

    name: str = ""
    filename: str = ""
    kind: str = "other"  # image | pdf | video | other
    mime_type: str = ""
    data_b64: str  # base64-encoded file contents


class WireRubric(BaseModel):
    """A rubric in mm_tasker's wire shape.

    Carries the parsed fields plus the original JSONL line. We rebuild a
    full ``RubricItem`` from these — ``raw_json`` is the source of truth
    for type-specific fields (paths, pattern, needles, etc.) that
    mm_tasker doesn't model directly.
    """

    number: int = Field(..., ge=1)
    type: str
    category: str = ""
    importance: str = "nice_to_have"
    points: int = 0
    criterion: str = ""
    raw_json: str = ""


class RunRequest(BaseModel):
    """POST /run body.

    `task_id` names the on-disk task folder. mm_tasker is expected to pass its
    own task record id (or a stable derived key); the service does not generate
    one.

    `run_index` names the per-run subdirectory under the model. Mirrors
    mm_tasker's mm.tasker.run.run_index — when mm_tasker dispatches the 3rd
    run of a task against a given model, it should send run_index=3 and the
    service writes to runs/<model>/run_3/. Defaults to 1 so older clients
    still produce a valid layout; mm_tasker should populate it explicitly.
    """

    task_id: str
    run_index: int = Field(default=1, ge=1, le=10_000)
    model: str
    system_prompt: str = "You are a helpful assistant."
    user_prompt: str
    media: list[WireMedia] = Field(default_factory=list)
    rubrics: list[WireRubric] = Field(default_factory=list)
    idempotency_key: str | None = None

    @field_validator("task_id")
    @classmethod
    def _task_id_shape(cls, v: str) -> str:
        # Filesystem-safe: alphanumeric + - _ . and not too long. No '/', no
        # '..', no leading dot. Mirrors the wire-level safety we apply to
        # media filenames at _materialize_media.
        if not v or len(v) > 128:
            raise ValueError("task_id must be non-empty and <=128 chars")
        if v.startswith(".") or "/" in v or "\\" in v or ".." in v:
            raise ValueError(
                "task_id must not start with '.' or contain '/', '\\\\', or '..'"
            )
        if not all(c.isalnum() or c in "-_." for c in v):
            raise ValueError("task_id may only contain alphanumeric + - _ .")
        return v

    @field_validator("idempotency_key")
    @classmethod
    def _idem_shape(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not v or len(v) > 64 or not all(c.isalnum() or c in "-_" for c in v):
            raise ValueError("idempotency_key must be alphanumeric + - _ , <=64 chars")
        return v


class WireScore(BaseModel):
    """One per-rubric verdict in the response."""

    rubric_number: int
    passed: bool
    triggered: bool = False
    judged_by: str  # 'probe' | 'regex' | 'llm_judge' | 'not_applicable' | 'error'
    rationale: str = ""
    awarded_points: int = 0
    # LLM-judge metadata; ignored by mm_tasker unless judged_by == 'llm_judge'
    judge_confidence: float | None = None
    judge_raw_response: str | None = None
    judge_tokens_in: int | None = None
    judge_tokens_out: int | None = None


class OutputFile(BaseModel):
    name: str
    file: str  # base64 contents
    sha256: str = ""
    size_bytes: int = 0


class RunResponse(BaseModel):
    """POST /run reply — also embedded in JobStatusResponse when status=='done'."""

    job_id: str
    response_text: str
    tokens_in: int = 0
    tokens_out: int = 0
    output_files: list[OutputFile] = Field(default_factory=list)
    scores: list[WireScore] = Field(default_factory=list)


class JobStatusResponse(BaseModel):
    """POST /run 202 reply and GET /jobs/{job_id} reply.

    status values:
      queued  — job accepted, not yet started
      running — agent is executing
      done    — completed; response_text / scores / output_files populated
      error   — failed; error field has the reason
    """

    job_id: str
    status: str
    # Populated when status == 'done'
    response_text: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    output_files: list[OutputFile] | None = None
    scores: list[WireScore] | None = None
    # Populated when status == 'error'
    error: str | None = None


class RegradeRequest(BaseModel):
    """POST /regrade body — re-score an existing run with new rubrics,
    without re-running the agent. Mirror of RunRequest minus the
    agent-execution fields (user_prompt, media, system_prompt) since
    those are already on disk from the original /run call.
    """

    task_id: str
    run_index: int = Field(default=1, ge=1, le=10_000)
    model: str
    rubrics: list[WireRubric] = Field(default_factory=list)
    response_text: str = ""
    # Optional bundle override. Defaults to today's UTC bundle; pass
    # explicitly if regrade-ing a run from an earlier day.
    bundle_name: str | None = None
    # Judge model key for LLM-judge rubrics. If provided, resolved via
    # .llm_config/ (same logic as /judge). Falls back to GOKU_JUDGE_MODEL
    # env var if absent.
    judge_model: str = ""


class RegradeResponse(BaseModel):
    """POST /regrade reply. Same envelope shape as RunResponse so the
    mm_tasker client's existing score-row parser works unchanged.
    output_files / tokens are omitted — nothing new was generated.
    """

    job_id: str
    response_text: str = ""
    scores: list[WireScore] = Field(default_factory=list)


class JudgeRequest(BaseModel):
    """POST /judge body — matches mm_tasker._call_judge's existing payload."""

    model: str
    system_prompt: str
    user_prompt: str
    media: list[WireMedia] = Field(default_factory=list)


class JudgeResponse(BaseModel):
    """POST /judge reply — same envelope mm_tasker's judge already parses."""

    response_text: str
    tokens_in: int = 0
    tokens_out: int = 0


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
def require_bearer(authorization: str = Header(default="")) -> None:
    """Reject any request missing/wrong Bearer token.

    SERVICE_TOKEN must be set at process start; an empty value fails closed
    so a misconfigured deploy doesn't silently accept anonymous traffic.
    """
    if not SERVICE_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GOKU_SERVICE_TOKEN not configured on server",
        )
    expected = f"Bearer {SERVICE_TOKEN}"
    if not hmac.compare_digest(authorization, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing bearer token",
        )


# ---------------------------------------------------------------------------
# Idempotency cache (in-memory, single-process)
#
# Real deployments behind multiple uvicorn workers should swap this for
# Redis or a shared store — see the README. For one worker / dev, this is
# good enough and dependency-free.
# ---------------------------------------------------------------------------
class _IdempotencyCache:
    """Maps idempotency_key → job_id so repeat submissions return the same job."""

    def __init__(self, ttl_s: int):
        self._ttl = ttl_s
        self._store: dict[str, tuple[float, str]] = {}

    def get(self, key: str) -> str | None:
        entry = self._store.get(key)
        if not entry:
            return None
        when, value = entry
        if (time.time() - when) > self._ttl:
            self._store.pop(key, None)
            return None
        return value

    def put(self, key: str, job_id: str) -> None:
        self._store[key] = (time.time(), job_id)
        # Opportunistic GC; cheap because most entries get TTL'd anyway.
        if len(self._store) > 10_000:
            cutoff = time.time() - self._ttl
            self._store = {k: v for k, v in self._store.items() if v[0] > cutoff}


_idempotency = _IdempotencyCache(IDEMPOTENCY_TTL_S)


# ---------------------------------------------------------------------------
# Async job store (in-memory, single-process)
#
# Holds the state of every background /run job: queued → running → done|error.
# Thread-safe via a lock; the background worker thread writes, the poll
# endpoint reads. Multi-worker deployments should swap this for Redis.
# ---------------------------------------------------------------------------
class _JobStore:
    def __init__(self, ttl_s: int):
        self._ttl = ttl_s
        self._store: dict[str, tuple[float, dict]] = {}
        self._lock = threading.Lock()

    def put(self, job_id: str, data: dict) -> None:
        with self._lock:
            self._store[job_id] = (time.time(), data)
            if len(self._store) > 10_000:
                cutoff = time.time() - self._ttl
                self._store = {k: v for k, v in self._store.items() if v[0] > cutoff}

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            entry = self._store.get(job_id)
        if not entry:
            return None
        when, data = entry
        if (time.time() - when) > self._ttl:
            with self._lock:
                self._store.pop(job_id, None)
            return None
        return data


_job_store = _JobStore(JOB_TTL_S)


# ---------------------------------------------------------------------------
# Workspace helpers
# ---------------------------------------------------------------------------
@contextmanager
def _workspace(prefix: str = "goku-"):
    """Per-request temp directory. Always cleaned up, even on errors."""
    path = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Image processing — resize/re-encode before writing to workspace so the
# agent never hits Bedrock's 8000px dimension or 5 MB per-image limits.
# ---------------------------------------------------------------------------
_MAX_IMAGE_DIMENSION = 7680  # Bedrock limit is 8000px; leave margin
_MAX_IMAGE_BYTES = 3_500_000  # ~4.7 MB base64, safely under Bedrock's 5 MB cap

_IMAGE_SIGNATURES: list[tuple[bytes, str]] = [
    (b"RIFF", "image/webp"),
    (b"\x89PNG", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF8", "image/gif"),
]
_IMAGE_MIME_TYPES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})


def _detect_image_mime(blob: bytes) -> str | None:
    """Return MIME type from magic bytes, or None if not a recognised image."""
    header = blob[:12]
    for sig, mime in _IMAGE_SIGNATURES:
        if header.startswith(sig):
            if sig == b"RIFF" and header[8:12] != b"WEBP":
                continue
            return mime
    return None


def _process_image(blob: bytes, filename: str) -> bytes:
    """Resize / re-encode image bytes to stay within Bedrock's limits.

    Returns the original bytes unchanged if no processing is needed.
    Lazy-imports PIL so mock-mode startup stays dependency-free.
    """
    mime = _detect_image_mime(blob)
    if mime not in _IMAGE_MIME_TYPES:
        return blob

    from PIL import Image

    img = Image.open(io.BytesIO(blob))
    max_dim = max(img.size)
    needs_resize = max_dim > _MAX_IMAGE_DIMENSION
    needs_compress = len(blob) > _MAX_IMAGE_BYTES
    ext_mime = mimetypes.guess_type(filename)[0] or ""
    needs_reencode = bool(ext_mime) and ext_mime != mime

    if not needs_resize and not needs_compress and not needs_reencode:
        return blob

    if needs_resize:
        scale = _MAX_IMAGE_DIMENSION / max_dim
        img = img.resize(
            (int(img.size[0] * scale), int(img.size[1] * scale)),
            Image.Resampling.LANCZOS,
        )
    elif needs_compress:
        scale = (_MAX_IMAGE_BYTES / len(blob)) ** 0.5
        img = img.resize(
            (int(img.size[0] * scale), int(img.size[1] * scale)),
            Image.Resampling.LANCZOS,
        )

    out_fmt = "JPEG" if mime == "image/jpeg" else "PNG"
    if out_fmt == "JPEG" and img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    buf = io.BytesIO()
    quality = 85
    img.save(buf, format=out_fmt, quality=quality, optimize=True)
    result = buf.getvalue()

    # Progressive recompress if still over the size limit
    for _ in range(3):
        if len(result) <= _MAX_IMAGE_BYTES:
            break
        quality = max(quality - 15, 30)
        img = img.resize(
            (int(img.size[0] * 0.75), int(img.size[1] * 0.75)),
            Image.Resampling.LANCZOS,
        )
        buf = io.BytesIO()
        img.save(buf, format=out_fmt, quality=quality, optimize=True)
        result = buf.getvalue()

    logger.info("image %r: %d → %d bytes", filename, len(blob), len(result))
    return result


def _materialize_media(workspace: Path, media: list[WireMedia]) -> None:
    """Decode base64 media into the workspace. Images are resized/re-encoded
    to stay within Bedrock's dimension and size limits before being written.
    """
    for m in media:
        fname = (m.filename or m.name or "").strip()
        if not fname:
            raise HTTPException(400, "media entry missing filename")
        if "/" in fname or "\\" in fname or fname.startswith("."):
            raise HTTPException(400, f"unsafe media filename: {fname!r}")
        try:
            blob = base64.b64decode(m.data_b64, validate=True)
        except (ValueError, TypeError) as e:
            raise HTTPException(400, f"media {fname!r} bad base64: {e}")
        (workspace / fname).write_bytes(_process_image(blob, fname))


# ---------------------------------------------------------------------------
# Rubric rebuild — wire shape → goku's internal RubricItem
# ---------------------------------------------------------------------------
def _rebuild_rubric_item(wire: WireRubric) -> RubricItem:
    """Merge parsed wire fields with raw_json for type-specific keys.

    mm_tasker only models the rubric fields it cares about (number, type,
    category, points, importance, criterion). Type-specific fields
    (paths, pattern, needles, raw_shell, source) live inside raw_json
    because mm_tasker never parsed them. We reconstruct the full shape
    here so goku's existing scorers see exactly what they'd see when
    reading rubrics.jsonl directly.
    """
    extras: dict[str, Any] = {}
    if wire.raw_json:
        try:
            extras = json.loads(wire.raw_json)
            if not isinstance(extras, dict):
                extras = {}
        except (ValueError, TypeError):
            extras = {}
    merged = {
        **extras,
        "number": wire.number,
        "type": wire.type,
        "category": wire.category or extras.get("category", "CORRECTNESS"),
        "importance": wire.importance or extras.get("importance", "nice_to_have"),
        "points": wire.points or extras.get("points", 0),
        "criterion": wire.criterion or extras.get("criterion", ""),
    }
    try:
        return RubricItem.model_validate(merged)
    except ValidationError as e:
        raise HTTPException(
            400,
            f"rubric #{wire.number} failed validation: {e.errors()}",
        ) from e


# ---------------------------------------------------------------------------
# On-disk persistence — writes the task folder layout under the dated bundle.
#
# Layout produced per /run call (matches the README delivery export shape):
#   <SERVICE_OUTPUT_DIR>/MM Agentic Pilot Samples-YYYY-MM-DD/
#     └── tasks/<task_id>/
#           ├── instruction.md     (from req.user_prompt)
#           ├── rubrics.jsonl      (from req.rubrics raw_json lines)
#           ├── data/input_files/  (decoded req.media)
#           └── runs/<model>/run_<N>/
#                 ├── scores.jsonl  (one row per deterministic rubric verdict)
#                 └── results/      (files the agent produced; live mode only)
#
# The task root (instruction/rubrics/media) is rewritten on every call —
# the same task_id is expected to produce the same content. The
# runs/<model>/run_<N>/ directory is wiped + rewritten so a repeat call for
# the same (task, model, run_index) replaces the previous result rather than
# accumulating stale state. Different run_index values coexist side-by-side.
# ---------------------------------------------------------------------------
_FS_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _fs_safe_segment(value: str, fallback: str = "unknown") -> str:
    """Coerce arbitrary text (e.g. a Bedrock ARN) into a filesystem-safe path
    segment. Collapses unsafe characters into '_' and trims leading dots.
    """
    cleaned = _FS_UNSAFE.sub("_", value).strip("._")
    return cleaned or fallback


def _persist_task_inputs(
    task_root: Path,
    instruction: str,
    rubrics: list[WireRubric],
    media: list[WireMedia],
) -> None:
    """Write instruction.md, rubrics.jsonl, and data/input_files/ under
    task_root. Idempotent — overwrites existing files so the on-disk copy
    always reflects what the service most recently saw on the wire.
    """
    task_root.mkdir(parents=True, exist_ok=True)
    (task_root / "instruction.md").write_text(instruction, encoding="utf-8")

    # rubrics.jsonl: prefer raw_json (lossless) when present, fall back to
    # reconstructing from wire fields. One JSON object per line, no trailing
    # newline on the last row to match the loader's behavior.
    rubric_lines: list[str] = []
    for r in rubrics:
        if r.raw_json:
            try:
                # Validate it's JSON, but write the parsed-and-re-serialized
                # form so a stray trailing newline / BOM in raw_json doesn't
                # corrupt the file.
                rubric_lines.append(json.dumps(json.loads(r.raw_json)))
                continue
            except (ValueError, TypeError):
                pass
        rubric_lines.append(
            json.dumps(
                {
                    "number": r.number,
                    "type": r.type,
                    "category": r.category or "CORRECTNESS",
                    "importance": r.importance or "nice_to_have",
                    "points": r.points,
                    "criterion": r.criterion,
                }
            )
        )
    (task_root / "rubrics.jsonl").write_text(
        "\n".join(rubric_lines) + ("\n" if rubric_lines else ""),
        encoding="utf-8",
    )

    input_dir = task_root / "data" / "input_files"
    input_dir.mkdir(parents=True, exist_ok=True)
    # Clear stale media so a task_id that previously had file foo.png but
    # no longer references it doesn't keep the old file around.
    for existing in input_dir.iterdir():
        if existing.is_file():
            existing.unlink()
    for m in media:
        fname = (m.filename or m.name or "").strip()
        if not fname or "/" in fname or "\\" in fname or fname.startswith("."):
            # Same safety as _materialize_media — the boundary already
            # rejected these, but guard here too in case helpers are reused.
            continue
        try:
            (input_dir / fname).write_bytes(base64.b64decode(m.data_b64, validate=True))
        except (ValueError, TypeError) as e:
            logger.warning("skipping persist of media %r: %s", fname, e)


def _persist_run_outputs(
    task_root: Path,
    model_segment: str,
    run_index: int,
    scores: list[WireScore],
    output_files: list[OutputFile],
    agent_workspace: Path | None,
) -> Path:
    """Write runs/<model>/run_<N>/{scores.jsonl, results/} under task_root.
    Returns the run directory path.

    Only the specific run_<N> dir is wiped on a repeat call; sibling runs
    (different run_index) for the same (task, model) are untouched.
    """
    run_dir = task_root / "runs" / model_segment / f"run_{run_index}"
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "scores.jsonl").open("w", encoding="utf-8") as f:
        for s in scores:
            f.write(
                json.dumps(
                    {
                        "number": s.rubric_number,
                        "passed": s.passed,
                        "judged_by": s.judged_by,
                        "rationale": s.rationale,
                        "awarded_points": s.awarded_points,
                    }
                )
                + "\n"
            )

    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Inline base64 output_files (today empty in mock, populated once the
    # live runner is wired). Live mode may additionally copy from a local
    # workspace; agent_workspace is the seam for that.
    for of in output_files:
        # Same filename-safety rules as materialize_media.
        fname = of.name.strip()
        if not fname or "/" in fname or "\\" in fname or fname.startswith("."):
            logger.warning("skipping unsafe output_file name: %r", fname)
            continue
        try:
            (results_dir / fname).write_bytes(base64.b64decode(of.file, validate=True))
        except (ValueError, TypeError) as e:
            logger.warning("skipping persist of output_file %r: %s", fname, e)

    _SKIP_DIRS = frozenset({"bash_events", ".git"})

    if agent_workspace is not None and agent_workspace.exists():
        for src in agent_workspace.rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(agent_workspace)
            if rel.parts and rel.parts[0] in _SKIP_DIRS:
                continue
            dest = results_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

    return run_dir


# ---------------------------------------------------------------------------
# Agent runner — mock returns a stub; live delegates to agent_runner
# ---------------------------------------------------------------------------
class AgentResult(BaseModel):
    response_text: str
    tokens_in: int = 0
    tokens_out: int = 0
    output_files: list[OutputFile] = Field(default_factory=list)
    # Local directory containing files the agent produced. Populated in live
    # mode (from agent_runner.AgentRunResult.output_dir), None in mock.
    # Persistence later copies this into runs/<model>/results/.
    agent_output_dir: Path | None = None


def _resolve_llm_config_path(model: str) -> Path:
    """Find the .llm_config/*.json that matches `model`, checking in order:
      1. filename stem (e.g. .llm_config/claude-opus-4.7.json)
      2. JSON field 'display_name'
      3. JSON field 'model'

    Raises HTTPException if no match is found, so the client sees a clear
    400 rather than an obscure auth error from LiteLLM.
    """
    if not LLM_CONFIG_DIR.is_dir():
        raise HTTPException(
            500,
            f"GOKU_LLM_CONFIG_DIR {LLM_CONFIG_DIR} does not exist on the goku host",
        )

    candidates = sorted(p for p in LLM_CONFIG_DIR.glob("*.json") if p.is_file())
    # Pass 1: filename stem (cheapest).
    for p in candidates:
        if p.stem == model:
            return p
    # Pass 2 + 3: open each file and check display_name / model.
    for p in candidates:
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(raw, dict):
            continue
        if raw.get("display_name") == model or raw.get("model") == model:
            return p

    raise HTTPException(
        400,
        f"no LLM config matches model={model!r} in {LLM_CONFIG_DIR}. "
        f"Available: {[p.stem for p in candidates]}",
    )


def _run_agent_in_workspace(
    model: str,
    system_prompt: str,
    user_prompt: str,
    workspace: Path,
) -> AgentResult:
    """Mock returns a deterministic stub. Live spins up a Docker workspace
    via ``benchmarks.goku.agent_runner.run_single_task`` and returns the real
    agent response + a path to the agent's downloaded output files.

    `system_prompt` is currently NOT forwarded — the shared agent_runner uses
    OpenHands' default cli_mode system prompt to match the CLI's behavior. If
    mm_tasker needs a custom system prompt, it can be threaded through later
    via run_single_task; today the wire field exists for forward compatibility.
    """
    if SERVICE_MODE == "mock":
        files_in_workspace = sorted(p.name for p in workspace.iterdir())
        text = (
            f"[MOCK agent · model={model}]\n"
            f"prompt: {user_prompt[:200]}...\n"
            f"saw files: {files_in_workspace}\n"
            f"(set GOKU_SERVICE_MODE=live to invoke the real runner)"
        )
        return AgentResult(
            response_text=text,
            tokens_in=len(user_prompt) // 4,
            tokens_out=len(text) // 4,
            output_files=[],
            agent_output_dir=None,
        )

    if SERVICE_MODE != "live":
        raise HTTPException(
            500,
            f"unknown GOKU_SERVICE_MODE={SERVICE_MODE!r} (expected 'mock' or 'live')",
        )

    # Live: import lazily so the heavy openhands stack doesn't load in mock
    # mode (faster tests, smaller failure surface for the common dev path).
    from benchmarks.goku.agent_runner import run_single_task
    from benchmarks.utils.llm_config import load_llm_config

    config_path = _resolve_llm_config_path(model)
    llm = load_llm_config(config_path)
    logger.info("live run: model=%s config=%s", model, config_path.name)

    # Forward Bedrock credentials into the Docker container.
    # The OpenHands SDK suppresses api_key for Bedrock in _get_litellm_api_key_value,
    # so we inject it as AWS_BEARER_TOKEN_BEDROCK which LiteLLM reads from env.
    # os.environ mutation here is intentionally per-thread; concurrent jobs with
    # different Bedrock keys would race — acceptable for single-model eval workloads.
    forward_env: list[str] = []
    if llm.api_key is not None and "bedrock" in llm.model.lower():
        from pydantic import SecretStr as _SecretStr

        raw_key = (
            llm.api_key.get_secret_value()
            if isinstance(llm.api_key, _SecretStr)
            else str(llm.api_key)
        )
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = raw_key
        forward_env.append("AWS_BEARER_TOKEN_BEDROCK")
    if llm.aws_region_name:
        os.environ["AWS_REGION_NAME"] = llm.aws_region_name
        forward_env.append("AWS_REGION_NAME")

    # The agent_runner expects local paths to input files. The workspace tempdir
    # already has them materialized; collect them by listing the workspace.
    input_files = [str(p) for p in sorted(workspace.iterdir()) if p.is_file()]

    result = run_single_task(
        llm=llm,
        instruction=user_prompt,
        input_files=input_files,
        instance_id=uuid.uuid4().hex[:12],
        max_iterations=SERVICE_MAX_ITERATIONS,
        forward_env=forward_env or None,
    )

    # Pull token counts out of the conversation metrics if available. Different
    # LiteLLM providers report keys differently — be lenient.
    tokens_in = 0
    tokens_out = 0
    if result.metrics:
        for k in ("prompt_tokens", "input_tokens", "tokens_in"):
            if k in result.metrics and isinstance(result.metrics[k], int):
                tokens_in = result.metrics[k]
                break
        for k in ("completion_tokens", "output_tokens", "tokens_out"):
            if k in result.metrics and isinstance(result.metrics[k], int):
                tokens_out = result.metrics[k]
                break

    return AgentResult(
        response_text=result.response_text,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        output_files=[],  # Files are streamed to disk via agent_output_dir;
        # we don't double-encode them inline.
        agent_output_dir=result.output_dir,
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _score_against_workspace(
    rubrics: list[WireRubric],
    workspace: Path,
    response_text: str,
) -> list[WireScore]:
    """Score every deterministic rubric. Skips LLM-judge types — those flow
    through the separate /judge endpoint after Run Models completes.

    Per-rubric error semantics: a failure on one rubric never tanks the
    whole batch. We emit a WireScore with judged_by='error' and the
    exception message so the client (and the operator) can see what went
    wrong without losing the rest of the verdicts.
    """
    out: list[WireScore] = []
    for wire in rubrics:
        if wire.type in LLM_JUDGE_TYPES:
            # Deliberately left for /judge; do not emit a score row.
            continue
        if wire.type not in DETERMINISTIC_TYPES:
            out.append(
                WireScore(
                    rubric_number=wire.number,
                    passed=False,
                    judged_by="error",
                    rationale=f"unknown rubric type: {wire.type!r}",
                    awarded_points=0,
                )
            )
            continue

        try:
            item = _rebuild_rubric_item(wire)
            result: ScorerResult = score_deterministic(
                item=item,
                output_dir=workspace,
                response=response_text,
            )
            out.append(
                WireScore(
                    rubric_number=result.number,
                    passed=result.passed,
                    triggered=False,
                    judged_by=_judged_by_for_type(wire.type),
                    rationale=result.judge_rationale,
                    awarded_points=result.points_awarded,
                )
            )
        except HTTPException:
            raise  # let validation errors propagate
        except Exception as e:
            logger.exception("rubric #%s scoring failed", wire.number)
            out.append(
                WireScore(
                    rubric_number=wire.number,
                    passed=False,
                    judged_by="error",
                    rationale=f"scorer raised {type(e).__name__}: {e}",
                    awarded_points=0,
                )
            )
    return out


def _judged_by_for_type(rtype: str) -> str:
    if rtype == "response_regex_present":
        return "regex"
    return "probe"


# ---------------------------------------------------------------------------
# Shared LiteLLM credential helper
# ---------------------------------------------------------------------------
def _build_completion_kwargs(
    model_id: str,
    messages: list[dict],
    api_key: str | None,
    base_url: str | None,
    aws_region: str | None,
    max_tokens: int = 1024,
) -> dict:
    """Build litellm.completion kwargs, routing credentials correctly for
    Bedrock (aws_bearer_token) vs standard providers (api_key).
    Single source of truth — used by /judge so the logic never diverges."""
    kwargs: dict = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "timeout": 120,
    }
    if base_url:
        kwargs["base_url"] = base_url
    is_bedrock = model_id.startswith("bedrock/")
    key_str = str(api_key) if api_key else ""
    if hasattr(api_key, "get_secret_value"):
        key_str = api_key.get_secret_value()  # type: ignore[union-attr]
    if is_bedrock and key_str:
        kwargs["aws_bearer_token"] = key_str
        if aws_region:
            kwargs["aws_region_name"] = aws_region
    elif key_str:
        kwargs["api_key"] = key_str
    return kwargs


# ---------------------------------------------------------------------------
# Logging helper — binds job_id to every log line within a request
# ---------------------------------------------------------------------------
class _JobAdapter(logging.LoggerAdapter):
    def process(self, msg: str, kwargs: Any) -> tuple[str, Any]:
        job_id = (self.extra or {}).get("job_id", "?")
        return f"[{job_id}] {msg}", kwargs


# ---------------------------------------------------------------------------
# Background job runner — executes one /run request in a daemon thread
# ---------------------------------------------------------------------------
def _run_job_background(job_id: str, req: RunRequest) -> None:
    """Full agent-run pipeline. Runs in a background thread; writes status to
    _job_store so GET /jobs/{job_id} can report progress and final results.

    Error handling: both HTTPException (validation / config errors) and
    unexpected exceptions are caught and stored as status='error' so the
    polling client always sees a terminal state rather than an expired job.
    """
    jlog = _JobAdapter(logger, {"job_id": job_id})
    _job_store.put(job_id, {"job_id": job_id, "status": "running"})

    try:
        jlog.info(
            "run start task=%s run=%d model=%s media=%d rubrics=%d",
            req.task_id,
            req.run_index,
            req.model,
            len(req.media),
            len(req.rubrics),
        )

        task_root = _current_bundle_dir() / "tasks" / req.task_id
        _persist_task_inputs(
            task_root=task_root,
            instruction=req.user_prompt,
            rubrics=req.rubrics,
            media=req.media,
        )

        with _workspace() as ws:
            _materialize_media(ws, req.media)
            agent = _run_agent_in_workspace(
                model=req.model,
                system_prompt=req.system_prompt,
                user_prompt=req.user_prompt,
                workspace=ws,
            )
            # Score against the agent's output dir (contains all files the agent
            # produced + kept); fall back to input workspace in mock mode.
            score_dir = (
                agent.agent_output_dir if agent.agent_output_dir is not None else ws
            )
            scores = _score_against_workspace(
                rubrics=req.rubrics,
                workspace=score_dir,
                response_text=agent.response_text,
            )

        model_segment = _fs_safe_segment(req.model)
        try:
            run_dir = _persist_run_outputs(
                task_root=task_root,
                model_segment=model_segment,
                run_index=req.run_index,
                scores=scores,
                output_files=agent.output_files,
                agent_workspace=agent.agent_output_dir,
            )
        finally:
            if agent.agent_output_dir is not None and agent.agent_output_dir.exists():
                shutil.rmtree(agent.agent_output_dir, ignore_errors=True)

        jlog.info(
            "run done task=%s run_dir=%s scores=%d passed=%d",
            req.task_id,
            run_dir,
            len(scores),
            sum(1 for s in scores if s.passed),
        )
        _job_store.put(
            job_id,
            {
                "job_id": job_id,
                "status": "done",
                "response_text": agent.response_text,
                "tokens_in": agent.tokens_in,
                "tokens_out": agent.tokens_out,
                "output_files": [of.model_dump() for of in agent.output_files],
                "scores": [s.model_dump() for s in scores],
            },
        )

    except HTTPException as e:
        jlog.error(
            "run validation error task=%s HTTP %s: %s",
            req.task_id,
            e.status_code,
            e.detail,
        )
        _job_store.put(
            job_id,
            {
                "job_id": job_id,
                "status": "error",
                "error": f"HTTP {e.status_code}: {e.detail}",
            },
        )
    except Exception as e:
        jlog.exception("run failed task=%s", req.task_id)
        _job_store.put(
            job_id,
            {
                "job_id": job_id,
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
            },
        )


# ---------------------------------------------------------------------------
# App + routes
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Goku Eval Service",
    version="1.0",
    description="HTTP wrapper around the goku multimodal agentic eval engine.",
)


@app.middleware("http")
async def _enforce_body_limit(request: Request, call_next):
    """Reject oversized requests before the route handler buffers the body."""
    cl = request.headers.get("content-length")
    if cl:
        try:
            if int(cl) > MAX_REQUEST_BYTES:
                return JSONResponse(
                    {"detail": f"request body too large (>{MAX_REQUEST_BYTES} bytes)"},
                    status_code=413,
                )
        except ValueError:
            pass
    return await call_next(request)


@app.exception_handler(ValidationError)
async def _pydantic_handler(_req: Request, exc: ValidationError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": exc.errors()})


@app.get("/health")
def health() -> dict[str, Any]:
    """Cheap liveness probe — does NOT require auth."""
    return {"status": "ok", "mode": SERVICE_MODE}


@app.get("/readyz")
def readyz() -> dict[str, Any]:
    """Readiness probe — fails if the service can't accept real traffic."""
    if not SERVICE_TOKEN:
        raise HTTPException(503, "GOKU_SERVICE_TOKEN not configured")
    return {"status": "ready", "mode": SERVICE_MODE}


@app.post(
    "/run",
    response_model=JobStatusResponse,
    status_code=202,
    dependencies=[Depends(require_bearer)],
    summary="Submit an agent run job (async, returns 202 immediately)",
)
def run(req: RunRequest, raw_request: Request) -> JSONResponse:
    """Accept a run job and return 202 with a job_id immediately.

    The agent runs in a background thread. Poll GET /jobs/{job_id} to check
    status. When status=='done' the full response (response_text, scores,
    output_files) is included. When status=='error' the error field explains
    what went wrong.

    response_criteria rubrics are NOT scored here — clients call /judge after.
    """
    if len(req.rubrics) > MAX_RUBRICS:
        raise HTTPException(
            400, f"too many rubrics ({len(req.rubrics)} > {MAX_RUBRICS})"
        )
    cl = raw_request.headers.get("content-length")
    if cl and int(cl) > MAX_REQUEST_BYTES:
        raise HTTPException(413, f"request body too large (>{MAX_REQUEST_BYTES} bytes)")

    # Idempotency: same key → same job_id. If that job is still in the store,
    # return its current state (the client gets live status on retry).
    if req.idempotency_key:
        cached_job_id = _idempotency.get(req.idempotency_key)
        if cached_job_id is not None:
            existing = _job_store.get(cached_job_id)
            if existing:
                logger.info(
                    "idempotency hit: %s → job %s", req.idempotency_key, cached_job_id
                )
                return JSONResponse(status_code=202, content=existing)
            # Job expired but key is still alive — fall through to create a new job.

    job_id = uuid.uuid4().hex
    if req.idempotency_key:
        _idempotency.put(req.idempotency_key, job_id)

    initial: dict = {"job_id": job_id, "status": "queued"}
    _job_store.put(job_id, initial)

    t = threading.Thread(target=_run_job_background, args=(job_id, req), daemon=True)
    t.start()

    logger.info("job %s queued task=%s model=%s", job_id, req.task_id, req.model)
    return JSONResponse(status_code=202, content=initial)


@app.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    dependencies=[Depends(require_bearer)],
    summary="Poll the status of an async /run job",
)
def get_job_status(job_id: str) -> JSONResponse:
    """Return the current state of a background /run job.

    status=='queued'|'running' → poll again later.
    status=='done'             → response_text / scores / output_files populated.
    status=='error'            → error field has the reason; job will not recover.
    404                        → job_id unknown or expired (GOKU_JOB_TTL_S elapsed).
    """
    data = _job_store.get(job_id)
    if data is None:
        raise HTTPException(404, f"job {job_id!r} not found or expired")
    return JSONResponse(content=data)


@app.post(
    "/regrade",
    response_model=RegradeResponse,
    dependencies=[Depends(require_bearer)],
    summary="Re-score an existing run against new rubrics (no agent re-run)",
)
def regrade(req: RegradeRequest) -> RegradeResponse:
    """Take new rubrics for an already-completed run; re-score deterministic
    rubrics against the persisted workspace (runs/<model>/run_<N>/results/)
    and the saved response. Overwrites scores.jsonl in place.

    Mirrors /run: response_criteria / response_not_criteria (LLM judge) are
    deliberately skipped — call /judge separately for those, exactly like the
    post-/run flow. This keeps regrade free of LLM API cost so users can
    iterate on rubrics cheaply.
    """
    if len(req.rubrics) > MAX_RUBRICS:
        raise HTTPException(
            400, f"too many rubrics ({len(req.rubrics)} > {MAX_RUBRICS})"
        )

    bundle_dir = (
        SERVICE_OUTPUT_DIR / req.bundle_name
        if req.bundle_name
        else _current_bundle_dir()
    )
    task_root = bundle_dir / "tasks" / req.task_id
    model_segment = _fs_safe_segment(req.model)
    run_dir = task_root / "runs" / model_segment / f"run_{req.run_index}"

    if not run_dir.exists():
        # Fallback: scan every bundle in SERVICE_OUTPUT_DIR for this
        # (task, model, run_index). Lets the client regrade older runs
        # without having to know the bundle name. First match wins;
        # we don't expect collisions across bundles.
        if not req.bundle_name and SERVICE_OUTPUT_DIR.exists():
            # Scan the 30 most recently modified bundles only — prevents the
            # scan from growing unboundedly as bundles accumulate over months.
            recent = sorted(
                (c for c in SERVICE_OUTPUT_DIR.iterdir() if c.is_dir()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:30]
            for candidate in recent:
                guess = (
                    candidate
                    / "tasks"
                    / req.task_id
                    / "runs"
                    / model_segment
                    / f"run_{req.run_index}"
                )
                if guess.exists():
                    run_dir = guess
                    task_root = candidate / "tasks" / req.task_id
                    bundle_dir = candidate
                    break

    if not run_dir.exists():
        raise HTTPException(
            404,
            f"run not found: bundle={bundle_dir.name} task={req.task_id} "
            f"model={req.model} run_index={req.run_index}",
        )

    results_dir = run_dir / "results"
    response_text = req.response_text or ""

    job_id = uuid.uuid4().hex
    logger.info(
        "regrade start job=%s task=%s run=%d model=%s rubrics=%d",
        job_id,
        req.task_id,
        req.run_index,
        req.model,
        len(req.rubrics),
    )

    if SERVICE_MODE == "mock":
        scores = [
            WireScore(
                rubric_number=wire.number,
                passed=True,
                triggered=False,
                judged_by=_judged_by_for_type(wire.type)
                if wire.type in DETERMINISTIC_TYPES
                else "not_applicable",
                rationale=f"[MOCK regrade] {wire.type} auto-passed.",
                awarded_points=wire.points if wire.points > 0 else 0,
            )
            for wire in req.rubrics
            if wire.type not in LLM_JUDGE_TYPES
        ]
    else:
        scores = _score_against_workspace(
            rubrics=req.rubrics,
            workspace=results_dir,
            response_text=response_text,
        )

    # Overwrite scores.jsonl in place. response.txt and results/ are
    # untouched — only the score verdicts change.
    with (run_dir / "scores.jsonl").open("w", encoding="utf-8") as f:
        for s in scores:
            f.write(
                json.dumps(
                    {
                        "number": s.rubric_number,
                        "passed": s.passed,
                        "triggered": s.triggered,
                        "judged_by": s.judged_by,
                        "rationale": s.rationale,
                        "awarded_points": s.awarded_points,
                    }
                )
                + "\n"
            )

    logger.info(
        "regrade done job=%s run_dir=%s scores=%d passed=%d",
        job_id,
        run_dir,
        len(scores),
        sum(1 for s in scores if s.passed),
    )
    return RegradeResponse(
        job_id=job_id,
        response_text=response_text,
        scores=scores,
    )


@app.post(
    "/judge",
    response_model=JudgeResponse,
    dependencies=[Depends(require_bearer)],
    summary="Single LLM judge call",
)
def judge(req: JudgeRequest) -> JudgeResponse:
    """Thin LLM proxy. mm_tasker calls this once per response_criteria
    rubric (existing flow). Body is the literal envelope mm_tasker.
    _call_judge sends today; reply matches what it already parses.
    In live mode, calls LiteLLM using credentials from env vars or .llm_config/.
    """
    if SERVICE_MODE == "mock":
        verdict = {
            "passed": len(req.user_prompt) > 50,
            "rationale": (
                f"[MOCK judge · model={req.model}] decided on prompt "
                f"length ({len(req.user_prompt)} chars > 50)"
            ),
            "confidence": 0.5,
        }
        return JudgeResponse(
            response_text=json.dumps(verdict),
            tokens_in=len(req.user_prompt) // 4,
            tokens_out=30,
        )

    # Live: call LiteLLM with the pre-formatted prompt mm_tasker already built.
    # Credentials are read per-request so AWS token rotation is picked up
    # without a service restart.
    # Resolution order (mirrors run_infer.py CLI):
    #   1. Env vars: GOKU_JUDGE_MODEL, AWS_BEARER_TOKEN_BEDROCK, AWS_REGION_NAME
    #   2. .llm_config/<req.model>.json fallback (for multi-model deployments)
    import litellm  # lazy import — keeps mock-mode startup dependency-free

    model_id: str = os.environ.get("GOKU_JUDGE_MODEL", _DEFAULT_JUDGE_MODEL).strip()
    api_key: str | None = os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "").strip() or None
    aws_region: str | None = os.environ.get("AWS_REGION_NAME", "").strip() or None
    base_url: str | None = None

    # Fallback: if no env-var key, try resolving from .llm_config
    if not api_key and LLM_CONFIG_DIR.is_dir():
        try:
            config_path = _resolve_llm_config_path(req.model)
            raw_config = json.loads(config_path.read_text(encoding="utf-8"))
            model_id = raw_config.get("model") or model_id
            api_key = raw_config.get("api_key")
            base_url = raw_config.get("base_url") or raw_config.get("api_base")
            aws_region = raw_config.get("aws_region_name") or aws_region
        except HTTPException:
            pass  # no matching config file — proceed with env vars only

    messages: list[dict] = []
    if req.system_prompt:
        messages.append({"role": "system", "content": req.system_prompt})
    messages.append({"role": "user", "content": req.user_prompt})

    completion_kwargs = _build_completion_kwargs(
        model_id=model_id,
        messages=messages,
        api_key=api_key,
        base_url=base_url,
        aws_region=aws_region,
    )

    try:
        llm_response = litellm.completion(**completion_kwargs)
        response_text = llm_response.choices[0].message.content or ""  # type: ignore[union-attr]
        usage = getattr(llm_response, "usage", None)
        tokens_in = int(getattr(usage, "prompt_tokens", 0) or 0)
        tokens_out = int(getattr(usage, "completion_tokens", 0) or 0)
    except Exception as e:
        logger.exception("Judge LLM call failed for model=%s", model_id)
        raise HTTPException(502, f"judge LLM call failed: {e}")

    logger.info(
        "judge done model=%s tokens_in=%d tokens_out=%d",
        model_id,
        tokens_in,
        tokens_out,
    )
    return JudgeResponse(
        response_text=response_text,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )


# ---------------------------------------------------------------------------
# Direct invocation (development convenience)
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "benchmarks.goku.service:app",
        host=os.environ.get("GOKU_SERVICE_HOST", "127.0.0.1"),
        port=int(os.environ.get("GOKU_SERVICE_PORT", "8000")),
        reload=bool(int(os.environ.get("GOKU_SERVICE_RELOAD", "0"))),
        log_level=os.environ.get("GOKU_SERVICE_LOG_LEVEL", "info").lower(),
    )
