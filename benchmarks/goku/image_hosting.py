"""S3-hosted image inputs for many-image agent + judge tasks.

Why this exists
---------------
For tasks with >20 images, inline base64 in the OpenHands SDK's
``ImageContent`` doesn't scale: Anthropic's 32 MB request-body cap and
Gemini's 20 MB inline cap break around 100 images, and the multi-turn
agent conversation re-sends ALL images per turn — so a 30-iteration run
busts the body cap even at moderate per-image sizes.

All 3 agent providers (Claude/Gemini/GPT) plus the Gemini Flash judge
natively accept HTTPS image URLs in their standard ``image_url`` content
block (validated empirically in spikes/image_files_api/spike7_url_multi.py).
This module uploads the task's images to a single S3 bucket, generates
24-hour presigned URLs, and lets the existing ``ImageContent`` pass those
URLs through unchanged — no SDK patches, no Files API integration,
no Responses API switch for OpenAI.

The per-request body shrinks from O(images × base64-bytes) to
O(images × URL-bytes), which fixes the multi-turn body amplification
without changing any of the provider-specific code paths.

Pre-conditioning
----------------
Before upload, each image is downscaled to ≤2000 px (Anthropic's
many-image cap) and re-encoded as JPEG q=75 (bounded file size). This
ensures EVERY provider receives identical bytes — preserves benchmark
integrity by removing per-provider preprocessing variance.

Env vars
--------
  AWS_REGION (or AWS_DEFAULT_REGION) — required
      Region where the bucket lives. us-east-1 recommended for lowest
      provider-fetch latency.
  AWS_BUCKET (or GOKU_S3_BUCKET — back-compat) — required
      Bucket name. Example: "production-grtlabs-tag".
  AWS_FOLDER (or GOKU_S3_KEY_PREFIX — back-compat) — optional, default "goku"
      Top-level prefix under the bucket. Each task adds its own
      ``<task_key>/<run_id>/`` below this.
  AWS_ACCESS_KEY_ID / AWS_ACCESS_SECRET_KEY — credential ID
  AWS_SECRET_ACCESS_KEY / AWS_SECRET_KEY — credential secret
      Either pair works (standard boto3 names OR GRT Labs aliases). If
      neither pair is set, boto3 falls back to its default chain
      (~/.aws/credentials, IAM role, etc).
  GOKU_S3_URL_TTL_SEC (optional, default 86400 = 24h)
      Presigned URL validity. 24h covers any agent run; bucket lifecycle
      policy should clean up old objects.

Usage
-----
::

    from benchmarks.goku.image_hosting import upload_task_images
    urls = upload_task_images(
        image_paths=[Path("/data/img_001.jpg"), ...],
        task_key="task_abc1234567890def",
    )
    # urls = ["https://bucket.s3.us-east-1.amazonaws.com/goku-tasks/.../img_001.jpg?X-Amz-Signature=...", ...]
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# LiteLLM providers whose API servers fetch remote image URLs themselves —
# so passing a presigned S3 URL is reliable. Providers NOT in this set
# (e.g. ``gemini``, ``vertex_ai``) require LiteLLM to download each URL
# from inside the agent container and re-encode inline before the call,
# which is fragile at scale (every fetch is exposed to container-side
# network conditions). For those providers we always pick the inline
# base64 path so no S3 fetches happen at the agent edge.
PROVIDERS_THAT_FETCH_URLS_SERVER_SIDE: frozenset[str] = frozenset({
    "anthropic",
    "openai",
})


# Defaults — overridable via env vars (see module docstring)
_DEFAULT_KEY_PREFIX = "goku"             # matches AWS_FOLDER default in prod
_DEFAULT_URL_TTL_SEC = 24 * 3600         # 24h
_DEFAULT_MAX_DIM = 2000                  # Anthropic many-image cap
_DEFAULT_JPEG_QUALITY = 75               # balance: fidelity vs size


# Module-level boto3 client cache (one client per process is fine —
# boto3 clients are thread-safe). Construction is lazy so importing the
# module doesn't require AWS creds.
_s3_client: Any = None

# In-process URL cache: (task_key, abs_path, size, mtime_ns) →
# (presigned_url, generated_at_epoch). Mirrors the Gemini Files API
# cache pattern in llm_judge.py. Lets agent + judge (in same Python
# process) share one upload across many rubric calls. Cache TTL matches
# the presigned URL TTL so we regenerate the URL before it expires
# without re-uploading.
_HOSTED_URL_CACHE: dict[tuple, tuple[str, float]] = {}


def _hosted_cache_key(task_key: str, path: Path) -> tuple:
    st = path.stat()
    return (task_key, str(path.resolve()), st.st_size, st.st_mtime_ns)


def _cache_clear() -> None:
    """Used by tests."""
    _HOSTED_URL_CACHE.clear()


def _get_s3_client() -> Any:
    """Return a cached boto3 S3 client. Raises if AWS creds/region are
    not configured.

    Credential lookup order (most-specific to default):
      1. Standard boto3 env vars: AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY
      2. Project-specific aliases: AWS_ACCESS_SECRET_KEY + AWS_SECRET_KEY
         (matches the names used in the GRT Labs production env)
      3. ~/.aws/credentials file (boto3 default chain)
    Whichever pair is present first wins. Explicit pass to boto3.client()
    when (1) or (2) is set; otherwise we let boto3 walk its default chain."""
    global _s3_client
    if _s3_client is not None:
        return _s3_client
    try:
        import boto3
        from botocore.config import Config
    except ImportError as e:
        raise RuntimeError(
            "boto3 required for image_hosting. Install with: uv pip install boto3"
        ) from e
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if not region:
        raise RuntimeError(
            "AWS_REGION (or AWS_DEFAULT_REGION) env var not set. "
            "Set to the region where the S3 bucket lives (e.g., us-east-1)."
        )
    # Credential discovery — accept both standard and project-alias names.
    access_key = (
        os.environ.get("AWS_ACCESS_KEY_ID")
        or os.environ.get("AWS_ACCESS_SECRET_KEY")
    )
    secret_key = (
        os.environ.get("AWS_SECRET_ACCESS_KEY")
        or os.environ.get("AWS_SECRET_KEY")
    )
    kwargs: dict[str, Any] = {
        "region_name": region,
        # signature_version='s3v4' is required for presigned URLs in
        # modern regions.
        "config": Config(signature_version="s3v4"),
    }
    if access_key and secret_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = secret_key
    # else: fall back to boto3's default credential chain (~/.aws/credentials,
    # IAM role, etc.) — useful for EC2/ECS or developer workstations.
    _s3_client = boto3.client("s3", **kwargs)
    return _s3_client


def _bucket() -> str:
    """Return the configured bucket name. Accepts AWS_BUCKET (preferred,
    matches GRT Labs convention) or GOKU_S3_BUCKET (back-compat). Raises
    if neither is set."""
    bucket = os.environ.get("AWS_BUCKET") or os.environ.get("GOKU_S3_BUCKET")
    if not bucket:
        raise RuntimeError(
            "AWS_BUCKET (or GOKU_S3_BUCKET) env var not set. Set to the "
            "S3 bucket name where task images should be uploaded."
        )
    return bucket


def _key_prefix() -> str:
    """Top-level prefix under the bucket. Accepts AWS_FOLDER (preferred)
    or GOKU_S3_KEY_PREFIX (back-compat); defaults to "goku"."""
    return (
        os.environ.get("AWS_FOLDER")
        or os.environ.get("GOKU_S3_KEY_PREFIX")
        or _DEFAULT_KEY_PREFIX
    )


def _url_ttl() -> int:
    raw = os.environ.get("GOKU_S3_URL_TTL_SEC")
    if raw:
        try:
            return int(raw)
        except ValueError:
            logger.warning("Invalid GOKU_S3_URL_TTL_SEC=%r; using default %ds",
                           raw, _DEFAULT_URL_TTL_SEC)
    return _DEFAULT_URL_TTL_SEC


# ─────────────────────────────────────────────────────────────────
# Image preconditioning (uniform across providers)
# ─────────────────────────────────────────────────────────────────

def _precondition_image(
    src: Path, dest: Path,
    *, max_dim: int = _DEFAULT_MAX_DIM,
    jpeg_quality: int = _DEFAULT_JPEG_QUALITY,
) -> Path:
    """Resize an image (if needed) to fit Anthropic's many-image dim cap
    and re-encode as JPEG. Applied to ALL providers' inputs uniformly so
    each model sees the same bytes — preserves benchmark integrity."""
    try:
        from PIL import Image
    except ImportError as e:
        raise RuntimeError(
            "Pillow required for image preconditioning. "
            "Install with: uv pip install pillow"
        ) from e
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im.load()
        w, h = im.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        im.save(dest, format="JPEG", quality=jpeg_quality, optimize=True)
    return dest


# ─────────────────────────────────────────────────────────────────
# Upload + presigned URL generation
# ─────────────────────────────────────────────────────────────────

@dataclass
class HostedImage:
    """One uploaded image's metadata."""
    src_path: Path           # original local path
    s3_key: str              # full S3 object key
    presigned_url: str       # 24h GET URL


@dataclass
class TaskUploadResult:
    """All hosted images for one task. ``urls`` is the only field most
    callers need — pass directly to ``ImageContent(image_urls=...)``."""
    task_key: str
    run_prefix: str          # e.g. "tasks/<hash>/run_<timestamp>"
    images: list[HostedImage] = field(default_factory=list)

    @property
    def urls(self) -> list[str]:
        return [img.presigned_url for img in self.images]


def upload_task_images(
    image_paths: list[Path],
    task_key: str,
    *,
    precondition: bool = True,
    run_id: str | None = None,
) -> TaskUploadResult:
    """Upload one task's images to S3 and return per-image presigned URLs.

    Args:
        image_paths: Local image paths (typically from data/input_files/).
        task_key: Used as a prefix for S3 keys — must match the task's
            hash directory name (e.g., "task_abc1234567890def").
        precondition: If True (default), resize/re-encode each image to
            ≤2000 px JPEG q=75 BEFORE upload. Set False only if you want
            to ship the original bytes verbatim (rare; bypasses Anthropic
            many-image dim cap protection).
        run_id: Accepted for backward compatibility; defaults to a sortable
            UTC timestamp. No longer affects S3 keys — those are now
            content-addressed (see below) so reruns of the same task
            reuse existing objects instead of duplicating per run.

    S3 key layout (content-addressed):
        ``{prefix}/{task_key}/{sha256_16}{ext}``

    Same preconditioned bytes → same key → no re-upload across reruns.
    A HEAD check before each upload skips objects that already exist in
    the bucket. Old timestamped layouts (``{prefix}/{task_key}/run_<ts>/``)
    are still resolvable while their presigned URLs are alive but won't
    be written by this function any more.

    Returns: TaskUploadResult with ``urls`` ready to pass to ImageContent.

    Raises:
        RuntimeError: missing AWS creds / bucket / region.
        boto3 ClientError: S3 upload failure (propagated). HEAD failures
            other than 404/NoSuchKey/NotFound are propagated.
    """
    if not image_paths:
        return TaskUploadResult(task_key=task_key, run_prefix="")
    if not task_key.startswith("task_"):
        logger.warning("task_key=%r doesn't start with 'task_' — using as-is", task_key)

    # Lazy import: keeps module import cheap and lets tests stub
    # _get_s3_client without dragging botocore into every import path.
    from botocore.exceptions import ClientError

    s3 = _get_s3_client()
    bucket = _bucket()
    prefix = _key_prefix()
    ttl = _url_ttl()

    # run_id is accepted + logged for human context only — keys below are
    # content-addressed so reruns are idempotent.
    run_id = run_id or "run_" + time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())
    run_prefix = f"{prefix}/{task_key}"

    # Pre-condition into a scratch dir if requested
    import tempfile
    scratch_dir: Path | None = None
    if precondition:
        scratch_dir = Path(tempfile.mkdtemp(prefix=f"goku_imghost_{task_key}_"))

    result = TaskUploadResult(task_key=task_key, run_prefix=run_prefix)
    t0 = time.time()
    uploaded = 0
    cache_hits = 0
    s3_reused = 0
    for src in image_paths:
        if not src.is_file():
            logger.warning("Skipping missing image: %s", src)
            continue

        # Cache check: same (task_key, abs_path, size, mtime) → reuse the
        # previously-generated presigned URL if still well within TTL.
        # Saves agent ↔ judge re-uploads within one Python process.
        ck = _hosted_cache_key(task_key, src)
        cached = _HOSTED_URL_CACHE.get(ck)
        if cached is not None:
            url, generated_at = cached
            # Regenerate URL if we're within 10% of TTL expiry — avoids
            # handing out a URL that'll die mid-conversation.
            if time.time() - generated_at < ttl * 0.9:
                result.images.append(HostedImage(
                    src_path=src, s3_key=ck[1], presigned_url=url,
                ))
                cache_hits += 1
                continue
            # Expired — fall through to re-upload
            _HOSTED_URL_CACHE.pop(ck, None)

        if precondition and scratch_dir is not None:
            # Always save as .jpg (precondition re-encodes to JPEG)
            local = _precondition_image(src, scratch_dir / (src.stem + ".jpg"))
            content_type = "image/jpeg"
        else:
            local = src
            content_type = _content_type_for(src.suffix)

        # Content-addressed key: hash the FINAL bytes (post-precondition)
        # so identical inputs across reruns collapse to one S3 object.
        with open(local, "rb") as _fh:
            content_hash = hashlib.sha256(_fh.read()).hexdigest()[:16]
        s3_key = f"{run_prefix}/{content_hash}{local.suffix.lower()}"

        # HEAD-check: skip upload when an object with this content already
        # exists in the bucket (idempotent reruns).
        object_exists = False
        try:
            s3.head_object(Bucket=bucket, Key=s3_key)
            object_exists = True
        except ClientError as head_err:
            code = head_err.response.get("Error", {}).get("Code", "")
            status = head_err.response.get("ResponseMetadata", {}).get(
                "HTTPStatusCode"
            )
            if code in ("404", "NoSuchKey", "NotFound") or status == 404:
                object_exists = False
            else:
                logger.error("S3 HEAD failed: s3://%s/%s : %s",
                             bucket, s3_key, head_err)
                raise

        if object_exists:
            s3_reused += 1
        else:
            try:
                s3.upload_file(
                    Filename=str(local),
                    Bucket=bucket,
                    Key=s3_key,
                    ExtraArgs={"ContentType": content_type},
                )
            except Exception as e:
                logger.error("S3 upload failed: %s → s3://%s/%s : %s",
                             local, bucket, s3_key, e)
                raise
            uploaded += 1

        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": s3_key},
            ExpiresIn=ttl,
        )
        result.images.append(HostedImage(src_path=src, s3_key=s3_key,
                                          presigned_url=url))
        _HOSTED_URL_CACHE[ck] = (url, time.time())

    elapsed = time.time() - t0
    logger.info(
        "image_hosting: %d uploaded, %d in-process cache, %d S3-reused, "
        "%.1fs (TTL=%ds, bucket=%s, prefix=%s, run_id=%s)",
        uploaded, cache_hits, s3_reused, elapsed, ttl, bucket, run_prefix, run_id,
    )
    return result


def _content_type_for(suffix: str) -> str:
    s = suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(s, "application/octet-stream")


# Image count past which we auto-promote to URL-hosting mode. Below this,
# inline base64 in the existing path is simpler and faster (no S3 RTT).
INLINE_IMAGE_THRESHOLD = 20


def s3_hosting_configured() -> bool:
    """Cheap check: are AWS region + bucket env vars both set?

    Accepts either the AWS_BUCKET (GRT Labs convention) or the
    GOKU_S3_BUCKET (back-compat) name for the bucket. Region must be
    AWS_REGION or AWS_DEFAULT_REGION.

    Used to fail fast at task-load time with a clear error rather than
    deep inside the agent inference path."""
    bucket = os.environ.get("AWS_BUCKET") or os.environ.get("GOKU_S3_BUCKET")
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    return bool(bucket and region)


def should_use_url_hosting(
    image_paths: list[Path],
    llm_provider: str | None = None,
) -> bool:
    """Returns True if the harness should route this task's images through
    S3 + URLs (vs the legacy inline base64 path).

    Signals (in priority order):
      1. ``GOKU_IMAGE_MODE=many`` env var (explicit override)
      2. ``GOKU_IMAGE_MODE=inline`` env var (explicit override)
      3. ``llm_provider`` not in PROVIDERS_THAT_FETCH_URLS_SERVER_SIDE
         → force inline. Reason: providers like Gemini/Vertex don't
         accept remote URLs natively; LiteLLM downloads each URL from
         inside the agent container, which is fragile when there are
         many images and the local network is intermittent.
      4. Image count > ``INLINE_IMAGE_THRESHOLD`` (auto-promotion)

    Returns False for tasks with 0 image inputs (irrelevant)."""
    if not image_paths:
        return False
    env_mode = (os.environ.get("GOKU_IMAGE_MODE") or "").lower()
    if env_mode == "many":
        return True
    if env_mode == "inline":
        return False
    if llm_provider:
        prov = llm_provider.lower().strip()
        if prov and prov not in PROVIDERS_THAT_FETCH_URLS_SERVER_SIDE:
            return False
    return len(image_paths) > INLINE_IMAGE_THRESHOLD


# ─────────────────────────────────────────────────────────────────
# Cleanup (optional; relies on bucket lifecycle policy otherwise)
# ─────────────────────────────────────────────────────────────────

def cleanup_task_uploads(result: TaskUploadResult) -> int:
    """Delete all S3 objects created by ``upload_task_images``.

    Optional — if your bucket has a lifecycle policy (e.g., delete
    objects after 7 days), this is unnecessary. Returns the number of
    objects deleted.
    """
    if not result.images:
        return 0
    s3 = _get_s3_client()
    bucket = _bucket()
    objects = [{"Key": img.s3_key} for img in result.images]
    deleted = 0
    # delete_objects max is 1000 keys per call
    for i in range(0, len(objects), 1000):
        batch = objects[i : i + 1000]
        resp = s3.delete_objects(Bucket=bucket, Delete={"Objects": batch})
        deleted += len(resp.get("Deleted", []))
    logger.info("Cleanup: deleted %d S3 object(s) under %s",
                deleted, result.run_prefix)
    return deleted
