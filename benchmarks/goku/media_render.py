"""Render PDFs and videos to image frames for multimodal judging.

Why this exists
---------------
Some providers/routes don't accept native PDF or video input:
  * Kimi K2.5 via Bedrock Converse → no PDF, no video.
  * Claude Opus 4.7 → PDF native, video unsupported (Anthropic has no video).
  * GPT-5.5 → PDF native, video unsupported (OpenAI API explicitly excludes video).
  * Gemini 3.x → both native.

For the AGENT path the model is fixed and we route PDFs natively per-provider
(see :mod:`benchmarks.goku.media_adapters`). Videos are uniformly rendered to
keyframes because going asymmetric across the three agent models would muddy
the eval.

For the JUDGE path, fallback rendering kicks in only for providers that don't
accept native PDF/video (e.g. Kimi-Bedrock). With the default Gemini judge,
PDFs flow as native `file` blocks and only video keyframe extraction runs.

This module is deliberately thin: just two helpers (`pdf_to_page_images`,
`video_to_keyframes`) plus a small cache so repeated calls on the same file
don't re-render. No provider routing — that lives in :mod:`media_adapters`.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import pypdfium2 as pdfium


logger = logging.getLogger(__name__)


# Cache dir for rendered media. Process-local; cleared by the OS on /tmp
# eviction. Sharing across processes is safe because filenames embed a
# content-hash of the source.
_CACHE_ROOT = Path(tempfile.gettempdir()) / "goku_media_render"


# Tunables — defaults chosen for "OCR-quality" rendering on dense PDFs and
# "enough temporal sampling" on hour-long clips. Override per call if needed.
# Caps tuned to the per-category hard limits in the annotator spec sheet:
#   * PDF tasks:   ≤100 pages, ≤30 MB per file
#   * Video tasks: ≤60 minutes, ≤200 MB per file
#   * Image tasks: ≤5 MB per file, up to 20 files
_DEFAULT_PDF_DPI = 200             # 200 DPI ≈ sharp enough for small body text
_DEFAULT_KEYFRAME_COUNT = 120      # 2 frames per minute on a 60-min video.
                                   # Was 60 (1/min) — too sparse to catch
                                   # popups, brief UI states, or anything
                                   # that appears for < 60s. JPEG output
                                   # (~150 KB/frame) makes 120 frames fit
                                   # easily in one judge call (~18 MB
                                   # vs ~120 MB for PNG).
_MAX_PDF_PAGES = 100               # Anthropic native-PDF cap
_MAX_KEYFRAMES = 180               # safety ceiling (1.5× default) for short
                                   # videos where 3 fpm is feasible. The
                                   # judge byte cap (90 MB) and block-count
                                   # cap (100) are the actual hard limits.

# Per-file size + duration caps. These are the upper bound the harness will
# accept — task_loader fails loud if a task's data/input_files/ exceeds them
# so annotators see the error at task-discovery time, not silently downstream.
MAX_PDF_BYTES = 30_000_000         # 30 MB  (Anthropic 32 MB cap − base64 margin)
MAX_VIDEO_BYTES = 250_000_000      # 250 MB. Was 200 MB — based on Gemini's
                                   # older Files API limit. Today's API
                                   # accepts 2 GB; 250 MB gives headroom for
                                   # ~60-min 1280x720 H.264 content (Cars.mp4
                                   # for the Aditya Joshi task lands at 200.2
                                   # MB just barely over the old cap). The
                                   # binding cost is workspace upload time
                                   # + ffmpeg keyframe-extract wall, not API.
MAX_IMAGE_BYTES = 5_000_000        # 5 MB   (Anthropic per-image cap)
MAX_VIDEO_DURATION_SEC = 60 * 60   # 60 min (Gemini native cap)


def _file_hash(path: Path) -> str:
    """Short stable hash of file contents — used to key the cache so an edited
    source file doesn't pick up a stale render."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def pdf_to_page_images(
    pdf_path: str | Path,
    *,
    dpi: int = _DEFAULT_PDF_DPI,
    max_pages: int = _MAX_PDF_PAGES,
) -> list[Path]:
    """Render every page of ``pdf_path`` to PNG and return their paths.

    Cached: identical (file-content, dpi) tuple → reuses the previous render.
    Caps at ``max_pages`` to bound payload size.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    key = f"{pdf_path.stem}_{_file_hash(pdf_path)}_dpi{dpi}"
    out_dir = _CACHE_ROOT / "pdf" / key
    if out_dir.is_dir() and any(out_dir.iterdir()):
        return sorted(out_dir.glob("page_*.png"))

    # Atomic-publish: render into a unique temp dir under the same parent,
    # then rename to the final cache key. Two workers racing produce one
    # winner and one harmless leftover; readers never observe partial output.
    tmp_dir = _CACHE_ROOT / "pdf" / f".tmp.{key}.{uuid.uuid4().hex[:8]}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        with pdfium.PdfDocument(str(pdf_path)) as pdf:
            n_pages = min(len(pdf), max_pages)
            scale = dpi / 72.0
            for i in range(n_pages):
                page = pdf[i]
                pil_image = page.render(scale=scale).to_pil()
                pil_image.save(tmp_dir / f"page_{i+1:03d}.png", format="PNG", optimize=True)
            full_page_count = len(pdf)
        try:
            tmp_dir.rename(out_dir)
        except OSError:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    if full_page_count > max_pages:
        logger.warning(
            "PDF %s has %d pages; capped at %d for judge payload",
            pdf_path.name, full_page_count, max_pages,
        )
    return sorted(out_dir.glob("page_*.png"))


def video_to_keyframes(
    video_path: str | Path,
    *,
    n_frames: int = _DEFAULT_KEYFRAME_COUNT,
    max_frames: int = _MAX_KEYFRAMES,
) -> list[Path]:
    """Extract ``n_frames`` evenly-spaced frames from ``video_path`` as
    JPEGs (quality 3, very high). Returns paths sorted by frame index.

    JPEG over PNG: per-frame size drops from ~1-3 MB to ~100-200 KB with
    visually negligible loss for the judge's purposes. 120 JPEG frames
    total ~18 MB; 120 PNG frames would be ~180 MB and trip the judge
    total-bytes cap. The format choice is part of the cache key so an
    upgrade from a PNG-era cache forces a re-render.

    Cached: identical (file-content, n_frames, format) → reuses the
    previous extract.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    n_frames = min(n_frames, max_frames)
    # Format tag in the key isolates PNG-era caches from the JPEG era so
    # an in-place upgrade doesn't return stale, oversized PNG renders.
    key = f"{video_path.stem}_{_file_hash(video_path)}_n{n_frames}_fmt-jpg"
    out_dir = _CACHE_ROOT / "video" / key
    if out_dir.is_dir() and any(out_dir.iterdir()):
        return sorted(out_dir.glob("frame_*.jpg"))

    tmp_dir = _CACHE_ROOT / "video" / f".tmp.{key}.{uuid.uuid4().hex[:8]}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        duration = _probe_duration_seconds(video_path)
        if duration is None or duration <= 0:
            cmd = [
                "ffmpeg", "-y", "-i", str(video_path),
                "-frames:v", str(n_frames),
                "-vf", "fps=1",
                "-q:v", "3",  # JPEG quality 1-31 (lower=better); 3 = visually lossless
                str(tmp_dir / "frame_%03d.jpg"),
            ]
            _run_ffmpeg(cmd)
        else:
            # Single-pass extraction: ask ffmpeg for a constant fps that yields
            # exactly n_frames over the video's duration. One subprocess instead
            # of one-per-frame — orders of magnitude faster for long videos.
            fps_target = n_frames / duration
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-vf", f"fps={fps_target:.6f}",
                "-frames:v", str(n_frames),
                "-q:v", "3",  # JPEG quality 1-31 (lower=better); 3 = visually lossless
                str(tmp_dir / "frame_%03d.jpg"),
            ]
            _run_ffmpeg(cmd, timeout=300)
        try:
            tmp_dir.rename(out_dir)
        except OSError:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return sorted(out_dir.glob("frame_*.jpg"))


def _probe_duration_seconds(video_path: Path) -> float | None:
    """Return video duration in seconds via ffprobe, or None on failure."""
    if not shutil.which("ffprobe"):
        return None
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return None
        return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError, OSError):
        return None


def _run_ffmpeg(cmd: list[str], timeout: int = 60) -> None:
    """Run ffmpeg quietly; raise on non-zero exit."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg not found on PATH. Install via `brew install ffmpeg` "
            "(macOS) or your package manager."
        )
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        tail = (result.stderr or "")[-800:]
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode}): {tail}")
