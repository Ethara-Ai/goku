"""Render PDFs and videos to image frames for multimodal judging.

Why this exists
---------------
Some providers/routes don't accept native PDF or video input:
  * Kimi K2.5 via Bedrock Converse → no PDF, no video.
  * Claude Opus 4.7 → PDF native, video unsupported (Anthropic has no video).
  * GPT-5.5 → PDF native, video unsupported (OpenAI API explicitly excludes video).
  * Gemini 3.1 Pro → both native.

For the AGENT path the model is fixed and we route PDFs natively per-provider
(see :mod:`benchmarks.goku.media_adapters`). Videos are uniformly rendered to
keyframes because only Gemini natively supports them and going asymmetric
across the three agent models would muddy the eval.

For the JUDGE path (currently Kimi-Bedrock), both PDFs and videos must be
rendered to images because Bedrock-Kimi accepts only images.

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
_DEFAULT_KEYFRAME_COUNT = 60       # 1 frame per minute for a 60-min video
_MAX_PDF_PAGES = 100               # Anthropic native-PDF cap
_MAX_KEYFRAMES = 120               # safety ceiling (~2× default) for finer sampling

# Per-file size + duration caps. These are the upper bound the harness will
# accept — task_loader fails loud if a task's data/input_files/ exceeds them
# so annotators see the error at task-discovery time, not silently downstream.
MAX_PDF_BYTES = 30_000_000         # 30 MB  (Anthropic 32 MB cap − base64 margin)
MAX_VIDEO_BYTES = 200_000_000      # 200 MB (workspace upload + ffmpeg time)
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

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(pdf_path))
    n_pages = min(len(pdf), max_pages)
    scale = dpi / 72.0  # pypdfium2 takes a scale where 1.0 == 72 DPI
    paths: list[Path] = []
    for i in range(n_pages):
        page = pdf[i]
        pil_image = page.render(scale=scale).to_pil()
        dest = out_dir / f"page_{i+1:03d}.png"
        pil_image.save(dest, format="PNG", optimize=True)
        paths.append(dest)
    if len(pdf) > max_pages:
        logger.warning(
            "PDF %s has %d pages; capped at %d for judge payload",
            pdf_path.name, len(pdf), max_pages,
        )
    return paths


def video_to_keyframes(
    video_path: str | Path,
    *,
    n_frames: int = _DEFAULT_KEYFRAME_COUNT,
    max_frames: int = _MAX_KEYFRAMES,
) -> list[Path]:
    """Extract ``n_frames`` evenly-spaced frames from ``video_path`` as PNGs.

    Cached: identical (file-content, n_frames) → reuses the previous extract.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    n_frames = min(n_frames, max_frames)
    key = f"{video_path.stem}_{_file_hash(video_path)}_n{n_frames}"
    out_dir = _CACHE_ROOT / "video" / key
    if out_dir.is_dir() and any(out_dir.iterdir()):
        return sorted(out_dir.glob("frame_*.png"))

    out_dir.mkdir(parents=True, exist_ok=True)

    # Probe duration via ffprobe to plan the timestamp grid. Fall back to
    # ffmpeg's `-vf select` if ffprobe is missing.
    duration = _probe_duration_seconds(video_path)
    if duration is None or duration <= 0:
        # Last-ditch fallback: extract first N frames at native rate.
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-frames:v", str(n_frames),
            "-vf", f"fps=1",  # 1 fps; bounded by -frames:v
            str(out_dir / "frame_%03d.png"),
        ]
        _run_ffmpeg(cmd)
        return sorted(out_dir.glob("frame_*.png"))

    # Evenly-spaced timestamps: avoid the absolute 0 and absolute end.
    # For n=8 and duration=6s → [0.375, 1.125, 1.875, 2.625, 3.375, 4.125, 4.875, 5.625]
    timestamps = [duration * (i + 0.5) / n_frames for i in range(n_frames)]
    for idx, ts in enumerate(timestamps, start=1):
        dest = out_dir / f"frame_{idx:03d}.png"
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{ts:.3f}",
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", "2",
            str(dest),
        ]
        _run_ffmpeg(cmd)

    return sorted(out_dir.glob("frame_*.png"))


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


def _run_ffmpeg(cmd: list[str]) -> None:
    """Run ffmpeg quietly; raise on non-zero exit."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg not found on PATH. Install via `brew install ffmpeg` "
            "(macOS) or your package manager."
        )
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        # ffmpeg writes progress to stderr too, so just take the tail.
        tail = (result.stderr or "")[-800:]
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode}): {tail}")
