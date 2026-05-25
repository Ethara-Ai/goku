"""PDF subsample+tool pipeline — the "do what video does" approach.

Why this exists
---------------
The default render-pages-to-images path (``media_render.pdf_to_page_images``
called from ``run_infer.py``) inlines every page as ImageContent in turn 1.
For PDFs above ~12 pages this exceeds Anthropic's 32 MB request cap (HTTP
413) and pushes our 8 GB Docker container toward OOM under OpenAI. Native
PDF via the SDK's DocumentContent type is blocked by the upstream
agent-server's PyInstaller bundle (see ``run_infer.main()`` docstring).

This module replaces the turn-1 payload with a tiny per-page text index +
low-DPI thumbnails (~500 KB total for 55 pages), and installs in-container
shell tools so the agent can fetch any specific page at full resolution on
demand. Same architecture as the video pipeline (ffmpeg in container + 120
keyframes + agent can ffprobe/ffmpeg for finer detail).

Per-turn body payload stays bounded at ~5 MB regardless of total PDF size.

Gating
------
The tool-mode pipeline only fires for tasks whose ``input_files`` contain a
PDF. Image and video pipelines are untouched. The legacy inline-render path
is kept reachable via ``GOKU_PDF_MODE=inline`` for smoke tests and the
≤5-page case where inline is simpler.
"""
from __future__ import annotations

import io
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pypdf
import pypdfium2 as pdfium
from PIL import Image


logger = logging.getLogger(__name__)


# Tunables. Per-turn payload math (55-page worst case):
#   THUMBNAIL_DPI=50 → ~10 KB/page JPEG q=70 → 55 pages × 10 KB ≈ 550 KB
#   Text index      → ~1 KB/page on text-rich pages → ~55 KB total
#   Combined turn-1 payload bound: ~700 KB ⇒ 45× under Anthropic's 32 MB cap.
THUMBNAIL_DPI = 50
THUMBNAIL_JPEG_QUALITY = 70
INDEX_TEXT_PER_PAGE_CHARS = 400   # truncation point for text shown in index
MIN_TEXT_CHARS_FOR_INDEX = 30     # below this we mark the page (image-only)

# Container layout — must match the strings the agent sees in its prompt and
# the strings inside the tool scripts. Keep these in one place so a path
# rename only touches this module.
CONTAINER_PDF_DIR = "/workspace"           # PDFs already uploaded here by run_infer
CONTAINER_THUMBS_DIR = "/workspace/pdf_pages"
CONTAINER_TOOLS_DIR = "/workspace/tools"


@dataclass(frozen=True)
class PDFToolSetupResult:
    """Result of preparing a PDF task for tool-mode operation.

    Attributes:
        index_markdown: The page-by-page text index to send in turn 1.
        thumbnail_paths: Local (host-side) JPEG paths to attach as ImageContent
            in turn 1. Each filename embeds the source PDF stem + page number
            so the agent can map a thumbnail back to (pdf, page).
        agent_prompt_suffix: Lines to append to the task instruction telling
            the agent what files exist and which tools to use.
    """
    index_markdown: str
    thumbnail_paths: list[Path]
    agent_prompt_suffix: str


# ─────────────────────────────────────────────────────────────────
# Host-side rendering + text extraction
# ─────────────────────────────────────────────────────────────────

def _render_thumbnails(pdf_path: Path, dpi: int = THUMBNAIL_DPI) -> list[bytes]:
    """Render every page of ``pdf_path`` to a JPEG thumbnail and return the
    raw bytes per page. Uses pypdfium2 (already a project dep).
    """
    out: list[bytes] = []
    scale = dpi / 72.0
    with pdfium.PdfDocument(str(pdf_path)) as pdf:
        for i in range(len(pdf)):
            pil = pdf[i].render(scale=scale).to_pil()
            if pil.mode != "RGB":
                pil = pil.convert("RGB")
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=THUMBNAIL_JPEG_QUALITY, optimize=True)
            out.append(buf.getvalue())
    return out


def _extract_per_page_text(pdf_path: Path) -> list[str]:
    """Per-page text via pypdf. Returns one string per page; empty strings
    for pages where extraction yielded nothing (image-only pages).
    """
    pages: list[str] = []
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        for p in reader.pages:
            pages.append((p.extract_text() or "").strip())
    except Exception as exc:
        # If pypdf chokes on the PDF, fall back to "unknown" markers so the
        # agent still gets thumbnails and can navigate visually.
        logger.warning("pypdf text extraction failed for %s: %s", pdf_path, exc)
        try:
            with pdfium.PdfDocument(str(pdf_path)) as pdf:
                pages = [""] * len(pdf)
        except Exception:
            pages = []
    return pages


def _build_index_markdown(per_pdf_pages: dict[str, list[str]]) -> str:
    """Build the turn-1 markdown index from {filename: [per-page text]}.

    Output shape (one section per PDF):

        ## file.pdf (N pages)
        - p1 (820 chars): First 400 chars of page text...
        - p2 (image-only, no extractable text)
        - p3 (412 chars): ...
    """
    lines: list[str] = []
    for fname, pages in per_pdf_pages.items():
        lines.append(f"## {fname} ({len(pages)} pages)")
        for i, txt in enumerate(pages, 1):
            if len(txt) < MIN_TEXT_CHARS_FOR_INDEX:
                lines.append(f"- p{i} (image-only, no extractable text)")
                continue
            preview = _flatten_whitespace(txt)[:INDEX_TEXT_PER_PAGE_CHARS]
            lines.append(f"- p{i} ({len(txt)} chars): {preview}…")
        lines.append("")
    return "\n".join(lines).rstrip()


def _flatten_whitespace(s: str) -> str:
    return " ".join(s.split())


# ─────────────────────────────────────────────────────────────────
# Container-side install + tool scripts
# ─────────────────────────────────────────────────────────────────

# Apt-installable for the in-container tool scripts. tesseract is for the
# OCR fallback path in pdf_text.py when a page has no extractable text;
# poppler-utils ships pdftoppm as a backup renderer.
_CONTAINER_APT_PACKAGES = ("tesseract-ocr", "poppler-utils")

# pymupdf wheel is small (~50 MB) and provides text/OCR/render in one
# import, which keeps the tool scripts short. Pinned conservatively.
_CONTAINER_PIP_PACKAGES = ("pymupdf>=1.24.0",)


def install_pdf_deps_in_container(workspace: Any) -> bool:
    """Install poppler/tesseract/pymupdf inside the agent_server container.

    Mirrors the ffmpeg install hook in ``run_infer.py`` (P2 / lines 326-370):
    one apt-get + one pip call, both quiet, both non-fatal on failure
    (we log and proceed). The agent can still navigate via the index +
    thumbnails if the install fails; only the on-demand high-res fetch
    breaks.

    Returns True on full success, False on any partial failure.
    """
    apt_pkgs = " ".join(_CONTAINER_APT_PACKAGES)
    pip_pkgs = " ".join(f"'{p}'" for p in _CONTAINER_PIP_PACKAGES)
    cmd = (
        "sudo apt-get update -qq && "
        f"sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq {apt_pkgs} "
        "< /dev/null && "
        f"pip install --quiet --disable-pip-version-check {pip_pkgs}"
    )
    logger.info(
        "PDF input detected; installing %s + %s in container "
        "(one-time cost, ~30-45s)…",
        apt_pkgs, ", ".join(_CONTAINER_PIP_PACKAGES),
    )
    t0 = time.time()
    result = workspace.execute_command(cmd, timeout=300.0)
    elapsed = time.time() - t0
    exit_code = getattr(result, "exit_code", -1)
    if exit_code == 0:
        logger.info("PDF deps installed in container in %.1fs", elapsed)
        return True
    stderr = (
        getattr(result, "stderr", "")
        or getattr(result, "stdout", "")
        or ""
    )
    logger.warning(
        "PDF dep install failed (exit=%s, elapsed=%.1fs); agent can still "
        "navigate via the turn-1 index + thumbnails but the per-page "
        "high-res tool will not work. Last output: %s",
        exit_code, elapsed, stderr[:300],
    )
    return False


# Tool scripts. Tiny — embedded inline so we can ship them with one
# execute_command per script rather than docker cp. Keep them dependency-light
# (only pymupdf + stdlib) and behavior-stable so changes here don't surprise
# downstream agent runs.

_TOOL_PDF_PAGE = r'''#!/usr/bin/env python3
"""Render a single PDF page at full resolution. Writes a JPEG to /workspace.

The output dimensions are CLAMPED to 2000 px on the longest side because
Anthropic's many-image requests reject any image whose dimension exceeds
2000 px (HTTP 400 invalid_request_error). At DPI 200 a letter-size page is
1700x2200 — the 2200 height blows the cap, so we auto-derive a safe DPI
from the page's natural point dimensions before rendering. The agent
sees the effective DPI in stderr so it can request a larger fetch only
when the source PDF page is small enough to support it.

Usage: python /workspace/tools/pdf_page.py <pdf_filename> <page_num> [--dpi 200]
"""
import fitz, sys, os, argparse

MAX_DIM_PX = 2000  # Anthropic many-image cap. OpenAI/Gemini are higher.

ap = argparse.ArgumentParser()
ap.add_argument("pdf"); ap.add_argument("page", type=int)
ap.add_argument("--dpi", type=int, default=180)  # safe for letter at 2000px cap
args = ap.parse_args()

pdf_path = args.pdf if args.pdf.startswith("/") else f"/workspace/{args.pdf}"
if not os.path.isfile(pdf_path):
    print(f"ERROR: PDF not found at {pdf_path}", file=sys.stderr); sys.exit(2)

doc = fitz.open(pdf_path)
if not (1 <= args.page <= len(doc)):
    print(f"ERROR: page {args.page} out of range [1, {len(doc)}]", file=sys.stderr); sys.exit(2)

page = doc[args.page-1]
# Compute max DPI that keeps the longer dimension under MAX_DIM_PX. The page
# rect is in points (72 pt = 1 inch); a -1 margin guards against fp rounding.
max_dim_pt = max(page.rect.width, page.rect.height)
max_safe_dpi = int(MAX_DIM_PX * 72 / max_dim_pt) - 1
effective_dpi = min(args.dpi, max_safe_dpi)
if effective_dpi < args.dpi:
    print(f"[note: clamped DPI from {args.dpi} to {effective_dpi} to fit "
          f"{MAX_DIM_PX}-px many-image API cap]", file=sys.stderr)

mat = fitz.Matrix(effective_dpi/72.0, effective_dpi/72.0)
pix = page.get_pixmap(matrix=mat)
stem = os.path.splitext(os.path.basename(pdf_path))[0]
out = f"/workspace/hires_{stem}_p{args.page:03d}.jpg"
pix.pil_save(out, format="JPEG", quality=85, optimize=True)
print(out)
'''

_TOOL_PDF_TEXT = r'''#!/usr/bin/env python3
"""Extract text from a single PDF page (with OCR fallback for image-only pages).

Usage: python /workspace/tools/pdf_text.py <pdf_filename> <page_num>
"""
import fitz, sys, os

if len(sys.argv) < 3:
    print("Usage: pdf_text.py <pdf> <page_num>", file=sys.stderr); sys.exit(2)
pdf_path = sys.argv[1] if sys.argv[1].startswith("/") else f"/workspace/{sys.argv[1]}"
page_n = int(sys.argv[2])
if not os.path.isfile(pdf_path):
    print(f"ERROR: PDF not found at {pdf_path}", file=sys.stderr); sys.exit(2)

doc = fitz.open(pdf_path)
if not (1 <= page_n <= len(doc)):
    print(f"ERROR: page {page_n} out of range [1, {len(doc)}]", file=sys.stderr); sys.exit(2)

page = doc[page_n-1]
txt = page.get_text().strip()
if len(txt) < 30:
    # OCR fallback. PyMuPDF shells out to tesseract; ~1-3s per page.
    try:
        tp = page.get_textpage_ocr(language="eng", dpi=200, full=True)
        txt = (page.get_text(textpage=tp) or "").strip() or "(no text extractable, even via OCR)"
    except Exception as exc:
        txt = f"(text extraction failed: {exc!r})"
print(txt)
'''

_TOOL_PDF_SEARCH = r'''#!/usr/bin/env python3
"""Search all PDFs in /workspace for a regex; print page numbers + context.

Usage: python /workspace/tools/pdf_search.py "<regex>" [<pdf_filename>]
       If pdf_filename omitted, searches every *.pdf in /workspace.
"""
import fitz, sys, os, re, glob

if len(sys.argv) < 2:
    print("Usage: pdf_search.py <regex> [<pdf>]", file=sys.stderr); sys.exit(2)
pattern = sys.argv[1]
pdfs = ([sys.argv[2] if sys.argv[2].startswith("/") else f"/workspace/{sys.argv[2]}"]
        if len(sys.argv) >= 3 else sorted(glob.glob("/workspace/*.pdf")))

try:
    rx = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
except re.error as exc:
    print(f"ERROR: bad regex: {exc}", file=sys.stderr); sys.exit(2)

hits = 0
for pdf_path in pdfs:
    if not os.path.isfile(pdf_path):
        print(f"WARN: {pdf_path} not found", file=sys.stderr); continue
    name = os.path.basename(pdf_path)
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc, 1):
        txt = page.get_text()
        for m in rx.finditer(txt):
            start = max(0, m.start()-40); end = min(len(txt), m.end()+40)
            ctx = " ".join(txt[start:end].split())
            print(f"{name} p{i}: …{ctx}…")
            hits += 1
print(f"\n[{hits} match{'es' if hits != 1 else ''}]", file=sys.stderr)
'''


def write_tool_scripts(workspace: Any) -> bool:
    """Write the three pdf_* helper scripts under /workspace/tools/.

    Uses execute_command with a heredoc per script. Each script is <2 KB so
    one round-trip per file is acceptable (≤ 3 round-trips total per task).

    Returns True if all three scripts landed; False on any failure.
    """
    workspace.execute_command(f"mkdir -p {CONTAINER_TOOLS_DIR}")
    scripts = {
        "pdf_page.py": _TOOL_PDF_PAGE,
        "pdf_text.py": _TOOL_PDF_TEXT,
        "pdf_search.py": _TOOL_PDF_SEARCH,
    }
    all_ok = True
    for name, body in scripts.items():
        path = f"{CONTAINER_TOOLS_DIR}/{name}"
        # Heredoc with a non-conflicting sentinel. We escape backslashes and
        # backticks to make sure shell expansion doesn't mangle the body.
        sentinel = "PDFPIPELINE_EOF_42"
        safe_body = body.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
        cmd = (
            f"cat > {path} <<'{sentinel}'\n"
            f"{body}\n"
            f"{sentinel}\n"
            f"chmod +x {path}"
        )
        del safe_body  # using single-quote heredoc means no escaping needed
        result = workspace.execute_command(cmd, timeout=15.0)
        if getattr(result, "exit_code", -1) != 0:
            stderr = getattr(result, "stderr", "") or getattr(result, "stdout", "")
            logger.warning(
                "Failed to write %s in container (exit=%s): %s",
                path, getattr(result, "exit_code", -1), stderr[:200],
            )
            all_ok = False
    return all_ok


# ─────────────────────────────────────────────────────────────────
# Public entry point — used by run_infer.py
# ─────────────────────────────────────────────────────────────────

def prepare_pdf_tool_mode(
    pdf_paths: Sequence[str | Path],
    *,
    workspace: Any | None = None,
    write_tools: bool = True,
) -> PDFToolSetupResult:
    """Prepare a tool-mode PDF task.

    Steps:
        1. Render every page of every PDF to a host-side thumbnail.
        2. Extract per-page text from every PDF.
        3. Build the markdown index that goes into turn 1's TextContent.
        4. (Optional) Install pymupdf+poppler+tesseract in the container.
        5. (Optional) Drop the three pdf_* shell tool scripts under
           /workspace/tools/.

    The host-side work (steps 1-3) happens regardless of ``workspace``;
    container-side work (steps 4-5) is a no-op when ``workspace is None``
    so this function is callable from unit tests without a live container.

    Args:
        pdf_paths: Host-side paths to the PDF input files.
        workspace: An OpenHands ``RemoteWorkspace`` (or compatible) with an
            ``execute_command(cmd, timeout=...) -> result`` method.
        write_tools: If False, skip writing tool scripts even when
            ``workspace`` is provided. Used by tests that only want the
            host-side index/thumbnails.

    Returns:
        :class:`PDFToolSetupResult` containing the turn-1 content pieces
        and a prompt suffix telling the agent what's available.
    """
    pdf_paths = [Path(p) for p in pdf_paths]
    if not pdf_paths:
        raise ValueError("prepare_pdf_tool_mode called with empty pdf_paths")

    # Host-side: thumbnails + text extraction (cached on disk per source).
    per_pdf_pages: dict[str, list[str]] = {}
    thumbnail_paths: list[Path] = []
    for pdf in pdf_paths:
        if not pdf.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf}")
        text_pages = _extract_per_page_text(pdf)
        per_pdf_pages[pdf.name] = text_pages

        thumbs = _render_thumbnails(pdf)
        # Write to a stable per-source cache dir so re-runs don't re-render.
        import tempfile
        cache_dir = Path(tempfile.gettempdir()) / "goku_pdf_thumbs" / pdf.stem
        cache_dir.mkdir(parents=True, exist_ok=True)
        for i, jpg_bytes in enumerate(thumbs, 1):
            tpath = cache_dir / f"{pdf.stem}_p{i:03d}.jpg"
            if not tpath.exists() or tpath.stat().st_size != len(jpg_bytes):
                tpath.write_bytes(jpg_bytes)
            thumbnail_paths.append(tpath)

    index_md = _build_index_markdown(per_pdf_pages)
    prompt_suffix = _build_prompt_suffix(per_pdf_pages, pdf_paths)

    # Container-side: dep install + tool scripts. Both non-fatal.
    if workspace is not None:
        try:
            install_pdf_deps_in_container(workspace)
        except Exception as exc:
            logger.warning("install_pdf_deps_in_container raised: %s", exc)
        if write_tools:
            try:
                write_tool_scripts(workspace)
            except Exception as exc:
                logger.warning("write_tool_scripts raised: %s", exc)

    return PDFToolSetupResult(
        index_markdown=index_md,
        thumbnail_paths=thumbnail_paths,
        agent_prompt_suffix=prompt_suffix,
    )


def _build_prompt_suffix(
    per_pdf_pages: dict[str, list[str]],
    pdf_paths: Sequence[Path],
) -> str:
    """The text appended to the task instruction in turn 1."""
    pdf_list = "\n".join(
        f"  - /workspace/{p.name} ({len(per_pdf_pages.get(p.name, []))} pages)"
        for p in pdf_paths
    )
    return f"""

---

**PDF reference materials are pre-processed and available as on-demand tools.**

Files in your workspace:
{pdf_list}

Below this section you will see:
  1. A per-page text index for each PDF (page numbers, character counts, opening text).
  2. Low-resolution thumbnails (one per page) so you can scan visually.

To inspect a specific page at FULL resolution (e.g. to read a small label or examine an image clearly):
  ```
  python /workspace/tools/pdf_page.py <pdf_filename> <page_num> [--dpi 200]
  ```
  This writes a high-res JPEG to /workspace/hires_<stem>_p<NNN>.jpg which you can then read.

To extract the full text of a specific page (with OCR fallback for image-only pages):
  ```
  python /workspace/tools/pdf_text.py <pdf_filename> <page_num>
  ```

To search across all PDFs for a substring or regex:
  ```
  python /workspace/tools/pdf_search.py "<regex>" [<pdf_filename>]
  ```

**Strategy hint**: scan the index + thumbnails to identify candidate pages, then fetch full-resolution images for the pages that need a close read. Do not request high-resolution images for pages you do not need — each fetch is several MB.

**Important — image dimension cap**: Any image you view (via `file_editor view` or attached automatically as a tool result) must have `max(width, height) ≤ 2000 px`. The LLM API will reject requests containing larger images.

  - `pdf_page.py` auto-clamps every output to this limit, so any `--dpi` value is safe (it will scale back if the requested DPI would produce >2000 px).
  - If you want a closer look at fine details, **pass a higher `--dpi`** (e.g. `--dpi 300`) to `pdf_page.py` — it will return the maximum safe resolution for that page's dimensions.
  - **Do NOT call `pymupdf` / `fitz` directly to render pages at high zoom** (e.g. `Matrix(4, 4)`) — the resulting image will likely exceed 2000 px and your next tool call will fail. Using `pymupdf` for *text* extraction is fine — the cap only applies to rendered images.
  - If you crop an image with PIL or any other tool, make sure the saved crop is ≤ 2000 px on its longest side before viewing it.
""".strip()
