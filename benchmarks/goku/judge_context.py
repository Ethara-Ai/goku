"""Shared judge-context utilities.

Single source of truth for how we collect the agent's output files for the
LLM judge. Both run_infer (live scoring) and rescore (re-judge existing
trajectories) need the same semantics — keeping the logic in one place
prevents drift that would otherwise produce different scores for the same
agent output depending on which CLI ran.
"""

from __future__ import annotations

from pathlib import Path


MEDIA_SUFFIXES = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp",
    ".pdf",
    ".mp4", ".mov", ".webm", ".avi", ".mkv",
})

TEXT_PREVIEW_BYTES = 50_000
TEXT_HARD_CAP_BYTES = 500_000


def collect_file_contents(
    results_dir: Path,
    *,
    exclude_top_dirs: frozenset[str] | set[str] | None = None,
) -> tuple[str, list[str]]:
    """Return ``(text_summary, output_media_paths)`` for an agent's results dir.

    Args:
        results_dir: Directory the agent wrote to (workspace download or
            delivery `results/` folder).
        exclude_top_dirs: Top-level subdirectory names to skip entirely
            (e.g. ``{"bash_events"}`` for the rescore path, which treats
            bash traces as debugging output rather than artifacts).

    Returns:
        ``text_summary`` — concatenated text content of text-readable
            files. Large text files are truncated to ``TEXT_PREVIEW_BYTES``;
            files exceeding ``TEXT_HARD_CAP_BYTES`` are skipped with a
            placeholder line.
        ``output_media_paths`` — absolute paths to image/PDF/video files
            the agent produced. The judge attaches these natively via the
            per-provider routing in :mod:`benchmarks.goku.scorers.llm_judge`.
    """
    if not results_dir.exists():
        return "(no output files)", []

    exclude = exclude_top_dirs or frozenset()
    contents: list[str] = []
    media_paths: list[str] = []

    for f in sorted(results_dir.rglob("*")):
        if not f.is_file():
            continue
        if exclude:
            try:
                rel = f.relative_to(results_dir)
            except ValueError:
                continue
            if rel.parts and rel.parts[0] in exclude:
                continue

        size = f.stat().st_size
        suffix = f.suffix.lower()

        if suffix in MEDIA_SUFFIXES:
            contents.append(
                f"--- {f.name} --- (attached as output media; {size:,} bytes)"
            )
            media_paths.append(str(f.resolve()))
            continue

        if size > TEXT_HARD_CAP_BYTES:
            contents.append(
                f"--- {f.name} --- ({size} bytes, skipped — exceeds hard cap)"
            )
            continue

        try:
            text = f.read_text(encoding="utf-8")
            if len(text) > TEXT_PREVIEW_BYTES:
                contents.append(
                    f"--- {f.name} --- ({size} bytes, first "
                    f"{TEXT_PREVIEW_BYTES} chars shown)\n{text[:TEXT_PREVIEW_BYTES]}"
                )
            else:
                contents.append(f"--- {f.name} ---\n{text}")
        except UnicodeDecodeError:
            contents.append(f"--- {f.name} --- (binary, {size} bytes)")

    text_summary = "\n\n".join(contents) if contents else "(no output files)"
    return text_summary, media_paths
