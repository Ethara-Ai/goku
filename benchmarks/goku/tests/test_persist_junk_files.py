"""Regression tests for OS-junk filtering in run-output persistence.

Guards two properties of ``_persist_run_outputs``:
  1. ``.DS_Store`` / ``Thumbs.db`` are never copied into ``results/``.
  2. ``.gitkeep`` IS copied, so a directory whose only content is ``.gitkeep``
     still exists under ``results/`` — otherwise ``probe_dir_exists`` rubrics
     would flip pass -> fail.
"""

from pathlib import Path

from benchmarks.goku.service import _persist_run_outputs


def _make_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "agent_workspace"
    ws.mkdir()
    (ws / "report.md").write_text("real output", encoding="utf-8")
    (ws / ".DS_Store").write_bytes(b"\x00\x01junk")
    (ws / "Thumbs.db").write_bytes(b"\x00\x01junk")
    empty_dir = ws / "expected_dir"
    empty_dir.mkdir()
    (empty_dir / ".gitkeep").write_text("", encoding="utf-8")
    return ws


def test_os_junk_excluded_but_gitkeep_dir_preserved(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    task_root = tmp_path / "task"
    task_root.mkdir()

    run_dir = _persist_run_outputs(
        task_root=task_root,
        model_segment="model_x",
        run_index=1,
        scores=[],
        output_files=[],
        agent_workspace=ws,
    )
    results = run_dir / "results"

    # Junk is gone.
    assert not (results / ".DS_Store").exists()
    assert not (results / "Thumbs.db").exists()

    # Real output survives.
    assert (results / "report.md").read_text(encoding="utf-8") == "real output"

    # The .gitkeep-only directory is still reproduced -> probe_dir_exists safe.
    assert (results / "expected_dir").is_dir()
    assert (results / "expected_dir" / ".gitkeep").exists()
