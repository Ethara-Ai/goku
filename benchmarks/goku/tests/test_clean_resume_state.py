"""Tests for benchmarks.goku.scripts.clean_resume_state.

Covers:
  1. Strip a task entry from output.jsonl, preserving others
  2. Strip from output.critic_attempt_*.jsonl
  3. No-op when task absent (no archive created)
  4. Multiple tasks in one invocation
  5. Malformed lines preserved (cleanup is conservative)
  6. Dry-run touches nothing
  7. Archive backup created before write
  8. Idempotent — re-running doesn't create second archive
  9. Per-task subdirectory and conversations tarball archived
 10. Model filter substring match
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.goku.scripts.clean_resume_state import (
    archive_task_artifacts,
    clean,
    find_model_dirs,
    strip_task_from_jsonl,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def _model_dir(tmp_path: Path, slug: str = "claude-opus-4.7_sdk_abc_maxiter_30") -> Path:
    """Make a fake eval_outputs/run_1/run_1/goku/<slug>/ structure."""
    d = tmp_path / "run_1" / "run_1" / "goku" / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "conversations").mkdir(exist_ok=True)
    return d


# ---------- strip_task_from_jsonl ----------


def test_strip_removes_matching_line(tmp_path: Path):
    f = tmp_path / "output.jsonl"
    _write_jsonl(f, [
        {"instance_id": "task_a", "data": 1},
        {"instance_id": "task_b", "data": 2},
        {"instance_id": "task_c", "data": 3},
    ])
    kept, removed = strip_task_from_jsonl(
        f, {"task_b"}, dry_run=False, archive_suffix=".bak"
    )
    assert kept == 2 and removed == 1
    lines = [json.loads(l) for l in f.read_text().strip().splitlines()]
    assert [l["instance_id"] for l in lines] == ["task_a", "task_c"]


def test_strip_creates_archive_on_first_write(tmp_path: Path):
    f = tmp_path / "output.jsonl"
    _write_jsonl(f, [
        {"instance_id": "task_a"},
        {"instance_id": "task_b"},
    ])
    archive = f.with_suffix(".jsonl.bak")
    assert not archive.exists()
    strip_task_from_jsonl(f, {"task_b"}, dry_run=False, archive_suffix=".bak")
    assert archive.exists()
    archived = [json.loads(l) for l in archive.read_text().strip().splitlines()]
    assert {a["instance_id"] for a in archived} == {"task_a", "task_b"}


def test_strip_idempotent_archive(tmp_path: Path):
    """Re-running should NOT overwrite the original archive."""
    f = tmp_path / "output.jsonl"
    _write_jsonl(f, [{"instance_id": "task_a"}, {"instance_id": "task_b"}])
    archive = f.with_suffix(".jsonl.bak")

    # First run
    strip_task_from_jsonl(f, {"task_b"}, dry_run=False, archive_suffix=".bak")
    original_archive_bytes = archive.read_bytes()

    # Re-add task_b, then re-run strip
    f.write_text(f.read_text() + json.dumps({"instance_id": "task_b"}) + "\n")
    strip_task_from_jsonl(f, {"task_b"}, dry_run=False, archive_suffix=".bak")

    # Archive must still be the original
    assert archive.read_bytes() == original_archive_bytes


def test_strip_noop_when_task_absent(tmp_path: Path):
    f = tmp_path / "output.jsonl"
    _write_jsonl(f, [{"instance_id": "task_a"}])
    archive = f.with_suffix(".jsonl.bak")
    kept, removed = strip_task_from_jsonl(
        f, {"task_z"}, dry_run=False, archive_suffix=".bak"
    )
    assert kept == 1 and removed == 0
    # No archive created for no-op
    assert not archive.exists()


def test_strip_multiple_task_keys(tmp_path: Path):
    f = tmp_path / "output.jsonl"
    _write_jsonl(f, [
        {"instance_id": "task_a"},
        {"instance_id": "task_b"},
        {"instance_id": "task_c"},
        {"instance_id": "task_d"},
    ])
    kept, removed = strip_task_from_jsonl(
        f, {"task_b", "task_d"}, dry_run=False, archive_suffix=".bak"
    )
    assert kept == 2 and removed == 2


def test_strip_preserves_malformed_lines(tmp_path: Path):
    """Conservative behavior: malformed JSON lines are kept, not silently dropped."""
    f = tmp_path / "output.jsonl"
    f.write_text(
        '{"instance_id": "task_a"}\n'
        '{not valid json\n'
        '{"instance_id": "task_b"}\n'
        '\n'  # blank line — skipped
        '{"instance_id": "task_c"}\n'
    )
    kept, removed = strip_task_from_jsonl(
        f, {"task_b"}, dry_run=False, archive_suffix=".bak"
    )
    # task_a, malformed, task_c kept; task_b removed; blank skipped
    assert kept == 3 and removed == 1
    out_lines = f.read_text().splitlines()
    assert any("not valid json" in l for l in out_lines)


def test_strip_dry_run_doesnt_modify(tmp_path: Path):
    f = tmp_path / "output.jsonl"
    original = [{"instance_id": "task_a"}, {"instance_id": "task_b"}]
    _write_jsonl(f, original)
    before = f.read_bytes()

    kept, removed = strip_task_from_jsonl(
        f, {"task_b"}, dry_run=True, archive_suffix=".bak"
    )
    # Counts still reported correctly
    assert kept == 1 and removed == 1
    # But file unchanged
    assert f.read_bytes() == before
    assert not f.with_suffix(".jsonl.bak").exists()


def test_strip_missing_file_returns_zero(tmp_path: Path):
    kept, removed = strip_task_from_jsonl(
        tmp_path / "nonexistent.jsonl", {"task_a"},
        dry_run=False, archive_suffix=".bak",
    )
    assert kept == 0 and removed == 0


# ---------- archive_task_artifacts ----------


def test_archive_task_subdir(tmp_path: Path):
    md = _model_dir(tmp_path)
    (md / "task_a").mkdir()
    (md / "task_a" / "scores.jsonl").write_text("{}\n")
    n = archive_task_artifacts(
        md, {"task_a"}, dry_run=False, archive_suffix=".bak"
    )
    assert n == 1
    assert (md / "task_a.bak").is_dir()
    assert not (md / "task_a").exists()


def test_archive_conversations_tarball(tmp_path: Path):
    md = _model_dir(tmp_path)
    tar = md / "conversations" / "task_a.tar.gz"
    tar.parent.mkdir(exist_ok=True)
    tar.write_bytes(b"fake gzip")
    n = archive_task_artifacts(
        md, {"task_a"}, dry_run=False, archive_suffix=".bak"
    )
    assert n == 1
    assert (md / "conversations" / "task_a.tar.gz.bak").is_file()
    assert not tar.exists()


def test_archive_noop_when_absent(tmp_path: Path):
    md = _model_dir(tmp_path)
    n = archive_task_artifacts(
        md, {"task_missing"}, dry_run=False, archive_suffix=".bak"
    )
    assert n == 0


# ---------- find_model_dirs ----------


def test_find_model_dirs_no_filter(tmp_path: Path):
    _model_dir(tmp_path, "claude-opus-4.7_sdk_abc_maxiter_30")
    _model_dir(tmp_path, "gemini-3.1_sdk_abc_maxiter_30")
    dirs = find_model_dirs(tmp_path, None)
    names = sorted(d.name for d in dirs)
    assert names == ["claude-opus-4.7_sdk_abc_maxiter_30", "gemini-3.1_sdk_abc_maxiter_30"]


def test_find_model_dirs_with_filter(tmp_path: Path):
    _model_dir(tmp_path, "claude-opus-4.7_sdk_abc_maxiter_30")
    _model_dir(tmp_path, "gemini-3.1_sdk_abc_maxiter_30")
    _model_dir(tmp_path, "gpt-5.5_sdk_abc_maxiter_30")
    dirs = find_model_dirs(tmp_path, ["claude-opus", "gpt"])
    names = sorted(d.name for d in dirs)
    assert names == [
        "claude-opus-4.7_sdk_abc_maxiter_30",
        "gpt-5.5_sdk_abc_maxiter_30",
    ]


def test_find_model_dirs_skips_non_sdk_dirs(tmp_path: Path):
    """Directories not matching the OpenHands SDK naming convention are skipped."""
    _model_dir(tmp_path, "claude-opus-4.7_sdk_abc_maxiter_30")
    other = tmp_path / "run_1" / "run_1" / "goku" / "random_dir"
    other.mkdir(parents=True)
    dirs = find_model_dirs(tmp_path, None)
    assert len(dirs) == 1
    assert "_sdk_" in dirs[0].name


# ---------- clean (end-to-end) ----------


def test_clean_strips_both_jsonl_types(tmp_path: Path):
    md = _model_dir(tmp_path)
    _write_jsonl(md / "output.jsonl", [
        {"instance_id": "task_a"},
        {"instance_id": "task_target"},
        {"instance_id": "task_c"},
    ])
    _write_jsonl(md / "output.critic_attempt_1.jsonl", [
        {"instance_id": "task_target"},
        {"instance_id": "task_d"},
    ])
    _write_jsonl(md / "output.critic_attempt_2.jsonl", [
        {"instance_id": "task_target"},
    ])

    n_dirs, n_removed, _ = clean(tmp_path, {"task_target"})

    assert n_dirs == 1
    assert n_removed == 3  # one from each of the 3 files

    # Verify content
    remaining = [json.loads(l) for l in (md / "output.jsonl").read_text().strip().splitlines()]
    assert {r["instance_id"] for r in remaining} == {"task_a", "task_c"}
    remaining = [json.loads(l) for l in (md / "output.critic_attempt_1.jsonl").read_text().strip().splitlines()]
    assert {r["instance_id"] for r in remaining} == {"task_d"}
    remaining_text = (md / "output.critic_attempt_2.jsonl").read_text().strip()
    assert remaining_text == ""  # all entries were target


def test_clean_with_model_filter(tmp_path: Path):
    md1 = _model_dir(tmp_path, "claude-opus-4.7_sdk_abc_maxiter_30")
    md2 = _model_dir(tmp_path, "gpt-5.5_sdk_abc_maxiter_30")
    _write_jsonl(md1 / "output.jsonl", [{"instance_id": "task_x"}])
    _write_jsonl(md2 / "output.jsonl", [{"instance_id": "task_x"}])

    n_dirs, n_removed, _ = clean(
        tmp_path, {"task_x"}, model_filters=["claude-opus"]
    )
    assert n_dirs == 1
    assert n_removed == 1
    # md2 untouched
    assert (md2 / "output.jsonl").read_text().strip() != ""


def test_clean_missing_base_raises(tmp_path: Path):
    import pytest
    with pytest.raises(FileNotFoundError):
        clean(tmp_path / "nonexistent", {"task_x"})


def test_clean_archives_per_task_dir_and_conversations(tmp_path: Path):
    md = _model_dir(tmp_path)
    (md / "task_target").mkdir()
    (md / "conversations" / "task_target.tar.gz").write_bytes(b"x")
    _write_jsonl(md / "output.jsonl", [{"instance_id": "task_target"}])

    _, _, n_archived = clean(tmp_path, {"task_target"})

    assert n_archived == 2  # subdir + tarball
    # Originals gone, archives present
    assert not (md / "task_target").exists()
    assert not (md / "conversations" / "task_target.tar.gz").exists()
    # Find the archives (suffix has timestamp)
    arch_dirs = list(md.glob("task_target.archive_pre_rerun_*"))
    arch_tars = list((md / "conversations").glob("task_target.tar.gz.archive_pre_rerun_*"))
    assert len(arch_dirs) == 1
    assert len(arch_tars) == 1
