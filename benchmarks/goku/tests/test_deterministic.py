"""Tests for benchmarks.goku.scorers.deterministic."""

from pathlib import Path

import pytest

from benchmarks.goku.models import RubricItem
from benchmarks.goku.scorers.deterministic import score_deterministic


def _make_item(**kwargs) -> RubricItem:
    """Helper to create a RubricItem with defaults."""
    defaults = {
        "number": 1,
        "category": "FORMAT",
        "points": 5,
        "importance": "mandatory",
        "criterion": "Test criterion",
    }
    defaults.update(kwargs)
    return RubricItem(**defaults)


class TestProbeFileExists:
    def test_file_exists(self, tmp_path: Path):
        (tmp_path / "output.json").write_text('{"data": 1}')
        item = _make_item(type="probe_file_exists", paths=["output.json"])
        result = score_deterministic(item, tmp_path, "")
        assert result.passed is True
        assert result.points_awarded == 5

    def test_file_missing(self, tmp_path: Path):
        item = _make_item(type="probe_file_exists", paths=["output.json"])
        result = score_deterministic(item, tmp_path, "")
        assert result.passed is False
        assert result.points_awarded == 0

    def test_multiple_files_all_exist(self, tmp_path: Path):
        (tmp_path / "a.json").write_text("{}")
        (tmp_path / "b.md").write_text("# Report")
        item = _make_item(type="probe_file_exists", paths=["a.json", "b.md"])
        result = score_deterministic(item, tmp_path, "")
        assert result.passed is True

    def test_multiple_files_one_missing(self, tmp_path: Path):
        (tmp_path / "a.json").write_text("{}")
        item = _make_item(type="probe_file_exists", paths=["a.json", "b.md"])
        result = score_deterministic(item, tmp_path, "")
        assert result.passed is False

    def test_no_paths_specified(self, tmp_path: Path):
        item = _make_item(type="probe_file_exists", paths=None)
        result = score_deterministic(item, tmp_path, "")
        assert result.passed is False


class TestProbeFileContains:
    def test_pattern_found(self, tmp_path: Path):
        (tmp_path / "data.json").write_text('{"items": [1, 2, 3]}')
        item = _make_item(
            type="probe_file_contains",
            path="data.json",
            pattern=r'"items":\s*\[',
        )
        result = score_deterministic(item, tmp_path, "")
        assert result.passed is True

    def test_pattern_not_found(self, tmp_path: Path):
        (tmp_path / "data.json").write_text('{"items": []}')
        item = _make_item(
            type="probe_file_contains",
            path="data.json",
            pattern=r'"count":\s*\d+',
        )
        result = score_deterministic(item, tmp_path, "")
        assert result.passed is False

    def test_ignore_case(self, tmp_path: Path):
        (tmp_path / "report.md").write_text("The Total Is $500")
        item = _make_item(
            type="probe_file_contains",
            path="report.md",
            pattern=r"the total is",
            ignore_case=True,
        )
        result = score_deterministic(item, tmp_path, "")
        assert result.passed is True

    def test_case_sensitive_by_default(self, tmp_path: Path):
        (tmp_path / "report.md").write_text("The Total Is $500")
        item = _make_item(
            type="probe_file_contains",
            path="report.md",
            pattern=r"the total is",
            ignore_case=False,
        )
        result = score_deterministic(item, tmp_path, "")
        assert result.passed is False

    def test_file_not_found(self, tmp_path: Path):
        item = _make_item(
            type="probe_file_contains",
            path="missing.json",
            pattern=r"test",
        )
        result = score_deterministic(item, tmp_path, "")
        assert result.passed is False


class TestProbeDirExists:
    def test_dir_exists(self, tmp_path: Path):
        (tmp_path / "images").mkdir()
        item = _make_item(type="probe_dir_exists", paths=["images"])
        result = score_deterministic(item, tmp_path, "")
        assert result.passed is True

    def test_dir_missing(self, tmp_path: Path):
        item = _make_item(type="probe_dir_exists", paths=["images"])
        result = score_deterministic(item, tmp_path, "")
        assert result.passed is False


class TestShellSucceedsReal:
    def test_exit_zero(self, tmp_path: Path):
        item = _make_item(type="shell_succeeds_real", raw_shell="echo hello")
        result = score_deterministic(item, tmp_path, "")
        assert result.passed is True

    def test_exit_nonzero(self, tmp_path: Path):
        item = _make_item(type="shell_succeeds_real", raw_shell="exit 1")
        result = score_deterministic(item, tmp_path, "")
        assert result.passed is False

    def test_uses_output_dir_as_cwd(self, tmp_path: Path):
        (tmp_path / "test_file.txt").write_text("content")
        item = _make_item(
            type="shell_succeeds_real",
            raw_shell="test -f test_file.txt",
        )
        result = score_deterministic(item, tmp_path, "")
        assert result.passed is True


class TestResponseContains:
    def test_needle_found(self, tmp_path: Path):
        item = _make_item(
            type="response_contains",
            needles=["inventory", "items"],
        )
        result = score_deterministic(
            item, tmp_path, "Here is the inventory with 12 items."
        )
        assert result.passed is True

    def test_needle_missing(self, tmp_path: Path):
        item = _make_item(
            type="response_contains",
            needles=["inventory", "calories"],
        )
        result = score_deterministic(
            item, tmp_path, "Here is the inventory."
        )
        assert result.passed is False
        assert "calories" in result.judge_rationale

    def test_case_insensitive(self, tmp_path: Path):
        item = _make_item(
            type="response_contains",
            needles=["HELLO"],
        )
        result = score_deterministic(item, tmp_path, "hello world")
        assert result.passed is True


class TestResponseRegexPresent:
    def test_regex_matches(self, tmp_path: Path):
        item = _make_item(
            type="response_regex_present",
            pattern=r"\d+ items",
        )
        result = score_deterministic(
            item, tmp_path, "Found 12 items in the pantry"
        )
        assert result.passed is True

    def test_regex_no_match(self, tmp_path: Path):
        item = _make_item(
            type="response_regex_present",
            pattern=r"\d{4}-\d{2}-\d{2}",
        )
        result = score_deterministic(
            item, tmp_path, "No date present here"
        )
        assert result.passed is False


class TestNegativePointsScoring:
    def test_negative_item_passed_deducts_points(self, tmp_path: Path):
        item = _make_item(
            type="response_contains",
            points=-5,
            needles=["fabricated"],
        )
        result = score_deterministic(
            item, tmp_path, "This is fabricated data"
        )
        assert result.passed is True
        assert result.points_awarded == -5

    def test_negative_item_not_passed_no_deduction(self, tmp_path: Path):
        item = _make_item(
            type="response_contains",
            points=-5,
            needles=["fabricated"],
        )
        result = score_deterministic(
            item, tmp_path, "This is accurate data"
        )
        assert result.passed is False
        assert result.points_awarded == 0


class TestInvalidType:
    def test_rejects_non_deterministic_type(self, tmp_path: Path):
        item = _make_item(type="response_criteria")
        with pytest.raises(ValueError, match="not deterministic"):
            score_deterministic(item, tmp_path, "")
