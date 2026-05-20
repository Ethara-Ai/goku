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


class TestCleanShellStderr:
    """Verify the harness boot-noise filter on subprocess stderr."""

    def test_empty_stderr_returns_placeholder(self):
        from benchmarks.goku.scorers.deterministic import _clean_shell_stderr
        assert _clean_shell_stderr("") == "(empty)"
        assert _clean_shell_stderr("   \n  ") == "(empty)"

    def test_strips_sitecustomize_banner(self):
        from benchmarks.goku.scorers.deterministic import _clean_shell_stderr
        err = (
            "benchmarks sitecustomize imported\n"
            "+----------------------------------+\n"
            "|  OpenHands SDK v1.22.0           |\n"
            "|  Set OPENHANDS_SUPPRESS_BANNER=1 |\n"
            "+----------------------------------+\n"
            "real error here\n"
        )
        assert _clean_shell_stderr(err) == "real error here"

    def test_strips_modal_sandbox_noise(self):
        from benchmarks.goku.scorers.deterministic import _clean_shell_stderr
        err = (
            "benchmarks injected modal sitecustomize into run_instance_modal image\n"
            "[benchmarks] modal sitecustomize: applied sandbox timing patch\n"
            "[benchmarks] modal sitecustomize: applied runtime debug patch\n"
            "[modal-client] 2026-05-18 Warning: function name 'run_instance_modal' collision\n"
            "[benchmarks] modal sitecustomize: patched function timeout to 14400s\n"
            "AssertionError\n"
        )
        assert _clean_shell_stderr(err) == "AssertionError"

    def test_prefers_traceback_when_present(self):
        from benchmarks.goku.scorers.deterministic import _clean_shell_stderr
        err = (
            "benchmarks sitecustomize imported\n"
            "[benchmarks] modal sitecustomize: applied runtime debug patch\n"
            "Traceback (most recent call last):\n"
            '  File "<string>", line 1, in <module>\n'
            "    assert \"food_items\" in d\n"
            "AssertionError\n"
        )
        result = _clean_shell_stderr(err)
        assert result.startswith("Traceback")
        assert "AssertionError" in result
        # Crucially: no sitecustomize / modal banner content
        assert "sitecustomize" not in result
        assert "modal" not in result

    def test_budget_respected_on_huge_input(self):
        from benchmarks.goku.scorers.deterministic import _clean_shell_stderr
        err = "garbage line\n" * 1000 + "FINAL ERROR\n"
        result = _clean_shell_stderr(err, budget=50)
        assert len(result) <= 50
        # Tail of the real content should still appear
        assert "FINAL ERROR" in result

    def test_traceback_budget_takes_tail(self):
        from benchmarks.goku.scorers.deterministic import _clean_shell_stderr
        # Very long traceback — we want the AssertionError at the end, not the start
        err = (
            "Traceback (most recent call last):\n"
            + "  File \"long\", line 1, in <module>\n" * 50
            + "AssertionError: the real error\n"
        )
        result = _clean_shell_stderr(err, budget=80)
        assert "AssertionError: the real error" in result
