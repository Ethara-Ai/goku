"""Tests for score_llm_judge_council() — the 3-judge majority-vote scorer.

Each judge call is mocked at the _score_llm_judge_single boundary so these
tests run instantly with zero LLM cost. The mock injects the per-judge
verdicts the test wants to assert against.

Covers (the matrix of council outcomes):
  - unanimous pass (3/3)
  - majority pass (2/1)
  - majority fail (1/2)
  - unanimous fail (0/3)
  - one judge crashes — council still produces a verdict
  - two judges crash — conservative fail (1/3 pass at most)
  - cost is summed across judges
  - per_judge_verdicts preserves each judge's model + rationale
  - input validation: ≥2 judges required; mismatched parallel list lengths rejected
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from benchmarks.goku.models import RubricItem, ScorerResult
from benchmarks.goku.scorers.llm_judge import score_llm_judge_council


def _item(number: int = 1, points: int = 5) -> RubricItem:
    return RubricItem(
        number=number,
        type="response_criteria",
        category="MM_REASONING",
        points=points,
        importance="mandatory",
        criterion="test criterion",
    )


def _make_mock(verdicts: list[bool], cost_each: float = 0.10):
    """Return a _score_llm_judge_single mock yielding verdicts in call order."""
    calls = {"idx": 0}

    def _mock(item, response, file_contents, trajectory, judge_model, **kw):
        idx = calls["idx"]
        passed = verdicts[idx]
        calls["idx"] = idx + 1
        return ScorerResult(
            number=item.number,
            passed=passed,
            judge_rationale=f"mock-{judge_model}: {'pass' if passed else 'fail'}",
            points_awarded=item.points if passed else 0,
            judge_cost_usd=cost_each,
        )

    return _mock


class TestCouncilAggregation:
    """The four canonical vote patterns + per_judge_verdicts preservation."""

    def test_unanimous_pass(self):
        with patch(
            "benchmarks.goku.scorers.llm_judge._score_llm_judge_single",
            side_effect=_make_mock([True, True, True]),
        ):
            r = score_llm_judge_council(
                item=_item(), response="x", file_contents="y", trajectory="z",
                judge_models=["m1", "m2", "m3"],
            )
        assert r.passed is True
        assert r.points_awarded == 5
        assert r.vote == "3/3"
        assert r.consensus == "unanimous"
        assert r.disagreement == 0
        assert len(r.per_judge_verdicts) == 3
        assert all(v.passed for v in r.per_judge_verdicts)

    def test_majority_pass_2_1(self):
        with patch(
            "benchmarks.goku.scorers.llm_judge._score_llm_judge_single",
            side_effect=_make_mock([True, True, False]),
        ):
            r = score_llm_judge_council(
                item=_item(), response="x", file_contents="y", trajectory="z",
                judge_models=["m1", "m2", "m3"],
            )
        assert r.passed is True
        assert r.points_awarded == 5
        assert r.vote == "2/3"
        assert r.consensus == "majority"
        assert r.disagreement == 1

    def test_majority_fail_1_2(self):
        with patch(
            "benchmarks.goku.scorers.llm_judge._score_llm_judge_single",
            side_effect=_make_mock([True, False, False]),
        ):
            r = score_llm_judge_council(
                item=_item(), response="x", file_contents="y", trajectory="z",
                judge_models=["m1", "m2", "m3"],
            )
        assert r.passed is False
        assert r.points_awarded == 0  # No points awarded on majority-fail
        assert r.vote == "1/3"
        assert r.consensus == "majority"
        assert r.disagreement == 1

    def test_unanimous_fail(self):
        with patch(
            "benchmarks.goku.scorers.llm_judge._score_llm_judge_single",
            side_effect=_make_mock([False, False, False]),
        ):
            r = score_llm_judge_council(
                item=_item(), response="x", file_contents="y", trajectory="z",
                judge_models=["m1", "m2", "m3"],
            )
        assert r.passed is False
        assert r.points_awarded == 0
        assert r.vote == "0/3"
        assert r.consensus == "unanimous"
        assert r.disagreement == 0

    def test_per_judge_verdicts_preserved(self):
        with patch(
            "benchmarks.goku.scorers.llm_judge._score_llm_judge_single",
            side_effect=_make_mock([True, False, True]),
        ):
            r = score_llm_judge_council(
                item=_item(), response="x", file_contents="y", trajectory="z",
                judge_models=["judge_a", "judge_b", "judge_c"],
            )
        assert len(r.per_judge_verdicts) == 3
        # Order is preserved by index even though calls happen concurrently.
        assert r.per_judge_verdicts[0].judge_model == "judge_a"
        assert r.per_judge_verdicts[1].judge_model == "judge_b"
        assert r.per_judge_verdicts[2].judge_model == "judge_c"
        assert r.per_judge_verdicts[0].passed is True
        assert r.per_judge_verdicts[1].passed is False
        assert r.per_judge_verdicts[2].passed is True
        assert "judge_a" in r.per_judge_verdicts[0].judge_rationale


class TestCouncilCost:
    """Council cost = sum of per-judge costs."""

    def test_cost_is_summed(self):
        with patch(
            "benchmarks.goku.scorers.llm_judge._score_llm_judge_single",
            side_effect=_make_mock([True, True, True], cost_each=0.07),
        ):
            r = score_llm_judge_council(
                item=_item(), response="x", file_contents="y", trajectory="z",
                judge_models=["m1", "m2", "m3"],
            )
        assert r.judge_cost_usd == pytest.approx(0.21)

    def test_cost_includes_failing_judges(self):
        # Even failing judges contributed compute (LiteLLM still bills for tokens)
        with patch(
            "benchmarks.goku.scorers.llm_judge._score_llm_judge_single",
            side_effect=_make_mock([True, False, True], cost_each=0.05),
        ):
            r = score_llm_judge_council(
                item=_item(), response="x", file_contents="y", trajectory="z",
                judge_models=["m1", "m2", "m3"],
            )
        assert r.judge_cost_usd == pytest.approx(0.15)


class TestCouncilFailureHandling:
    """One or more judges crashing → council still produces a verdict."""

    def test_one_judge_crashes_others_decide(self):
        def _flaky(item, response, file_contents, trajectory, judge_model, **kw):
            if judge_model == "broken":
                raise TimeoutError("simulated timeout")
            return ScorerResult(
                number=item.number, passed=True,
                judge_rationale=f"mock {judge_model}: pass",
                points_awarded=item.points, judge_cost_usd=0.05,
            )

        with patch(
            "benchmarks.goku.scorers.llm_judge._score_llm_judge_single",
            side_effect=_flaky,
        ):
            r = score_llm_judge_council(
                item=_item(), response="x", file_contents="y", trajectory="z",
                judge_models=["m1", "broken", "m3"],
            )

        # 2 judges said pass, 1 crashed (counted as fail vote) → 2/3 majority pass
        assert r.passed is True
        assert r.vote == "2/3"
        assert r.disagreement == 1
        errored = [v for v in r.per_judge_verdicts if v.error]
        assert len(errored) == 1
        assert errored[0].judge_model == "broken"
        assert "TimeoutError" in errored[0].error

    def test_two_judges_crash_conservative_fail(self):
        def _flaky(item, response, file_contents, trajectory, judge_model, **kw):
            if judge_model in ("broken1", "broken2"):
                raise ConnectionError("network down")
            return ScorerResult(
                number=item.number, passed=True,
                judge_rationale=f"mock {judge_model}: pass",
                points_awarded=item.points, judge_cost_usd=0.05,
            )

        with patch(
            "benchmarks.goku.scorers.llm_judge._score_llm_judge_single",
            side_effect=_flaky,
        ):
            r = score_llm_judge_council(
                item=_item(), response="x", file_contents="y", trajectory="z",
                judge_models=["broken1", "broken2", "m3"],
            )

        # 1 judge pass + 2 crashed (each = fail vote) → 1/3 → majority fail
        assert r.passed is False
        assert r.vote == "1/3"
        assert sum(1 for v in r.per_judge_verdicts if v.error) == 2


class TestCouncilInputValidation:
    """Bad inputs reject cleanly."""

    def test_requires_at_least_2_judges(self):
        with pytest.raises(ValueError, match="requires ≥2 judges"):
            score_llm_judge_council(
                item=_item(), response="x", file_contents="y", trajectory="z",
                judge_models=["only_one"],
            )

    def test_mismatched_api_keys_length(self):
        with pytest.raises(ValueError, match="length .* != judge_models length"):
            score_llm_judge_council(
                item=_item(), response="x", file_contents="y", trajectory="z",
                judge_models=["m1", "m2", "m3"],
                judge_api_keys=["k1", "k2"],  # only 2, but 3 models
            )

    def test_api_keys_none_padded_to_match(self):
        # When judge_api_keys=None, function should pad with [None]*n internally.
        with patch(
            "benchmarks.goku.scorers.llm_judge._score_llm_judge_single",
            side_effect=_make_mock([True, True, True]),
        ):
            r = score_llm_judge_council(
                item=_item(), response="x", file_contents="y", trajectory="z",
                judge_models=["m1", "m2", "m3"],
                judge_api_keys=None,  # explicitly None
            )
        assert r.passed is True


class TestCouncilSchemaSerialization:
    """ScorerResult with per_judge_verdicts must JSON-roundtrip cleanly."""

    def test_council_result_jsonl_roundtrip(self):
        with patch(
            "benchmarks.goku.scorers.llm_judge._score_llm_judge_single",
            side_effect=_make_mock([True, False, True]),
        ):
            r = score_llm_judge_council(
                item=_item(), response="x", file_contents="y", trajectory="z",
                judge_models=["m1", "m2", "m3"],
            )
        s = r.model_dump_json()
        r2 = ScorerResult.model_validate_json(s)
        assert r2.passed == r.passed
        assert r2.vote == r.vote
        assert r2.consensus == r.consensus
        assert r2.disagreement == r.disagreement
        assert len(r2.per_judge_verdicts) == 3
        assert r2.per_judge_verdicts[0].judge_model == "m1"

    def test_single_judge_result_excludes_council_fields_in_jsonl(self):
        """Backward-compat: old-style ScorerResult.model_dump(exclude_none=True)
        must NOT include the new council fields (per_judge_verdicts, vote, etc.).
        Existing scores.jsonl readers must parse council-mode output unchanged.
        """
        r = ScorerResult(
            number=1, passed=True, judge_rationale="single judge",
            points_awarded=5, judge_cost_usd=0.02,
        )
        dumped = r.model_dump(exclude_none=True)
        assert "per_judge_verdicts" not in dumped
        assert "vote" not in dumped
        assert "consensus" not in dumped
        assert "disagreement" not in dumped
        # Old fields remain
        assert dumped["passed"] is True
        assert dumped["judge_cost_usd"] == 0.02
