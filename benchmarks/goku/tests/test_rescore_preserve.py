"""Regression tests for the --skip-llm-judge preserve path in rescore_single.

Locks the C1 fix: scores.jsonl stores the DISPLAY value of 'passed' (inverted
for negative / points<0 rubric items). When a rescore preserves those verdicts,
it must re-invert negatives back to INTERNAL semantics before feeding them to
compute_task_score — otherwise a detected hallucination penalty is silently
dropped and the task score is inflated.

All tests are OFFLINE: skip_llm_judge=True never calls the judge, and an autouse
fixture booby-traps litellm.completion so any accidental network call fails loud.
"""

import json
from typing import Literal

import pytest

from benchmarks.goku import rescore as rescore_mod
from benchmarks.goku.models import RubricItem, ScorerResult
from benchmarks.goku.scoring import compute_task_score, write_scores_jsonl


@pytest.fixture(autouse=True)
def _no_real_llm(monkeypatch):
    """Any real LLM call during these tests is a bug — make it explode."""

    def _boom(*_a, **_k):
        raise AssertionError("real LLM call attempted in an offline test")

    import litellm

    monkeypatch.setattr(litellm, "completion", _boom, raising=False)
    monkeypatch.setattr(litellm, "acompletion", _boom, raising=False)


def _item(
    number,
    points,
    *,
    negative=False,
    importance: Literal["mandatory", "nice_to_have"] = "nice_to_have",
):
    return RubricItem(
        number=number,
        type="response_not_criteria" if negative else "response_criteria",
        category="HALLUCINATION" if negative else "CORRECTNESS",
        points=points,
        importance=importance,
        criterion=f"Criterion #{number}",
    )


def _write_live_scores(tmp_path, rubric_items, internal_passed):
    """Run the real live scoring + write scores.jsonl (display-inverted)."""
    (tmp_path / "results").mkdir(exist_ok=True)
    live_results = [
        ScorerResult(
            number=it.number,
            passed=internal_passed[it.number],
            judge_rationale=f"real verdict for item {it.number}",
            points_awarded=(it.points if internal_passed[it.number] else 0),
        )
        for it in rubric_items
    ]
    live_score = compute_task_score(live_results, rubric_items)
    write_scores_jsonl(live_score, tmp_path / "scores.jsonl", rubric_items)
    return live_score


def _rescore(tmp_path, rubric_items):
    results, _ = rescore_mod.rescore_single(
        task_dir=tmp_path,
        rubric_items=rubric_items,
        response_text="",
        file_contents="",
        trajectory="",
        judge_model="",
        judge_api_key=None,
        judge_region=None,
        skip_llm_judge=True,
    )
    return results


# --- exhaustive polarity x outcome recovery --------------------------------
@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("internal", [False, True])
def test_preserve_recovers_internal_passed(tmp_path, negative, internal):
    points = -5 if negative else 5
    rubric = [_item(1, points, negative=negative)]
    _write_live_scores(tmp_path, rubric, {1: internal})

    results = _rescore(tmp_path, rubric)

    assert len(results) == 1
    # The recovered INTERNAL 'passed' must equal what the live judge decided,
    # regardless of the display inversion applied on write.
    assert results[0].passed is internal


# --- the headline C1 case: negative penalty must survive a rescore ---------
def test_negative_penalty_not_dropped_on_rescore(tmp_path):
    rubric = [_item(1, -5, negative=True, importance="mandatory")]  # hallucination
    live = _write_live_scores(tmp_path, rubric, {1: True})  # agent DID hallucinate

    # On disk the display value is inverted (False), which is the trap.
    rows = [
        json.loads(line)
        for line in (tmp_path / "scores.jsonl").read_text().splitlines()
        if line.strip()
    ]
    disk = [r for r in rows if "number" in r]
    assert disk[0]["passed"] is False  # display-inverted, as written

    results = _rescore(tmp_path, rubric)
    rescored = compute_task_score(results, rubric)

    # Penalty preserved + mandatory gate still fails, exactly like the live run.
    assert results[0].passed is True
    assert rescored.awarded == live.awarded == -5
    assert rescored.passed is live.passed is False


# --- rescored aggregate must equal the live aggregate (mixed rubric) -------
def test_rescore_matches_live_score(tmp_path):
    rubric = [
        _item(1, -5, negative=True, importance="mandatory"),
        _item(2, -3, negative=True),
        _item(3, 5, importance="mandatory"),
        _item(4, 3),
    ]
    internal = {1: True, 2: False, 3: True, 4: False}
    live = _write_live_scores(tmp_path, rubric, internal)

    results = _rescore(tmp_path, rubric)
    rescored = compute_task_score(results, rubric)

    assert rescored.passed == live.passed
    assert rescored.awarded == live.awarded
    assert rescored.max_total == live.max_total
    assert abs(rescored.per_task_score - live.per_task_score) < 1e-9


# --- idempotency: rescore twice must be stable -----------------------------
def test_rescore_is_idempotent(tmp_path):
    rubric = [_item(1, -5, negative=True), _item(2, 5)]
    live = _write_live_scores(tmp_path, rubric, {1: True, 2: True})

    r1 = _rescore(tmp_path, rubric)
    s1 = compute_task_score(r1, rubric)
    # persist again (as the real flow does) and rescore a second time
    write_scores_jsonl(s1, tmp_path / "scores.jsonl", rubric)
    r2 = _rescore(tmp_path, rubric)
    s2 = compute_task_score(r2, rubric)

    assert [r.passed for r in r1] == [r.passed for r in r2]
    assert s1.awarded == s2.awarded == live.awarded
    assert s1.passed == s2.passed == live.passed


# --- placeholder verdicts must NOT be preserved (stubbed instead) ----------
def test_placeholder_not_preserved(tmp_path):
    rubric = [_item(1, -5, negative=True)]
    (tmp_path / "results").mkdir(exist_ok=True)
    (tmp_path / "scores.jsonl").write_text(
        json.dumps(
            {
                "number": 1,
                "passed": False,
                "judge_rationale": "(skipped — --skip-llm-judge)",
            }
        )
        + "\n"
    )
    results = _rescore(tmp_path, rubric)
    # Stubbed (not a preserved real verdict): passed=False, placeholder rationale.
    assert results[0].passed is False
    assert "skipped" in results[0].judge_rationale


# --- missing scores.jsonl => all stubbed, no crash -------------------------
def test_missing_scores_file_stubs_all(tmp_path):
    rubric = [_item(1, -5, negative=True), _item(2, 5)]
    (tmp_path / "results").mkdir(exist_ok=True)
    results = _rescore(tmp_path, rubric)
    assert [r.passed for r in results] == [False, False]


# ==========================================================================
# RUBRIC-CHANGE correctness: after the rubric is edited between the original
# run and the rescore, the FINAL score must equal an INDEPENDENT oracle of
# what the score should be under the NEW rubric, given the preserved verdicts.
# ==========================================================================
def _oracle_score(new_rubric, original_internal):
    """Ground-truth score under the NEW rubric.

    Preserved items reuse the ORIGINAL internal verdict; items with no prior
    verdict (newly added) are stubbed passed=False. compute_task_score then
    applies the NEW rubric's points/importance. This is what a *correct*
    rescore must reproduce.
    """
    oracle_results = [
        ScorerResult(
            number=it.number,
            passed=original_internal.get(it.number, False),
            judge_rationale="oracle",
            points_awarded=0,
        )
        for it in new_rubric
    ]
    return compute_task_score(oracle_results, new_rubric)


def test_rubric_magnitude_change_scores_under_new_points(tmp_path):
    # Original rubric.
    orig = [_item(1, -5, negative=True), _item(2, 5)]
    original_internal = {1: True, 2: True}  # hallucinated; correct
    _write_live_scores(tmp_path, orig, original_internal)

    # New rubric: same polarity, DIFFERENT magnitudes.
    new = [_item(1, -3, negative=True), _item(2, 10)]
    results = _rescore(tmp_path, new)
    final = compute_task_score(results, new)
    oracle = _oracle_score(new, original_internal)

    # Final score uses the NEW points (-3 penalty, +10 award) and matches oracle.
    assert final.awarded == oracle.awarded == 7  # -3 + 10
    assert final.max_total == oracle.max_total == 10
    assert abs(final.per_task_score - oracle.per_task_score) < 1e-9
    assert final.passed == oracle.passed


def test_rubric_added_and_removed_items(tmp_path):
    # Original rubric had items 1,2,3.
    orig = [_item(1, -5, negative=True), _item(2, 5), _item(3, 5)]
    original_internal = {1: True, 2: True, 3: False}
    _write_live_scores(tmp_path, orig, original_internal)

    # New rubric: item 3 REMOVED, item 4 ADDED (no prior verdict -> stub).
    new = [_item(1, -5, negative=True), _item(2, 5), _item(4, 5)]
    results = _rescore(tmp_path, new)
    final = compute_task_score(results, new)
    oracle = _oracle_score(new, original_internal)  # item4 stubs to False

    assert final.awarded == oracle.awarded
    assert final.max_total == oracle.max_total
    assert final.passed == oracle.passed
    # Item 4 (new) must be stubbed, not accidentally inherited from item 3.
    assert next(r.passed for r in results if r.number == 4) is False


def test_rubric_importance_change_regates(tmp_path):
    # Original: negative item is nice_to_have (does not gate).
    orig = [_item(1, -5, negative=True, importance="nice_to_have"), _item(2, 5)]
    original_internal = {1: True, 2: True}  # hallucinated but non-gating originally
    live = _write_live_scores(tmp_path, orig, original_internal)
    assert live.passed is True  # hallucination didn't fail the task originally

    # New rubric: same item promoted to MANDATORY -> hallucination now fails gate.
    new = [_item(1, -5, negative=True, importance="mandatory"), _item(2, 5)]
    results = _rescore(tmp_path, new)
    final = compute_task_score(results, new)
    oracle = _oracle_score(new, original_internal)

    assert final.passed == oracle.passed is False  # new mandatory gate trips
    assert final.awarded == oracle.awarded
