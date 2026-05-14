"""Tests for benchmarks.goku.calibration."""

import logging

from benchmarks.goku.calibration import check_calibration


class TestCheckCalibration:
    def test_well_calibrated(self):
        per_model_scores = {
            "model_a": {"task_1": 0.8, "task_2": 0.6},
            "model_b": {"task_1": 0.7, "task_2": 0.5},
            "model_c": {"task_1": 0.6, "task_2": 0.4},
        }
        results = check_calibration(per_model_scores)
        assert len(results) == 2
        assert all(r["flag"] == "well_calibrated" for r in results)

    def test_too_easy(self):
        per_model_scores = {
            "model_a": {"task_1": 0.95},
            "model_b": {"task_1": 0.98},
            "model_c": {"task_1": 0.92},
        }
        results = check_calibration(per_model_scores)
        assert len(results) == 1
        assert results[0]["flag"] == "too_easy"
        assert results[0]["task_key"] == "task_1"

    def test_too_hard(self):
        per_model_scores = {
            "model_a": {"task_1": 0.1},
            "model_b": {"task_1": 0.2},
            "model_c": {"task_1": 0.15},
        }
        results = check_calibration(per_model_scores)
        assert len(results) == 1
        assert results[0]["flag"] == "too_hard"

    def test_mixed_calibration(self):
        per_model_scores = {
            "model_a": {"easy": 0.95, "hard": 0.1, "good": 0.7},
            "model_b": {"easy": 0.92, "hard": 0.2, "good": 0.6},
            "model_c": {"easy": 0.98, "hard": 0.05, "good": 0.8},
        }
        results = check_calibration(per_model_scores)
        flags = {r["task_key"]: r["flag"] for r in results}
        assert flags["easy"] == "too_easy"
        assert flags["hard"] == "too_hard"
        assert flags["good"] == "well_calibrated"

    def test_custom_thresholds(self):
        per_model_scores = {
            "model_a": {"task_1": 0.85},
            "model_b": {"task_1": 0.88},
        }
        # Default threshold is 0.9, so this should be well_calibrated
        results = check_calibration(per_model_scores)
        assert results[0]["flag"] == "well_calibrated"

        # With lower threshold, it becomes too_easy
        results = check_calibration(
            per_model_scores, too_easy_threshold=0.8
        )
        assert results[0]["flag"] == "too_easy"

    def test_empty_input(self):
        results = check_calibration({})
        assert results == []

    def test_warns_below_target_min(self, caplog):
        per_model_scores = {
            "model_a": {"task_1": 0.5},
            "model_b": {"task_1": 0.6},
        }
        with caplog.at_level(logging.WARNING):
            check_calibration(per_model_scores, target_min_score=0.7)
        assert any("no model exceeds target" in r.message for r in caplog.records)

    def test_results_sorted_by_task_key(self):
        per_model_scores = {
            "m1": {"z_task": 0.5, "a_task": 0.8, "m_task": 0.6},
        }
        results = check_calibration(per_model_scores)
        keys = [r["task_key"] for r in results]
        assert keys == sorted(keys)
