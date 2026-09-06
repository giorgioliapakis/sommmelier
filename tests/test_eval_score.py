"""Recovery scoring must preserve outcome units and quality failures."""

import pytest

from evals.score import score_recovery


def test_recovery_scores_zero_effect_and_does_not_override_failed_quality():
    truth = {
        "scenario": "zero_effect",
        "seed": 42,
        "expected_validation": "pass",
        "roi_is_monetary": True,
        "true_roi": {"meta": 0.0, "video": 2.0},
    }
    results = {
        "metadata": {"roi_is_monetary": True},
        "run_manifest": {"status": "complete", "quality_status": "failed"},
        "roi": {
            "meta": {"mean": 0.1, "ci_lower": 0.0, "ci_upper": 0.2},
            "video": {"mean": 1.0, "ci_lower": 0.5, "ci_upper": 1.5},
        },
    }
    score = score_recovery(results, truth)
    assert score["mean_absolute_error"] == pytest.approx(0.55)
    assert score["interval_coverage"] == 0.5
    assert score["decision_ready"] is False
    results["metadata"]["roi_is_monetary"] = False
    with pytest.raises(ValueError, match="units do not match"):
        score_recovery(results, truth)
