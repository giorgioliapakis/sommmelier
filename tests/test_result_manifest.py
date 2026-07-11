"""Tests for technical run state and independent model-quality state."""

from mmm.result_manifest import (
    create_run_manifest,
    decision_readiness,
    finalize_run_manifest,
    record_section_error,
)


def _complete_results():
    return {
        "run_manifest": create_run_manifest("2026-01-01T00:00:00+00:00"),
        "roi": {"meta": {"mean": 1.0}},
        "contributions": {"meta": {"absolute": 10.0}},
        "model_fit": {"r_squared": 0.8, "mape": 0.1},
        "diagnostics": {"diagnostics_available": True, "convergence_ok": True},
        "model_review": {"passed": True},
        "charts": {"roi": "roi.png"},
    }


def test_complete_run_can_still_fail_statistical_quality():
    results = _complete_results()
    results["diagnostics"]["convergence_ok"] = False
    results["model_review"]["passed"] = False

    manifest = finalize_run_manifest(results, "2026-01-01T01:00:00+00:00")

    assert manifest["status"] == "complete"
    assert manifest["quality_status"] == "failed"
    assert manifest["sections"]["optimal_frequency"] == "not_generated"
    assert manifest["completed_at"] == "2026-01-01T01:00:00+00:00"


def test_missing_required_output_fails_run():
    results = _complete_results()
    results["roi"] = {}

    manifest = finalize_run_manifest(results)

    assert manifest["status"] == "failed"
    assert manifest["sections"]["roi"] == "missing"


def test_optional_extraction_error_marks_run_degraded():
    results = _complete_results()
    error = RuntimeError("optimizer unavailable")
    record_section_error(results, "optimization", error)

    manifest = finalize_run_manifest(results)

    assert manifest["status"] == "degraded"
    assert manifest["errors"] == [
        {
            "section": "optimization",
            "required": False,
            "error_type": "RuntimeError",
            "message": "optimizer unavailable",
        }
    ]


def test_required_extraction_error_is_machine_readable():
    results = _complete_results()
    results["diagnostics"] = {}
    record_section_error(results, "diagnostics", ValueError("unknown R-hat schema"), required=True)

    manifest = finalize_run_manifest(results)

    assert manifest["status"] == "failed"
    assert manifest["sections"]["diagnostics"] == "error"
    assert manifest["quality_status"] == "unknown"


def test_decision_readiness_requires_both_completeness_and_quality():
    results = _complete_results()
    finalize_run_manifest(results)
    assert decision_readiness(results) == (
        True,
        "run complete and model quality passed",
    )

    results["run_manifest"]["quality_status"] = "failed"
    assert decision_readiness(results) == (False, "model quality status is failed")
