"""Tests for durable model-quality history."""

import pytest

from mmm.tracking.model_quality import (
    ModelMetrics,
    ModelQualityTracker,
    extract_metrics_from_results,
)


def test_corrupt_history_is_reported_instead_of_silently_erased(tmp_path):
    history = tmp_path / "history.json"
    history.write_text("not json")

    with pytest.raises(ValueError, match="invalid JSON"):
        ModelQualityTracker(history)


def test_add_run_replaces_history_without_leaving_temporary_file(tmp_path):
    history = tmp_path / "history.json"
    tracker = ModelQualityTracker(history)

    tracker.add_run(ModelMetrics(timestamp="2026-07-11", data_file="data.csv"))

    reloaded = ModelQualityTracker(history)
    assert reloaded.history[0]["data_file"] == "data.csv"
    assert not history.with_suffix(".json.tmp").exists()


def test_extract_metrics_preserves_run_state_and_fails_closed_on_diagnostics():
    metrics = extract_metrics_from_results(
        {
            "timestamp": "2026-07-11T00:00:00Z",
            "run_manifest": {
                "run_id": "run-123",
                "status": "complete",
                "quality_status": "failed",
            },
            "metadata": {"n_time_periods": 52, "n_geos": 3, "channels": ["meta"]},
            "roi": {"meta": {"mean": 2.0, "ci_lower": 1.0, "ci_upper": 3.0}},
        },
        "data.csv",
    )

    assert metrics.run_id == "run-123"
    assert metrics.technical_status == "complete"
    assert metrics.quality_status == "failed"
    assert metrics.convergence_ok is False
    assert metrics.avg_roi_ci_width == 1.0


def test_quality_report_includes_correlated_run_state(tmp_path):
    tracker = ModelQualityTracker(tmp_path / "history.json")
    tracker.add_run(
        ModelMetrics(
            timestamp="2026-07-11",
            data_file="data.csv",
            run_id="run-123",
            technical_status="complete",
            quality_status="passed",
        )
    )

    report = tracker.generate_quality_report()

    assert "Run ID: run-123" in report
    assert "Run status: complete / quality passed" in report
