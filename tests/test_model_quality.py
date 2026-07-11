"""Tests for durable model-quality history."""

import pytest

from mmm.tracking.model_quality import ModelMetrics, ModelQualityTracker


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
