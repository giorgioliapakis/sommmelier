"""CLI contract tests."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from mmm.analysis.insights import Insight, InsightPriority, InsightType
from mmm.cli.main import app, find_latest_results, load_json_object
from mmm.model.mmm import ModelResults

runner = CliRunner()


def test_validate_returns_nonzero_for_invalid_dataset():
    sample = Path(__file__).parent.parent / "data" / "examples" / "sample_data.csv"

    result = runner.invoke(app, ["validate", str(sample)])

    assert result.exit_code == 1
    assert "Result: FAILED" in result.output


def test_analyze_blocks_results_that_failed_quality_checks(tmp_path):
    results = tmp_path / "results.json"
    results.write_text('{"run_manifest": {"status": "complete", "quality_status": "failed"}}')

    result = runner.invoke(app, ["analyze", str(results)])

    assert result.exit_code == 2
    assert "Recommendations blocked" in result.output
    assert "model quality status is failed" in result.output


def _decision_ready_results() -> dict[str, object]:
    return {
        "timestamp": "2026-01-02T00:00:00Z",
        "run_manifest": {"status": "complete", "quality_status": "passed"},
        "metadata": {
            "roi_is_monetary": True,
            "n_time_periods": 52,
            "n_geos": 3,
            "channels": ["meta"],
            "total_spend": {"meta": 100.0},
            "estimated_impression_channels": [],
        },
        "roi": {"meta": {"mean": 1.2, "ci_lower": 1.0, "ci_upper": 1.4}},
        "contributions": {"meta": {"absolute": 10.0, "percentage": 50}},
        "marginal_roi": {"meta": 1.1},
        "model_fit": {"r_squared": 0.8, "mape": 0.1},
        "diagnostics": {"diagnostics_available": True, "convergence_ok": True, "rhat_warnings": 0},
        "optimization": {"current": {"optimal_allocation": {"meta": 100.0}}},
    }


def test_analyze_json_is_machine_parseable_without_human_preamble(tmp_path, monkeypatch):
    results = tmp_path / "full_results_current.json"
    results.write_text(json.dumps(_decision_ready_results()))
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["analyze", str(results), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["model_health"]["convergence"] == "good"
    assert "Analyzing:" not in result.output


@pytest.mark.parametrize("command", ["analyze", "report"])
def test_result_commands_report_invalid_json_without_traceback(tmp_path, command):
    results = tmp_path / "broken.json"
    results.write_text("not-json")

    result = runner.invoke(app, [command, str(results)])

    assert result.exit_code == 1
    assert "Invalid JSON" in result.output
    assert "Traceback" not in result.output


def test_report_generates_html_and_uses_safe_file_uri(tmp_path, monkeypatch):
    results = tmp_path / "result with spaces.json"
    results.write_text(json.dumps(_decision_ready_results()))
    opened = Mock(return_value=True)
    monkeypatch.setattr("webbrowser.open", opened)

    result = runner.invoke(app, ["report", str(results), "--open"])

    report_path = results.with_suffix(".html")
    assert result.exit_code == 0
    assert report_path.exists()
    opened.assert_called_once_with(report_path.resolve().as_uri())


def test_find_latest_results_prefers_newest_full_result(tmp_path):
    older = tmp_path / "full_results_old.json"
    newer = tmp_path / "full_results_new.json"
    older.write_text("{}")
    newer.write_text("{}")
    older.touch()
    newer.touch()
    older_mtime = older.stat().st_mtime - 10
    older.chmod(0o644)
    import os

    os.utime(older, (older_mtime, older_mtime))

    assert find_latest_results(tmp_path) == newer
    assert find_latest_results(tmp_path / "missing") is None


def test_load_json_object_rejects_non_object(tmp_path):
    path = tmp_path / "array.json"
    path.write_text("[]")

    with pytest.raises(ValueError, match="Expected a JSON object"):
        load_json_object(path)


def test_load_json_object_reports_read_errors(tmp_path):
    with pytest.raises(ValueError, match="Could not read"):
        load_json_object(tmp_path / "missing.json")


def test_analyze_writes_human_readable_report(tmp_path, monkeypatch):
    results = tmp_path / "full_results_current.json"
    results.write_text(json.dumps(_decision_ready_results()))
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["analyze", str(results)])

    analysis = tmp_path / "analysis_current.md"
    assert result.exit_code == 0
    assert "SOMMMELIER ANALYSIS REPORT" in result.output
    assert analysis.exists()
    assert "MODEL HEALTH" in analysis.read_text()


def test_run_executes_local_model_pipeline(tmp_path, monkeypatch):
    dataset = SimpleNamespace(
        n_time_periods=52,
        n_geos=3,
        summary=Mock(return_value="dataset summary"),
    )
    validation = SimpleNamespace(passed=True, summary=Mock(return_value="valid"))
    fitted = SimpleNamespace(summary=Mock(return_value="fit summary"))

    class FakeAutoMMM:
        def __init__(self, supplied_dataset, config):
            assert supplied_dataset is dataset
            self.config = config

        def prepare(self):
            return None

        def fit(self):
            return fitted

        def save(self, path):
            assert path == tmp_path / "model"

    generated_report = Mock()
    monkeypatch.setattr("mmm.data.load_mmm_data", Mock(return_value=dataset))
    monkeypatch.setattr("mmm.data.validate_dataset", Mock(return_value=validation))
    monkeypatch.setattr("mmm.model.AutoMMM", FakeAutoMMM)
    monkeypatch.setattr("mmm.analysis.reports.generate_report", generated_report)

    result = runner.invoke(
        app,
        ["run", "input.csv", "--output-dir", str(tmp_path), "--n-chains", "2", "--n-keep", "10"],
    )

    assert result.exit_code == 0
    assert "Model prepared" in result.output
    assert "fit summary" in result.output
    generated_report.assert_called_once()


def test_run_stops_before_modeling_when_validation_fails(monkeypatch):
    validation = SimpleNamespace(passed=False, summary=Mock(return_value="invalid dataset"))
    monkeypatch.setattr("mmm.data.load_mmm_data", Mock(return_value=SimpleNamespace()))
    monkeypatch.setattr("mmm.data.validate_dataset", Mock(return_value=validation))

    result = runner.invoke(app, ["run", "input.csv"])

    assert result.exit_code == 1
    assert "invalid dataset" in result.output


def test_quality_renders_summary_and_history(monkeypatch):
    tracker = SimpleNamespace(
        history=[
            {
                "timestamp": "2026-01-01",
                "data_file": "input.csv",
                "n_time_periods": 52,
                "r_squared": 0.8,
                "mape": 0.1,
                "convergence_ok": True,
            }
        ],
        generate_quality_report=Mock(return_value="quality summary"),
    )
    monkeypatch.setattr("mmm.tracking.ModelQualityTracker", Mock(return_value=tracker))

    summary = runner.invoke(app, ["quality"])
    history = runner.invoke(app, ["quality", "--history"])

    assert summary.exit_code == 0
    assert "quality summary" in summary.output
    assert history.exit_code == 0
    assert "2026-01-01" in history.output
    assert "Convergence: OK" in history.output


def test_optimize_rejects_nonpositive_budget_before_loading_model():
    result = runner.invoke(app, ["optimize", "model", "--budget", "0"])

    assert result.exit_code == 1
    assert "Budget must be greater than zero" in result.output


def test_optimize_cli_reports_readiness_failure(monkeypatch):
    from mmm.model.mmm import AutoMMM

    model = AutoMMM.__new__(AutoMMM)
    model._meridian = object()
    model._results = ModelResults()
    model.dataset = SimpleNamespace(total_spend=100.0)
    monkeypatch.setattr(AutoMMM, "load", Mock(return_value=model))
    result = runner.invoke(app, ["optimize", "model"])
    assert result.exit_code == 2
    assert "Recommendations blocked" in result.output


def test_optimize_renders_model_allocation(monkeypatch):
    import pandas as pd

    model = SimpleNamespace(
        dataset=SimpleNamespace(
            total_spend=100.0,
            config=SimpleNamespace(media_channels=[{"name": "meta", "spend_column": "meta_spend"}]),
            df=pd.DataFrame({"meta_spend": [40.0, 60.0]}),
        ),
        optimize_budget=Mock(return_value={"meta": 120.0}),
    )
    monkeypatch.setattr("mmm.model.AutoMMM.load", Mock(return_value=model))

    result = runner.invoke(app, ["optimize", "model", "--budget", "120"])

    assert result.exit_code == 0
    model.optimize_budget.assert_called_once_with(budget=120.0)
    assert "meta" in result.output
    assert "+20.0%" in result.output


def test_insights_blocks_unfitted_model(monkeypatch):
    model = SimpleNamespace(results=None)
    monkeypatch.setattr("mmm.model.AutoMMM.load", Mock(return_value=model))

    result = runner.invoke(app, ["insights", "model"])

    assert result.exit_code == 1
    assert "Model has no results" in result.output


def test_insights_renders_generated_insight(monkeypatch):
    import pandas as pd

    model = SimpleNamespace(
        results=ModelResults(channel_roi={"meta": 1.2}, roi_is_monetary=True),
        dataset=SimpleNamespace(
            config=SimpleNamespace(media_channels=[{"name": "meta", "spend_column": "meta_spend"}]),
            df=pd.DataFrame({"meta_spend": [100.0]}),
        ),
    )
    insight = Insight(
        type=InsightType.EFFICIENCY,
        priority=InsightPriority.MEDIUM,
        channel="meta",
        title="A useful finding",
        description="Evidence",
        recommendation="Act carefully",
    )
    monkeypatch.setattr("mmm.model.AutoMMM.load", Mock(return_value=model))
    monkeypatch.setattr("mmm.analysis.generate_insights", Mock(return_value=[insight]))

    result = runner.invoke(app, ["insights", "model"])

    assert result.exit_code == 0
    assert "A useful finding" in result.output
    assert "Act carefully" in result.output
