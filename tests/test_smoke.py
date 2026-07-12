"""Tests for Modal smoke artifact verification."""

import json
from pathlib import Path

import pytest

from mmm.smoke import EXPECTED_CHARTS, latest_result, verify_modal_smoke


def _write_smoke_result(tmp_path: Path) -> Path:
    outputs = tmp_path / "outputs"
    charts_dir = outputs / "charts_run"
    charts_dir.mkdir(parents=True)
    charts = {}
    for index, name in enumerate(sorted(EXPECTED_CHARTS)):
        chart = charts_dir / f"{name}.png"
        chart.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([index]))
        charts[name] = str(chart)

    result = outputs / "full_results_run.json"
    result.write_text(
        json.dumps(
            {
                "run_manifest": {
                    "status": "complete",
                    "quality_status": "failed",
                    "required_sections": ["roi", "diagnostics"],
                    "sections": {"roi": "complete", "diagnostics": "complete"},
                    "errors": [],
                },
                "charts": charts,
                "optimization": {"current": {"optimal_allocation": {"meta": 100.0}}},
            }
        )
    )
    result.with_suffix(".html").write_text("<html>report</html>")
    return result


def test_verify_modal_smoke_accepts_complete_under_sampled_run(tmp_path):
    result = _write_smoke_result(tmp_path)

    verified = verify_modal_smoke(result)

    assert verified["run_manifest"]["quality_status"] == "failed"
    assert latest_result(result.parent) == result


def test_verify_modal_smoke_rejects_duplicate_charts(tmp_path):
    result = _write_smoke_result(tmp_path)
    data = json.loads(result.read_text())
    chart_paths = list(data["charts"].values())
    data["charts"][sorted(EXPECTED_CHARTS)[-1]] = chart_paths[0]
    result.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="duplicate chart"):
        verify_modal_smoke(result)


def test_latest_result_requires_an_artifact(tmp_path):
    with pytest.raises(ValueError, match="No full_results"):
        latest_result(tmp_path)
