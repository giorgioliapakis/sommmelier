"""CLI contract tests."""

from pathlib import Path

from typer.testing import CliRunner

from mmm.cli.main import app

runner = CliRunner()


def test_validate_returns_nonzero_for_invalid_dataset():
    sample = Path(__file__).parent.parent / "data" / "examples" / "sample_data.csv"

    result = runner.invoke(app, ["validate", str(sample)])

    assert result.exit_code == 1
    assert "Result: FAILED" in result.output


def test_analyze_blocks_results_that_failed_quality_checks(tmp_path):
    results = tmp_path / "results.json"
    results.write_text(
        '{"run_manifest": {"status": "complete", "quality_status": "failed"}}'
    )

    result = runner.invoke(app, ["analyze", str(results)])

    assert result.exit_code == 2
    assert "Recommendations blocked" in result.output
    assert "model quality status is failed" in result.output
