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
