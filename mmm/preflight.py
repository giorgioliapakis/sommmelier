"""Shared local preflight for any path that can submit paid model compute."""

from pathlib import Path

from mmm.data import load_mmm_data, validate_dataset
from mmm.data.schema import DataConfig, MMMDataset
from mmm.data.validator import check_meridian_compatibility


def preflight_data_path(path: Path | str, *, kpi_column: str = "conversions") -> MMMDataset:
    """Validate a data path completely before a Modal remote call is allowed."""
    dataset = load_mmm_data(path, DataConfig(kpi_column=kpi_column))
    report = validate_dataset(dataset)
    compatibility_issues = check_meridian_compatibility(dataset)
    if not report.passed or compatibility_issues:
        problems = [
            result.message
            for result in report.results
            if not result.passed and result.severity == "error"
        ]
        problems.extend(compatibility_issues)
        raise ValueError("MMM data preflight failed: " + "; ".join(problems))
    return dataset
