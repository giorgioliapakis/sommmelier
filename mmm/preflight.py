"""Shared local preflight for any path that can submit paid model compute."""

from pathlib import Path

import pandas as pd

from mmm.data import load_mmm_data, load_mmm_dataframe, validate_dataset
from mmm.data.schema import DataConfig, MMMDataset
from mmm.data.validator import check_meridian_compatibility
from mmm.validation.holdout import generate_holdout_mask


def preflight_data_path(
    path: Path | str,
    *,
    kpi_column: str = "conversions",
    allow_population_estimates: bool = False,
    allow_impression_estimates: bool = False,
) -> MMMDataset:
    """Validate a data path completely before a Modal remote call is allowed."""
    dataset = load_mmm_data(
        path,
        DataConfig(
            kpi_column=kpi_column,
            allow_population_estimates=allow_population_estimates,
            allow_impression_estimates=allow_impression_estimates,
        ),
    )
    return _require_valid_dataset(dataset)


def preflight_dataframe(
    frame: pd.DataFrame,
    *,
    kpi_column: str = "conversions",
    allow_population_estimates: bool = False,
    allow_impression_estimates: bool = False,
) -> MMMDataset:
    """Validate an in-memory frame before model construction or paid compute."""
    dataset = load_mmm_dataframe(
        frame,
        DataConfig(
            kpi_column=kpi_column,
            allow_population_estimates=allow_population_estimates,
            allow_impression_estimates=allow_impression_estimates,
        ),
    )
    return _require_valid_dataset(dataset)


def _require_valid_dataset(dataset: MMMDataset) -> MMMDataset:
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


def validate_paid_run_request(dataset: MMMDataset, *, holdout_weeks: int = 0) -> None:
    """Reject invalid paid-run options before any remote function is invoked."""
    if holdout_weeks < 0:
        raise ValueError("holdout_weeks cannot be negative")
    if holdout_weeks:
        generate_holdout_mask(
            n_geos=dataset.n_geos,
            n_periods=dataset.n_time_periods,
            holdout_weeks=holdout_weeks,
        )
