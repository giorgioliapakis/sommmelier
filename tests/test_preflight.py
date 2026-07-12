"""Tests for the pre-paid-compute data boundary."""

from pathlib import Path

import pytest

from mmm.preflight import preflight_data_path, validate_paid_run_request

EXAMPLES = Path(__file__).parent.parent / "data" / "examples"


def test_preflight_accepts_complete_meridian_sample():
    dataset = preflight_data_path(EXAMPLES / "meridian_sample.csv")

    assert dataset.n_time_periods == 156
    assert dataset.n_geos == 40


def test_preflight_rejects_short_data_before_remote_submission():
    with pytest.raises(ValueError, match="MMM data preflight failed.*8 periods"):
        preflight_data_path(EXAMPLES / "sample_data_extended.csv")


def test_preflight_honors_explicit_population_estimate_fallback(tmp_path):
    import pandas as pd

    frame = pd.read_csv(EXAMPLES / "meridian_sample.csv").drop(columns="population")
    path = tmp_path / "without_population.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="population column is required"):
        preflight_data_path(path)

    dataset = preflight_data_path(path, allow_population_estimates=True)

    assert dataset.config.allow_population_estimates is True


def test_preflight_honors_explicit_impression_estimate_fallback(tmp_path):
    import pandas as pd

    frame = pd.read_csv(EXAMPLES / "meridian_sample.csv").drop(columns="Channel0_impression")
    path = tmp_path / "without_impressions.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Missing impressions"):
        preflight_data_path(path)

    dataset = preflight_data_path(path, allow_impression_estimates=True)

    assert dataset.config.allow_impression_estimates is True


@pytest.mark.parametrize("holdout_weeks", [-1, 79])
def test_paid_run_rejects_invalid_holdout_before_remote_submission(holdout_weeks):
    dataset = preflight_data_path(EXAMPLES / "meridian_sample.csv")

    with pytest.raises(ValueError, match="holdout_weeks"):
        validate_paid_run_request(dataset, holdout_weeks=holdout_weeks)


def test_paid_run_accepts_valid_holdout():
    dataset = preflight_data_path(EXAMPLES / "meridian_sample.csv")

    validate_paid_run_request(dataset, holdout_weeks=12)
