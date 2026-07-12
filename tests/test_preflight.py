"""Tests for the pre-paid-compute data boundary."""

from pathlib import Path

import pytest

from mmm.preflight import preflight_data_path

EXAMPLES = Path(__file__).parent.parent / "data" / "examples"


def test_preflight_accepts_complete_meridian_sample():
    dataset = preflight_data_path(EXAMPLES / "meridian_sample.csv")

    assert dataset.n_time_periods == 156
    assert dataset.n_geos == 40


def test_preflight_rejects_short_data_before_remote_submission():
    with pytest.raises(ValueError, match="MMM data preflight failed.*8 periods"):
        preflight_data_path(EXAMPLES / "sample_data_extended.csv")
