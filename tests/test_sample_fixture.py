"""Tests for deterministic live-compatibility fixtures."""

from pathlib import Path

import pytest

from mmm.data import load_mmm_data, validate_dataset
from mmm.sample_fixture import build_model_shape_fixture

SOURCE = Path(__file__).parent.parent / "data" / "examples" / "meridian_sample.csv"


@pytest.mark.parametrize(("national", "expected_geos"), [(False, 8), (True, 1)])
def test_build_model_shape_fixture_is_valid_and_exercises_optional_inputs(
    tmp_path, national, expected_geos
):
    destination = tmp_path / "fixture.csv"

    frame = build_model_shape_fixture(SOURCE, destination, national=national)
    dataset = load_mmm_data(destination)
    report = validate_dataset(dataset)

    assert len(frame) == 52 * expected_geos
    assert dataset.n_geos == expected_geos
    assert dataset.n_time_periods == 52
    assert report.passed
    assert dataset.config.organic_channels[0].name == "newsletter"
    assert dataset.config.treatment_columns == ["promotion_treatment"]
    assert dataset.config.control_columns == [
        "competitor_sales_control",
        "sentiment_score_control",
    ]
    video = next(channel for channel in dataset.config.media_channels if channel.name == "video")
    assert video.reach_column == "video_reach"
    assert video.frequency_column == "video_frequency"


def test_fixture_rejects_dimensions_too_small(tmp_path):
    with pytest.raises(ValueError, match="at least 1 geo and 26"):
        build_model_shape_fixture(SOURCE, tmp_path / "fixture.csv", n_periods=8)
