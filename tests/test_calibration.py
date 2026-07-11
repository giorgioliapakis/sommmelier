"""Tests for calibration-to-prior numerical correctness."""

import math

import pytest

from mmm.calibration.calibration_data import (
    CalibrationData,
    ExperimentResult,
    PlatformConversions,
    PriorBelief,
    calculate_channel_priors,
    experiment_to_prior,
)


def test_platform_prior_location_is_converted_to_log_space():
    calibration = CalibrationData(
        platform_conversions=[
            PlatformConversions(
                channel="meta",
                platform_conversions=100,
                period_weeks=4,
                spend=100,
            )
        ]
    )

    prior = calculate_channel_priors(calibration)["meta"]

    assert prior["roi_mean"] == pytest.approx(math.log(0.7))


def test_platform_and_belief_means_are_combined_in_log_space():
    calibration = CalibrationData(
        platform_conversions=[
            PlatformConversions(
                channel="meta",
                platform_conversions=200,
                period_weeks=4,
                spend=100,
            )
        ],
        prior_beliefs=[
            PriorBelief(
                channel="meta",
                expected_roi_low=1.0,
                expected_roi_high=1.0,
            )
        ],
    )

    prior = calculate_channel_priors(calibration)["meta"]

    assert prior["roi_mean"] == pytest.approx((math.log(1.0) + math.log(1.0)) / 2)


def test_lift_without_outcome_count_is_not_treated_as_roi():
    experiment = ExperimentResult(
        channel="meta",
        experiment_type="geo_lift",
        lift_estimate=0.15,
        test_spend=50_000,
    )

    with pytest.raises(ValueError, match="lift alone is not an ROI"):
        experiment_to_prior(experiment)

    assert calculate_channel_priors(CalibrationData(experiments=[experiment])) == {}


def test_platform_calibration_requires_positive_spend():
    calibration = CalibrationData(
        platform_conversions=[
            PlatformConversions(
                channel="meta",
                platform_conversions=100,
                period_weeks=4,
                spend=0,
            )
        ]
    )

    with pytest.raises(ValueError, match="requires positive spend"):
        calculate_channel_priors(calibration)
