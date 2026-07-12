"""Tests for calibration-to-prior numerical correctness."""

import math

import pytest

from mmm.calibration.calibration_data import (
    CalibrationData,
    ExperimentResult,
    PlatformConversions,
    PriorBelief,
    calculate_channel_priors,
    create_calibration_template,
    experiment_to_prior,
    infer_calibration_metric,
    load_calibration,
    save_calibration,
)


def test_platform_prior_location_is_converted_to_log_space():
    calibration = CalibrationData(
        platform_conversions=[
            PlatformConversions(
                channel="meta",
                platform_conversions=100,
                period_weeks=4,
                spend=100,
                metric="incremental_kpi_per_currency",
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
                metric="incremental_kpi_per_currency",
            )
        ],
        prior_beliefs=[
            PriorBelief(
                channel="meta",
                expected_roi_low=1.0,
                expected_roi_high=1.0,
                metric="incremental_kpi_per_currency",
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
        metric="incremental_kpi_per_currency",
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
                metric="incremental_kpi_per_currency",
            )
        ]
    )

    with pytest.raises(ValueError, match="requires positive spend"):
        calculate_channel_priors(calibration)


def test_rejects_mixed_calibration_metrics():
    calibration = CalibrationData(
        prior_beliefs=[
            PriorBelief(
                channel="meta",
                expected_roi_low=1.0,
                expected_roi_high=2.0,
                metric="monetary_roi",
            )
        ],
        platform_conversions=[
            PlatformConversions(
                channel="search",
                platform_conversions=100,
                period_weeks=4,
                spend=100,
                metric="incremental_kpi_per_currency",
            )
        ],
    )

    with pytest.raises(ValueError, match="mix incompatible metrics"):
        calculate_channel_priors(calibration)


def test_rejects_calibration_that_does_not_match_model_outcome():
    calibration = CalibrationData(
        prior_beliefs=[
            PriorBelief(
                channel="meta",
                expected_roi_low=1.0,
                expected_roi_high=2.0,
                metric="monetary_roi",
            )
        ]
    )

    with pytest.raises(ValueError, match="does not match the model outcome"):
        calculate_channel_priors(calibration, expected_metric="incremental_kpi_per_currency")


def test_infers_calibration_metric_from_model_columns():
    assert infer_calibration_metric(["date", "conversions"], "conversions") == (
        "incremental_kpi_per_currency"
    )
    assert (
        infer_calibration_metric(["date", "conversions", "revenue_per_conversion"], "conversions")
        == "monetary_roi"
    )
    assert infer_calibration_metric(["date", "revenue"], "revenue") == "monetary_roi"


def test_calibration_metric_round_trips(tmp_path):
    calibration = CalibrationData(
        prior_beliefs=[
            PriorBelief(
                channel="meta",
                expected_roi_low=1.0,
                expected_roi_high=2.0,
                metric="monetary_roi",
            )
        ]
    )
    path = tmp_path / "calibration.json"

    save_calibration(calibration, path)

    assert load_calibration(path) == calibration


def test_legacy_ambiguous_calibration_is_rejected(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(
        '{"prior_beliefs": [{"channel": "meta", "expected_roi_low": 1, "expected_roi_high": 2}]}'
    )

    with pytest.raises(ValueError, match="must declare metric"):
        load_calibration(path)


@pytest.mark.parametrize(
    ("low", "high"),
    [(0.0, 1.0), (2.0, 1.0)],
)
def test_prior_range_must_be_positive_and_ordered(low, high):
    with pytest.raises(ValueError, match="positive and ordered"):
        PriorBelief(
            channel="meta",
            expected_roi_low=low,
            expected_roi_high=high,
            metric="monetary_roi",
        )


def test_calibration_metric_must_be_supported():
    with pytest.raises(ValueError, match="Calibration metric must be"):
        PriorBelief(
            channel="meta",
            expected_roi_low=1.0,
            expected_roi_high=2.0,
            metric="roas",  # type: ignore[arg-type]
        )


def test_generated_template_cannot_apply_fake_priors(tmp_path):
    path = tmp_path / "calibration.json"

    create_calibration_template(path)
    calibration = load_calibration(path)

    assert calculate_channel_priors(calibration) == {}
    assert "Add only measured calibration data" in calibration.notes
