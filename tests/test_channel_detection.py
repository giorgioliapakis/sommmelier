"""Tests for production column detection logic."""

from mmm.detection import detect_columns


def test_detects_spend_impressions_and_reach_frequency_channels():
    detected = detect_columns(
        [
            "meta_spend",
            "meta_impressions",
            "YouTube_Spend",
            "YouTube_Reach",
            "YouTube_Frequency",
        ]
    )

    assert detected.media_channels == (
        {
            "name": "meta",
            "spend_column": "meta_spend",
            "impressions_column": "meta_impressions",
            "reach_column": None,
            "frequency_column": None,
        },
        {
            "name": "YouTube",
            "spend_column": "YouTube_Spend",
            "impressions_column": None,
            "reach_column": "YouTube_Reach",
            "frequency_column": "YouTube_Frequency",
        },
    )


def test_spend_must_be_a_suffix():
    detected = detect_columns(["meta_spend_control", "spend_note"])

    assert detected.media_channels == ()


def test_detects_organic_treatments_and_documented_controls():
    detected = detect_columns(
        [
            "newsletter_organic",
            "pricing_treatment",
            "competitor_control",
            "is_holiday",
            "product_launch",
            "is_promotion",
        ]
    )

    assert detected.organic_columns == ("newsletter_organic",)
    assert detected.treatment_columns == ("pricing_treatment",)
    assert detected.control_columns == (
        "competitor_control",
        "is_holiday",
        "product_launch",
        "is_promotion",
    )
