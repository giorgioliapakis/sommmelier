"""Tests for channel auto-detection logic (R&F, organic, treatment)."""

import pandas as pd
import pytest


def detect_channels(df: pd.DataFrame):
    """
    Replicate the auto-detection logic from modal_mmm_full.py for testing.
    Returns dicts of detected channel types.
    """
    spend_cols_all = [col for col in df.columns if '_spend' in col.lower()]

    si_channels = []
    rf_channels = []
    organic_channels = []
    treatment_cols = []
    control_cols = []

    for spend_col in spend_cols_all:
        ch = spend_col.replace('_spend', '').replace('_Spend', '')
        reach_col = next((c for c in df.columns if c.lower() == f"{ch.lower()}_reach"), None)
        freq_col = next((c for c in df.columns if c.lower() == f"{ch.lower()}_frequency"), None)

        if reach_col and freq_col:
            rf_channels.append(ch)
        else:
            si_channels.append(ch)

    organic_cols = [col for col in df.columns if col.lower().endswith('_organic')]
    organic_channels = [col.rsplit('_organic', 1)[0] for col in organic_cols]

    treatment_cols = [col for col in df.columns if col.lower().endswith('_treatment')]
    treatment_names = [col.rsplit('_treatment', 1)[0] for col in treatment_cols]

    control_cols = [col for col in df.columns if '_control' in col.lower()]
    control_cols = [c for c in control_cols if c not in treatment_cols]

    return {
        "si_channels": si_channels,
        "rf_channels": rf_channels,
        "organic_channels": organic_channels,
        "treatment_names": treatment_names,
        "control_cols": control_cols,
    }


class TestChannelDetection:
    def test_mixed_si_and_rf(self):
        """3 channels: 1 has R&F, 2 are spend+impressions."""
        df = pd.DataFrame({
            "meta_spend": [100], "meta_impressions": [1000],
            "youtube_spend": [200], "youtube_reach": [5000], "youtube_frequency": [3.2],
            "google_spend": [150], "google_impressions": [1500],
        })
        result = detect_channels(df)
        assert "youtube" in result["rf_channels"]
        assert "meta" in result["si_channels"]
        assert "google" in result["si_channels"]
        assert len(result["rf_channels"]) == 1

    def test_no_rf_columns(self):
        """No R&F columns — all spend+impressions. Backward compatible."""
        df = pd.DataFrame({
            "meta_spend": [100], "meta_impressions": [1000],
            "google_spend": [150], "google_impressions": [1500],
        })
        result = detect_channels(df)
        assert len(result["rf_channels"]) == 0
        assert len(result["si_channels"]) == 2

    def test_reach_without_frequency(self):
        """Channel with _reach but no _frequency — treated as spend+impressions."""
        df = pd.DataFrame({
            "meta_spend": [100], "meta_reach": [5000],
        })
        result = detect_channels(df)
        assert "meta" in result["si_channels"]
        assert len(result["rf_channels"]) == 0

    def test_organic_detection(self):
        """Columns ending in _organic are detected."""
        df = pd.DataFrame({
            "meta_spend": [100],
            "newsletter_organic": [500],
            "blog_organic": [300],
        })
        result = detect_channels(df)
        assert "newsletter" in result["organic_channels"]
        assert "blog" in result["organic_channels"]

    def test_no_organic(self):
        """No organic columns — empty list."""
        df = pd.DataFrame({"meta_spend": [100]})
        result = detect_channels(df)
        assert result["organic_channels"] == []

    def test_treatment_detection(self):
        """Columns ending in _treatment are detected."""
        df = pd.DataFrame({
            "meta_spend": [100],
            "pricing_treatment": [0.9],
            "promotion_discount_treatment": [0.2],
        })
        result = detect_channels(df)
        assert "pricing" in result["treatment_names"]
        assert "promotion_discount" in result["treatment_names"]

    def test_no_treatments(self):
        """No treatment columns — backward compatible."""
        df = pd.DataFrame({"meta_spend": [100], "holiday_control": [1]})
        result = detect_channels(df)
        assert result["treatment_names"] == []

    def test_control_not_confused_with_treatment(self):
        """Control columns are separate from treatment columns."""
        df = pd.DataFrame({
            "meta_spend": [100],
            "weather_control": [72],
            "pricing_treatment": [0.9],
        })
        result = detect_channels(df)
        assert "weather_control" in result["control_cols"]
        assert "pricing_treatment" not in result["control_cols"]
        assert "pricing" in result["treatment_names"]
