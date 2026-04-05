"""Tests for per-channel prior construction."""

import pytest


def _build_prior_list(channels: list[str], calibration_priors: dict, default_mean=0.2, default_sigma=0.9):
    """Replicate the per-channel prior logic from modal_mmm_full.py for testing."""
    roi_m_list = []
    for ch in channels:
        if ch in calibration_priors:
            p = calibration_priors[ch]
            roi_m_list.append({"mean": p["roi_mean"], "sigma": p["roi_sigma"], "source": p.get("source", "calibration")})
        else:
            roi_m_list.append({"mean": default_mean, "sigma": default_sigma, "source": "default"})
    return roi_m_list


class TestPerChannelPriors:
    """Test per-channel prior construction logic."""

    def test_mixed_calibrated_and_default(self):
        """3 channels, 2 have calibration data, 1 uses default."""
        channels = ["meta", "google", "tiktok"]
        calibration_priors = {
            "meta": {"roi_mean": 0.3, "roi_sigma": 0.8, "source": "experiment:geo_lift"},
            "google": {"roi_mean": 0.5, "roi_sigma": 1.2, "source": "belief:historical"},
        }
        result = _build_prior_list(channels, calibration_priors)

        assert result[0]["mean"] == 0.3
        assert result[0]["sigma"] == 0.8
        assert result[1]["mean"] == 0.5
        assert result[1]["sigma"] == 1.2
        assert result[2]["mean"] == 0.2  # default
        assert result[2]["sigma"] == 0.9  # default
        # Each channel has its own prior, not an average
        assert result[0]["mean"] != result[1]["mean"]

    def test_no_calibration_data(self):
        """No calibration data — all channels get default."""
        channels = ["meta", "google", "tiktok"]
        calibration_priors = {}
        result = _build_prior_list(channels, calibration_priors)

        for entry in result:
            assert entry["mean"] == 0.2
            assert entry["sigma"] == 0.9
            assert entry["source"] == "default"

    def test_extra_calibration_data_ignored(self):
        """Calibration data for channels not in dataset is ignored."""
        channels = ["meta", "google"]
        calibration_priors = {
            "meta": {"roi_mean": 0.3, "roi_sigma": 0.8, "source": "experiment"},
            "facebook": {"roi_mean": 0.4, "roi_sigma": 1.0, "source": "experiment"},  # not in channels
        }
        result = _build_prior_list(channels, calibration_priors)

        assert len(result) == 2
        assert result[0]["mean"] == 0.3  # meta gets its prior
        assert result[1]["mean"] == 0.2  # google gets default

    def test_prior_list_length_matches_channels(self):
        """Prior list must have exactly one entry per channel."""
        channels = ["a", "b", "c", "d"]
        calibration_priors = {"b": {"roi_mean": 1.0, "roi_sigma": 0.5, "source": "test"}}
        result = _build_prior_list(channels, calibration_priors)

        assert len(result) == len(channels)
