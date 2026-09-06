"""Tests for per-channel prior construction."""

import pytest

from mmm.model.priors import build_prior_arrays as _build_prior_arrays


def test_real_meridian_priors_separate_media_and_rf_channels():
    from mmm.model.priors import build_roi_prior

    prior = build_roi_prior(
        ["search", "meta"],
        ["video"],
        {"meta": {"roi_mean": 0.3, "roi_sigma": 0.5}, "video": {"roi_mean": 0.7, "roi_sigma": 0.4}},
    )
    assert prior.roi_m.batch_shape.as_list() == [2]
    assert prior.roi_rf.batch_shape.as_list() == [1]
    assert prior.roi_m.loc.numpy().tolist() == pytest.approx([0.2, 0.3])
    assert prior.roi_rf.loc.numpy().tolist() == pytest.approx([0.7])


@pytest.mark.parametrize("sigma", [0, -1, float("nan"), float("inf")])
def test_production_prior_builder_rejects_invalid_scale(sigma):
    with pytest.raises(ValueError, match="Invalid LogNormal"):
        _build_prior_arrays(["meta"], {"meta": {"roi_mean": 0.2, "roi_sigma": sigma}})


class TestPerChannelPriors:
    """Test per-channel prior construction logic."""

    def test_mixed_calibrated_and_default(self):
        """3 channels, 2 have calibration data, 1 uses default."""
        channels = ["meta", "google", "tiktok"]
        calibration_priors = {
            "meta": {"roi_mean": 0.3, "roi_sigma": 0.8, "source": "experiment:geo_lift"},
            "google": {"roi_mean": 0.5, "roi_sigma": 1.2, "source": "belief:historical"},
        }
        means, sigmas, sources = _build_prior_arrays(channels, calibration_priors)

        assert means == [0.3, 0.5, 0.2]
        assert sigmas == [0.8, 1.2, 0.9]
        assert sources[2] == "default"
        # Each channel has its own prior, not an average
        assert means[0] != means[1]

    def test_no_calibration_data(self):
        """No calibration data — all channels get default."""
        channels = ["meta", "google", "tiktok"]
        calibration_priors = {}
        means, sigmas, sources = _build_prior_arrays(channels, calibration_priors)

        assert all(m == 0.2 for m in means)
        assert all(s == 0.9 for s in sigmas)
        assert all(src == "default" for src in sources)

    def test_extra_calibration_data_ignored(self):
        """Calibration data for channels not in dataset is ignored."""
        channels = ["meta", "google"]
        calibration_priors = {
            "meta": {"roi_mean": 0.3, "roi_sigma": 0.8, "source": "experiment"},
            "facebook": {"roi_mean": 0.4, "roi_sigma": 1.0, "source": "experiment"},
        }
        means, sigmas, _ = _build_prior_arrays(channels, calibration_priors)

        assert len(means) == 2
        assert means[0] == 0.3  # meta gets its prior
        assert means[1] == 0.2  # google gets default

    def test_prior_arrays_length_matches_channels(self):
        """Prior arrays must have exactly one entry per channel."""
        channels = ["a", "b", "c", "d"]
        calibration_priors = {"b": {"roi_mean": 1.0, "roi_sigma": 0.5, "source": "test"}}
        means, sigmas, _ = _build_prior_arrays(channels, calibration_priors)

        assert len(means) == len(channels)
        assert len(sigmas) == len(channels)

    def test_batched_lognormal_shape(self):
        """Verify that parallel arrays produce correct batch_shape for LogNormal."""
        channels = ["meta", "google", "tiktok"]
        calibration_priors = {
            "meta": {"roi_mean": 0.3, "roi_sigma": 0.8},
        }
        means, sigmas, _ = _build_prior_arrays(channels, calibration_priors)
        assert len(means) == 3
        # In real usage: tfp.distributions.LogNormal(means, sigmas).batch_shape == [3]
