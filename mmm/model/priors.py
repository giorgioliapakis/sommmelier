"""Shared, ordered ROI priors for paid media and reach/frequency channels."""

from typing import Any

from mmm.result_manifest import finite_number


def build_prior_arrays(
    channels: list[str],
    calibration_priors: dict[str, dict[str, Any]],
    default_mean: float = 0.2,
    default_sigma: float = 0.9,
) -> tuple[list[float], list[float], list[str]]:
    """Return LogNormal locations/scales in the model's channel order."""
    means, sigmas, sources = [], [], []
    for channel in channels:
        values = calibration_priors.get(channel, {})
        mean = values.get("roi_mean", default_mean)
        sigma = values.get("roi_sigma", default_sigma)
        if not finite_number(mean) or not finite_number(sigma) or sigma <= 0:
            raise ValueError(f"Invalid LogNormal prior for {channel}")
        means.append(float(mean))
        sigmas.append(float(sigma))
        sources.append(str(values.get("source", "calibration" if values else "default")))
    return means, sigmas, sources


def build_roi_prior(
    media_channels: list[str],
    rf_channels: list[str],
    calibration_priors: dict[str, dict[str, Any]] | None = None,
    *,
    default_mean: float = 0.2,
    default_sigma: float = 0.9,
) -> Any:
    """Construct separate Meridian priors for each execution family."""
    import tensorflow_probability as tfp
    from meridian.model import prior_distribution

    distributions = {}
    for name, channels in (("roi_m", media_channels), ("roi_rf", rf_channels)):
        if channels:
            means, sigmas, _ = build_prior_arrays(
                channels, calibration_priors or {}, default_mean, default_sigma
            )
            distributions[name] = tfp.distributions.LogNormal(means, sigmas)
    return prior_distribution.PriorDistribution(**distributions)
