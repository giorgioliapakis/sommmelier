"""Balanced holdout sampling for Meridian model evaluation."""

import numpy as np


def generate_holdout_mask(
    n_geos: int,
    n_periods: int,
    holdout_weeks: int,
) -> np.ndarray:
    """
    Generate a balanced holdout mask for Meridian's ModelSpec.

    Holdout observations are distributed across both geographies and time.
    Meridian is not a KPI forecaster, so contiguous trailing windows can distort
    knot-based trend estimation and are deliberately avoided.

    Args:
        n_geos: Number of geographies.
        n_periods: Number of time periods.
        holdout_weeks: Number of observations to hold out per geography.

    Returns:
        Boolean array of shape (n_geos, n_periods) where True = holdout.

    Raises:
        ValueError: If holdout_weeks is invalid.
    """
    if n_geos <= 0 or n_periods <= 0:
        raise ValueError("n_geos and n_periods must be positive")
    if holdout_weeks <= 0:
        raise ValueError("holdout_weeks must be positive")
    if holdout_weeks > n_periods // 2:
        raise ValueError(
            f"holdout_weeks ({holdout_weeks}) exceeds half the data "
            f"({n_periods // 2}). Not enough training data."
        )

    mask = np.zeros((n_geos, n_periods), dtype=bool)
    periods = np.arange(holdout_weeks)
    for geo_index in range(n_geos):
        phase = (geo_index + 0.5) / n_geos
        holdout_indices = np.floor((periods + phase) * n_periods / holdout_weeks).astype(int)
        mask[geo_index, holdout_indices] = True
    return mask
