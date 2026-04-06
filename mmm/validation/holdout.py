"""Holdout validation for out-of-time model evaluation."""

import numpy as np


def generate_holdout_mask(
    n_geos: int,
    n_periods: int,
    holdout_weeks: int,
) -> np.ndarray:
    """
    Generate a balanced holdout mask for Meridian's ModelSpec.

    The last `holdout_weeks` time periods are held out for all geos,
    giving a clean out-of-time validation split.

    Args:
        n_geos: Number of geographies.
        n_periods: Number of time periods.
        holdout_weeks: Number of trailing weeks to hold out.

    Returns:
        Boolean array of shape (n_geos, n_periods) where True = holdout.

    Raises:
        ValueError: If holdout_weeks is invalid.
    """
    if holdout_weeks <= 0:
        raise ValueError("holdout_weeks must be positive")
    if holdout_weeks > n_periods // 2:
        raise ValueError(
            f"holdout_weeks ({holdout_weeks}) exceeds half the data "
            f"({n_periods // 2}). Not enough training data."
        )

    mask = np.zeros((n_geos, n_periods), dtype=bool)
    mask[:, -holdout_weeks:] = True
    return mask
