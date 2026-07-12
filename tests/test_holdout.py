"""Tests for holdout validation mask generation."""

import numpy as np
import pytest

from mmm.validation.holdout import generate_holdout_mask


class TestGenerateHoldoutMask:
    def test_basic_holdout(self):
        """Holdout observations are balanced instead of forming a trailing block."""
        mask = generate_holdout_mask(n_geos=3, n_periods=52, holdout_weeks=8)
        assert mask.shape == (3, 52)
        assert mask.dtype == bool
        assert (mask.sum(axis=1) == 8).all()
        assert mask.sum(axis=0).max() - mask.sum(axis=0).min() <= 1
        assert not mask[:, -8:].all()

    def test_holdout_patterns_are_staggered_across_geos(self):
        mask = generate_holdout_mask(n_geos=5, n_periods=52, holdout_weeks=4)
        assert any(not np.array_equal(mask[0], mask[g]) for g in range(1, 5))

    def test_no_holdout_raises(self):
        """holdout_weeks=0 raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            generate_holdout_mask(n_geos=3, n_periods=52, holdout_weeks=0)

    def test_negative_holdout_raises(self):
        with pytest.raises(ValueError, match="positive"):
            generate_holdout_mask(n_geos=3, n_periods=52, holdout_weeks=-1)

    def test_too_large_holdout_raises(self):
        """holdout_weeks > n_periods/2 is rejected."""
        with pytest.raises(ValueError, match="Not enough training data"):
            generate_holdout_mask(n_geos=3, n_periods=52, holdout_weeks=30)

    def test_exact_half_is_allowed(self):
        """holdout_weeks == n_periods/2 is the boundary — allowed."""
        mask = generate_holdout_mask(n_geos=2, n_periods=52, holdout_weeks=26)
        assert (mask.sum(axis=1) == 26).all()
        assert mask.sum(axis=0).max() - mask.sum(axis=0).min() <= 1

    def test_holdout_count(self):
        """Total held-out observations = n_geos * holdout_weeks."""
        mask = generate_holdout_mask(n_geos=3, n_periods=52, holdout_weeks=8)
        assert mask.sum() == 3 * 8

    @pytest.mark.parametrize(("n_geos", "n_periods"), [(0, 52), (3, 0)])
    def test_invalid_dimensions_raise(self, n_geos, n_periods):
        with pytest.raises(ValueError, match="n_geos and n_periods"):
            generate_holdout_mask(n_geos=n_geos, n_periods=n_periods, holdout_weeks=1)
