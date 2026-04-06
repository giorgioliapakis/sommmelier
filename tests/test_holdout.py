"""Tests for holdout validation mask generation."""

import numpy as np
import pytest

from mmm.validation.holdout import generate_holdout_mask


class TestGenerateHoldoutMask:
    def test_basic_holdout(self):
        """52-week dataset, holdout last 8 weeks."""
        mask = generate_holdout_mask(n_geos=3, n_periods=52, holdout_weeks=8)
        assert mask.shape == (3, 52)
        assert mask.dtype == bool
        # Last 8 columns should be True
        assert mask[:, -8:].all()
        # First 44 columns should be False
        assert not mask[:, :44].any()

    def test_holdout_all_geos(self):
        """All geos have the same holdout pattern."""
        mask = generate_holdout_mask(n_geos=5, n_periods=52, holdout_weeks=4)
        for g in range(5):
            np.testing.assert_array_equal(mask[0], mask[g])

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
        assert mask[:, -26:].all()
        assert not mask[:, :26].any()

    def test_holdout_count(self):
        """Total held-out observations = n_geos * holdout_weeks."""
        mask = generate_holdout_mask(n_geos=3, n_periods=52, holdout_weeks=8)
        assert mask.sum() == 3 * 8
