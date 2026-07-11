"""Tests for recommendation safety and optimizer handoff."""

from mmm.recommendations.engine import analyze_roi, calculate_budget_reallocation


def test_non_monetary_roi_is_not_compared_to_breakeven():
    recommendations = analyze_roi(
        {
            "metadata": {"roi_is_monetary": False},
            "roi": {"meta": {"mean": 0.2}, "google": {"mean": 0.05}},
        }
    )

    assert len(recommendations) == 1
    assert "not a monetary return" in recommendations[0].title
    assert "pause" not in recommendations[0].action.lower()


def test_budget_reallocation_uses_meridian_optimizer_output():
    result = calculate_budget_reallocation(
        {
            "metadata": {"total_spend": {"meta": 100.0, "google": 100.0}},
            "marginal_roi": {"meta": 100.0, "google": 1.0},
            "optimization": {
                "current": {
                    "optimal_allocation": {"meta": 80.0, "google": 120.0},
                }
            },
        }
    )

    assert result["suggested"] == {"meta": 80.0, "google": 120.0}
    assert result["change"] == {"meta": -20.0, "google": 20.0}


def test_budget_reallocation_does_not_invent_allocation_without_optimizer():
    result = calculate_budget_reallocation(
        {
            "metadata": {"total_spend": {"meta": 100.0, "google": 100.0}},
            "marginal_roi": {"meta": 2.0, "google": 1.0},
        }
    )

    assert result["suggested"] == {}
